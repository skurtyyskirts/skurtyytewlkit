"""
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from __future__ import annotations

__all__ = [
    "ScatterBrushController",
    "ScatterMode",
    "destroy_scatter_brush_controller",
    "get_scatter_brush_controller",
]

import os
import secrets
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING

import carb
import carb.settings
import omni.kit.commands
import omni.kit.undo
import omni.usd
from omni.flux.utils.common import Event, EventSubscription
from pxr import Gf, Sdf, Usd, UsdGeom
from pydantic import ValidationError

from .assets import read_asset_up_axis
from .picking import create_surface_picker
from .placement import existing_placement_points, generate_flood
from .presets import PresetStore, get_default_presets_directory
from .sampling import PaddingIndex, stage_up_axis, stamp_rng
from .settings import (
    ScatterAssetEntry,
    ScatterBrushSettings,
    TargetMode,
    UpAxis,
    load_assets_from_carb,
    save_assets_to_carb,
)
from .targets import (
    ScatterTarget,
    canonical_instance,
    find_capture_mesh,
    instance_count,
    selected_prototypes,
    validated_anchor_prototype,
)

if TYPE_CHECKING:
    from .geometry import MeshSurfaceCache
    from .picking import SurfacePicker

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"


class ScatterMode(Enum):
    """Interaction mode of the viewport brush tool."""

    OFF = 0
    PAINT = 1
    ERASE = 2


def _normalize_asset_path(path: str) -> str:
    """Return the asset path trimmed and with forward slashes."""
    return str(path).strip().replace("\\", "/")


def _asset_key(path: str) -> str:
    """Return the comparison key used to detect duplicate asset paths on the current platform."""
    return os.path.normcase(os.path.normpath(_normalize_asset_path(path)))


def _describe_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line suitable for a status message."""
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


class ScatterBrushController:
    """Process-wide brush state shared by the Scatter window and the viewport brush tool.

    Owns the validated settings, the asset palette, the preset store, the paint/erase mode and the stroke counter,
    persists settings and assets through carb, and publishes every change through events. Use it from the main loop
    only; it is not thread-safe.
    """

    def __init__(self):
        self._settings = ScatterBrushSettings.load_from_carb()
        self._assets = load_assets_from_carb()
        self._presets = PresetStore(get_default_presets_directory())
        self._mode = ScatterMode.OFF
        self._stroke_index = 0
        self._picker_factory: Callable[[object, MeshSurfaceCache], SurfacePicker] | None = None

        self.__on_mode_changed = Event(copy=True)
        self.__on_settings_changed = Event(copy=True)
        self.__on_assets_changed = Event(copy=True)
        self.__on_stroke_committed = Event(copy=True)
        self.__on_status_message = Event(copy=True)

    @property
    def settings(self) -> ScatterBrushSettings:
        """Current brush settings. Mutate through ``update_settings`` or ``replace_settings``."""
        return self._settings

    @property
    def assets(self) -> list[ScatterAssetEntry]:
        """Copy of the asset palette in palette order."""
        return [entry.model_copy() for entry in self._assets]

    @property
    def mode(self) -> ScatterMode:
        """Current interaction mode."""
        return self._mode

    def set_mode(self, mode: ScatterMode) -> None:
        """Switch the interaction mode and notify subscribers when it actually changes."""
        if mode == self._mode:
            return
        self._mode = mode
        self.__on_mode_changed(mode)

    def toggle_paint(self) -> None:
        """Toggle between off and paint; erase mode turns off."""
        self.set_mode(ScatterMode.PAINT if self._mode == ScatterMode.OFF else ScatterMode.OFF)

    def update_settings(self, **kwargs) -> bool:
        """Apply field updates after validating the complete resulting settings.

        Returns False and posts an error status, leaving the settings untouched, when a field is unknown or the merged
        settings fail validation. On success the settings are persisted and ``settings_changed`` fires.
        """
        unknown = sorted(set(kwargs) - set(ScatterBrushSettings.model_fields))
        if unknown:
            self.post_status(f"Unknown brush setting: {', '.join(unknown)}", is_error=True)
            return False
        try:
            validated = ScatterBrushSettings.model_validate({**self._settings.model_dump(), **kwargs})
        except ValidationError as exc:
            self.post_status(f"Invalid brush setting: {_describe_validation_error(exc)}", is_error=True)
            return False
        self._set_settings(validated)
        return True

    def replace_settings(self, settings: ScatterBrushSettings) -> None:
        """Replace the whole settings object with a copy of the given one, persist it and notify subscribers."""
        self._set_settings(settings.model_copy(deep=True))

    def add_asset(self, path: str, up_axis: UpAxis | None = None) -> bool:
        """Append an ingested asset to the palette.

        Paths are compared after normalization so the same file cannot be added twice. When no up axis is given it is
        read from the asset file. Returns False when the path is empty or already present.
        """
        normalized = _normalize_asset_path(path)
        if not normalized:
            self.post_status("Cannot add an empty asset path to the brush", is_error=True)
            return False
        if self._find_asset(normalized) is not None:
            self.post_status(f"{normalized} is already in the brush")
            return False
        if up_axis is None:
            up_axis = read_asset_up_axis(normalized)
        self._assets.append(ScatterAssetEntry(path=normalized, up_axis=up_axis))
        self._assets_changed()
        return True

    def remove_asset(self, path: str) -> bool:
        """Remove the palette entry matching the path. Returns False when there is none."""
        entry = self._find_asset(path)
        if entry is None:
            return False
        self._assets.remove(entry)
        self._assets_changed()
        return True

    def set_asset_enabled(self, path: str, enabled: bool) -> bool:
        """Enable or disable a palette entry. Returns False when the path is unknown."""
        return self._update_asset(path, "enabled", enabled)

    def set_asset_weight(self, path: str, weight: float) -> bool:
        """Set the selection weight of a palette entry. Returns False when the path is unknown or the weight invalid."""
        return self._update_asset(path, "weight", weight)

    def set_asset_up_axis(self, path: str, up_axis: UpAxis) -> bool:
        """Override the authored up axis of a palette entry. Returns False when the path is unknown."""
        return self._update_asset(path, "up_axis", up_axis)

    def enabled_assets(self) -> list[ScatterAssetEntry]:
        """Copy of the enabled palette entries in palette order."""
        return [entry.model_copy() for entry in self._assets if entry.enabled]

    def preset_names(self) -> list[str]:
        """Sorted names of the stored presets."""
        return self._presets.list_names()

    def apply_preset(self, name: str) -> bool:
        """Load a preset and make it the current settings. Returns False and posts an error when it cannot be loaded."""
        try:
            settings = self._presets.load(name)
        except ValueError as exc:
            self.post_status(f"Cannot load preset '{name}': {exc}", is_error=True)
            return False
        self.replace_settings(settings)
        return True

    def save_preset(self, name: str | None = None) -> bool:
        """Store the current settings as a preset, under ``name`` or the current preset name.

        A new name becomes the current ``preset_name`` once the file is written. Returns False and posts an error when
        the name is invalid or the file cannot be written.
        """
        preset_name = name or self._settings.preset_name
        updated = self._settings.model_copy(update={"preset_name": preset_name})
        try:
            self._presets.save(preset_name, updated)
        except (ValueError, OSError) as exc:
            self.post_status(f"Cannot save preset '{preset_name}': {exc}", is_error=True)
            return False
        if preset_name != self._settings.preset_name:
            self._set_settings(updated)
        return True

    def rename_preset(self, old: str, new: str) -> bool:
        """Rename a stored preset, following the current ``preset_name`` when it matches."""
        try:
            self._presets.rename(old, new)
        except (ValueError, OSError) as exc:
            self.post_status(f"Cannot rename preset '{old}': {exc}", is_error=True)
            return False
        if self._settings.preset_name == old:
            self._set_settings(self._settings.model_copy(update={"preset_name": new}))
        return True

    def clone_preset(self, src: str, dst: str) -> bool:
        """Duplicate a stored preset under a new name."""
        try:
            self._presets.clone(src, dst)
        except (ValueError, OSError) as exc:
            self.post_status(f"Cannot clone preset '{src}': {exc}", is_error=True)
            return False
        return True

    def delete_preset(self, name: str) -> bool:
        """Delete a stored preset; a missing preset is not an error."""
        try:
            self._presets.delete(name)
        except (ValueError, OSError) as exc:
            self.post_status(f"Cannot delete preset '{name}': {exc}", is_error=True)
            return False
        return True

    def set_anchor_from_selection(self, usd_context_name: str) -> bool:
        """Use the first selected prototype of the USD context as the anchor prototype.

        Returns False and posts an error when the selection contains no captured mesh or instance.
        """
        prototypes = selected_prototypes(omni.usd.get_context(usd_context_name))
        if not prototypes:
            self.post_status("Select a captured mesh or instance to use as the anchor", is_error=True)
            return False
        anchor = min(prototypes, key=str)
        return self.update_settings(anchor_prototype_path=str(anchor))

    def next_stroke(self) -> tuple[int, int]:
        """Return ``(seed, stroke_index)`` for a new stroke and advance the stroke counter.

        The seed is the configured one unless ``randomize_seed`` is set, in which case a fresh 31-bit value is drawn.
        """
        seed = secrets.randbits(31) if self._settings.randomize_seed else self._settings.seed
        stroke_index = self._stroke_index
        self._stroke_index += 1
        return seed, stroke_index

    def create_picker(self, viewport_api, cache: MeshSurfaceCache) -> SurfacePicker:
        """Build the surface picker for a viewport, through the factory override when one is set."""
        if self._picker_factory is not None:
            return self._picker_factory(viewport_api, cache)
        return create_surface_picker(viewport_api, cache)

    def set_picker_factory(self, factory: Callable[[object, MeshSurfaceCache], SurfacePicker] | None) -> None:
        """Override how ``create_picker`` builds pickers, or restore the default with None."""
        self._picker_factory = factory

    def flood(
        self,
        usd_context_name: str,
        cache: MeshSurfaceCache,
        targets: Sequence[ScatterTarget] | None = None,
    ) -> int:
        """Fill whole target surfaces with placements in one undo step.

        Without explicit targets the anchor prototype (in anchor target mode) or the selected prototypes are used,
        each resolved through its canonical instance and capture mesh. ``flood_max_instances`` caps the total across
        all targets. Returns the number of placements created; problems are reported through status messages.
        """
        usd_context = omni.usd.get_context(usd_context_name)
        stage = usd_context.get_stage()
        if stage is None:
            self.post_status("Open a project before flooding", is_error=True)
            return 0
        assets = self.enabled_assets()
        if not assets:
            self.post_status("Add and enable at least one asset before flooding", is_error=True)
            return 0
        if targets is None:
            targets = self._default_flood_targets(stage, usd_context)
        if not targets:
            if self._settings.target_mode == TargetMode.ANCHOR:
                self.post_status("Set a captured mesh as the anchor prototype before flooding", is_error=True)
            else:
                self.post_status("Select a captured mesh or instance to flood", is_error=True)
            return 0

        layer = stage.GetEditTarget().GetLayer()
        stage_up = stage_up_axis(stage)
        seed, stroke_index = self.next_stroke()
        remaining = self._settings.flood_max_instances
        total = 0
        with omni.kit.undo.group():
            for target_index, target in enumerate(targets):
                if remaining <= 0:
                    break
                padding_index = PaddingIndex(max(self._settings.padding, 1.0))
                existing = existing_placement_points(stage, target)
                if len(existing):
                    padding_index.add_many(existing)
                records = generate_flood(
                    cache,
                    target,
                    self._settings,
                    assets,
                    stamp_rng(seed, stroke_index, target_index),
                    padding_index,
                    stage_up,
                    layer,
                    remaining,
                )
                if not records:
                    continue
                omni.kit.commands.execute(
                    "ScatterFloodCommand",
                    context_name=usd_context_name,
                    layer_identifier=layer.identifier,
                    records=[record.to_dict() for record in records],
                )
                total += len(records)
                remaining -= len(records)
        self.notify_stroke_committed(total, False)
        return total

    def subscribe_mode_changed(self, fn: Callable[[ScatterMode], None]) -> EventSubscription:
        """Subscribe to mode changes; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_mode_changed, fn)

    def subscribe_settings_changed(self, fn: Callable[[ScatterBrushSettings], None]) -> EventSubscription:
        """Subscribe to settings replacements; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_settings_changed, fn)

    def subscribe_assets_changed(self, fn: Callable[[list[ScatterAssetEntry]], None]) -> EventSubscription:
        """Subscribe to palette changes; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_assets_changed, fn)

    def subscribe_stroke_committed(self, fn: Callable[[int, bool], None]) -> EventSubscription:
        """Subscribe to ``(count, erase)`` notifications sent after a stroke or flood commits."""
        return EventSubscription(self.__on_stroke_committed, fn)

    def subscribe_status_message(self, fn: Callable[[str, bool], None]) -> EventSubscription:
        """Subscribe to ``(text, is_error)`` status messages meant for the user."""
        return EventSubscription(self.__on_status_message, fn)

    def notify_stroke_committed(self, count: int, erase: bool) -> None:
        """Tell subscribers that a stroke or flood committed ``count`` placements or erasures."""
        self.__on_stroke_committed(count, erase)

    def post_status(self, text: str, is_error: bool = False) -> None:
        """Log a user-facing status line and forward it to subscribers."""
        if is_error:
            carb.log_warn(f"{_LOG_PREFIX} {text}")
        else:
            carb.log_info(f"{_LOG_PREFIX} {text}")
        self.__on_status_message(text, is_error)

    def destroy(self) -> None:
        """Drop every subscriber and override so the controller can be released."""
        self.__on_mode_changed = Event(copy=True)
        self.__on_settings_changed = Event(copy=True)
        self.__on_assets_changed = Event(copy=True)
        self.__on_stroke_committed = Event(copy=True)
        self.__on_status_message = Event(copy=True)
        self._picker_factory = None

    def _set_settings(self, settings: ScatterBrushSettings) -> None:
        """Adopt validated settings, persist them and notify subscribers."""
        self._settings = settings
        self._settings.save_to_carb()
        self.__on_settings_changed(self._settings)

    def _find_asset(self, path: str) -> ScatterAssetEntry | None:
        """Return the palette entry whose normalized path matches, or None."""
        key = _asset_key(path)
        return next((entry for entry in self._assets if _asset_key(entry.path) == key), None)

    def _update_asset(self, path: str, field_name: str, value: object) -> bool:
        """Assign one validated field on a palette entry, persist and notify."""
        entry = self._find_asset(path)
        if entry is None:
            return False
        try:
            setattr(entry, field_name, value)
        except ValidationError as exc:
            self.post_status(f"Invalid asset {field_name}: {_describe_validation_error(exc)}", is_error=True)
            return False
        self._assets_changed()
        return True

    def _assets_changed(self) -> None:
        """Persist the palette and notify subscribers."""
        save_assets_to_carb(self._assets)
        self.__on_assets_changed(self.assets)

    def _default_flood_targets(self, stage: Usd.Stage, usd_context) -> list[ScatterTarget]:
        """Resolve the anchor prototype or the selected prototypes into flood targets."""
        if self._settings.target_mode == TargetMode.ANCHOR:
            anchor = validated_anchor_prototype(stage, self._settings.anchor_prototype_path)
            prototypes = [anchor] if anchor is not None else []
        else:
            prototypes = sorted(selected_prototypes(usd_context), key=str)
        targets = []
        for prototype_root in prototypes:
            target = self._build_target(stage, prototype_root)
            if target is not None:
                targets.append(target)
        return targets

    @staticmethod
    def _build_target(stage: Usd.Stage, prototype_root: Sdf.Path) -> ScatterTarget | None:
        """Build a flood target from a prototype using its canonical instance for geometry and transform."""
        instance_root = canonical_instance(stage, prototype_root)
        mesh_path = find_capture_mesh(stage, prototype_root if instance_root is None else instance_root)
        if mesh_path is None:
            carb.log_warn(f"{_LOG_PREFIX} No capture mesh found under {prototype_root}; skipping flood target")
            return None
        parent_world = Gf.Matrix4d(1.0)
        if instance_root is not None:
            instance_prim = stage.GetPrimAtPath(instance_root)
            if instance_prim and instance_prim.IsValid():
                parent_world = UsdGeom.Xformable(instance_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return ScatterTarget(
            prototype_root=prototype_root,
            parent_instance_root=instance_root,
            mesh_path=mesh_path,
            parent_world=parent_world,
            instance_count=instance_count(stage, prototype_root),
        )


_CONTROLLER_INSTANCE: ScatterBrushController | None = None


def get_scatter_brush_controller() -> ScatterBrushController:
    """Return the process-wide brush controller, creating it on first use."""
    global _CONTROLLER_INSTANCE
    if _CONTROLLER_INSTANCE is None:
        _CONTROLLER_INSTANCE = ScatterBrushController()
    return _CONTROLLER_INSTANCE


def destroy_scatter_brush_controller() -> None:
    """Destroy the process-wide brush controller, if any, so the next access creates a fresh one."""
    global _CONTROLLER_INSTANCE
    if _CONTROLLER_INSTANCE is not None:
        _CONTROLLER_INSTANCE.destroy()
    _CONTROLLER_INSTANCE = None
