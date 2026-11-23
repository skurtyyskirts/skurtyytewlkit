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
    "FloodOutcome",
    "ScatterBrushController",
    "ScatterMode",
    "destroy_reapply_drivers",
    "destroy_scatter_brush_controller",
    "flood_targets",
    "get_reapply_driver",
    "get_scatter_brush_controller",
]

import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import carb
import carb.settings
import omni.kit.commands
import omni.kit.undo
import omni.usd
from omni.flux.utils.common import Event, EventSubscription
from omni.flux.utils.common.omni_url import OmniUrl
from pxr import Sdf, Usd
from pydantic import ValidationError

from .assets import (
    asset_comparison_key,
    normalize_asset_path,
    read_asset_meters_per_unit,
    read_asset_natural_extent,
    read_asset_up_axis,
    read_authored_asset_up_axis,
)
from .commands import ScatterStrokeCommand, ScatterStrokeKind
from .instancing import instanced_points, instancers_of_selection
from .materials import asset_source_hash, build_materials_payload, resolve_asset_texture_set
from .picking import create_surface_picker
from .placement import existing_placement_points, flood_sample_count, generate_flood
from .presets import PresetStore, get_default_presets_directory
from .geometry import MeshSurfaceCache
from .reapply import ReapplyDriver, ReapplyScope, scope_of_placements
from .sampling import PaddingIndex, stage_meters_per_unit, stage_up_axis, stamp_rng
from .settings import (
    RANGE_PAIRS,
    ScatterAssetEntry,
    ScatterBrushSettings,
    TargetMode,
    UpAxis,
    load_assets_from_carb,
    save_assets_to_carb,
    stroke_prim_limit,
)
from .targets import (
    ScatterTarget,
    build_prototype_target,
    selected_prototypes,
    validated_anchor_prototype,
)

if TYPE_CHECKING:
    from .picking import SurfacePicker

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
# Samples a flood draws for every placement its prim limit leaves room for. Padding and the surface angle reject some
# samples, so drawing only as many as there is room for would leave a limited flood short of its limit, while drawing
# everything the surface takes would compose thousands of placements to keep a handful.
_FLOOD_OVERSAMPLE = 4


class ScatterMode(Enum):
    """Interaction mode of the viewport brush tool."""

    OFF = 0
    PAINT = 1
    ERASE = 2


def _asset_display_name(path: str) -> str:
    """Return the file name of an asset, for status messages."""
    return OmniUrl(path).name or path


def _input_display_name(input_name: str) -> str:
    """Return a shader input name without its ``inputs:`` prefix, for status messages."""
    return input_name.removeprefix("inputs:")


def _with_ranges_ordered(current: dict, updates: dict) -> dict:
    """Merge ``updates`` into ``current``, moving the untouched end of an inverted low/high pair out of the way.

    Only the end the caller did not set is moved: when both ends arrive together and still conflict, the caller asked
    for something contradictory and validation should say so.
    """
    merged = {**current, **updates}
    for low_name, high_name in RANGE_PAIRS:
        low, high = merged[low_name], merged[high_name]
        if low <= high:
            continue
        if high_name in updates and low_name not in updates:
            merged[low_name] = high
        elif low_name in updates and high_name not in updates:
            merged[high_name] = low
    return merged


def _describe_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line suitable for a status message."""
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


@dataclass(frozen=True)
class _StrokePreview:
    """The stroke the brush painted last, while live edits still belong to it.

    Attributes:
        usd_context_name: USD context the stroke was painted in.
        layer_identifier: Layer the stroke authored its placements in.
        scope: The stroke's placements.
        command: The stroke's command; the preview lasts while it is the one on top of the undo stack.
    """

    usd_context_name: str
    layer_identifier: str
    scope: ReapplyScope
    command: ScatterStrokeCommand


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
        # Paths already reported as disagreeing with their file, so a stroke does not repeat the warning.
        self._up_axis_warned: set[str] = set()
        # Up-axis mismatches reported since the last stroke summary, by asset path, for that summary to repeat.
        self._stroke_notes: dict[str, str] = {}
        self._picker_factory: Callable[[object, MeshSurfaceCache], SurfacePicker] | None = None

        # The turn the viewport gizmo and the Scatter window's turn fields have between them put on one patch,
        # kept with the scope it was put on: a different selection has not been turned, so it reads as none. Held
        # here because the gizmo and the window resolve their own scope and neither owns the other.
        self._selection_turn: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._selection_turn_scope: ReapplyScope = ReapplyScope()

        # The stroke the brush painted last, which live edits change without undo steps of their own until anything
        # else happens, so that one undo takes the stroke away with everything done to it and keeps the settings.
        self._preview: _StrokePreview | None = None

        self.__on_mode_changed = Event(copy=True)
        self.__on_settings_changed = Event(copy=True)
        self.__on_assets_changed = Event(copy=True)
        self.__on_stroke_committed = Event(copy=True)
        self.__on_status_message = Event(copy=True)
        self.__on_selection_turn_changed = Event(copy=True)
        self.__on_preview_changed = Event(copy=True)
        self.__on_history_stepped = Event(copy=True)
        omni.kit.undo.subscribe_on_change(self._on_undo_stack_changed)
        # The undo stack reports a command the same way whether it was done, undone or redone, so the two commands
        # that step through the history are listened to directly.
        self._history_callbacks = [
            omni.kit.commands.register_callback(name, omni.kit.commands.POST_DO_CALLBACK, self._on_history_stepped)
            for name in ("Undo", "Redo")
        ]

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

        Moving one end of a low/high pair past the other carries the other end along, because a drag field reports
        every value it passes through and would otherwise be unable to cross its partner at all.

        Returns False and posts an error status, leaving the settings untouched, when a field is unknown or the merged
        settings fail validation. On success the settings are persisted and ``settings_changed`` fires.
        """
        unknown = sorted(set(kwargs) - set(ScatterBrushSettings.model_fields))
        if unknown:
            self.post_status(f"Unknown brush setting: {', '.join(unknown)}", is_error=True)
            return False
        try:
            validated = ScatterBrushSettings.model_validate(_with_ranges_ordered(self._settings.model_dump(), kwargs))
        except ValidationError as exc:
            self.post_status(f"Invalid brush setting: {_describe_validation_error(exc)}", is_error=True)
            return False
        self._set_settings(validated)
        return True

    def replace_settings(self, settings: ScatterBrushSettings) -> None:
        """Replace the whole settings object with a copy of the given one, persist it and notify subscribers."""
        self._set_settings(settings.model_copy(deep=True))

    def reset_settings(self) -> None:
        """Put every brush setting back to its default, the way loading a preset replaces them all.

        The asset palette and the presets on disk are left alone: they are what the user collected, not settings. Like
        a preset, the change reaches the stroke just painted when live editing is on, and one undo still removes it.

        Two fields are kept. The preset name says which preset is being edited and names the container the next
        stroke goes into, so resetting it would send new scatter into another patch than the one being painted.
        ``edit_selection_live`` says whether a settings change may rewrite scatter that is already painted, so turning
        it back on here would rewrite the selection with the very defaults being restored.
        """
        self.replace_settings(
            ScatterBrushSettings(
                preset_name=self._settings.preset_name, edit_selection_live=self._settings.edit_selection_live
            )
        )
        self.post_status("Brush settings reset to their defaults")

    def add_asset(self, path: str, up_axis: UpAxis | None = None) -> bool:
        """Append an ingested asset to the palette.

        Paths are compared after normalization so the same file cannot be added twice. When no up axis is given it is
        read from the asset file, as are the unit scale and natural size the brush needs to place the asset at a
        predictable world size. Returns False when the path is empty or already present.
        """
        normalized = normalize_asset_path(path)
        if not normalized:
            self.post_status("Cannot add an empty asset path to the brush", is_error=True)
            return False
        if self._find_asset(normalized) is not None:
            self.post_status(f"{normalized} is already in the brush")
            return False
        if up_axis is None:
            up_axis = read_asset_up_axis(normalized)
        entry = ScatterAssetEntry(
            path=normalized,
            up_axis=up_axis,
            meters_per_unit=read_asset_meters_per_unit(normalized),
            natural_extent=read_asset_natural_extent(normalized),
        )
        entry.textures = resolve_asset_texture_set(normalized)
        self._assets.append(entry)
        self._report_texture_set(entry)
        self._assets_changed()
        return True

    def remove_asset(self, path: str) -> bool:
        """Remove the palette entry matching the path. Returns False when there is none."""
        entry = self._find_asset(path)
        if entry is None:
            return False
        self._assets.remove(entry)
        self._stroke_notes.pop(entry.path, None)
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
        # Forgotten so the entry is judged again: an override that still disagrees with the file is worth saying.
        normalized = normalize_asset_path(path)
        self._up_axis_warned.discard(normalized)
        self._stroke_notes.pop(normalized, None)
        return self._update_asset(path, "up_axis", up_axis)

    def set_asset_texture_source(self, path: str, texture_path: str) -> bool:
        """Point a palette entry at a texture set instead of auto-detecting one, or clear the override with "".

        The picked texture's siblings become the only candidates, which is how an asset whose textures are named after
        something else gets them assigned. Returns False when the path is unknown.
        """
        entry = self._find_asset(path)
        if entry is None:
            return False
        entry.texture_source = normalize_asset_path(texture_path)
        entry.textures = resolve_asset_texture_set(entry.path, entry.texture_source)
        self._report_texture_set(entry)
        self._assets_changed()
        return True

    def ensure_asset_textures(self) -> bool:
        """Resolve the texture set of every palette entry that has none or whose asset was re-ingested.

        Entries restored from carb never went through ``add_asset``, so this is what fills them in. Call it when the
        palette is displayed or before a flood, never from the ``assets`` properties, which run on every UI refresh.

        Returns:
            Whether any entry changed.
        """
        changed = False
        for entry in self._assets:
            texture_set = entry.textures
            if texture_set is not None and texture_set.source_hash == asset_source_hash(entry.path):
                continue
            entry.textures = resolve_asset_texture_set(entry.path, entry.texture_source)
            changed = True
        if changed:
            self._assets_changed()
        return changed

    def ensure_asset_measurements(self) -> bool:
        """Read the unit scale and natural size of every palette entry that has none.

        Entries restored from carb never went through ``add_asset``, so this is what fills them in. Without a
        measurement the brush cannot turn a target height into a scale, so an unmeasured asset falls back to its own
        size instead of being placed at the height the user asked for.

        Returns:
            Whether any entry changed.
        """
        changed = False
        for entry in self._assets:
            if entry.natural_extent != (0.0, 0.0, 0.0):
                continue
            extent = read_asset_natural_extent(entry.path)
            if extent == (0.0, 0.0, 0.0):
                # Left unmeasured rather than recorded as measured-and-empty, so the next call retries once the file
                # is readable again. Staying silent is deliberate: this runs on every stroke, and an asset that is
                # simply gone is already reported by the palette.
                continue
            entry.meters_per_unit = read_asset_meters_per_unit(entry.path)
            entry.natural_extent = extent
            changed = True
        if changed:
            self._assets_changed()
        return changed

    def report_mismatched_asset_up_axes(self) -> list[str]:
        """Report palette entries whose up axis disagrees with the one their own file authors.

        A mismatch is silent otherwise: the up-axis correction is skipped and the asset is scattered lying on its
        side, which looks like a brush fault rather than a palette setting. Files that author no axis are left alone,
        because there is nothing to disagree with.

        Returns:
            The paths reported by this call; each path is reported once until its up axis is set again.
        """
        reported: list[str] = []
        for entry in self._assets:
            if entry.path in self._up_axis_warned:
                continue
            authored = read_authored_asset_up_axis(entry.path)
            if authored is None or authored == entry.up_axis:
                continue
            self._up_axis_warned.add(entry.path)
            reported.append(entry.path)
            message = (
                f"{_asset_display_name(entry.path)} is authored {authored.value}-up but the brush is set to "
                f"{entry.up_axis.value}-up, so it will be scattered on its side; set it to {authored.value} in "
                "Placement > Asset Up Axis"
            )
            # Kept for the summary of the stroke this is reported at the start of, which would otherwise replace it.
            self._stroke_notes[entry.path] = message
            self.post_status(message)
        return reported

    def post_stroke_summary(self, text: str, is_error: bool = False) -> None:
        """Post what a stroke or a flood did, behind whatever was reported about its assets when it started.

        An up-axis mismatch is reported once, as the stroke that will paint the asset on its side begins, and the
        summary arrives when that stroke ends. Posted on its own, the summary would take the one line that explains
        what is wrong with the placements just painted off the status line a moment after it appeared.

        Args:
            text: What the stroke or the flood did.
            is_error: Whether the summary reports something that went wrong.
        """
        notes, self._stroke_notes = list(self._stroke_notes.values()), {}
        self.post_status(". ".join([*notes, text]), is_error)

    def materials_payload(self, layer: Sdf.Layer, assets: Sequence[ScatterAssetEntry] | None = None) -> dict:
        """Build the shader-override payload for ``assets`` (the enabled palette by default).

        Returns an empty payload when ``assign_textures`` is off, so callers can pass the result straight through.
        """
        if not self._settings.assign_textures:
            return {}
        return build_materials_payload(layer, self.enabled_assets() if assets is None else assets)

    def enabled_assets(self) -> list[ScatterAssetEntry]:
        """Copy of the enabled palette entries in palette order."""
        return [entry.model_copy() for entry in self._assets if entry.enabled]

    def preset_names(self) -> list[str]:
        """Sorted names of the stored presets."""
        return self._presets.list_names()

    def stored_preset_name(self, name: str) -> str | None:
        """Return the name of the stored preset a save under ``name`` would overwrite, or None when there is none.

        The stored name can differ from ``name`` in case, on a file system that ignores it. A name that is not a
        valid preset name overwrites nothing, so it returns None and is left for the save to refuse.

        Args:
            name: Name a preset would be saved under.
        """
        try:
            return self._presets.stored_name(name)
        except ValueError:
            return None

    def apply_preset(self, name: str) -> bool:
        """Load a preset and make it the current settings. Returns False and posts an error when it cannot be loaded.

        ``edit_selection_live`` keeps its current value: a preset describes a brush, and a preset saved with live
        editing on would otherwise turn it back on and rewrite the selected scatter in the same step.
        """
        try:
            settings = self._presets.load(name)
        except ValueError as exc:
            self.post_status(f"Cannot load preset '{name}': {exc}", is_error=True)
            return False
        self.replace_settings(settings.model_copy(update={"edit_selection_live": self._settings.edit_selection_live}))
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

        This is also where a stroke's palette is brought up to date. Textures and measurements are not persisted, so
        a palette restored from carb has neither until something resolves them; the Scatter window does it when it
        opens, but the brush can be used from the viewport without ever opening that window. Resolving here, rather
        than in ``StrokeSession``, is what makes it take effect: the session is handed a copy of the palette taken
        straight after this call.
        """
        self.ensure_asset_textures()
        self.ensure_asset_measurements()
        self.report_mismatched_asset_up_axes()
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
        # Resolved before the palette is copied; next_stroke() below is too late for the copy taken here.
        self.ensure_asset_textures()
        self.ensure_asset_measurements()
        self.report_mismatched_asset_up_axes()
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
        materials = self.materials_payload(layer, assets)
        seed, stroke_index = self.next_stroke()
        with omni.kit.undo.group():
            outcome = flood_targets(
                stage, usd_context_name, layer, cache, targets, self._settings, assets, materials, seed, stroke_index
            )
        self.notify_stroke_committed(outcome.placed, False)
        self._report_flood(outcome)
        return outcome.placed

    def instance_all(self, usd_context_name: str) -> int:
        """Merge every loose placement of the edit layer into the instancers of their patches, in one undo step.

        Painting merges the stroke before the one being painted on its own; this covers the rest, such as the last
        stroke of a session, scatter painted before instancing existed, or a patch that was split to be edited.

        Args:
            usd_context_name: USD context whose edit layer is merged.

        Returns:
            How many placements were merged; problems and the result are reported through status messages.
        """
        stage = omni.usd.get_context(usd_context_name).get_stage()
        if stage is None:
            self.post_status("Open a project before instancing scatter", is_error=True)
            return 0
        layer = stage.GetEditTarget().GetLayer()
        # The stroke being previewed is merged too, so it stops being the preview before its prims go away.
        self.end_preview()
        _, command = omni.kit.commands.execute(
            "ScatterInstanceCommand", context_name=usd_context_name, layer_identifier=layer.identifier
        )
        count = command if isinstance(command, int) else 0
        if count:
            self.post_status(f"Instanced {count} placement{'s' if count != 1 else ''}")
        else:
            self.post_status("No loose placements to instance in the edit layer")
        return count

    def split_selection(self, usd_context_name: str) -> int:
        """Turn the instancers the selection names back into loose placements the brush can edit, in one undo step.

        The split placements are selected, which makes them what live edits and the gizmo change. Painting the next
        stroke merges them again.

        Args:
            usd_context_name: USD context whose selection names the instancers.

        Returns:
            How many placements the split created; problems are reported through status messages.
        """
        usd_context = omni.usd.get_context(usd_context_name)
        stage = usd_context.get_stage()
        if stage is None:
            self.post_status("Open a project before splitting scatter", is_error=True)
            return 0
        instancers = instancers_of_selection(stage, usd_context.get_selection().get_selected_prim_paths())
        if not instancers:
            self.post_status("Select instanced scatter, or the mesh it is painted on, to split it", is_error=True)
            return 0
        layer = stage.GetEditTarget().GetLayer()
        _, paths = omni.kit.commands.execute(
            "ScatterSplitCommand",
            context_name=usd_context_name,
            layer_identifier=layer.identifier,
            instancer_paths=[str(path) for path in instancers],
        )
        paths = list(paths or [])
        if not paths:
            self.post_status(
                "The selected instanced scatter is authored in another layer; set that layer as the edit target to "
                "split it",
                is_error=True,
            )
            return 0
        containers = sorted({str(Sdf.Path(path).GetParentPath()) for path in paths})
        usd_context.get_selection().set_selected_prim_paths(containers, True)
        self.post_status(f"Split {len(paths)} placement{'s' if len(paths) != 1 else ''} for editing")
        return len(paths)

    def subscribe_mode_changed(self, fn: Callable[[ScatterMode], None]) -> EventSubscription:
        """Subscribe to mode changes; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_mode_changed, fn)

    def selection_turn(self, scope: ReapplyScope) -> tuple[float, float, float]:
        """Return the X, Y, Z degrees a scope has been turned by so far, or no turn when it is a different scope.

        Args:
            scope: Placements the caller is about to turn, or is showing the turn of.

        Returns:
            The turn in degrees, which is ``(0, 0, 0)`` for a scope nothing has turned yet.
        """
        return self._selection_turn if scope == self._selection_turn_scope else (0.0, 0.0, 0.0)

    def set_selection_turn(self, scope: ReapplyScope, turn: Sequence[float]) -> None:
        """Record what a scope has been turned by and tell everyone showing it.

        Args:
            scope: Placements that were turned.
            turn: X, Y, Z degrees the scope now stands turned by, measured from where it was when it was selected.
        """
        value = (float(turn[0]), float(turn[1]), float(turn[2]))
        if scope == self._selection_turn_scope and value == self._selection_turn:
            return
        self._selection_turn_scope = scope
        self._selection_turn = value
        self.__on_selection_turn_changed(scope, value)

    def subscribe_selection_turn_changed(self, fn: Callable[[ReapplyScope, tuple], None]) -> EventSubscription:
        """Subscribe to the turn a patch has been given; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_selection_turn_changed, fn)

    def begin_preview(self, usd_context_name: str) -> bool:
        """Make the stroke just painted in a USD context the preview that live edits change.

        The stroke is the one on top of the undo stack, which is where committing it has just put it. It stays the
        preview until that is no longer so, the selection changes or the stage closes; until then an edit of it is
        part of the stroke rather than an undo step of its own.

        Args:
            usd_context_name: USD context the stroke was painted in.

        Returns:
            Whether a preview started; False when the top of the undo stack is not a stroke that placed something.
        """
        stack = omni.kit.undo.get_undo_stack()
        command = stack[-1].command if stack else None
        if (
            not isinstance(command, ScatterStrokeCommand)
            or command.kind != ScatterStrokeKind.PLACE
            or not command.created_paths
        ):
            self.end_preview()
            return False
        self._preview = _StrokePreview(
            usd_context_name=usd_context_name,
            layer_identifier=command.layer_identifier,
            scope=scope_of_placements(command.created_paths),
            command=command,
        )
        self.__on_preview_changed()
        return True

    def end_preview(self) -> None:
        """Stop previewing the last stroke; later edits of it are undo steps of their own."""
        if self._preview is None:
            return
        self._preview = None
        self.__on_preview_changed()

    def preview_scope(self, usd_context_name: str) -> ReapplyScope | None:
        """Return the placements of the stroke being previewed in a USD context, or None when there is none.

        Args:
            usd_context_name: USD context asking.

        Returns:
            The scope, or None when no stroke of that context is still the preview.
        """
        preview = self._current_preview()
        if preview is None or preview.usd_context_name != usd_context_name:
            return None
        return preview.scope

    def folds_into_preview(self, usd_context_name: str, layer_identifier: str, scope: ReapplyScope) -> bool:
        """Whether an edit of ``scope`` in a layer belongs to the stroke being previewed.

        Such an edit adds no undo step: undoing the stroke takes it away, and redoing the stroke brings it back.

        Args:
            usd_context_name: USD context of the edit.
            layer_identifier: Layer the edit was authored in.
            scope: Placements the edit changed.

        Returns:
            True only while the preview is that exact stroke in that exact layer.
        """
        preview = self._current_preview()
        return (
            preview is not None
            and preview.usd_context_name == usd_context_name
            and preview.layer_identifier == layer_identifier
            and preview.scope == scope
        )

    def subscribe_preview_changed(self, fn: Callable[[], None]) -> EventSubscription:
        """Subscribe to the preview starting or ending; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_preview_changed, fn)

    def subscribe_history_stepped(self, fn: Callable[[], None]) -> EventSubscription:
        """Subscribe to an undo or a redo having run, which can add or remove scatter without a stroke committing."""
        return EventSubscription(self.__on_history_stepped, fn)

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
        self.__on_selection_turn_changed = Event(copy=True)
        self.__on_preview_changed = Event(copy=True)
        self.__on_history_stepped = Event(copy=True)
        omni.kit.undo.unsubscribe_on_change(self._on_undo_stack_changed)
        for callback_id in self._history_callbacks:
            omni.kit.commands.unregister_callback(callback_id)
        self._history_callbacks = []
        self._preview = None
        self._picker_factory = None

    def _current_preview(self) -> _StrokePreview | None:
        """Return the preview while its stroke is still the one on top of the undo stack, else None."""
        preview = self._preview
        if preview is None:
            return None
        stack = omni.kit.undo.get_undo_stack()
        if not stack or stack[-1].command is not preview.command:
            return None
        return preview

    def _on_undo_stack_changed(self, _command_names) -> None:
        """End the preview once anything else, an undo included, has put a different command on top of the stack.

        Called from inside whatever command just ran, so nothing may escape it: a fault here would fail a command
        that has nothing to do with scatter.
        """
        if self._preview is None or self._current_preview() is not None:
            return
        try:
            self.end_preview()
        except Exception as exc:  # noqa: BLE001 - runs inside an unrelated command, which must not fail because of it.
            self._preview = None
            carb.log_error(f"{_LOG_PREFIX} Ending the scatter stroke preview failed: {exc}")

    def _on_history_stepped(self, _info: dict) -> None:
        """Start the turn count of the selection again and tell subscribers that an undo or a redo has run.

        What a patch has been turned by is recorded here, not read from the stage, and an undo takes a turn away
        without saying so. Kept, the record would show a turn the patch no longer has and the next turn would be
        measured from it, landing the patch somewhere other than the angle typed.

        Called from inside the undo or redo command, so nothing may escape it: a fault here would fail a command that
        has nothing to do with scatter.
        """
        try:
            self.set_selection_turn(self._selection_turn_scope, (0.0, 0.0, 0.0))
            self.__on_history_stepped()
        except Exception as exc:  # noqa: BLE001 - runs inside the undo command, which must not fail because of it.
            carb.log_error(f"{_LOG_PREFIX} Following an undo or a redo failed: {exc}")

    def _set_settings(self, settings: ScatterBrushSettings) -> None:
        """Adopt validated settings, persist them and notify subscribers."""
        self._settings = settings
        self._settings.save_to_carb()
        self.__on_settings_changed(self._settings)

    def _find_asset(self, path: str) -> ScatterAssetEntry | None:
        """Return the palette entry whose normalized path matches, or None."""
        key = asset_comparison_key(path)
        return next((entry for entry in self._assets if asset_comparison_key(entry.path) == key), None)

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

    def _report_texture_set(self, entry: ScatterAssetEntry) -> None:
        """Post one status line when an entry's textures could not be resolved, and stay quiet when they are fine.

        An asset that supplies its own textures is silent even though it leaves most AperturePBR inputs empty: almost
        no asset uses all of them, so reporting that would be noise on every healthy asset.
        """
        texture_set = entry.textures
        if texture_set is None:
            return
        if texture_set.overrides or texture_set.assigned or not texture_set.reason:
            # Nothing to say: the asset either got its textures or already had them. Which maps a texture set does
            # not contain is shown in the row's tooltip rather than announced, because no asset carries every input
            # AperturePBR declares and saying so on every add would be pure noise.
            return
        self.post_status(f"{_asset_display_name(entry.path)}: {texture_set.reason}")

    def _report_flood(self, outcome: FloodOutcome) -> None:
        """Post what a flood placed, and say so when the prim limit kept it from placing more.

        A flood that places nothing looks like a button that does nothing, and one the prim limit trimmed looks like
        a flood cap that was ignored, so both are said rather than left to be guessed.
        """
        placements = f"{outcome.placed:,} placement{'s' if outcome.placed != 1 else ''}"
        prims = f"{outcome.rendered_prims:,} prim{'s' if outcome.rendered_prims != 1 else ''}"
        if outcome.limited and not outcome.placed:
            self.post_stroke_summary(
                "Flood placed nothing: one placement on the target renders on more instances than the flood's limit "
                f"of {outcome.prim_limit:,} rendered prims allows. Raise Target > Prim Budget to flood it",
                is_error=True,
            )
        elif outcome.limited:
            self.post_stroke_summary(
                f"Flood placed {placements}, rendering {prims}, and was held back by its limit of "
                f"{outcome.prim_limit:,} rendered prims. Raise Target > Prim Budget to allow more in one flood",
                is_error=True,
            )
        elif outcome.placed:
            self.post_stroke_summary(f"Flood placed {placements}, rendering {prims}")
        else:
            self.post_stroke_summary(
                "Flood placed nothing: the surface may already be filled at this Padding or be steeper than Max "
                "Surface Angle, or no enabled asset has a Weight above zero",
                is_error=True,
            )

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
            target = build_prototype_target(stage, prototype_root)
            if target is not None:
                targets.append(target)
        return targets


@dataclass(frozen=True)
class FloodOutcome:
    """What a flood authored.

    Attributes:
        placed: Placements created across every target.
        rendered_prims: Prims they add to the rendered stage; a placement renders once per instance of the prototype
            it was authored under.
        limited: Whether the prim limit, rather than the surfaces or ``flood_max_instances``, decided how many
            placements at least one target received: it was filled up to the limit, or had no room left at all.
        prim_limit: Rendered prims the flood was allowed, from ``stroke_prim_limit``.
    """

    placed: int = 0
    rendered_prims: int = 0
    limited: bool = False
    prim_limit: int = 0


def flood_targets(
    stage: Usd.Stage,
    usd_context_name: str,
    layer: Sdf.Layer,
    cache: MeshSurfaceCache,
    targets: Sequence[ScatterTarget],
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    materials: dict,
    seed: int,
    stroke_index: int,
) -> FloodOutcome:
    """Fill whole target surfaces with placements, one flood command per target that received any.

    The brush floods with its own settings and palette; a scatter group floods with the ones it recorded, so neither
    changes the other. Callers open the undo group, so a flood can share one undo step with the edits around it.

    Two caps bound a flood. ``settings.flood_max_instances`` is the number of samples it may draw, shared by the
    targets in order. ``stroke_prim_limit`` is the number of prims its placements may render, which is what a
    prototype with many instances runs into first, since each placement under it renders once per instance; a target
    that would pass it keeps only the placements there is room for.

    Args:
        stage: Stage the targets belong to.
        usd_context_name: USD context the flood commands are executed in.
        layer: Layer the placements are authored in.
        cache: Mesh cache providing the geometry of the targets.
        targets: Surfaces to fill, in the order they share the two caps.
        settings: Settings the placements are generated with.
        assets: Enabled assets to choose from, with their textures and measurements resolved.
        materials: Shader-override payload authored with the placements, empty to author none.
        seed: Seed of the flood's random draws.
        stroke_index: Stroke counter value mixed into the seed.

    Returns:
        What the flood authored.
    """
    stage_up = stage_up_axis(stage)
    stage_units = stage_meters_per_unit(stage)
    prim_limit = stroke_prim_limit(settings.prim_budget)
    remaining = settings.flood_max_instances
    placed = 0
    rendered_prims = 0
    limited = False
    for target_index, target in enumerate(targets):
        if remaining <= 0:
            break
        replicas = max(target.instance_count, 1)
        wanted = min(remaining, flood_sample_count(cache, target, settings))
        if wanted <= 0:
            continue
        room = (prim_limit - rendered_prims) // replicas
        if room <= 0:
            # A later target with fewer instances may still fit under the prim limit.
            limited = True
            continue
        padding_index = PaddingIndex(max(settings.padding, 1.0))
        if settings.padding > 0.0:
            # A flood is merged into its patch's instancer as soon as it is authored, so the placements an earlier
            # flood left here are instancer entries; reading the loose ones alone would stack every flood on the last.
            for points in (existing_placement_points(stage, target), instanced_points(stage, target)):
                if len(points):
                    padding_index.add_many(points)
        records = generate_flood(
            cache,
            target,
            settings,
            assets,
            stamp_rng(seed, stroke_index, target_index),
            padding_index,
            stage_up,
            stage_units,
            layer,
            min(wanted, room * _FLOOD_OVERSAMPLE),
        )
        if room < wanted and len(records) >= room:
            # The samples are drawn independently of one another, so the first of them cover the surface as evenly
            # as all of them would.
            records = records[:room]
            limited = True
        if not records:
            continue
        omni.kit.commands.execute(
            "ScatterFloodCommand",
            context_name=usd_context_name,
            layer_identifier=layer.identifier,
            records=records,
            materials=materials,
            instance=settings.instance_strokes,
        )
        placed += len(records)
        remaining -= len(records)
        rendered_prims += len(records) * replicas
    return FloodOutcome(placed=placed, rendered_prims=rendered_prims, limited=limited, prim_limit=prim_limit)


_DRIVERS: dict[str, ReapplyDriver] = {}


def get_reapply_driver(usd_context_name: str = "") -> ReapplyDriver:
    """Return the live-edit driver of a USD context, creating it with a mesh cache of its own on first use.

    One driver per context is shared by the Scatter window and the viewport gizmo, so a settings change is applied
    once whichever of them made it, and the gizmo can edit the preview while the window is closed. A driver follows
    the controller it was made with, so one left over from a controller that has since been replaced is replaced too.

    Args:
        usd_context_name: USD context the edits are authored in.

    Returns:
        The context's driver.
    """
    controller = get_scatter_brush_controller()
    driver = _DRIVERS.get(usd_context_name)
    if driver is not None and driver.controller is not controller:
        # The controller it follows was replaced, so its subscriptions lead nowhere any more.
        driver.destroy()
        driver = None
    if driver is None:

        def stage_getter() -> Usd.Stage | None:
            context = omni.usd.get_context(usd_context_name)
            return None if context is None else context.get_stage()

        driver = ReapplyDriver(controller, usd_context_name, MeshSurfaceCache(stage_getter), owns_cache=True)
        _DRIVERS[usd_context_name] = driver
    return driver


def destroy_reapply_drivers() -> None:
    """Destroy every driver :func:`get_reapply_driver` created, so the next access creates fresh ones."""
    drivers = list(_DRIVERS.values())
    _DRIVERS.clear()
    for driver in drivers:
        driver.destroy()


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
    # Each live-edit driver holds the controller it was made with, so they go first.
    destroy_reapply_drivers()
    if _CONTROLLER_INSTANCE is not None:
        _CONTROLLER_INSTANCE.destroy()
    _CONTROLLER_INSTANCE = None
