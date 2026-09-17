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
    "ReapplyDriver",
    "ReapplyOutcome",
    "ReapplyScope",
    "ScopeEntry",
    "changed_reappliable_fields",
    "changed_recompose_assets",
    "is_hand_edited",
    "read_placement_origin",
    "read_placement_turn",
    "read_placement_yaw",
    "reapply_transforms",
    "resolve_scope",
]

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import carb
import omni.kit.commands
import omni.usd
from omni.flux.utils.common import Event, EventSubscription
from omni.flux.utils.common.interactive_usd_notices import begin_interaction, end_interaction
from pxr import Gf, Sdf, Usd

from .assets import (
    asset_comparison_key,
    read_asset_meters_per_unit,
    read_asset_natural_extent,
    read_asset_up_axis,
)
from .constants import (
    CONTAINER_PREFIX,
    SCATTER_ASSET_ATTR,
    SCATTER_KEY_ATTR,
    SCATTER_ORIGIN_ATTR,
    SCATTER_TRANSFORM_KEY_ATTR,
    SCATTER_TURN_ATTR,
    SCATTER_YAW_ATTR,
)
from .placement import PlacementTransform, asset_scale_factor, author_placement_transforms, transform_key
from .sampling import (
    turn_matrix,
    ParentSpaceOrigin,
    compose_parent_space_transform,
    placement_rng,
    stage_meters_per_unit,
    stage_up_axis,
    to_parent_space_origin,
)
from .settings import LIVE_EDIT_SOFT_LIMIT, REAPPLIABLE_FIELDS, ScatterAssetEntry, ScatterBrushSettings
from .targets import build_prototype_target, get_prototype_root, is_scatter_prim, prototype_side_path

if TYPE_CHECKING:
    from .controller import ScatterBrushController
    from .geometry import MeshSurfaceCache
    from .targets import ScatterTarget

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
# How far a placement with no recorded surface may sit from the mesh and still be recognized as standing on it. The
# brush radius bounds how far a stamp reaches and the vertical offset is the only thing that lifts a placement off
# the surface, so twice the larger of the two covers a placement painted with settings close to the current ones.
_RECOVERY_DISTANCE_FACTOR = 2.0


@dataclass(frozen=True)
class ScopeEntry:
    """The placements of one prototype that a live edit is allowed to write to.

    Attributes:
        prototype_root: ``/RootNode/meshes/mesh_<HASH>`` the placements are authored under.
        container_names: Names of the scatter containers the placements came from, for the readout.
        placement_paths: Prototype-side paths of the placements themselves.
    """

    prototype_root: Sdf.Path
    container_names: tuple[str, ...]
    placement_paths: tuple[Sdf.Path, ...]


@dataclass(frozen=True)
class ReapplyScope:
    """Everything a settings change may write to, and nothing else.

    Resolved from the selection before any edit happens, so that a change can never reach a placement the user did
    not select. An empty scope means a settings change only affects the next stroke.
    """

    entries: tuple[ScopeEntry, ...] = ()

    @property
    def placement_count(self) -> int:
        """Total number of placements in the scope."""
        return sum(len(entry.placement_paths) for entry in self.entries)

    @property
    def is_empty(self) -> bool:
        """Whether the scope holds no placement at all."""
        return not self.entries

    def describe(self) -> str:
        """Return a one-line summary of the scope for the Scatter window, empty when the scope is empty."""
        if self.is_empty:
            return ""
        count = self.placement_count
        noun = "placement" if count == 1 else "placements"
        if len(self.entries) > 1:
            return f"{count} {noun} on {len(self.entries)} meshes"
        entry = self.entries[0]
        containers = ", ".join(name.removeprefix(CONTAINER_PREFIX) for name in entry.container_names)
        return f"{count} {noun} - {entry.prototype_root.name} / {containers}"


@dataclass
class ReapplyOutcome:
    """What one re-apply did, so the user can be told why a placement did not move.

    Attributes:
        reapplied: Placements that were given the current settings.
        hand_edited: Placements whose transform no longer matches the checksum the brush left, so the user changed
            them by hand and they were left alone.
        recovered: Placements that had no recorded surface and were projected back onto the mesh instead; their
            rotation and scale draws could not be recovered and were drawn afresh.
        foreign_layer: Placements with no opinion in the edit target layer, which this layer may not change.
        unresolved: Placements whose surface or asset could not be resolved at all.
    """

    reapplied: int = 0
    hand_edited: int = 0
    recovered: int = 0
    foreign_layer: int = 0
    unresolved: int = 0

    def describe(self) -> str:
        """Return the notes worth showing after a re-apply; empty when every placement in scope was re-applied."""
        notes = []
        if self.hand_edited:
            notes.append(f"{self.hand_edited} hand-edited left alone")
        if self.recovered:
            notes.append(f"{self.recovered} re-rolled (painted before the brush recorded surfaces)")
        if self.foreign_layer:
            notes.append(f"{self.foreign_layer} in another layer")
        if self.unresolved:
            notes.append(f"{self.unresolved} could not be resolved")
        return "; ".join(notes)


def changed_reappliable_fields(previous: ScatterBrushSettings, current: ScatterBrushSettings) -> set[str]:
    """Return the re-appliable settings fields that differ between two settings objects.

    Args:
        previous: Settings before the change.
        current: Settings after the change.

    Returns:
        The names of the changed fields that a live edit honours; empty when the change only touched fields that
        decide where the next stroke lands.
    """
    return {name for name in REAPPLIABLE_FIELDS if getattr(previous, name) != getattr(current, name)}


def changed_recompose_assets(previous: Sequence[ScatterAssetEntry], current: Sequence[ScatterAssetEntry]) -> set[str]:
    """Return the paths of palette entries whose change a placement that is already painted would follow.

    Only the up axis and the measurements count. Those decide the correction a placement is turned by and the factor
    Absolute Size divides its height by, so changing one has to reach a patch that is already painted, the same way
    moving a slider does. The rest of an entry decides what the *next* stroke does: the weight and the enabled flag
    pick which asset is stamped, and textures are assigned when a placement is authored, so including them would
    rewrite a patch for nothing.

    An entry appearing or disappearing is not a change either. A placement references its own file, so a palette
    gaining an entry cannot alter one, and a palette losing an entry leaves its placements with the transforms they
    already have rather than rewriting them from an entry that is gone.

    Args:
        previous: Palette before the change.
        current: Palette after it.

    Returns:
        The paths that changed; empty when a recompose would produce the transforms it already produced.
    """
    was = _recompose_values(previous)
    return {path for path, values in _recompose_values(current).items() if path in was and was[path] != values}


def _recompose_values(assets: Sequence[ScatterAssetEntry]) -> dict[str, tuple]:
    """Return, per palette entry path, only the values a recompose reads from that entry."""
    return {entry.path: (entry.up_axis, entry.meters_per_unit, tuple(entry.natural_extent)) for entry in assets}


def resolve_scope(
    stage: Usd.Stage | None, usd_context: omni.usd.UsdContext, settings: ScatterBrushSettings
) -> ReapplyScope:
    """Resolve the current selection into the placements a settings change may write to.

    The narrowest selection wins. Selecting placements or a scatter container scopes the edit to exactly those
    placements. Selecting anything else inside the capture topology, such as a mesh or an instance, scopes it to the
    container of the current preset on that prototype, which is what keeps two patches painted with two presets on
    one mesh independent.

    Args:
        stage: Stage the selection belongs to, or None when no project is open.
        usd_context: USD context whose selection is read.
        settings: Brush settings, for the preset whose container a prototype-wide selection resolves to.

    Returns:
        The scope, empty when nothing relevant is selected.
    """
    if stage is None:
        return ReapplyScope()
    explicit: dict[Sdf.Path, set[Sdf.Path]] = {}
    prototypes: set[Sdf.Path] = set()
    for selected in usd_context.get_selection().get_selected_prim_paths():
        prototype = get_prototype_root(selected)
        if prototype is None:
            continue
        placements = _selected_placements(stage, selected)
        if placements is None:
            prototypes.add(prototype)
        else:
            explicit.setdefault(prototype, set()).update(placements)
    if explicit:
        return _build_scope(explicit)
    container_name = f"{CONTAINER_PREFIX}{settings.slug()}"
    by_prototype: dict[Sdf.Path, set[Sdf.Path]] = {}
    for prototype in prototypes:
        placements = _placements_of(stage.GetPrimAtPath(prototype.AppendChild(container_name)))
        if placements:
            by_prototype[prototype] = set(placements)
    return _build_scope(by_prototype)


def reapply_transforms(
    stage: Usd.Stage,
    layer: Sdf.Layer,
    scope: ReapplyScope,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    cache: MeshSurfaceCache,
) -> tuple[list[PlacementTransform], list[PlacementTransform], ReapplyOutcome]:
    """Recompose every placement in ``scope`` under ``settings`` without writing anything.

    Each placement is recomposed by the same function that painted it, from the surface it recorded and its own
    random key, so a re-apply at unchanged settings produces the transform the placement already has.

    Args:
        stage: Stage holding the placements.
        layer: Layer whose opinions are read and that the result is meant to be authored in.
        scope: Placements the caller is allowed to touch.
        settings: Settings to apply.
        assets: Brush palette, used to recover each placement's up axis and measurement.
        cache: Mesh cache, used to project placements that recorded no surface back onto their mesh.

    Returns:
        ``(before, after, outcome)``: the transforms as they are now, the transforms to author, and what happened to
        the placements that are in neither list.
    """
    outcome = ReapplyOutcome()
    before: list[PlacementTransform] = []
    after: list[PlacementTransform] = []
    resolver = _AssetResolver(layer, assets)
    up_axis = stage_up_axis(stage)
    meters_per_unit = stage_meters_per_unit(stage)
    for entry in scope.entries:
        target = build_prototype_target(stage, entry.prototype_root)
        if target is None:
            outcome.unresolved += len(entry.placement_paths)
            continue
        for path in entry.placement_paths:
            spec = layer.GetPrimAtPath(path)
            current = None if spec is None else _read_transform(spec, path)
            if current is None:
                outcome.foreign_layer += 1
                continue
            if is_hand_edited(spec, current):
                outcome.hand_edited += 1
                continue
            asset = resolver.resolve(spec)
            if asset is None:
                outcome.unresolved += 1
                continue
            origin = read_placement_origin(spec)
            if origin is None:
                origin = _recover_origin(cache, target, current, settings)
                if origin is None:
                    outcome.unresolved += 1
                    continue
                outcome.recovered += 1
            translate, rotate_xyz, scale = compose_parent_space_transform(
                origin,
                target.parent_world,
                settings,
                placement_rng(_read_key(spec, path), settings.seed),
                up_axis,
                asset.up_axis,
                asset_scale_factor(asset, settings, meters_per_unit),
                read_placement_yaw(spec),
                read_placement_turn(spec),
            )
            before.append(current)
            after.append(
                PlacementTransform(
                    prim_path=str(path),
                    translate=_triple(translate),
                    rotate_xyz=_triple(rotate_xyz),
                    scale=_triple(scale),
                )
            )
            outcome.reapplied += 1
    return before, after, outcome


class ReapplyDriver:
    """Applies re-appliable settings changes to the selected placements, live, as one undo step per edit.

    Owned by the Scatter window, which also supplies the mesh cache and tells the driver when an edit is finished so
    that a whole slider drag commits once. Changing a setting that decides where a stroke lands is ignored, and so is
    every change made while ``edit_selection_live`` is off or nothing relevant is selected.

    Use from the main loop only; it is not thread-safe.
    """

    def __init__(self, controller: ScatterBrushController, usd_context_name: str, cache: MeshSurfaceCache):
        """Create the driver and start listening for settings and selection changes.

        Args:
            controller: Brush controller whose settings changes drive the edits.
            usd_context_name: USD context the Scatter window edits.
            cache: Mesh cache used to project placements that recorded no surface.
        """
        self._controller = controller
        self._usd_context_name = usd_context_name
        self._cache = cache
        self._known_settings = controller.settings
        self._known_assets = controller.assets
        self._scope = ReapplyScope()
        self._outcome = ReapplyOutcome()
        self._layer: Sdf.Layer | None = None
        self._interaction_token = None
        self._before: list[PlacementTransform] = []
        self._after: list[PlacementTransform] = []
        self._warned_foreign_layer = False

        self.__on_scope_changed = Event(copy=True)
        self._settings_sub = controller.subscribe_settings_changed(self._on_settings_changed)
        self._assets_sub = controller.subscribe_assets_changed(self._on_assets_changed)
        self._stage_event_sub = (
            omni.usd.get_context(usd_context_name)
            .get_stage_event_stream()
            .create_subscription_to_pop(self._on_stage_event, name="ScatterReapplyDriver")
        )
        self.refresh_scope()

    @property
    def scope(self) -> ReapplyScope:
        """Placements the next settings change would write to."""
        return self._scope

    @property
    def active(self) -> bool:
        """Whether an edit is open and waiting to be committed."""
        return self._interaction_token is not None

    def summary(self) -> str:
        """Return the line the Scatter window shows: the scope, plus what the last re-apply could not do."""
        if not self._controller.settings.edit_selection_live:
            return ""
        if self._scope.is_empty:
            return "Select painted scatter to edit it"
        notes = self._outcome.describe()
        summary = self._scope.describe()
        if self._scope.placement_count > LIVE_EDIT_SOFT_LIMIT:
            summary = f"{summary} (updated when the edit ends)"
        return f"{summary} - {notes}" if notes else summary

    def refresh_scope(self) -> None:
        """Resolve the selection into a scope again and notify subscribers when it changed."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        scope = resolve_scope(usd_context.get_stage(), usd_context, self._controller.settings)
        if scope == self._scope:
            return
        self._scope = scope
        self._outcome = ReapplyOutcome()
        self.__on_scope_changed(scope)

    def edit_finished(self) -> None:
        """Commit the open edit as one undoable command; does nothing when no edit is open.

        Called by the Scatter window when a field stops being edited, so that a slider drag is a single undo step.
        """
        if self._interaction_token is None:
            return
        if self._scope.placement_count > LIVE_EDIT_SOFT_LIMIT:
            self._apply()
        before, after = self._before, self._after
        self._close()
        if not after:
            return
        omni.kit.commands.execute(
            "ScatterReapplyCommand",
            context_name=self._usd_context_name,
            layer_identifier=self._layer.identifier,
            before=[transform.to_dict() for transform in before],
            after=[transform.to_dict() for transform in after],
            already_applied=True,
        )

    def subscribe_scope_changed(self, fn) -> EventSubscription:
        """Subscribe to scope changes; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_scope_changed, fn)

    def destroy(self) -> None:
        """Commit any open edit and drop every subscription so the driver can be released."""
        self.edit_finished()
        self._settings_sub = None
        self._assets_sub = None
        self._stage_event_sub = None
        self.__on_scope_changed = Event(copy=True)

    def _on_settings_changed(self, settings: ScatterBrushSettings) -> None:
        """Re-apply the change to the scope when it is one a placement that already exists can follow."""
        previous, self._known_settings = self._known_settings, settings
        if not settings.edit_selection_live:
            self.edit_finished()
            return
        if not changed_reappliable_fields(previous, settings):
            return
        if self._interaction_token is None and not self._begin():
            return
        if self._scope.placement_count <= LIVE_EDIT_SOFT_LIMIT:
            self._apply()

    def _on_assets_changed(self, assets: Sequence[ScatterAssetEntry]) -> None:
        """Re-apply a palette change to the scope, the way a settings change is applied.

        Setting an asset's up axis is the fix for one scattered on its side, and it is a palette field rather than a
        brush setting, so without this the patch the user is looking at would keep its old orientation until they
        happened to nudge a slider.
        """
        previous, self._known_assets = self._known_assets, list(assets)
        if not self._controller.settings.edit_selection_live:
            self.edit_finished()
            return
        if not changed_recompose_assets(previous, self._known_assets):
            return
        if self._interaction_token is None and not self._begin():
            return
        if self._scope.placement_count <= LIVE_EDIT_SOFT_LIMIT:
            self._apply()

    def _on_stage_event(self, event) -> None:
        """Commit before the ground moves: a new selection or a closing stage ends the edit it was opened for."""
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self.edit_finished()
            self.refresh_scope()
        elif event.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.OPENED)):
            self._close()
            self.refresh_scope()

    def _begin(self) -> bool:
        """Open an edit on the current scope, or report why there is nothing to open."""
        self.refresh_scope()
        if self._scope.is_empty:
            return False
        stage = omni.usd.get_context(self._usd_context_name).get_stage()
        if stage is None:
            return False
        self._layer = stage.GetEditTarget().GetLayer()
        self._before = []
        self._after = []
        self._warned_foreign_layer = False
        self._interaction_token = begin_interaction(stage)
        return True

    def _apply(self) -> None:
        """Recompose the scope under the current settings and author the result straight into the edit layer."""
        stage = omni.usd.get_context(self._usd_context_name).get_stage()
        if stage is None:
            return
        before, after, outcome = reapply_transforms(
            stage, self._layer, self._scope, self._controller.settings, self._controller.assets, self._cache
        )
        self._outcome = outcome
        if outcome.foreign_layer and not self._warned_foreign_layer:
            self._warned_foreign_layer = True
            carb.log_warn(
                f"{_LOG_PREFIX} {outcome.foreign_layer} selected scattered prim(s) are authored outside the edit "
                f"layer '{self._layer.identifier}' and were not updated; set that layer as the edit target to edit them"
            )
        # Kept from the first pass of the edit: every later pass recomposes the same placements from the same
        # recorded surfaces, so the transforms to undo to are the ones they had before the edit started.
        if not self._before:
            self._before = before
        self._after = after
        author_placement_transforms(self._layer, after)
        self.__on_scope_changed(self._scope)

    def _close(self) -> None:
        """End the notice interaction and forget the open edit."""
        token = self._interaction_token
        self._interaction_token = None
        if token is not None:
            end_interaction(token)


class _AssetResolver:
    """Finds the palette entry a placement was painted from, reading the asset file when it has left the palette."""

    def __init__(self, layer: Sdf.Layer, assets: Sequence[ScatterAssetEntry]):
        self._layer = layer
        self._palette = {asset_comparison_key(entry.path): entry for entry in assets}
        self._from_file: dict[str, ScatterAssetEntry] = {}

    def resolve(self, spec: Sdf.PrimSpec) -> ScatterAssetEntry | None:
        """Return the asset entry a placement spec references, or None when it names no asset.

        An entry rebuilt from the file may be unmeasured, which leaves the placement at its own size exactly as
        painting an unmeasured asset does.

        Args:
            spec: Placement prim spec.

        Returns:
            The entry, or None when the spec carries no scatter asset attribute.
        """
        attribute = spec.attributes.get(SCATTER_ASSET_ATTR)
        if attribute is None or not attribute.default:
            return None
        path = self._layer.ComputeAbsolutePath(str(attribute.default))
        key = asset_comparison_key(path)
        entry = self._palette.get(key)
        if entry is not None:
            return entry
        if key not in self._from_file:
            self._from_file[key] = ScatterAssetEntry(
                path=path,
                up_axis=read_asset_up_axis(path),
                meters_per_unit=read_asset_meters_per_unit(path),
                natural_extent=read_asset_natural_extent(path),
            )
        return self._from_file[key]


def _selected_placements(stage: Usd.Stage, selected: Sdf.Path | str) -> list[Sdf.Path] | None:
    """Return the placements a selected path names directly, or None when it names no scatter prim.

    None is what makes the selection fall back to the prototype, so it is deliberately different from an empty list,
    which means an empty scatter container was selected and nothing should be edited.
    """
    path = prototype_side_path(selected)
    if path is None:
        return None
    prim = stage.GetPrimAtPath(path)
    if not is_scatter_prim(prim):
        return None
    if prim.GetName().startswith(CONTAINER_PREFIX):
        return _placements_of(prim)
    return [path]


def _placements_of(container: Usd.Prim) -> list[Sdf.Path]:
    """Return the paths of the scatter placements directly under a container prim."""
    if not container:
        return []
    return [child.GetPath() for child in container.GetChildren() if is_scatter_prim(child)]


def _build_scope(by_prototype: dict[Sdf.Path, set[Sdf.Path]]) -> ReapplyScope:
    """Turn a prototype-to-placements mapping into a scope with stable ordering."""
    entries = []
    for prototype in sorted(by_prototype, key=str):
        placements = sorted(by_prototype[prototype], key=str)
        if not placements:
            continue
        containers = sorted({path.GetParentPath().name for path in placements})
        entries.append(
            ScopeEntry(
                prototype_root=prototype,
                container_names=tuple(containers),
                placement_paths=tuple(placements),
            )
        )
    return ReapplyScope(entries=tuple(entries))


def _read_transform(spec: Sdf.PrimSpec, path: Sdf.Path) -> PlacementTransform | None:
    """Return the transform a layer authors for a placement, or None when it authors none of the three ops."""
    translate = spec.attributes.get("xformOp:translate")
    rotate = spec.attributes.get("xformOp:rotateXYZ")
    scale = spec.attributes.get("xformOp:scale")
    if translate is None or rotate is None or scale is None:
        return None
    return PlacementTransform(
        prim_path=str(path),
        translate=_triple(translate.default),
        rotate_xyz=_triple(rotate.default),
        scale=_triple(scale.default),
    )


def is_hand_edited(spec: Sdf.PrimSpec, current: PlacementTransform) -> bool:
    """Whether a placement's transform is no longer the one the brush gave it.

    A placement with no recorded checksum was painted before the brush recorded one, so nothing can be concluded and
    it counts as untouched rather than as edited: refusing to move it would make the whole feature useless on
    everything painted so far.
    """
    recorded = spec.attributes.get(SCATTER_TRANSFORM_KEY_ATTR)
    if recorded is None or recorded.default is None:
        return False
    return int(recorded.default) != transform_key(current.translate, current.rotate_xyz, current.scale)


def read_placement_origin(spec: Sdf.PrimSpec) -> ParentSpaceOrigin | None:
    """Return the surface a placement recorded when it was painted, or None when it recorded none."""
    attribute = spec.attributes.get(SCATTER_ORIGIN_ATTR)
    if attribute is None or attribute.default is None:
        return None
    return ParentSpaceOrigin.from_values(list(attribute.default))


def read_placement_turn(spec: Sdf.PrimSpec) -> Gf.Matrix4d | None:
    """Return the whole turn a placement records, or None when nothing has turned it.

    Args:
        spec: Placement prim spec.

    Returns:
        The turn as a rotation in parent space, or None for a placement that records none, which is every placement
        painted before turns were recorded.
    """
    attribute = spec.attributes.get(SCATTER_TURN_ATTR)
    if attribute is None or attribute.default is None or len(attribute.default) < 3:
        return None
    return turn_matrix(tuple(float(value) for value in attribute.default[:3]))


def read_placement_yaw(spec: Sdf.PrimSpec) -> float:
    """Return the degrees the move gizmo has turned a placement by about the stage up axis; 0 when it never was."""
    attribute = spec.attributes.get(SCATTER_YAW_ATTR)
    if attribute is None or attribute.default is None:
        return 0.0
    return float(attribute.default)


def _read_key(spec: Sdf.PrimSpec, path: Sdf.Path) -> int:
    """Return a placement's random key, deriving one from its prim name when it recorded none.

    The derived key is stable for a given prim, so a placement painted before the brush recorded keys still draws the
    same rotation and scale every time it is re-applied; it is simply not the draw it was painted with.
    """
    attribute = spec.attributes.get(SCATTER_KEY_ATTR)
    if attribute is not None and attribute.default is not None:
        return int(attribute.default)
    return int.from_bytes(hashlib.blake2b(path.name.encode("utf-8"), digest_size=8).digest(), "big")


def _recover_origin(
    cache: MeshSurfaceCache,
    target: ScatterTarget,
    current: PlacementTransform,
    settings: ScatterBrushSettings,
) -> ParentSpaceOrigin | None:
    """Recover the surface under a placement that recorded none, by projecting it back onto the capture mesh.

    The projection also removes whatever vertical offset the placement was painted with, which is what the recorded
    surface would have held. The stroke heading is not recoverable, so such a placement follows every setting except
    align to stroke.

    Returns:
        The recovered origin, or None when the placement is too far from the mesh to be standing on it.
    """
    world = target.parent_world.Transform(Gf.Vec3d(*current.translate))
    reach = max(settings.radius, abs(settings.vertical_offset)) * _RECOVERY_DISTANCE_FACTOR
    sample = cache.closest_point(target.mesh_path, world, reach)
    if sample is None:
        return None
    return to_parent_space_origin(
        Gf.Vec3d(*(float(value) for value in sample.position)),
        Gf.Vec3d(*(float(value) for value in sample.normal)),
        target.parent_world,
    )


def _triple(values) -> tuple[float, float, float]:
    """Return any three-component value as a tuple of three Python floats."""
    x, y, z = (float(value) for value in values)
    return (x, y, z)
