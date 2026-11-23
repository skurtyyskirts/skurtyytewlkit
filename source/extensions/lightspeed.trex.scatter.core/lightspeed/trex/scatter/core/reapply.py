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
    "PreparedReapply",
    "ReapplyDriver",
    "ReapplyOutcome",
    "ReapplyScope",
    "ScopeEntry",
    "changed_reappliable_fields",
    "changed_recompose_assets",
    "is_hand_edited",
    "is_inside_instancer",
    "prepare_reapply",
    "read_placement_origin",
    "read_placement_turn",
    "read_placement_yaw",
    "reapply_transforms",
    "resolve_scope",
    "scope_of_placements",
]

import dataclasses
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
    SCATTER_PREVIEW_OF_ATTR,
    SCATTER_TRANSFORM_KEY_ATTR,
    SCATTER_TURN_ATTR,
    SCATTER_YAW_ATTR,
)
from .instancing import is_scatter_instancer
from .placement import PlacementTransform, asset_scale_factor, author_placement_transforms, transform_key
from .sampling import (
    ParentSpaceOrigin,
    compose_parent_space_transform,
    placement_rng,
    stage_meters_per_unit,
    stage_up_axis,
    to_parent_space_origin,
    turn_matrix,
)
from .settings import (
    LIVE_EDIT_SOFT_LIMIT,
    REAPPLIABLE_FIELDS,
    ApplyTo,
    ScatterAssetEntry,
    ScatterBrushSettings,
    UpAxis,
)
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
# Draws a placement's generator makes at most when it is composed: three rotations, then up to three scales. Two spare
# so that a composition drawing one more than expected fails loudly in the tests rather than reading past the end.
_PLACEMENT_DRAWS = 8


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
    one mesh independent. That last rule only holds while the brush paints on every mesh: when it paints on the
    selection, a selected mesh is where the next stroke goes, and editing everything already on it would make every
    setting changed between two strokes rewrite the whole field.

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
    if settings.apply_to == ApplyTo.SELECTED:
        return ReapplyScope()
    container_name = f"{CONTAINER_PREFIX}{settings.slug()}"
    by_prototype: dict[Sdf.Path, set[Sdf.Path]] = {}
    for prototype in prototypes:
        placements = _placements_of(stage.GetPrimAtPath(prototype.AppendChild(container_name)))
        if placements:
            by_prototype[prototype] = set(placements)
    return _build_scope(by_prototype)


def scope_of_placements(placement_paths: Sequence[Sdf.Path | str]) -> ReapplyScope:
    """Return the scope holding exactly the given placements, grouped by the prototype each one is authored under.

    Args:
        placement_paths: Placement paths; an instance-side path is rewritten to the prototype side it is authored
            on, and paths outside the capture topology are left out.

    Returns:
        The scope, empty when no path names a placement under a prototype.
    """
    by_prototype: dict[Sdf.Path, set[Sdf.Path]] = {}
    for placement_path in placement_paths:
        path = prototype_side_path(str(placement_path))
        if path is None:
            continue
        by_prototype.setdefault(get_prototype_root(path), set()).add(path)
    return _build_scope(by_prototype)


class _ReplayedDraws:
    """Stands in for a placement's generator by replaying draws taken from it once.

    Building a generator costs more than composing a placement, and a slider drag composes every placement of the
    scope at every step with the same seed. ``uniform`` is the arithmetic numpy performs on the same double, so a
    replay produces exactly the values the generator would have.
    """

    __slots__ = ("_index", "_values")

    def __init__(self, values: tuple[float, ...]):
        self._values = values
        self._index = 0

    def random(self) -> float:
        """Return the next draw in [0, 1)."""
        value = self._values[self._index]
        self._index += 1
        return value

    def uniform(self, low: float, high: float) -> float:
        """Return the next draw spread over [low, high), as ``numpy.random.Generator.uniform`` computes it."""
        return low + (high - low) * self.random()


@dataclass
class _PreparedPlacement:
    """Everything a re-apply reads from one placement, read once when an edit opens.

    Attributes:
        path: Prototype-side path of the placement.
        parent_world: World transform of the instance that parents it.
        origin: Surface the placement is composed from.
        key: The placement's own random key.
        yaw: Degrees the move gizmo turned it by about the stage up axis.
        turn: Whole turn it records, or None.
        asset_key: Comparison key of the asset it references, to find its palette entry at every step.
        fallback_asset: Entry used when the asset is not in the palette at a step: the palette entry when the edit
            opened, or one read from the asset file.
    """

    path: Sdf.Path
    parent_world: Gf.Matrix4d
    origin: ParentSpaceOrigin
    key: int
    yaw: float
    turn: Gf.Matrix4d | None
    asset_key: str
    fallback_asset: ScatterAssetEntry
    draws_seed: int | None = None
    seeded_draws: tuple[float, ...] = ()

    def draws(self, seed: int) -> _ReplayedDraws:
        """Return the draws this placement's generator makes under ``seed``, taken from it once per seed."""
        if self.draws_seed != seed:
            values = placement_rng(self.key, seed).random(_PLACEMENT_DRAWS)
            self.seeded_draws = tuple(float(value) for value in values)
            self.draws_seed = seed
        return _ReplayedDraws(self.seeded_draws)


class PreparedReapply:
    """The placements of a scope read once, so that every step of a live edit only recomposes and writes them.

    Reading a placement back from its layer costs more than composing it, and nothing a re-apply reads from a
    placement changes while a setting is being dragged: the edit is the only thing writing to it.
    """

    def __init__(
        self,
        placements: list[_PreparedPlacement],
        before: list[PlacementTransform],
        outcome: ReapplyOutcome,
        stage_up: UpAxis,
        meters_per_unit: float,
    ):
        """
        Args:
            placements: The placements a re-apply may write to.
            before: Their transforms when the edit opened, in the same order.
            outcome: What happened to the placements of the scope that were left out.
            stage_up: Up axis of the stage.
            meters_per_unit: Meters per stage unit.
        """
        self._placements = placements
        self._before = before
        self._outcome = outcome
        self._stage_up = stage_up
        self._meters_per_unit = meters_per_unit

    @property
    def before(self) -> list[PlacementTransform]:
        """The transforms the re-appliable placements had when the edit opened, in scope order."""
        return list(self._before)

    def compose(
        self, settings: ScatterBrushSettings, assets: Sequence[ScatterAssetEntry]
    ) -> tuple[list[PlacementTransform], ReapplyOutcome]:
        """Recompose every prepared placement under ``settings`` and ``assets`` without writing anything.

        Args:
            settings: Settings to apply.
            assets: Brush palette, which may have changed since the edit opened.

        Returns:
            ``(after, outcome)``: the transforms to author, in the order of :attr:`before`, and what happened to the
            placements that are not among them.
        """
        palette = {asset_comparison_key(entry.path): entry for entry in assets}
        after: list[PlacementTransform] = []
        for item in self._placements:
            asset = palette.get(item.asset_key, item.fallback_asset)
            translate, rotate_xyz, scale = compose_parent_space_transform(
                item.origin,
                item.parent_world,
                settings,
                item.draws(settings.seed),
                self._stage_up,
                asset.up_axis,
                asset_scale_factor(asset, settings, self._meters_per_unit),
                item.yaw,
                item.turn,
            )
            after.append(
                PlacementTransform(
                    prim_path=str(item.path),
                    translate=_triple(translate),
                    rotate_xyz=_triple(rotate_xyz),
                    scale=_triple(scale),
                )
            )
        return after, dataclasses.replace(self._outcome, reapplied=len(after))


def prepare_reapply(
    stage: Usd.Stage,
    layer: Sdf.Layer,
    scope: ReapplyScope,
    settings: ScatterBrushSettings,
    assets: Sequence[ScatterAssetEntry],
    cache: MeshSurfaceCache,
) -> PreparedReapply:
    """Read every placement in ``scope`` that a re-apply may write to, once.

    Placements another layer authors, that were moved by hand, or whose asset or surface cannot be resolved are
    counted on the outcome and left out.

    Args:
        stage: Stage holding the placements.
        layer: Layer whose opinions are read and that the result is meant to be authored in.
        scope: Placements the caller is allowed to touch.
        settings: Settings when the edit opens, used only to recover a surface a placement never recorded.
        assets: Brush palette, used to recover each placement's up axis and measurement.
        cache: Mesh cache, used to project placements that recorded no surface back onto their mesh.

    Returns:
        The prepared placements, ready to be composed at every step of the edit.
    """
    outcome = ReapplyOutcome()
    before: list[PlacementTransform] = []
    placements: list[_PreparedPlacement] = []
    resolver = _AssetResolver(layer, assets)
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
            resolved = resolver.resolve_with_key(spec)
            if resolved is None:
                outcome.unresolved += 1
                continue
            origin = read_placement_origin(spec)
            if origin is None:
                origin = _recover_origin(cache, target, current, settings)
                if origin is None:
                    outcome.unresolved += 1
                    continue
                outcome.recovered += 1
            asset_key, asset = resolved
            placements.append(
                _PreparedPlacement(
                    path=path,
                    parent_world=target.parent_world,
                    origin=origin,
                    key=_read_key(spec, path),
                    yaw=read_placement_yaw(spec),
                    turn=read_placement_turn(spec),
                    asset_key=asset_key,
                    fallback_asset=asset,
                )
            )
            before.append(current)
    return PreparedReapply(placements, before, outcome, stage_up_axis(stage), stage_meters_per_unit(stage))


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
    random key, so a re-apply at unchanged settings produces the transform the placement already has. A live edit
    calls :func:`prepare_reapply` once and composes the result at every step instead.

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
    prepared = prepare_reapply(stage, layer, scope, settings, assets, cache)
    after, outcome = prepared.compose(settings, assets)
    return prepared.before, after, outcome


class ReapplyDriver:
    """Applies re-appliable settings changes to painted scatter, live, as one undo step per edit.

    What an edit changes is the stroke the brush just painted while it is the preview, and otherwise the selected
    scatter. An edit of the preview adds no undo step of its own: it is part of that stroke, so one undo takes the
    stroke and everything done to it away while the settings stay as they were left. An edit of a selection commits
    one ``ScatterReapplyCommand`` when the edit finishes, so a whole slider drag is a single undo step.

    One driver per USD context, from ``controller.get_reapply_driver``, is shared by the Scatter window and the viewport
    gizmo; each tells it when an edit it drove is finished. Changing a setting that decides where a stroke lands is
    ignored, and so is every change made while ``edit_selection_live`` is off or nothing relevant is in scope.

    Use from the main loop only; it is not thread-safe.
    """

    def __init__(
        self,
        controller: ScatterBrushController,
        usd_context_name: str,
        cache: MeshSurfaceCache,
        owns_cache: bool = False,
    ):
        """Create the driver and start listening for settings, preview and selection changes.

        Args:
            controller: Brush controller whose settings changes drive the edits.
            usd_context_name: USD context the edits are authored in.
            cache: Mesh cache used to project placements that recorded no surface.
            owns_cache: Whether :meth:`destroy` also destroys ``cache``.
        """
        self._controller = controller
        self._usd_context_name = usd_context_name
        self._cache = cache
        self._owns_cache = owns_cache
        self._known_settings = controller.settings
        self._known_assets = controller.assets
        self._scope = ReapplyScope()
        self._previewing = False
        # Whether the empty scope is empty because the selection is instanced scatter, which Split makes editable.
        self._instanced_selected = False
        self._outcome = ReapplyOutcome()
        self._layer: Sdf.Layer | None = None
        self._interaction_token = None
        self._prepared: PreparedReapply | None = None
        self._after: list[PlacementTransform] = []
        self._warned_foreign_layer = False
        self._holds = 0

        self.__on_scope_changed = Event(copy=True)
        self._settings_sub = controller.subscribe_settings_changed(self.follow_settings)
        self._assets_sub = controller.subscribe_assets_changed(self._on_assets_changed)
        self._preview_sub = controller.subscribe_preview_changed(self._on_preview_changed)
        usd_context = omni.usd.get_context(usd_context_name)
        self._stage_event_sub = (
            None
            if usd_context is None
            else usd_context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event, name="ScatterReapplyDriver"
            )
        )
        self.refresh_scope()

    @property
    def scope(self) -> ReapplyScope:
        """Placements the next settings change would write to."""
        return self._scope

    @property
    def previewing(self) -> bool:
        """Whether the scope is the stroke the brush just painted rather than the selection."""
        return self._previewing

    @property
    def controller(self) -> ScatterBrushController:
        """Brush controller whose settings changes drive the edits."""
        return self._controller

    @property
    def active(self) -> bool:
        """Whether an edit is open and waiting to be committed."""
        return self._interaction_token is not None

    def summary(self) -> str:
        """Return the line the Scatter window shows: the scope, plus what the last re-apply could not do."""
        if not self._controller.settings.edit_selection_live:
            return ""
        if self._scope.is_empty:
            if self._instanced_selected:
                return "Instanced scatter selected - Split it to edit it"
            return "Select painted scatter to edit it"
        notes = self._outcome.describe()
        summary = self._scope.describe()
        if self._previewing:
            summary = f"Last stroke: {summary}"
        if self._scope.placement_count > LIVE_EDIT_SOFT_LIMIT:
            summary = f"{summary} (updated when the edit ends)"
        return f"{summary} - {notes}" if notes else summary

    def refresh_scope(self) -> None:
        """Resolve the preview or the selection into a scope again and notify subscribers when it changed."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        settings = self._controller.settings
        preview = self._controller.preview_scope(self._usd_context_name) if settings.edit_selection_live else None
        instanced_selected = False
        if preview is not None:
            scope, previewing = preview, True
        elif usd_context is None:
            # The context this driver edits is gone, so there is nothing left to select.
            scope, previewing = ReapplyScope(), False
        else:
            stage = usd_context.get_stage()
            scope, previewing = resolve_scope(stage, usd_context, settings), False
            instanced_selected = scope.is_empty and _selection_inside_instancer(stage, usd_context)
        changed_hint = instanced_selected != self._instanced_selected
        self._instanced_selected = instanced_selected
        if scope == self._scope and previewing == self._previewing:
            if changed_hint:
                self.__on_scope_changed(scope)
            return
        self._scope = scope
        self._previewing = previewing
        self._outcome = ReapplyOutcome()
        self.__on_scope_changed(scope)

    def hold(self) -> None:
        """Keep the next or open edit open across :meth:`edit_finished` until :meth:`release`.

        For a drag that changes settings from outside the Scatter window, such as the gizmo's rings on the preview:
        the window ends an edit whenever a setting changes from outside its own fields, which would otherwise close
        and reopen the edit at every step of the drag.
        """
        self._holds += 1

    def release(self) -> None:
        """Undo one :meth:`hold` and finish the edit once nothing holds it any more."""
        self._holds = max(0, self._holds - 1)
        self.edit_finished()

    def edit_finished(self) -> None:
        """Commit the open edit; does nothing when no edit is open or something holds it open.

        An edit of the preview is left in the layer as part of the stroke it belongs to. Any other edit becomes one
        undoable command. Called by whatever drove the edit when it stops, so that a slider or gizmo drag is a single
        undo step.
        """
        if self._interaction_token is None or self._holds:
            return
        if self._scope.placement_count > LIVE_EDIT_SOFT_LIMIT:
            self._apply()
        before = [] if self._prepared is None else self._prepared.before
        after = self._after
        folded = self._previewing and self._controller.folds_into_preview(
            self._usd_context_name, self._layer.identifier, self._scope
        )
        self._close()
        if not after or folded:
            return
        omni.kit.commands.execute(
            "ScatterReapplyCommand",
            context_name=self._usd_context_name,
            layer_identifier=self._layer.identifier,
            before=before,
            after=after,
            already_applied=True,
        )

    def subscribe_scope_changed(self, fn) -> EventSubscription:
        """Subscribe to scope changes; the subscription ends when the returned object is released."""
        return EventSubscription(self.__on_scope_changed, fn)

    def destroy(self) -> None:
        """Commit any open edit and drop every subscription so the driver can be released."""
        self._holds = 0
        self.edit_finished()
        self._settings_sub = None
        self._assets_sub = None
        self._preview_sub = None
        self._stage_event_sub = None
        self.__on_scope_changed = Event(copy=True)
        if self._owns_cache and self._cache is not None:
            self._cache.destroy()
        self._cache = None

    def follow_settings(self, settings: ScatterBrushSettings) -> None:
        """Re-apply a settings change to the scope when it is one a placement that already exists can follow.

        Subscribed to the controller, and also called by whoever is about to finish the edit the change opened. The
        controller notifies its subscribers in no particular order, so a caller that finishes the edit from its own
        handler would otherwise do so, some of the time, before this had opened it: the edit would then stay open
        with no undo step until something else happened to close it. A second call with the same settings changes
        nothing.

        Args:
            settings: Settings the controller adopted.
        """
        previous, self._known_settings = self._known_settings, settings
        if previous.edit_selection_live != settings.edit_selection_live or previous.apply_to != settings.apply_to:
            # The preview only counts while live editing is on, and a selected mesh only counts while the brush paints
            # on every mesh, so either change moves the scope.
            self.edit_finished()
            self.refresh_scope()
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

        The edit is finished here as well. A palette change is one click, never a drag, and no control reports the
        end of one, so an edit left open would have no undo step until a new selection happened to close it.
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
        self.edit_finished()

    def _on_preview_changed(self) -> None:
        """Commit what was done to the old scope, then follow the stroke the brush painted or stopped previewing."""
        self.edit_finished()
        self.refresh_scope()

    def _on_stage_event(self, event) -> None:
        """Commit before the ground moves: a new selection or a closing stage ends the edit it was opened for.

        Selecting anything also ends the preview, so the next edit goes to what the user has just picked.
        """
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self.edit_finished()
            self._controller.end_preview()
            self.refresh_scope()
        elif event.type in (int(omni.usd.StageEventType.CLOSING), int(omni.usd.StageEventType.OPENED)):
            self._close()
            self._controller.end_preview()
            self.refresh_scope()

    def _begin(self) -> bool:
        """Open an edit on the current scope, reading its placements once, or report there is nothing to open."""
        self.refresh_scope()
        if self._scope.is_empty:
            return False
        stage = self._stage()
        if stage is None:
            return False
        self._layer = stage.GetEditTarget().GetLayer()
        self._prepared = prepare_reapply(
            stage, self._layer, self._scope, self._controller.settings, self._controller.assets, self._cache
        )
        self._after = []
        self._warned_foreign_layer = False
        self._interaction_token = begin_interaction(stage)
        return True

    def _apply(self) -> None:
        """Recompose the prepared scope under the current settings and author it straight into the edit layer."""
        if self._prepared is None:
            return
        after, outcome = self._prepared.compose(self._controller.settings, self._controller.assets)
        self._outcome = outcome
        if outcome.foreign_layer and not self._warned_foreign_layer:
            self._warned_foreign_layer = True
            carb.log_warn(
                f"{_LOG_PREFIX} {outcome.foreign_layer} selected scattered prim(s) are authored outside the edit "
                f"layer '{self._layer.identifier}' and were not updated; set that layer as the edit target to edit them"
            )
        self._after = after
        author_placement_transforms(self._layer, after)
        self.__on_scope_changed(self._scope)

    def _stage(self) -> Usd.Stage | None:
        """Return the stage of the driver's USD context, or None when the context or its stage is gone."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        return None if usd_context is None else usd_context.get_stage()

    def _close(self) -> None:
        """End the notice interaction and forget the open edit."""
        token = self._interaction_token
        self._interaction_token = None
        self._prepared = None
        self._after = []
        if token is not None:
            end_interaction(token)


class _AssetResolver:
    """Finds the palette entry a placement was painted from, reading the asset file when it has left the palette."""

    def __init__(self, layer: Sdf.Layer, assets: Sequence[ScatterAssetEntry]):
        self._layer = layer
        self._palette = {asset_comparison_key(entry.path): entry for entry in assets}
        self._from_file: dict[str, ScatterAssetEntry] = {}

    def resolve_with_key(self, spec: Sdf.PrimSpec) -> tuple[str, ScatterAssetEntry] | None:
        """Return the comparison key of the asset a placement references together with its entry.

        Args:
            spec: Placement prim spec.

        Returns:
            ``(key, entry)``, or None when the spec carries no scatter asset attribute.
        """
        attribute = spec.attributes.get(SCATTER_ASSET_ATTR)
        if attribute is None or not attribute.default:
            return None
        entry = self.resolve(spec)
        if entry is None:
            return None
        return (asset_comparison_key(self._layer.ComputeAbsolutePath(str(attribute.default))), entry)

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
    which means an empty scatter container or instanced scatter was selected and nothing should be edited: an
    instancer entry is not a prim a live edit can write to until the instancer is split.
    """
    path = prototype_side_path(selected)
    if path is None:
        return None
    prim = stage.GetPrimAtPath(path)
    if is_inside_instancer(prim):
        return []
    if not is_scatter_prim(prim):
        return None
    if prim.GetName().startswith(CONTAINER_PREFIX):
        return _placements_of(prim)
    return [path]


def _selection_inside_instancer(stage: Usd.Stage | None, usd_context: omni.usd.UsdContext) -> bool:
    """Whether anything selected is instanced scatter, so the readout can say how to make it editable."""
    if stage is None:
        return False
    for selected in usd_context.get_selection().get_selected_prim_paths():
        path = prototype_side_path(selected)
        if path is not None and is_inside_instancer(stage.GetPrimAtPath(path)):
            return True
    return False


def is_inside_instancer(prim: Usd.Prim) -> bool:
    """Whether ``prim`` is instanced scatter: an instancer, one of its prototypes, or the viewport stand-in of one."""
    while prim and not prim.IsPseudoRoot():
        if is_scatter_instancer(prim) or prim.HasAttribute(SCATTER_PREVIEW_OF_ATTR):
            return True
        prim = prim.GetParent()
    return False


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
    # Indexed rather than unpacked from a generator: this runs for every placement at every step of a live edit.
    return (float(values[0]), float(values[1]), float(values[2]))
