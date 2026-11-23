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

__all__ = ["StrokeSession"]

import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import carb
import numpy as np
import omni.kit.commands
import omni.usd
from omni.flux.utils.common.interactive_usd_notices import begin_interaction, end_interaction
from pxr import Gf, Sdf, Usd

from .commands import ScatterStrokeKind
from .instancing import InstancedEraser, InstancedPool, instanced_points, instanced_pool
from .materials import build_materials_payload
from .placement import (
    PlacementPool,
    author_placements,
    existing_placement_points,
    generate_stamp,
    placement_pool,
    remove_placements,
    snapshot_prims,
    to_world_points,
)
from .sampling import PaddingIndex, stage_meters_per_unit, stamp_rng
from .settings import ScatterAssetEntry, ScatterBrushSettings, UpAxis, stroke_prim_limit

if TYPE_CHECKING:
    from omni.flux.utils.common.interactive_usd_notices import InteractionToken

    from .geometry import MeshSurfaceCache, SurfaceSample
    from .picking import SurfaceHit
    from .placement import PlacementRecord
    from .targets import ScatterTarget

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
# Mouse jitter below this world distance is not a segment worth walking.
_MIN_SEGMENT_LENGTH = 1.0e-6
# Tolerance that lets a stamp land exactly on the segment end despite float accumulation.
_SPACING_EPSILON = 1.0e-9
# A single mouse event can span a huge world distance (grazing pick angle, cursor jumping across the sky); walking it
# would author thousands of stamps in one frame, so longer jumps restart the segment at the new sample instead.
_MAX_STAMPS_PER_UPDATE = 256
# Time one update, or one resume, may spend stamping before it leaves the rest of the walk to the next frame. A frame
# at 60 Hz lasts 16.7 ms, so this leaves the renderer and the rest of the UI most of it however heavy a stamp is.
_STAMP_TIME_BUDGET = 0.008
# Time the end of a stroke may spend catching up before it commits. Mouse-up cannot hand the rest of the walk to a
# later frame, so finishing everything there froze the application for as long as the stroke was behind the cursor,
# which heavy settings and a fast drag make arbitrarily long. What this cannot reach is left unpainted and reported.
_END_DRAIN_BUDGET = 1.0
# Stamp centres along a segment projected onto the mesh in one query rather than one query each.
_PROJECTION_BATCH = 32
# Clock the time budget is measured with; tests replace it to decide when a budget runs out.
_clock = time.perf_counter


@dataclass
class _ReceivedSample:
    """A sample the stroke has been given and not yet walked to.

    Attributes:
        target: Target the sample lies on.
        sample: The surface sample itself.
        position: ``sample.position`` as a float64 array.
        restart: Whether the walk starts over at this sample with one stamp instead of walking to it, as it does on a
            new target or after a jump too long to walk.
        heading: Stroke direction of that restarting stamp, or None.
    """

    target: ScatterTarget
    sample: SurfaceSample
    position: np.ndarray
    restart: bool
    heading: Gf.Vec3d | None


class StrokeSession:
    """State machine for one paint or erase stroke.

    ``begin`` opens a USD notice interaction and stamps at the first sample. ``update`` walks the segment from the
    previous sample in steps of ``stamp_spacing`` (carrying the remainder to the next call) and stamps at every step
    that projects onto the target mesh; a change of target restarts the segment at the new sample. ``end`` commits a
    single ``ScatterStrokeCommand`` covering everything the stroke authored or removed and closes the interaction.
    Placements are written to the edit layer as soon as they are generated so the viewport updates while the mouse
    moves; the command's first ``do()`` is therefore a no-op.

    A paint stroke stops adding placements once what it authored renders ``stroke_prim_limit`` prims, and ``end``
    catches up for at most ``_END_DRAIN_BUDGET``; :attr:`limit_reached` and :attr:`cut_short` report either.

    Each stamp draws its randomness from ``stamp_rng(seed, stroke_index, stamp_index)`` so a stroke is reproducible.
    """

    def __init__(
        self,
        usd_context_name: str,
        settings: ScatterBrushSettings,
        assets: Sequence[ScatterAssetEntry],
        cache: MeshSurfaceCache,
        erase: bool,
        stage_up: UpAxis,
        seed: int,
        stroke_index: int,
        view_point: Gf.Vec3d | None = None,
    ):
        """
        Args:
            usd_context_name: USD context the stroke paints in.
            settings: Brush settings the stroke keeps from start to end.
            assets: Brush assets to choose from.
            cache: Mesh cache stamps project onto.
            erase: Whether the stroke erases instead of painting.
            stage_up: Up axis of the stage.
            seed: Seed of the stroke's random draws.
            stroke_index: Index of the stroke, mixed into the random draws.
            view_point: World point the stroke is painted from, normally the camera, so that every stamp paints the
                side of the surface the user is looking at. None paints the side facing the stage up axis.
        """
        self._usd_context_name = usd_context_name
        self._settings = settings
        self._assets = list(assets)
        self._cache = cache
        self._erase = erase
        self._stage_up = stage_up
        self._seed = seed
        self._stroke_index = stroke_index
        self._view_point = None if view_point is None else Gf.Vec3d(view_point)

        self._active = False
        self._stage: Usd.Stage | None = None
        # Derived in begin() rather than taken from the caller, so the viewport tool's constructor call is unchanged.
        self._stage_meters_per_unit = 1.0
        self._layer: Sdf.Layer | None = None
        self._interaction_token: InteractionToken | None = None
        self._padding_index: PaddingIndex | None = None
        self._materials: dict[str, list[dict[str, Any]]] = {}
        # Padding and erasing are measured in the world, where a prototype stands once per instance. What the stroke
        # knows of a prototype is therefore kept per instance it reached the prototype through, as (prototype,
        # instance) keys: a stroke that crosses from one instance of a mesh to the next would otherwise measure the
        # second against where the first stands, and paint a second layer over what already shows there.
        self._seeded: set[tuple[Sdf.Path, Sdf.Path | None]] = set()
        # Parent-space surface points the stroke placed under each prototype, and how many of them the padding index
        # holds at the world position they have on each instance.
        self._placed_points: dict[Sdf.Path, list[tuple[float, ...]]] = {}
        self._mirrored: dict[tuple[Sdf.Path, Sdf.Path | None], int] = {}
        self._erase_pools: dict[tuple[Sdf.Path, Sdf.Path | None], PlacementPool] = {}
        self._instanced_pools: dict[tuple[Sdf.Path, Sdf.Path | None], InstancedPool] = {}
        self._eraser: InstancedEraser | None = None
        self._stamp_index = 0
        self._records: list[PlacementRecord] = []
        self._rendered_prims = 0
        self._prim_limit = stroke_prim_limit(settings.prim_budget)
        self._limit_reached = False
        # Prototypes the stroke went over without room for one placement on them, with their instance counts.
        self._blocked: dict[Sdf.Path, int] = {}
        self._cut_short = False
        self._failed = False
        self._erased_paths: list[str] = []
        self._erased_lookup: set[str] = set()
        self._snapshot: Sdf.Layer | None = None
        # Prototype and position of the last sample the stroke was given, which the next one is measured from.
        self._current_prototype: Sdf.Path | None = None
        self._received_position: np.ndarray | None = None
        # Where the walk stands: the last sample it reached or the last stamp it made, and how far it went since.
        self._previous_position: np.ndarray | None = None
        self._distance_since_stamp = 0.0
        # Samples given but not walked yet, because walking to them took longer than a frame's budget.
        self._pending: deque[_ReceivedSample] = deque()
        self._warned_foreign_layer = False

    @property
    def active(self) -> bool:
        """Whether ``begin`` succeeded and ``end`` or ``abort`` has not run yet."""
        return self._active

    @property
    def erase(self) -> bool:
        """Whether this stroke removes placements instead of creating them."""
        return self._erase

    @property
    def settings(self) -> ScatterBrushSettings:
        """Brush settings captured when the stroke started."""
        return self._settings

    @property
    def placed_count(self) -> int:
        """Number of placements authored so far, or of placements removed so far for an erase stroke.

        An erase counts instancer entries it removed as well as prims: either way one placement is gone.
        """
        if not self._erase:
            return len(self._records)
        return len(self._erased_paths) + (0 if self._eraser is None else self._eraser.erased_count)

    @property
    def stamp_count(self) -> int:
        """Stamps the stroke has made, whether or not they placed or erased anything; 0 for one that never began."""
        return self._stamp_index

    @property
    def pending(self) -> bool:
        """Whether the stroke is behind the samples it was given and needs :meth:`resume` to catch up."""
        return bool(self._pending)

    @property
    def rendered_prims(self) -> int:
        """Prims the placements authored so far add to the rendered stage; 0 for an erase stroke.

        A placement authored under a prototype renders once per instance of that prototype's hash, so this is the
        cost the player pays, while ``placed_count`` is only what the stroke wrote.
        """
        return self._rendered_prims

    @property
    def prim_limit(self) -> int:
        """Rendered prims at which the stroke stops adding placements."""
        return self._prim_limit

    @property
    def limit_reached(self) -> bool:
        """Whether the stroke stopped adding placements because it filled :attr:`prim_limit`.

        Always False for an erase stroke, which only removes.
        """
        return self._limit_reached

    @property
    def blocked_prototypes(self) -> Mapping[Sdf.Path, int]:
        """Prototypes the stroke went over but could not paint on, with the instances a placement on each renders on.

        One placement on such a prototype already renders more than the stroke had left of its limit, so the stroke
        skipped it and carried on: a mesh with fewer instances may still have fitted.
        """
        return MappingProxyType(self._blocked)

    @property
    def cut_short(self) -> bool:
        """Whether ``end`` stopped before the stroke had walked to every sample it was given.

        It ran out of time, or the walk failed (see :attr:`failed`). The path past the point the walk reached was left
        unpainted. Only known once ``end`` has run.
        """
        return self._cut_short

    @property
    def failed(self) -> bool:
        """Whether an error stopped ``end`` from catching up; what the stroke authored before it is still committed."""
        return self._failed

    def can_place_on(self, target: ScatterTarget) -> bool:
        """Return whether a paint stamp on ``target`` could still add a placement.

        False once the stroke has filled its prim limit, and for a target with more instances than the stroke has
        room left for, since each placement on it renders once per instance.

        Args:
            target: Target a stamp would author under.
        """
        return not self._limit_reached and self._placements_left(target) > 0

    def begin(self, hit: SurfaceHit, target: ScatterTarget, sample: SurfaceSample) -> None:
        """Start the stroke on the stage of the USD context and stamp once at the sample.

        Does nothing when the stroke is already active or no stage is open.
        """
        if self._active:
            carb.log_warn(f"{_LOG_PREFIX} Scatter stroke already active; ignoring begin")
            return
        stage = omni.usd.get_context(self._usd_context_name).get_stage()
        if stage is None:
            carb.log_warn(f"{_LOG_PREFIX} Cannot start a scatter stroke without an open stage")
            return
        self._stage = stage
        self._stage_meters_per_unit = stage_meters_per_unit(stage)
        self._layer = stage.GetEditTarget().GetLayer()
        # Cells as large as the padding: a query then visits the 27 cells around it, each holding a handful of
        # points, however small the padding is.
        self._padding_index = PaddingIndex(self._settings.padding)
        # Resolved once per stroke: the payload is the same for every stamp and touches disk to relativize paths.
        self._materials = build_materials_payload(self._layer, self._assets) if self._settings.assign_textures else {}
        if self._erase:
            self._eraser = InstancedEraser(self._layer)
        self._interaction_token = begin_interaction(stage)
        self._active = True
        self._current_prototype = target.prototype_root
        self._received_position = np.asarray(sample.position, dtype=np.float64)
        self._start_segment(sample)
        self._stamp(target, sample, None)
        self._flush_erased_entries()

    def update(self, hit: SurfaceHit, target: ScatterTarget, sample: SurfaceSample) -> None:
        """Advance the stroke to a new sample, stamping every ``stamp_spacing`` along the way.

        A different target prototype restarts the segment at this sample with one stamp; no stamps are interpolated
        across targets. A jump longer than ``_MAX_STAMPS_PER_UPDATE`` stamps restarts the segment the same way so one
        mouse event cannot flood the layer. Does nothing before ``begin``.

        Stamping stops once ``_STAMP_TIME_BUDGET`` has been spent and the rest of the walk waits for the next update
        or :meth:`resume`, so heavy settings make the brush trail the cursor instead of freezing the application. The
        walk resumes from the last stamp it made, so the stamps are the ones an unhurried walk would make.

        A stroke that reached its prim limit ignores further samples: nothing more can be placed along them.
        """
        if not self._active or self._limit_reached:
            return
        position = np.asarray(sample.position, dtype=np.float64)
        if target.prototype_root != self._current_prototype:
            self._receive(_ReceivedSample(target, sample, position, restart=True, heading=None))
        else:
            delta = position - self._received_position
            length = float(np.linalg.norm(delta))
            if length > _MIN_SEGMENT_LENGTH:
                heading = delta / length
                too_far = (length + _SPACING_EPSILON) // self._settings.stamp_spacing > _MAX_STAMPS_PER_UPDATE
                heading_world = Gf.Vec3d(float(heading[0]), float(heading[1]), float(heading[2])) if too_far else None
                self._receive(_ReceivedSample(target, sample, position, restart=too_far, heading=heading_world))
        self._drain(_clock() + _STAMP_TIME_BUDGET)

    def resume(self) -> None:
        """Keep walking towards the samples the stroke is behind on, for another frame's budget.

        Called once per frame while :attr:`pending`, so a stroke that fell behind catches up even when the mouse has
        stopped moving.
        """
        if self._active and self._pending:
            self._drain(_clock() + _STAMP_TIME_BUDGET)

    def end(self) -> int:
        """Finish the walk, commit the stroke as one undoable command and close the notice interaction.

        Whatever the stroke is still behind on is stamped first, for at most ``_END_DRAIN_BUDGET``; the path that
        leaves unwalked stays unpainted and :attr:`cut_short` reports it. A failure while catching up is logged and
        the stroke still commits everything it authored, so it stays undoable. The interaction is closed even when the
        commit raises. Returns the number of placements authored (or prims removed) by the stroke; zero when nothing
        happened or the stroke was not active.
        """
        if not self._active:
            return 0
        try:
            try:
                self._drain(_clock() + _END_DRAIN_BUDGET)
            except Exception as exc:  # noqa: BLE001 - what was authored must still reach the undo stack.
                carb.log_error(f"{_LOG_PREFIX} Scatter stroke stopped catching up: {exc}")
                self._failed = True
            # A failed walk left its path unfinished as surely as one that ran out of time, whatever is still queued.
            self._cut_short = self._failed or bool(self._pending)
            self._pending.clear()
            return self._commit()
        finally:
            self._close()

    def abort(self) -> None:
        """Same as ``end`` but never raises, for use in gesture teardown paths."""
        try:
            self.end()
        except Exception as exc:  # noqa: BLE001 - abort runs during gesture teardown and must not propagate.
            carb.log_error(f"{_LOG_PREFIX} Scatter stroke abort failed: {exc}")

    def _start_segment(self, sample: SurfaceSample) -> None:
        """Make this sample the origin of the spacing walk."""
        self._previous_position = np.asarray(sample.position, dtype=np.float64)
        self._distance_since_stamp = 0.0

    def _receive(self, received: _ReceivedSample) -> None:
        """Queue a sample for the walk and measure the next one from it."""
        self._pending.append(received)
        self._current_prototype = received.target.prototype_root
        self._received_position = received.position

    def _drain(self, deadline: float) -> None:
        """Walk towards the queued samples in order until they are all reached or ``deadline`` has passed.

        At least one stamp is made per call whenever one is due, so the walk always moves on. A stroke that reaches
        its prim limit drops the samples it has not walked to: nothing more can be placed along them.
        """
        while self._pending and not self._limit_reached:
            received = self._pending[0]
            if received.restart:
                self._pending.popleft()
                self._start_segment(received.sample)
                self._stamp(received.target, received.sample, received.heading)
            elif self._walk_towards(received, deadline):
                self._pending.popleft()
            else:
                break
            if _clock() >= deadline:
                break
        if self._limit_reached:
            self._pending.clear()
        self._flush_erased_entries()

    def _flush_erased_entries(self) -> None:
        """Remove the instancer entries the stamps made so far erased, once for all of them.

        Each instancer is rewritten whole however few entries leave it, so the stamps of one frame are written
        together rather than one rewrite each.
        """
        if self._eraser is not None:
            self._eraser.flush()

    def _walk_towards(self, received: _ReceivedSample, deadline: float) -> bool:
        """Stamp every ``stamp_spacing`` from where the walk stands to a sample, until done or out of time.

        The stamp centres are projected onto the mesh in batches. When time runs out the walk stands on the last stamp
        it made, so the next call carries on from there. The walk also stops where the stroke reaches its prim limit.

        Returns:
            Whether the sample was reached.
        """
        end = received.position
        delta = end - self._previous_position
        length = float(np.linalg.norm(delta))
        if length <= _MIN_SEGMENT_LENGTH:
            self._previous_position = end
            return True
        spacing = self._settings.stamp_spacing
        first = spacing - self._distance_since_stamp
        if first > length + _SPACING_EPSILON:
            self._distance_since_stamp += length
            self._previous_position = end
            return True
        heading = delta / length
        heading_world = Gf.Vec3d(float(heading[0]), float(heading[1]), float(heading[2]))
        count = int((length + _SPACING_EPSILON - first) // spacing) + 1
        offsets = first + spacing * np.arange(count, dtype=np.float64)
        points = self._previous_position[None, :] + heading[None, :] * offsets[:, None]
        target = received.target
        for batch_start in range(0, count, _PROJECTION_BATCH):
            batch = points[batch_start : batch_start + _PROJECTION_BATCH]
            centers = self._cache.closest_samples(target.mesh_path, batch, self._settings.radius)
            for step, center in enumerate(centers, start=batch_start):
                if center is not None:
                    self._stamp(target, center, heading_world)
                    if self._limit_reached:
                        return False
                if step < count - 1 and _clock() >= deadline:
                    self._previous_position = points[step]
                    self._distance_since_stamp = 0.0
                    return False
        self._distance_since_stamp = max(0.0, length - float(offsets[-1]))
        self._previous_position = end
        return True

    def _stamp(self, target: ScatterTarget, center: SurfaceSample, heading_world: Gf.Vec3d | None) -> None:
        """Place or erase one stamp centered on a surface sample."""
        stamp_index = self._stamp_index
        self._stamp_index += 1
        if self._erase:
            self._erase_at(target, center)
            return
        self._place_at(target, center, stamp_rng(self._seed, self._stroke_index, stamp_index), heading_world)

    def _place_at(
        self,
        target: ScatterTarget,
        center: SurfaceSample,
        rng: np.random.Generator,
        heading_world: Gf.Vec3d | None,
    ) -> None:
        """Generate the placements of one stamp and author them immediately, up to what the stroke may still render.

        A stamp that would take the stroke past its prim limit is trimmed to what fits, and the stroke places nothing
        after it. A target with no room for even one placement is skipped and noted instead: it has more instances
        than the stroke has prims left, which says nothing about the next mesh the stroke reaches.
        """
        room = self._placements_left(target)
        if room <= 0:
            self._blocked[target.prototype_root] = max(target.instance_count, 1)
            return
        if self._settings.padding > 0.0:
            # Only the padding check reads the index, so with no padding the placements already on the prototype
            # are never read at all.
            self._seed_padding(target)
        records = generate_stamp(
            self._cache,
            target,
            center,
            self._settings,
            self._assets,
            rng,
            self._padding_index,
            heading_world,
            self._stage_up,
            self._stage_meters_per_unit,
            self._layer,
            view_point=self._view_point,
        )
        if len(records) >= room:
            records = records[:room]
            self._limit_reached = True
        if not records:
            return
        try:
            author_placements(self._layer, records, self._materials)
        except Exception:
            # Whatever was authored before the failure is in the layer, so it has to be in the command as well, or
            # undoing the stroke would leave it behind.
            self._count_placed(
                target, [record for record in records if self._layer.GetPrimAtPath(record.prim_path) is not None]
            )
            raise
        self._count_placed(target, records)

    def _count_placed(self, target: ScatterTarget, records: list[PlacementRecord]) -> None:
        """Add authored records to the stroke and to what it costs the rendered stage."""
        self._records.extend(records)
        # A prototype with no instance on the stage still renders its own placements once.
        self._rendered_prims += len(records) * max(target.instance_count, 1)
        if self._settings.padding > 0.0:
            # The padding index holds these already, at the world position they have on the instance just stamped;
            # kept in parent space as well, so they can be placed on any other instance the stroke moves on to.
            placed = self._placed_points.setdefault(target.prototype_root, [])
            placed.extend(record.origin[:3] for record in records if len(record.origin) >= 3)
            self._mirrored[target.prototype_root, target.parent_instance_root] = len(placed)

    def _placements_left(self, target: ScatterTarget) -> int:
        """Return how many more placements the stroke may author on a target before it reaches its prim limit.

        Fewer on a prototype with many instances, because each placement there renders once per instance.
        """
        return (self._prim_limit - self._rendered_prims) // max(target.instance_count, 1)

    def _erase_at(self, target: ScatterTarget, center: SurfaceSample) -> None:
        """Snapshot and remove every placement of the target within the brush radius of the sample.

        Only placements authored in the stroke's edit layer can be removed from it; candidates that live in another
        layer are left alone, reported once per stroke and never counted as erased. Every stamp snapshots into the
        one layer handed to the erase command so each container the stroke empties stays restorable. Instanced
        placements are erased from their instancer the same way, entry by entry.
        """
        center_world = Gf.Vec3d(float(center.position[0]), float(center.position[1]), float(center.position[2]))
        brush_asset_paths = [asset.path for asset in self._assets]
        key = (target.prototype_root, target.parent_instance_root)
        pool = self._erase_pools.get(key)
        if pool is None:
            # Read once per prototype and instance per stroke: erasing removes placements, and nothing else adds any
            # while the stroke runs, so what the pool holds stays true for the rest of the stroke.
            pool = placement_pool(self._stage, target, self._settings.erase_scope, brush_asset_paths)
            self._erase_pools[key] = pool
        instanced = self._instanced_pools.get(key)
        if instanced is None:
            # Entries noted through another instance of the prototype have to be gone before this one is read.
            self._flush_erased_entries()
            instanced = instanced_pool(self._stage, self._layer, target, self._settings.erase_scope, brush_asset_paths)
            self._instanced_pools[key] = instanced
        for instancer_path, ids in instanced.take_within(center_world, self._settings.radius).items():
            self._eraser.erase(instancer_path, ids)
        # A pool read through another instance of the prototype may still hold what this one has erased since.
        candidates = [
            path
            for path in pool.take_within(center_world, self._settings.radius)
            if str(path) not in self._erased_lookup
        ]
        prim_paths = [str(path) for path in candidates if self._layer.GetPrimAtPath(path) is not None]
        if len(prim_paths) < len(candidates):
            self._warn_foreign_layer_candidates(len(candidates) - len(prim_paths))
        if not prim_paths:
            return
        self._snapshot = snapshot_prims(self._layer, prim_paths, into=self._snapshot)
        remove_placements(self._layer, prim_paths)
        self._erased_paths.extend(prim_paths)
        self._erased_lookup.update(prim_paths)

    def _warn_foreign_layer_candidates(self, count: int) -> None:
        """Log once per stroke that placements under the brush are authored outside the edit layer."""
        if self._warned_foreign_layer:
            return
        self._warned_foreign_layer = True
        carb.log_warn(
            f"{_LOG_PREFIX} {count} scattered prim(s) under the brush are authored outside the edit layer "
            f"'{self._layer.identifier}' and were not erased; set that layer as the edit target to erase them"
        )

    def _seed_padding(self, target: ScatterTarget) -> None:
        """Bring the padding index up to date with what stands on the instance about to be stamped.

        A placement authored under a prototype stands on every instance of it, so the index holds its world position
        once per instance the stroke has reached the prototype through. The first stamp on an instance reads what
        the stage holds there, which includes what this stroke has painted through the prototype's other instances.
        A later one adds what the stroke has painted through those other instances since.
        """
        key = (target.prototype_root, target.parent_instance_root)
        placed = self._placed_points.get(target.prototype_root, [])
        if key not in self._seeded:
            self._seeded.add(key)
            for points in (existing_placement_points(self._stage, target), instanced_points(self._stage, target)):
                if len(points):
                    self._padding_index.add_many(points)
        elif self._mirrored[key] < len(placed):
            self._padding_index.add_many(to_world_points(placed[self._mirrored[key] :], target.parent_world))
        self._mirrored[key] = len(placed)

    def _commit(self) -> int:
        """Execute the single stroke command when the stroke changed anything and return the change count."""
        if self._erase:
            erased_entries = [] if self._eraser is None else self._eraser.edits
            if not self._erased_paths and not erased_entries:
                return 0
            omni.kit.commands.execute(
                "ScatterStrokeCommand",
                context_name=self._usd_context_name,
                layer_identifier=self._layer.identifier,
                kind=ScatterStrokeKind.ERASE.value,
                prim_paths=list(self._erased_paths),
                snapshot_layer=self._snapshot,
                already_applied=True,
                erased_entries=erased_entries,
            )
            return self.placed_count
        if not self._records:
            return 0
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name=self._usd_context_name,
            layer_identifier=self._layer.identifier,
            kind=ScatterStrokeKind.PLACE.value,
            # The records themselves rather than dictionaries of them: the command accepts both, and converting a
            # stroke of thousands there and straight back costs a hitch at mouse-up and a copy on the undo stack.
            records=list(self._records),
            already_applied=True,
            materials=self._materials,
            # The stroke before this one stops being the preview now, so it no longer needs to be prims.
            instance_others=self._settings.instance_strokes,
        )
        return len(self._records)

    def _close(self) -> None:
        """Deactivate the stroke and flush the deferred USD notices."""
        token = self._interaction_token
        self._interaction_token = None
        self._active = False
        if token is not None:
            end_interaction(token)
