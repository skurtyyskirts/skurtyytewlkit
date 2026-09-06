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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import numpy as np
import omni.kit.commands
import omni.usd
from omni.flux.utils.common.interactive_usd_notices import begin_interaction, end_interaction
from pxr import Gf, Sdf, Usd

from .commands import ScatterStrokeKind
from .placement import (
    author_placements,
    erase_candidates,
    existing_placement_points,
    generate_stamp,
    remove_placements,
    snapshot_prims,
)
from .sampling import PaddingIndex, stamp_rng
from .settings import ScatterAssetEntry, ScatterBrushSettings, UpAxis

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


class StrokeSession:
    """State machine for one paint or erase stroke.

    ``begin`` opens a USD notice interaction and stamps at the first sample. ``update`` walks the segment from the
    previous sample in steps of ``stamp_spacing`` (carrying the remainder to the next call) and stamps at every step
    that projects onto the target mesh; a change of target restarts the segment at the new sample. ``end`` commits a
    single ``ScatterStrokeCommand`` covering everything the stroke authored or removed and closes the interaction.
    Placements are written to the edit layer as soon as they are generated so the viewport updates while the mouse
    moves; the command's first ``do()`` is therefore a no-op.

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
    ):
        self._usd_context_name = usd_context_name
        self._settings = settings
        self._assets = list(assets)
        self._cache = cache
        self._erase = erase
        self._stage_up = stage_up
        self._seed = seed
        self._stroke_index = stroke_index

        self._active = False
        self._stage: Usd.Stage | None = None
        self._layer: Sdf.Layer | None = None
        self._interaction_token: InteractionToken | None = None
        self._padding_index: PaddingIndex | None = None
        self._seeded_prototypes: set[Sdf.Path] = set()
        self._stamp_index = 0
        self._records: list[PlacementRecord] = []
        self._erased_paths: list[str] = []
        self._snapshot: Sdf.Layer | None = None
        self._current_prototype: Sdf.Path | None = None
        self._previous_position: np.ndarray | None = None
        self._distance_since_stamp = 0.0
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
        """Number of placements authored so far, or of prims removed so far for an erase stroke."""
        return len(self._erased_paths) if self._erase else len(self._records)

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
        self._layer = stage.GetEditTarget().GetLayer()
        self._padding_index = PaddingIndex(max(self._settings.padding, 1.0))
        self._interaction_token = begin_interaction(stage)
        self._active = True
        self._start_segment(target, sample)
        self._stamp(target, sample, None)

    def update(self, hit: SurfaceHit, target: ScatterTarget, sample: SurfaceSample) -> None:
        """Advance the stroke to a new sample, stamping every ``stamp_spacing`` along the way.

        A different target prototype restarts the segment at this sample with one stamp; no stamps are interpolated
        across targets. A jump longer than ``_MAX_STAMPS_PER_UPDATE`` stamps restarts the segment the same way so one
        mouse event cannot flood the layer. Does nothing before ``begin``.
        """
        if not self._active:
            return
        if target.prototype_root != self._current_prototype:
            self._start_segment(target, sample)
            self._stamp(target, sample, None)
            return

        position = np.asarray(sample.position, dtype=np.float64)
        delta = position - self._previous_position
        length = float(np.linalg.norm(delta))
        if length <= _MIN_SEGMENT_LENGTH:
            return
        heading = delta / length
        heading_world = Gf.Vec3d(float(heading[0]), float(heading[1]), float(heading[2]))
        spacing = self._settings.stamp_spacing
        if (length + self._distance_since_stamp + _SPACING_EPSILON) // spacing > _MAX_STAMPS_PER_UPDATE:
            self._start_segment(target, sample)
            self._stamp(target, sample, heading_world)
            return

        travelled = spacing - self._distance_since_stamp
        while travelled <= length + _SPACING_EPSILON:
            point = self._previous_position + heading * travelled
            center = self._cache.closest_point(target.mesh_path, point, self._settings.radius)
            if center is not None:
                self._stamp(target, center, heading_world)
            travelled += spacing
        self._distance_since_stamp = max(0.0, length - (travelled - spacing))
        self._previous_position = position

    def end(self) -> int:
        """Commit the stroke as one undoable command and close the notice interaction.

        The interaction is closed even when the commit raises. Returns the number of placements authored (or prims
        removed) by the stroke; zero when nothing happened or the stroke was not active.
        """
        if not self._active:
            return 0
        try:
            return self._commit()
        finally:
            self._close()

    def abort(self) -> None:
        """Same as ``end`` but never raises, for use in gesture teardown paths."""
        try:
            self.end()
        except Exception as exc:  # noqa: BLE001 - abort runs during gesture teardown and must not propagate.
            carb.log_error(f"{_LOG_PREFIX} Scatter stroke abort failed: {exc}")

    def _start_segment(self, target: ScatterTarget, sample: SurfaceSample) -> None:
        """Make this sample the origin of the spacing walk for the given target."""
        self._current_prototype = target.prototype_root
        self._previous_position = np.asarray(sample.position, dtype=np.float64)
        self._distance_since_stamp = 0.0

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
        """Generate the placements of one stamp and author them immediately."""
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
            self._layer,
        )
        if not records:
            return
        author_placements(self._layer, records)
        self._records.extend(records)

    def _erase_at(self, target: ScatterTarget, center: SurfaceSample) -> None:
        """Snapshot and remove every placement of the target within the brush radius of the sample.

        Only placements authored in the stroke's edit layer can be removed from it; candidates that live in another
        layer are left alone, reported once per stroke and never counted as erased. Every stamp snapshots into the
        one layer handed to the erase command so each container the stroke empties stays restorable.
        """
        center_world = Gf.Vec3d(float(center.position[0]), float(center.position[1]), float(center.position[2]))
        candidates = erase_candidates(
            self._stage,
            target,
            center_world,
            self._settings.radius,
            self._settings.erase_scope,
            [asset.path for asset in self._assets],
        )
        prim_paths = [str(path) for path in candidates if self._layer.GetPrimAtPath(path) is not None]
        if len(prim_paths) < len(candidates):
            self._warn_foreign_layer_candidates(len(candidates) - len(prim_paths))
        if not prim_paths:
            return
        self._snapshot = snapshot_prims(self._layer, prim_paths, into=self._snapshot)
        remove_placements(self._layer, prim_paths)
        self._erased_paths.extend(prim_paths)

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
        """Feed the padding index with the placements that already exist under the target, once per prototype."""
        if target.prototype_root in self._seeded_prototypes:
            return
        self._seeded_prototypes.add(target.prototype_root)
        points = existing_placement_points(self._stage, target)
        if len(points):
            self._padding_index.add_many(points)

    def _commit(self) -> int:
        """Execute the single stroke command when the stroke changed anything and return the change count."""
        if self._erase:
            if not self._erased_paths:
                return 0
            omni.kit.commands.execute(
                "ScatterStrokeCommand",
                context_name=self._usd_context_name,
                layer_identifier=self._layer.identifier,
                kind=ScatterStrokeKind.ERASE.value,
                prim_paths=list(self._erased_paths),
                snapshot_layer=self._snapshot,
                already_applied=True,
            )
            return len(self._erased_paths)
        if not self._records:
            return 0
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name=self._usd_context_name,
            layer_identifier=self._layer.identifier,
            kind=ScatterStrokeKind.PLACE.value,
            records=[record.to_dict() for record in self._records],
            already_applied=True,
        )
        return len(self._records)

    def _close(self) -> None:
        """Deactivate the stroke and flush the deferred USD notices."""
        token = self._interaction_token
        self._interaction_token = None
        self._active = False
        if token is not None:
            end_interaction(token)
