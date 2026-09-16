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

__all__ = ["MoveOutcome", "move_scope", "scope_pivot"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd

from .constants import SCATTER_ORIGIN_ATTR
from .placement import PlacementTransform
from .reapply import ReapplyScope, is_hand_edited, read_placement_origin, read_placement_yaw
from .sampling import (
    ParentSpaceOrigin,
    rotation_to_xyz_degrees,
    stage_up_axis,
    to_parent_space_origin,
    twist_degrees,
    up_axis_vector,
)
from .targets import build_prototype_target

if TYPE_CHECKING:
    from .geometry import MeshSurfaceCache
    from .targets import ScatterTarget

# Floor for how far a dragged placement looks for the surface, for a mesh too small to reach across on its own.
_MIN_SETTLE_DISTANCE = 100.0
_TRANSLATE_OP = "xformOp:translate"


@dataclass
class MoveOutcome:
    """What one drag of the move gizmo did, so the user can be told why a placement did not follow.

    Attributes:
        moved: Placements that followed the drag.
        settled: Placements of ``moved`` that found the surface again and were re-oriented onto it.
        unsettled: Placements of ``moved`` that could not reach the capture mesh and kept the rigid result.
        recovered: Placements that had recorded no surface; the move projects one and records it, so a move is what
            upgrades a placement painted before the brush recorded surfaces.
        hand_edited: Placements the user had moved by hand, which travel with the patch and stay theirs.
        foreign_layer: Placements with no opinion in the edit target layer, which this layer may not change.
    """

    moved: int = 0
    settled: int = 0
    unsettled: int = 0
    recovered: int = 0
    hand_edited: int = 0
    foreign_layer: int = 0

    def describe(self) -> str:
        """Return the notes worth showing after a drag; empty when every placement moved and settled."""
        notes = []
        if self.unsettled:
            notes.append(f"{self.unsettled} could not reach the surface")
        if self.recovered:
            notes.append(f"{self.recovered} gained a surface they never recorded")
        if self.hand_edited:
            notes.append(f"{self.hand_edited} hand-edited kept their own transform")
        if self.foreign_layer:
            notes.append(f"{self.foreign_layer} in another layer")
        return "; ".join(notes)


def scope_pivot(stage: Usd.Stage, layer: Sdf.Layer, scope: ReapplyScope) -> Gf.Vec3d | None:
    """Return the world point a move gizmo should sit on for ``scope``.

    The centre of the placements' bounding box rather than their average, so one dense corner of a patch cannot drag
    the handle away from the middle of what the user can see.

    Args:
        stage: Stage holding the placements.
        layer: Layer whose authored transforms are read.
        scope: Placements the gizmo would move.

    Returns:
        The pivot in world space, or None when the scope holds no placement this layer authors.
    """
    lowest = highest = None
    for entry in scope.entries:
        target = build_prototype_target(stage, entry.prototype_root)
        if target is None:
            continue
        for position in _authored_positions(layer, entry.placement_paths):
            world = target.parent_world.Transform(position)
            lowest = world if lowest is None else _corner(lowest, world, min)
            highest = world if highest is None else _corner(highest, world, max)
    if lowest is None:
        return None
    return (lowest + highest) * 0.5


def move_scope(
    stage: Usd.Stage,
    layer: Sdf.Layer,
    scope: ReapplyScope,
    delta: Gf.Matrix4d,
    cache: MeshSurfaceCache,
) -> tuple[list[PlacementTransform], list[PlacementTransform], MoveOutcome]:
    """Move every placement in ``scope`` by a world-space delta and settle it onto the surface it lands on.

    The brush settings are deliberately not consulted. A drag transforms the transform a placement already carries,
    so moving a patch can never also resize or re-rotate it from whatever happens to be in the Scatter window;
    recomposing would turn every move into a re-derive. The surface each placement records moves with it, so a
    settings change afterwards still recomposes it where it now sits.

    Args:
        stage: Stage holding the placements.
        layer: Layer to read the current transforms from, and that the result is meant to be authored in.
        scope: Placements the drag is allowed to touch, and nothing else.
        delta: World transform of the drag, the rotation about the gizmo pivot with its translation, such that a
            world point moves to ``delta.Transform(point)``.
        cache: Mesh cache used to find the surface under each moved placement.

    Returns:
        ``(before, after, outcome)``: the placements as they are now, as the drag leaves them, and what happened to
        the ones in neither list.
    """
    outcome = MoveOutcome()
    before: list[PlacementTransform] = []
    after: list[PlacementTransform] = []
    up = up_axis_vector(stage_up_axis(stage))
    rotation = delta.ExtractRotation()
    for entry in scope.entries:
        target = build_prototype_target(stage, entry.prototype_root)
        if target is None:
            outcome.foreign_layer += len(entry.placement_paths)
            continue
        parent_delta = target.parent_world * delta * target.parent_world.GetInverse()
        reach = _settle_reach(target, cache)
        for path in entry.placement_paths:
            spec = layer.GetPrimAtPath(path)
            current = None if spec is None else _read_transform(spec, path)
            if current is None:
                outcome.foreign_layer += 1
                continue
            after.append(_move_placement(spec, current, target, parent_delta, rotation, up, cache, reach, outcome))
            before.append(current)
            outcome.moved += 1
    return before, after, outcome


def _move_placement(
    spec: Sdf.PrimSpec,
    current: PlacementTransform,
    target: ScatterTarget,
    parent_delta: Gf.Matrix4d,
    rotation: Gf.Rotation,
    up: Gf.Vec3d,
    cache: MeshSurfaceCache,
    reach: float,
    outcome: MoveOutcome,
) -> PlacementTransform:
    """Return one placement as the drag leaves it, recording on ``outcome`` what happened to it.

    ``current.brush_owned`` is set here as well, so that undoing the drag restores a hand-edited placement as the
    user's rather than stamping it as the brush's.
    """
    origin = read_placement_origin(spec)
    if origin is None:
        origin = _origin_under(current, target, cache, reach)
        outcome.recovered += 1
    current.brush_owned = not is_hand_edited(spec, current)
    if not current.brush_owned:
        outcome.hand_edited += 1

    # The offset a placement stands at above its surface, carried along as it is rather than recomputed, because the
    # settings that produced it are not this function's business.
    offset = Gf.Vec3d(*current.translate) - origin.position
    rigid_position = parent_delta.Transform(origin.position)
    rigid_normal = rotation.TransformDir(origin.normal).GetNormalized()

    settled = _surface_under(rigid_position, target, cache, reach)
    if settled is None:
        outcome.unsettled += 1
        position, normal, settle = rigid_position, rigid_normal, Gf.Rotation()
    else:
        outcome.settled += 1
        position, normal = settled
        settle = Gf.Rotation(rigid_normal, normal)

    turn = Gf.Matrix4d().SetRotate(rotation) * Gf.Matrix4d().SetRotate(settle)
    heading = None if origin.heading is None else turn.TransformDir(origin.heading)
    return PlacementTransform(
        prim_path=current.prim_path,
        translate=_triple(position + turn.TransformDir(offset)),
        rotate_xyz=_rotated_euler(current.rotate_xyz, turn),
        scale=current.scale,
        origin=ParentSpaceOrigin(position=position, normal=normal, heading=heading).to_values(),
        yaw=read_placement_yaw(spec) + twist_degrees(turn.ExtractRotation(), up),
        brush_owned=current.brush_owned,
    )


def _rotated_euler(rotate_xyz: Sequence[float], turn: Gf.Matrix4d) -> tuple[float, float, float]:
    """Return the XYZ euler angles of a placement's rotation once ``turn`` is applied after it.

    The angles are rebuilt as X then Y then Z because that is the order ``compose_parent_space_transform`` composes
    them in, so a rotation read back from a placement and turned here stays the rotation that placement authors.
    """
    existing = (
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.XAxis(), float(rotate_xyz[0])))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.YAxis(), float(rotate_xyz[1])))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.ZAxis(), float(rotate_xyz[2])))
    )
    return _triple(rotation_to_xyz_degrees(existing * turn))


def _origin_under(
    current: PlacementTransform, target: ScatterTarget, cache: MeshSurfaceCache, reach: float
) -> ParentSpaceOrigin:
    """Build the surface a placement that recorded none is standing on, from where it sits now.

    Falling back to the stage up axis when the mesh cannot be reached keeps the drag rigid for that placement rather
    than dropping it, which is the same bargain ``move_scope`` makes for a placement that cannot settle.
    """
    position = Gf.Vec3d(*current.translate)
    settled = _surface_under(position, target, cache, reach)
    if settled is None:
        return ParentSpaceOrigin(position=position, normal=Gf.Vec3d(0.0, 0.0, 1.0))
    return ParentSpaceOrigin(position=settled[0], normal=settled[1])


def _surface_under(
    position: Gf.Vec3d, target: ScatterTarget, cache: MeshSurfaceCache, reach: float
) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    """Project a parent-space point onto the target's capture mesh; None when the mesh is out of reach."""
    sample = cache.closest_point(target.mesh_path, target.parent_world.Transform(position), reach)
    if sample is None:
        return None
    origin = to_parent_space_origin(
        Gf.Vec3d(*(float(value) for value in sample.position)),
        Gf.Vec3d(*(float(value) for value in sample.normal)),
        target.parent_world,
    )
    return origin.position, origin.normal


def _settle_reach(target: ScatterTarget, cache: MeshSurfaceCache) -> float:
    """Return how far a moved placement may look for the surface: as far across as the mesh it belongs to.

    Scaling this to the mesh rather than to the drag is what makes the patch behave like something lying on that
    mesh. Lift it and it settles back down, slide it past the edge and it stops at the edge, but it never snaps onto
    geometry that is not the one it was painted on, because only that one mesh is ever searched.
    """
    geometry = cache.get(target.mesh_path)
    if geometry is None:
        return _MIN_SETTLE_DISTANCE
    extent = Gf.Vec3d(*(float(high - low) for low, high in zip(geometry.bbox_min, geometry.bbox_max)))
    return max(extent.GetLength(), _MIN_SETTLE_DISTANCE)


def _authored_positions(layer: Sdf.Layer, paths: Sequence[Sdf.Path]):
    """Yield the parent-space translation a layer authors for each placement that has one."""
    for path in paths:
        spec = layer.GetPrimAtPath(path)
        translate = None if spec is None else spec.attributes.get(_TRANSLATE_OP)
        if translate is None or translate.default is None:
            continue
        yield Gf.Vec3d(*translate.default)


def _corner(current: Gf.Vec3d, candidate: Gf.Vec3d, pick) -> Gf.Vec3d:
    """Return the component-wise ``min`` or ``max`` of two points."""
    return Gf.Vec3d(*(pick(a, b) for a, b in zip(current, candidate)))


def _read_transform(spec: Sdf.PrimSpec, path: Sdf.Path) -> PlacementTransform | None:
    """Return the transform a layer authors for a placement, or None when it authors none of the three ops.

    The surface and the turn it records come along, so that undoing a drag restores where the placement was, what it
    was standing on and how far it had been turned, all in one step.
    """
    translate = spec.attributes.get(_TRANSLATE_OP)
    rotate = spec.attributes.get("xformOp:rotateXYZ")
    scale = spec.attributes.get("xformOp:scale")
    if translate is None or rotate is None or scale is None:
        return None
    recorded = spec.attributes.get(SCATTER_ORIGIN_ATTR)
    return PlacementTransform(
        prim_path=str(path),
        translate=_triple(translate.default),
        rotate_xyz=_triple(rotate.default),
        scale=_triple(scale.default),
        origin=() if recorded is None or recorded.default is None else tuple(float(v) for v in recorded.default),
        yaw=read_placement_yaw(spec),
    )


def _triple(values) -> tuple[float, float, float]:
    """Return any three-component value as a tuple of three Python floats."""
    x, y, z = (float(value) for value in values)
    return (x, y, z)
