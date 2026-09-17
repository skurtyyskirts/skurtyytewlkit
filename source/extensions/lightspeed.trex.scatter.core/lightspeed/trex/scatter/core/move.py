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

__all__ = ["MoveOutcome", "ScopeDragSession", "move_scope", "scope_pivot"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import omni.kit.commands
import omni.usd
from omni.flux.utils.common.interactive_usd_notices import begin_interaction, end_interaction
from pxr import Gf, Sdf, Usd

from .constants import SCATTER_ORIGIN_ATTR, SCATTER_TURN_ATTR
from .placement import PlacementTransform, author_placement_transforms
from .reapply import ReapplyScope, is_hand_edited, read_placement_origin, read_placement_turn, read_placement_yaw
from .sampling import (
    ParentSpaceOrigin,
    rotation_to_xyz_degrees,
    to_parent_space_origin,
    turn_to_degrees,
)
from .targets import build_prototype_target

if TYPE_CHECKING:
    from .geometry import MeshSurfaceCache
    from .targets import ScatterTarget

# Floor for how far a dragged placement looks for the surface, for a mesh too small to reach across on its own.
_MIN_SETTLE_DISTANCE = 100.0
# Degrees a drag may turn the patch and still count as a slide. Anything above it is the user aiming the patch, and
# the aim is kept rather than settled away.
_TURN_EPSILON = 1e-6
_TRANSLATE_OP = "xformOp:translate"
# An explicit identity. A default-constructed ``Gf.Rotation`` leaves its angle uninitialized rather than standing
# for no rotation, so using one as no rotation turns a placement by whatever happened to be in that memory:
# nearly always nothing, occasionally a wild angle that tips a patch over on a drag meant only to slide it.
_NO_ROTATION = Gf.Rotation(Gf.Vec3d.XAxis(), 0.0)


@dataclass
class MoveOutcome:
    """What one drag of the move gizmo did, so the user can be told why a placement did not follow.

    Attributes:
        moved: Placements that followed the drag.
        settled: Placements a slide re-planted on the surface it carried them to. A turn settles nothing, because
            it never moves a placement off the surface it already recorded.
        unsettled: Placements a slide carried beyond the reach of the capture mesh, which kept the rigid result.
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
    conform: bool = True,
) -> tuple[list[PlacementTransform], list[PlacementTransform], MoveOutcome]:
    """Move every placement in ``scope`` by a world-space delta and settle it onto the surface it lands on.

    A drag transforms the transform a placement already carries, so moving a patch can never also resize or
    re-rotate it from whatever happens to be in the Scatter window; recomposing would turn every move into a
    re-derive. ``conform`` is the single exception, and it only ever decides whether the surface is allowed to say
    which way a placement points. The surface each placement records moves with it, so a settings change afterwards
    still recomposes it where it now sits.

    A turn aims the patch without moving it: every placement keeps its position and the surface it recorded, and
    only the way it points changes. Carrying them around the gizmo instead would swing the far side of a patch up
    into the air on any ring but the one about the surface normal, and then drop it somewhere else on the mesh.

    A turn is kept for as long as the patch is only moved. It is not recorded beyond the yaw, so a settings change
    that recomposes the patch puts a tilt back onto the surface; the up axis of the palette entry, not the gizmo, is
    where a permanently mis-oriented asset is corrected.

    Args:
        stage: Stage holding the placements.
        layer: Layer to read the current transforms from, and that the result is meant to be authored in.
        scope: Placements the drag is allowed to touch, and nothing else.
        delta: World transform of the drag. A slide is a translation, and a world point moves to
            ``delta.Transform(point)``. A turn is a rotation, and only its rotation is used: carried into the space
            of the instance that parents each placement, it aims that placement where it stands, so whatever pivot
            the rotation was built about does not move anything.
        cache: Mesh cache used to find the surface under each moved placement.
        conform: Whether the brush is set to conform to the surface. This is the one setting a move does consult,
            because a placement the user asked to stand straight up must not be tipped over by the ground it is
            slid across.

    Returns:
        ``(before, after, outcome)``: the placements as they are now, as the drag leaves them, and what happened to
        the ones in neither list.
    """
    outcome = MoveOutcome()
    before: list[PlacementTransform] = []
    after: list[PlacementTransform] = []
    rotation = delta.ExtractRotation()
    # The gizmo's arrows and its rings are separate gestures, so a drag is either a slide or a turn, and the two
    # mean different things. A slide carries the patch across the mesh and re-plants it on what it lands on. A turn
    # only aims it. Every placement of a drag is treated the same way, so a patch cannot come apart.
    turning = abs(rotation.GetAngle()) > _TURN_EPSILON
    for entry in scope.entries:
        target = build_prototype_target(stage, entry.prototype_root)
        if target is None:
            outcome.foreign_layer += len(entry.placement_paths)
            continue
        parent_delta = target.parent_world * delta * target.parent_world.GetInverse()
        # Everything a placement holds - its transform, the surface it recorded, its heading - is in the space of
        # the instance that parents it, and a capture's instances stand at whatever angle the game left them at.
        # The rings send a world rotation, so it is carried into that space here: applied as if the instance stood
        # square to the stage, a ring would aim the patch about some other axis than the one the user grabbed.
        # Orthonormalized first, because an instance is free to carry scale and a scaled conjugation is no longer a
        # rotation.
        parent_rotation = parent_delta.GetOrthonormalized().ExtractRotation()
        reach = _settle_reach(target, cache)
        for path in entry.placement_paths:
            spec = layer.GetPrimAtPath(path)
            current = None if spec is None else _read_transform(spec, path)
            if current is None:
                outcome.foreign_layer += 1
                continue
            after.append(
                _move_placement(
                    spec, current, target, parent_delta, parent_rotation, turning, conform, cache, reach, outcome
                )
            )
            before.append(current)
            outcome.moved += 1
    return before, after, outcome


def _move_placement(
    spec: Sdf.PrimSpec,
    current: PlacementTransform,
    target: ScatterTarget,
    parent_delta: Gf.Matrix4d,
    parent_rotation: Gf.Rotation,
    turning: bool,
    conform: bool,
    cache: MeshSurfaceCache,
    reach: float,
    outcome: MoveOutcome,
) -> PlacementTransform:
    """Return one placement as the drag leaves it, recording on ``outcome`` what happened to it.

    A turn leaves the placement standing exactly where it is and only changes the way it points. A slide moves it,
    re-plants it on the surface it lands on, and turns it onto that surface when ``conform`` is set.

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

    if turning:
        # Aimed where it stands: nothing is projected, so a turn cannot lift a placement off the mesh or drop it on
        # a different part of it, and the surface it recorded is still the one underneath it. The placement turns
        # about that recorded point rather than about its own origin, so a blade pivots where it meets the ground.
        position, normal, settle = origin.position, origin.normal, _NO_ROTATION
    else:
        rigid_position = parent_delta.Transform(origin.position)
        settled = _surface_under(rigid_position, target, cache, reach)
        if settled is None:
            outcome.unsettled += 1
            position, normal, settle = rigid_position, origin.normal, _NO_ROTATION
        else:
            outcome.settled += 1
            position, normal = settled
            # Turned onto the ground it landed on only when the brush conforms, so a patch of assets the brush was
            # told to stand straight up is not tipped over by the surface it is slid across.
            settle = Gf.Rotation(origin.normal, normal) if conform else _NO_ROTATION

    turn = Gf.Matrix4d().SetRotate(parent_rotation) * Gf.Matrix4d().SetRotate(settle)
    heading = None if origin.heading is None else turn.TransformDir(origin.heading)
    return PlacementTransform(
        prim_path=current.prim_path,
        translate=_triple(position + turn.TransformDir(offset)),
        rotate_xyz=_rotated_euler(current.rotate_xyz, turn),
        scale=current.scale,
        origin=ParentSpaceOrigin(position=position, normal=normal, heading=heading).to_values(),
        # Left exactly as it is: the yaw was only ever a turn about the up axis, and it keeps its old meaning
        # so that scatter painted before turns were recorded composes to what it always did. This drag's turn goes
        # to the whole-turn record below, which a recompose applies on top of the composed placement.
        yaw=None,
        turn=_recorded_turn(spec, turn),
        brush_owned=current.brush_owned,
    )


def _recorded_turn(spec: Sdf.PrimSpec, turn: Gf.Matrix4d) -> tuple[float, float, float]:
    """Return the whole turn to record on a placement once this drag's turn is added to the one it already holds.

    Recorded so that a settings change recomposing the patch aims it the way the user left it, instead of standing
    every placement back up on its surface and throwing away everything but the yaw.
    """
    recorded = read_placement_turn(spec)
    return turn_to_degrees(turn if recorded is None else recorded * turn)


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
    recorded_turn = spec.attributes.get(SCATTER_TURN_ATTR)
    return PlacementTransform(
        prim_path=str(path),
        translate=_triple(translate.default),
        rotate_xyz=_triple(rotate.default),
        scale=_triple(scale.default),
        origin=() if recorded is None or recorded.default is None else tuple(float(v) for v in recorded.default),
        yaw=read_placement_yaw(spec),
        # Always a value, never None: a drag rewinds the patch to this before measuring each of its updates, and a
        # turn left behind by the update before it would be composed on top of all over again.
        turn=(0.0, 0.0, 0.0)
        if recorded_turn is None or recorded_turn.default is None
        else _triple(recorded_turn.default[:3]),
    )


def _triple(values) -> tuple[float, float, float]:
    """Return any three-component value as a tuple of three Python floats."""
    x, y, z = (float(value) for value in values)
    return (x, y, z)


class ScopeDragSession:
    """Holds a patch still while a drag is dialled in, and commits the whole drag as one undo step.

    Every update is measured from the state the patch was in when the session opened: the patch is rewound to that
    state and the total delta applied to it, so a long drag cannot accumulate the error of each step before it, and
    one command undoes the drag whole.

    The viewport gizmo and the Scatter window's turn fields are two ways of doing the same thing, so they share this
    rather than each keeping their own idea of where a drag started.

    Use from the main loop only; it is not thread-safe.
    """

    def __init__(self, usd_context_name: str, cache: MeshSurfaceCache):
        """
        Args:
            usd_context_name: USD context the drag is authored in.
            cache: Mesh cache used to find the surface under each moved placement.
        """
        self._usd_context_name = usd_context_name
        self._cache = cache
        self._scope = ReapplyScope()
        self._layer: Sdf.Layer | None = None
        self._start: list[PlacementTransform] = []
        self._moved: list[PlacementTransform] = []
        self._conform = True
        self._interaction_token = None

    @property
    def active(self) -> bool:
        """Whether a drag is open and waiting to be committed."""
        return self._interaction_token is not None

    def begin(self, scope: ReapplyScope, conform: bool) -> bool:
        """Take the patch as it stands, so every update is measured from there rather than from the last one.

        Args:
            scope: Placements the drag is allowed to touch, and nothing else.
            conform: Whether the brush is set to conform to the surface.

        Returns:
            Whether a drag was opened; there is nothing to open with no stage or an empty scope.
        """
        stage = self._stage()
        if stage is None or scope.is_empty:
            return False
        self._scope = scope
        self._conform = conform
        self._layer = stage.GetEditTarget().GetLayer()
        self._start, _after, _outcome = move_scope(
            stage, self._layer, scope, Gf.Matrix4d(1.0), self._cache, conform=conform
        )
        self._moved = []
        self._interaction_token = begin_interaction(stage)
        return True

    def update(self, delta: Gf.Matrix4d) -> MoveOutcome:
        """Move the patch to where ``delta`` from the start of the drag would put it, and author the result."""
        stage = self._stage()
        if stage is None or self._interaction_token is None:
            return MoveOutcome()
        author_placement_transforms(self._layer, self._start)
        _before, after, outcome = move_scope(stage, self._layer, self._scope, delta, self._cache, conform=self._conform)
        author_placement_transforms(self._layer, after)
        self._moved = after
        return outcome

    def commit(self) -> None:
        """Commit the drag as one undoable command; does nothing when no drag is open."""
        token, self._interaction_token = self._interaction_token, None
        if token is not None:
            end_interaction(token)
        if self._moved:
            omni.kit.commands.execute(
                "ScatterReapplyCommand",
                context_name=self._usd_context_name,
                layer_identifier=self._layer.identifier,
                before=[transform.to_dict() for transform in self._start],
                after=[transform.to_dict() for transform in self._moved],
                already_applied=True,
            )
        self._start = []
        self._moved = []

    def cancel(self) -> None:
        """Put the patch back as it was and close the drag without committing anything."""
        if self._interaction_token is not None and self._start:
            author_placement_transforms(self._layer, self._start)
        self._moved = []
        self.commit()

    def _stage(self) -> Usd.Stage | None:
        """Return the stage of the session's USD context, or None when there is none."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        return None if usd_context is None else usd_context.get_stage()
