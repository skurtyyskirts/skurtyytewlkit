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

__all__ = ["MoveOutcome", "PreparedMove", "ScopeDragSession", "move_scope", "prepare_move", "scope_pivot"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import omni.kit.commands
import omni.usd
from omni.flux.utils.common.interactive_usd_notices import begin_interaction, end_interaction
from pxr import Gf, Sdf, Usd

from .constants import SCATTER_ORIGIN_ATTR, SCATTER_TURN_ATTR
from .controller import get_scatter_brush_controller
from .placement import PlacementTransform, author_placement_transforms
from .reapply import ReapplyScope, is_hand_edited, read_placement_origin, read_placement_turn, read_placement_yaw
from .settings import LIVE_EDIT_SOFT_LIMIT
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
# Relative slack on how far a slid placement looks for the surface, for the rounding between where a surface was
# recorded and where it is projected again.
_REACH_SLACK = 1e-6


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

    A drag reads the patch once through :func:`prepare_move` and applies every step to that; this is the same thing
    in one call.

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
    prepared = prepare_move(stage, layer, scope, cache)
    after, outcome = prepared.apply(delta, cache, conform)
    return prepared.start, after, outcome


@dataclass
class _PreparedPlacement:
    """What a drag reads from one placement, read once when the drag opens.

    Attributes:
        current: The placement as it stood, with the surface and turn it recorded and whether it is the brush's.
        origin: Surface it stands on: the recorded one, or one projected for a placement that recorded none.
        recorded_turn: Whole turn it recorded, or None.
        recovered: Whether ``origin`` had to be projected because the placement recorded none.
        on_surface: Whether ``origin`` lies on the capture mesh, which bounds how far a slide has to look for it.
    """

    current: PlacementTransform
    origin: ParentSpaceOrigin
    recorded_turn: Gf.Matrix4d | None
    recovered: bool
    on_surface: bool


@dataclass
class _PreparedEntry:
    """The prepared placements of one prototype and the target they are authored under."""

    target: ScatterTarget
    placements: list[_PreparedPlacement]
    reach: float


class PreparedMove:
    """A patch read once, so every step of a drag is measured from the same start without reading it back."""

    def __init__(self, entries: list[_PreparedEntry], foreign_layer: int):
        """
        Args:
            entries: The prepared placements, per prototype.
            foreign_layer: Placements of the scope the layer authors no transform for.
        """
        self._entries = entries
        self._foreign_layer = foreign_layer

    @property
    def start(self) -> list[PlacementTransform]:
        """The placements as they stood when the drag opened, in scope order."""
        return [placement.current for entry in self._entries for placement in entry.placements]

    def apply(
        self, delta: Gf.Matrix4d, cache: MeshSurfaceCache, conform: bool = True
    ) -> tuple[list[PlacementTransform], MoveOutcome]:
        """Return the patch as a drag of ``delta`` from the start leaves it, without writing anything.

        Args:
            delta: World transform of the drag; see :func:`move_scope`.
            cache: Mesh cache used to find the surface under each moved placement.
            conform: Whether the brush is set to conform to the surface.

        Returns:
            ``(after, outcome)``: the placements in the order of :attr:`start`, and what happened to them.
        """
        outcome = MoveOutcome(foreign_layer=self._foreign_layer)
        after: list[PlacementTransform] = []
        rotation = delta.ExtractRotation()
        # The gizmo's arrows and its rings are separate gestures, so a drag is either a slide or a turn, and the two
        # mean different things. A slide carries the patch across the mesh and re-plants it on what it lands on. A
        # turn only aims it. Every placement of a drag is treated the same way, so a patch cannot come apart.
        turning = abs(rotation.GetAngle()) > _TURN_EPSILON
        for entry in self._entries:
            target = entry.target
            parent_delta = target.parent_world * delta * target.parent_world.GetInverse()
            # Everything a placement holds - its transform, the surface it recorded, its heading - is in the space
            # of the instance that parents it, and a capture's instances stand at whatever angle the game left them
            # at. The rings send a world rotation, so it is carried into that space here: applied as if the instance
            # stood square to the stage, a ring would aim the patch about some other axis than the one grabbed.
            # Orthonormalized first, because an instance is free to carry scale and a scaled conjugation is no
            # longer a rotation.
            parent_rotation = parent_delta.GetOrthonormalized().ExtractRotation()
            surfaces = None if turning else _settle_entry(entry, parent_delta, cache)
            for index, placement in enumerate(entry.placements):
                after.append(
                    _move_placement(
                        placement,
                        parent_rotation,
                        None if surfaces is None else surfaces[index],
                        parent_delta,
                        turning,
                        conform,
                        outcome,
                    )
                )
                outcome.moved += 1
        return after, outcome


def prepare_move(stage: Usd.Stage, layer: Sdf.Layer, scope: ReapplyScope, cache: MeshSurfaceCache) -> PreparedMove:
    """Read every placement of ``scope`` a drag may move, once.

    Args:
        stage: Stage holding the placements.
        layer: Layer to read the current transforms from.
        scope: Placements the drag is allowed to touch.
        cache: Mesh cache, used to project a surface for a placement that recorded none.

    Returns:
        The prepared patch.
    """
    entries: list[_PreparedEntry] = []
    foreign_layer = 0
    for entry in scope.entries:
        target = build_prototype_target(stage, entry.prototype_root)
        if target is None:
            foreign_layer += len(entry.placement_paths)
            continue
        reach = _settle_reach(target, cache)
        placements: list[_PreparedPlacement] = []
        for path in entry.placement_paths:
            spec = layer.GetPrimAtPath(path)
            current = None if spec is None else _read_transform(spec, path)
            if current is None:
                foreign_layer += 1
                continue
            # Set here so that undoing the drag restores a hand-edited placement as the user's rather than stamping
            # it as the brush's.
            current.brush_owned = not is_hand_edited(spec, current)
            origin = read_placement_origin(spec)
            recovered = origin is None
            on_surface = True
            if origin is None:
                origin, on_surface = _origin_under(current, target, cache, reach)
            placements.append(
                _PreparedPlacement(
                    current=current,
                    origin=origin,
                    recorded_turn=read_placement_turn(spec),
                    recovered=recovered,
                    on_surface=on_surface,
                )
            )
        entries.append(_PreparedEntry(target=target, placements=placements, reach=reach))
    return PreparedMove(entries, foreign_layer)


def _settle_entry(
    entry: _PreparedEntry, parent_delta: Gf.Matrix4d, cache: MeshSurfaceCache
) -> list[tuple[Gf.Vec3d, Gf.Vec3d] | None]:
    """Find the surface every slid placement of one prototype lands on, in one query of its capture mesh.

    The query only reaches as far as the slide carried a placement: a placement stood on the mesh before the slide,
    so the nearest surface is never farther away than that, and limiting the reach to it finds exactly the point a
    search of the whole mesh would. A placement whose surface was never on the mesh gets the whole mesh's reach.

    Returns:
        Per placement, the ``(position, normal)`` it settles on in parent space, or None when the mesh is out of
        reach.
    """
    if not entry.placements:
        return []
    target = entry.target
    world_points = []
    reach = 0.0
    for placement in entry.placements:
        rigid_world = target.parent_world.Transform(parent_delta.Transform(placement.origin.position))
        world_points.append((rigid_world[0], rigid_world[1], rigid_world[2]))
        if placement.on_surface:
            moved = (rigid_world - target.parent_world.Transform(placement.origin.position)).GetLength()
            reach = max(reach, moved)
        else:
            reach = entry.reach
    # Slack for the rounding between where the surface was recorded and where it is projected again.
    reach = min(entry.reach, reach * (1.0 + _REACH_SLACK) + _REACH_SLACK * entry.reach)
    positions, normals, valid = cache.closest_points(
        target.mesh_path, np.asarray(world_points, dtype=np.float64), reach
    )
    surfaces: list[tuple[Gf.Vec3d, Gf.Vec3d] | None] = []
    for position, normal, is_valid in zip(positions, normals, valid):
        if not is_valid:
            surfaces.append(None)
            continue
        origin = to_parent_space_origin(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
            Gf.Vec3d(float(normal[0]), float(normal[1]), float(normal[2])),
            target.parent_world,
        )
        surfaces.append((origin.position, origin.normal))
    return surfaces


def _move_placement(
    placement: _PreparedPlacement,
    parent_rotation: Gf.Rotation,
    surface: tuple[Gf.Vec3d, Gf.Vec3d] | None,
    parent_delta: Gf.Matrix4d,
    turning: bool,
    conform: bool,
    outcome: MoveOutcome,
) -> PlacementTransform:
    """Return one placement as the drag leaves it, recording on ``outcome`` what happened to it.

    A turn leaves the placement standing exactly where it is and only changes the way it points. A slide moves it,
    re-plants it on the surface it lands on, and turns it onto that surface when ``conform`` is set.
    """
    current = placement.current
    origin = placement.origin
    if placement.recovered:
        outcome.recovered += 1
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
    elif surface is None:
        outcome.unsettled += 1
        position, normal, settle = parent_delta.Transform(origin.position), origin.normal, _NO_ROTATION
    else:
        outcome.settled += 1
        position, normal = surface
        # Turned onto the ground it landed on only when the brush conforms, so a patch of assets the brush was told
        # to stand straight up is not tipped over by the surface it is slid across.
        settle = Gf.Rotation(origin.normal, normal) if conform else _NO_ROTATION

    if not turning and (settle is _NO_ROTATION or normal == origin.normal):
        # A slide over ground facing the way it faced where the placement started turns nothing, so the rotation and
        # the recorded turn it already carries are kept exactly instead of being decomposed and composed again.
        return PlacementTransform(
            prim_path=current.prim_path,
            translate=_triple(position + offset),
            rotate_xyz=current.rotate_xyz,
            scale=current.scale,
            origin=ParentSpaceOrigin(position=position, normal=normal, heading=origin.heading).to_values(),
            yaw=None,
            turn=(0.0, 0.0, 0.0) if placement.recorded_turn is None else current.turn,
            brush_owned=current.brush_owned,
        )

    turn = Gf.Matrix4d().SetRotate(parent_rotation) * Gf.Matrix4d().SetRotate(settle)
    heading = None if origin.heading is None else turn.TransformDir(origin.heading)
    recorded = placement.recorded_turn
    return PlacementTransform(
        prim_path=current.prim_path,
        translate=_triple(position + turn.TransformDir(offset)),
        rotate_xyz=_rotated_euler(current.rotate_xyz, turn),
        scale=current.scale,
        origin=ParentSpaceOrigin(position=position, normal=normal, heading=heading).to_values(),
        # Left exactly as it is: the yaw was only ever a turn about the up axis, and it keeps its old meaning so that
        # scatter painted before turns were recorded composes to what it always did. This drag's turn goes to the
        # whole-turn record, which a recompose applies on top of the composed placement.
        yaw=None,
        # Recorded so that a settings change recomposing the patch aims it the way the user left it, instead of
        # standing every placement back up on its surface and throwing away everything but the yaw.
        turn=turn_to_degrees(turn if recorded is None else recorded * turn),
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
) -> tuple[ParentSpaceOrigin, bool]:
    """Build the surface a placement that recorded none is standing on, from where it sits now.

    Falling back to the stage up axis when the mesh cannot be reached keeps the drag rigid for that placement rather
    than dropping it, which is the same bargain ``move_scope`` makes for a placement that cannot settle.

    Returns:
        ``(origin, on_surface)``, where ``on_surface`` is False for the fallback that is not on the mesh.
    """
    position = Gf.Vec3d(*current.translate)
    settled = _surface_under(position, target, cache, reach)
    if settled is None:
        return ParentSpaceOrigin(position=position, normal=Gf.Vec3d(0.0, 0.0, 1.0)), False
    return ParentSpaceOrigin(position=settled[0], normal=settled[1]), True


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
    return (float(values[0]), float(values[1]), float(values[2]))


class ScopeDragSession:
    """Holds a patch still while a drag is dialled in, and commits the whole drag as one undo step.

    Every update is measured from the state the patch was in when the session opened, read once then: the total
    delta is applied to that state, so a long drag cannot accumulate the error of each step before it, and one
    command undoes the drag whole. A drag of the stroke the brush is previewing adds no undo step at all, because it
    is part of that stroke. A patch of more than ``LIVE_EDIT_SOFT_LIMIT`` placements moves once, when the drag ends,
    the way a live settings edit of one does.

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
        self._prepared: PreparedMove | None = None
        self._start: list[PlacementTransform] = []
        self._moved: list[PlacementTransform] = []
        self._conform = True
        self._interaction_token = None
        # For a patch too large to follow every step: the delta the drag has reached, applied when it ends.
        self._deferred = False
        self._pending_delta: Gf.Matrix4d | None = None

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
        self._prepared = prepare_move(stage, self._layer, scope, self._cache)
        self._start = self._prepared.start
        self._moved = []
        self._deferred = len(self._start) > LIVE_EDIT_SOFT_LIMIT
        self._pending_delta = None
        self._interaction_token = begin_interaction(stage)
        return True

    def update(self, delta: Gf.Matrix4d) -> MoveOutcome:
        """Move the patch to where ``delta`` from the start of the drag would put it, and author the result."""
        if self._stage() is None or self._interaction_token is None or self._prepared is None:
            return MoveOutcome()
        if self._deferred:
            self._pending_delta = delta
            return MoveOutcome()
        # Every placement of the scope is written, so the result replaces the previous step without the patch having
        # to be put back first.
        after, outcome = self._prepared.apply(delta, self._cache, self._conform)
        author_placement_transforms(self._layer, after)
        self._moved = after
        return outcome

    def commit(self) -> None:
        """Commit the drag as one undoable command; does nothing when no drag is open."""
        if self._pending_delta is not None and self._prepared is not None and self._interaction_token is not None:
            self._moved, _outcome = self._prepared.apply(self._pending_delta, self._cache, self._conform)
            author_placement_transforms(self._layer, self._moved)
        self._pending_delta = None
        token, self._interaction_token = self._interaction_token, None
        if token is not None:
            end_interaction(token)
        if self._moved and not self._folds_into_preview():
            omni.kit.commands.execute(
                "ScatterReapplyCommand",
                context_name=self._usd_context_name,
                layer_identifier=self._layer.identifier,
                before=self._start,
                after=self._moved,
                already_applied=True,
            )
        self._prepared = None
        self._start = []
        self._moved = []

    def cancel(self) -> None:
        """Put the patch back as it was and close the drag without committing anything."""
        self._pending_delta = None
        if self._interaction_token is not None and self._start:
            author_placement_transforms(self._layer, self._start)
        self._moved = []
        self.commit()

    def _folds_into_preview(self) -> bool:
        """Whether the drag moved the stroke the brush is previewing, which keeps it instead of an undo step."""
        return get_scatter_brush_controller().folds_into_preview(
            self._usd_context_name, self._layer.identifier, self._scope
        )

    def _stage(self) -> Usd.Stage | None:
        """Return the stage of the session's USD context, or None when there is none."""
        usd_context = omni.usd.get_context(self._usd_context_name)
        return None if usd_context is None else usd_context.get_stage()
