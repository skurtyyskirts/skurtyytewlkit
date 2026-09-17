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
    "PaddingIndex",
    "ParentSpaceOrigin",
    "asset_up_correction",
    "choose_asset",
    "compose_parent_space_transform",
    "falloff_weight",
    "parent_uniform_scale",
    "placement_rng",
    "rotation_to_xyz_degrees",
    "sample_disk",
    "sample_rotation_degrees",
    "sample_scale",
    "stage_meters_per_unit",
    "stage_up_axis",
    "stamp_rng",
    "tangent_basis",
    "to_parent_space_origin",
    "turn_matrix",
    "turn_to_degrees",
    "twist_degrees",
    "up_axis_vector",
]

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .settings import Falloff, ScaleMode, ScatterAssetEntry, ScatterBrushSettings, UpAxis

_MIN_CELL_SIZE = 1e-6
_EPSILON = 1e-9
_GAUSSIAN_SIGMA = 0.4

# Radial acceptance curves over the normalized distance t in [0, 1]
_FALLOFF_CURVES: dict[Falloff, Callable[[float], float]] = {
    Falloff.CONSTANT: lambda _t: 1.0,
    Falloff.LINEAR: lambda t: 1.0 - t,
    Falloff.SMOOTH: lambda t: 1.0 - t * t * (3.0 - 2.0 * t),
    Falloff.SPHERE: lambda t: math.sqrt(max(0.0, 1.0 - t * t)),
    Falloff.GAUSSIAN: lambda t: math.exp(-0.5 * (t / _GAUSSIAN_SIGMA) ** 2),
}


def _as_array3(value) -> np.ndarray:
    """Copy any 3-component indexable (Gf vector, tuple, numpy array) into a float64 array."""
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _as_vec3d(value) -> Gf.Vec3d:
    """Copy any 3-component indexable (Gf vector, tuple, numpy array) into a Gf.Vec3d."""
    return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))


def falloff_weight(kind: Falloff, t: float) -> float:
    """Acceptance weight of a sample at normalized distance t from the brush center.

    Args:
        kind: Falloff curve to evaluate.
        t: Distance from the center divided by the brush radius. Negative values count as the center.

    Returns:
        Weight in [0, 1]; 0 for any t greater than 1.

    Raises:
        ValueError: If kind is not a supported falloff.
    """
    curve = _FALLOFF_CURVES.get(kind)
    if curve is None:
        raise ValueError(f"Unsupported falloff kind: {kind!r}")
    if t > 1.0:
        return 0.0
    return float(curve(max(0.0, float(t))))


def sample_disk(rng: np.random.Generator, count: int) -> np.ndarray:
    """Draw points uniformly distributed over the unit disk.

    Args:
        rng: Random generator that provides every draw.
        count: Number of points.

    Returns:
        (count, 2) float64 array of (x, y) offsets inside the unit disk.
    """
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    # sqrt keeps the area density uniform instead of piling samples near the center
    radii = np.sqrt(rng.random(count))
    angles = rng.random(count) * (2.0 * np.pi)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def tangent_basis(normal: np.ndarray, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal tangent frame around a surface normal.

    Args:
        normal: Surface normal; does not need to be unit length.
        hint: Optional direction whose projection on the tangent plane becomes the tangent. Ignored when it is
            parallel to the normal.

    Returns:
        (tangent, bitangent) unit vectors with cross(tangent, bitangent) pointing along the normal.

    Raises:
        ValueError: If normal has zero length.
    """
    unit_normal = _as_array3(normal)
    length = float(np.linalg.norm(unit_normal))
    if length < _EPSILON:
        raise ValueError("normal must be non-zero")
    unit_normal /= length

    tangent: np.ndarray | None = None
    if hint is not None:
        hint_array = _as_array3(hint)
        projected = hint_array - np.dot(hint_array, unit_normal) * unit_normal
        projected_length = float(np.linalg.norm(projected))
        if projected_length > _EPSILON:
            tangent = projected / projected_length
    if tangent is None:
        # The axis least aligned with the normal gives the most stable cross product
        helper = np.zeros(3, dtype=np.float64)
        helper[int(np.argmin(np.abs(unit_normal)))] = 1.0
        tangent = np.cross(helper, unit_normal)
        tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(unit_normal, tangent)
    return tangent, bitangent


def choose_asset(rng: np.random.Generator, assets: Sequence[ScatterAssetEntry]) -> ScatterAssetEntry | None:
    """Pick one enabled asset with probability proportional to its weight.

    Returns:
        The chosen entry, or None when no entry is enabled with a positive weight.
    """
    candidates = [asset for asset in assets if asset.enabled and asset.weight > 0.0]
    if not candidates:
        return None
    weights = np.array([asset.weight for asset in candidates], dtype=np.float64)
    index = int(rng.choice(len(candidates), p=weights / weights.sum()))
    return candidates[index]


def _shape_unit_sample(u: float, weight: float, bias: float) -> float:
    """Reshape a uniform draw: weight concentrates it around 0.5, bias pushes it toward 1 (positive) or 0."""
    t = 0.5 + math.copysign(abs(2.0 * u - 1.0) ** weight, u - 0.5) / 2.0
    return t ** (2.0 ** (-bias))


def sample_scale(rng: np.random.Generator, settings: ScatterBrushSettings) -> tuple[float, float, float]:
    """Draw a placement scale from the brush scale settings.

    In ``ABSOLUTE_SIZE`` the draw is a target height in stage units rather than a multiplier, so one value always
    drives all three axes: a height has no per-axis meaning. Disabling randomness there pins every placement to
    ``size_min``, not to a multiplier of 1, because a fixed height is still a height.

    Returns:
        (x, y, z) scale factors, or target heights in ``ABSOLUTE_SIZE``. Uniform mode returns the same value on every
        axis, per-axis mode draws each axis independently from its own range.
    """
    if settings.scale_mode == ScaleMode.ABSOLUTE_SIZE:
        if not settings.scale_enabled:
            return (settings.size_min, settings.size_min, settings.size_min)
        t = _shape_unit_sample(rng.random(), settings.scale_weight, settings.scale_bias)
        height = settings.size_min + t * (settings.size_max - settings.size_min)
        return (height, height, height)
    if not settings.scale_enabled:
        return (1.0, 1.0, 1.0)
    if settings.scale_uniform:
        t = _shape_unit_sample(rng.random(), settings.scale_weight, settings.scale_bias)
        value = settings.scale_min + t * (settings.scale_max - settings.scale_min)
        return (value, value, value)
    ranges = (
        (settings.scale_x_min, settings.scale_x_max),
        (settings.scale_y_min, settings.scale_y_max),
        (settings.scale_z_min, settings.scale_z_max),
    )
    values = []
    for low, high in ranges:
        t = _shape_unit_sample(rng.random(), settings.scale_weight, settings.scale_bias)
        values.append(low + t * (high - low))
    return (values[0], values[1], values[2])


def sample_rotation_degrees(rng: np.random.Generator, settings: ScatterBrushSettings) -> tuple[float, float, float]:
    """Draw (x, y, z) rotation angles in degrees, each uniform within its configured range."""
    return (
        float(rng.uniform(settings.rotation_x_min, settings.rotation_x_max)),
        float(rng.uniform(settings.rotation_y_min, settings.rotation_y_max)),
        float(rng.uniform(settings.rotation_z_min, settings.rotation_z_max)),
    )


def stamp_rng(seed: int, stroke_index: int, stamp_index: int) -> np.random.Generator:
    """Create the generator for one stamp so that a stroke replays identically for the same seed."""
    return np.random.default_rng([int(seed), int(stroke_index), int(stamp_index)])


def placement_rng(key: int, seed: int) -> np.random.Generator:
    """Create the generator that composes one placement, from its own key and the brush seed.

    Giving every placement its own generator is what lets a placement be recomposed on its own: the stamp generator
    is shared by all the placements of a stamp, so no placement's draws can be replayed from it afterwards. The brush
    seed is mixed in so that changing the seed, or rerolling it, draws a new arrangement for placements that already
    exist; the stroke seed is already part of ``key`` through the stamp generator that drew it.

    Args:
        key: The placement's own key, drawn once when it was painted.
        seed: ``ScatterBrushSettings.seed``.

    Returns:
        The generator the placement's rotation and scale are drawn from.
    """
    return np.random.default_rng([int(key), int(seed)])


def up_axis_vector(up_axis: UpAxis) -> Gf.Vec3d:
    """Unit vector of a stage or asset up axis."""
    if UpAxis(up_axis) == UpAxis.Y:
        return Gf.Vec3d(0.0, 1.0, 0.0)
    return Gf.Vec3d(0.0, 0.0, 1.0)


def stage_up_axis(stage: Usd.Stage) -> UpAxis:
    """Up axis authored on the stage (UsdGeom fallback is Y)."""
    return UpAxis.Z if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else UpAxis.Y


def stage_meters_per_unit(stage: Usd.Stage) -> float:
    """Meters per stage unit.

    Returns:
        The authored unit scale; 1.0 when the stage authors none, which matches ``read_asset_meters_per_unit`` so an
        asset and a stage that both declare nothing convert into each other unchanged.
    """
    if not stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit):
        return 1.0
    return float(UsdGeom.GetStageMetersPerUnit(stage))


def parent_uniform_scale(matrix: Gf.Matrix4d) -> float:
    """Average scale a transform applies, as the cube root of the absolute determinant of its linear part.

    The absolute value keeps a mirroring parent positive: the mirror itself is already carried by the parent's linear
    map, so only its magnitude belongs here.

    Returns:
        The factor; 1.0 for a near-degenerate matrix, so a collapsed instance cannot divide a placement's size up to
        infinity.
    """
    linear = Gf.Matrix3d(
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2],
    )  # fmt: skip
    determinant = abs(linear.GetDeterminant())
    if determinant < _EPSILON:
        return 1.0
    return float(determinant ** (1.0 / 3.0))


def asset_up_correction(asset_up: UpAxis, stage_up: UpAxis) -> Gf.Rotation:
    """Rotation that turns an asset authored with asset_up so that it stands up on a stage_up stage.

    Returns:
        Identity when the axes match, +90 degrees about X for a Y-up asset on a Z-up stage and -90 degrees about X
        for a Z-up asset on a Y-up stage.
    """
    if UpAxis(asset_up) == UpAxis(stage_up):
        return Gf.Rotation(Gf.Vec3d.XAxis(), 0.0)
    angle = 90.0 if UpAxis(stage_up) == UpAxis.Z else -90.0
    return Gf.Rotation(Gf.Vec3d.XAxis(), angle)


def rotation_to_xyz_degrees(matrix_or_rotation: Gf.Rotation | Gf.Matrix4d | Gf.Matrix3d) -> Gf.Vec3f:
    """Convert a rotation into the Euler angles of a UsdGeom rotateXYZ op.

    The op applies X, then Y, then Z; in row-vector terms the matrix is Rx * Ry * Rz. Decomposing about
    (Z, Y, X) and reversing the result is the same idiom Kit's TransformPrimSRT uses for rotateXYZ.

    Args:
        matrix_or_rotation: Rotation, or a matrix whose rotation part is used (scale and shear are discarded).

    Returns:
        (x, y, z) angles in degrees.

    Raises:
        TypeError: If the value is not a Gf rotation or matrix.
    """
    if isinstance(matrix_or_rotation, Gf.Rotation):
        rotation = matrix_or_rotation
    elif isinstance(matrix_or_rotation, Gf.Matrix4d | Gf.Matrix3d):
        rotation = matrix_or_rotation.GetOrthonormalized().ExtractRotation()
    else:
        raise TypeError(f"Expected Gf.Rotation, Gf.Matrix4d or Gf.Matrix3d, got {type(matrix_or_rotation)!r}")
    angles = rotation.Decompose(Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis())
    return Gf.Vec3f(angles[2], angles[1], angles[0])


def _normal_to_parent_space(normal_world: Gf.Vec3d, parent_world: Gf.Matrix4d) -> Gf.Vec3d:
    """Bring a world normal into parent space with the inverse-transpose of the parent's direction transform.

    Directions go world -> parent through the inverse of parent_world; normals go through the inverse-transpose
    of that, which is the transpose of parent_world itself.
    """
    normal = parent_world.GetTranspose().TransformDir(normal_world)
    if normal.GetLength() < _EPSILON:
        raise ValueError("normal_world must be non-zero")
    return normal.GetNormalized()


def _heading_yaw_degrees(heading: Gf.Vec3d, effective_up: Gf.Vec3d, conform: Gf.Matrix4d) -> float:
    """Yaw about the stage up axis that turns the asset's +X toward the heading after the conform rotation.

    Returns:
        Signed angle in degrees, 0 when the heading has no component in the tangent plane.
    """
    # +X is perpendicular to both supported up axes, so the yaw is measured from where conform sends it
    forward = conform.TransformDir(Gf.Vec3d.XAxis())
    side = Gf.Cross(effective_up, forward)
    along = Gf.Dot(heading, forward)
    across = Gf.Dot(heading, side)
    if math.hypot(along, across) < _EPSILON:
        return 0.0
    return math.degrees(math.atan2(across, along))


@dataclass(frozen=True)
class ParentSpaceOrigin:
    """The surface a placement was composed from, in the space of the instance that parents the placement.

    Held in parent space rather than world space so that it survives the instance moving: a placement is a child of
    that instance and keeps its relationship to the surface, which a stored world point would not.

    Attributes:
        position: Surface point the placement sits on, before the vertical offset is applied.
        normal: Surface normal at that point, normalized.
        heading: Stroke direction the stamp travelled in, or None when the stamp had none.
    """

    position: Gf.Vec3d
    normal: Gf.Vec3d
    heading: Gf.Vec3d | None = None

    def to_values(self) -> tuple[float, ...]:
        """Flatten the origin into the six or nine doubles a placement authors; no heading means six."""
        values = [*self.position, *self.normal, *(self.heading or ())]
        return tuple(float(value) for value in values)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> ParentSpaceOrigin | None:
        """Rebuild an origin from authored doubles.

        Args:
            values: Six or nine doubles as produced by :meth:`to_values`.

        Returns:
            The origin, or None when the values are neither six nor nine, which is how a placement authored by
            something other than this brush is recognized.
        """
        if len(values) not in (6, 9):
            return None
        heading = Gf.Vec3d(*(float(value) for value in values[6:9])) if len(values) == 9 else None
        return cls(
            position=Gf.Vec3d(*(float(value) for value in values[0:3])),
            normal=Gf.Vec3d(*(float(value) for value in values[3:6])),
            heading=heading,
        )


def to_parent_space_origin(
    position_world: Gf.Vec3d,
    normal_world: Gf.Vec3d,
    parent_world: Gf.Matrix4d,
    heading_world: Gf.Vec3d | None = None,
) -> ParentSpaceOrigin:
    """Bring a sampled surface into the space of the instance that will parent the placement.

    Args:
        position_world: Surface point in world space.
        normal_world: Surface normal in world space.
        parent_world: World transform of the instance that parents the placement.
        heading_world: Stroke direction in world space, or None when the stamp had none.

    Returns:
        The origin :func:`compose_parent_space_transform` composes from.

    Raises:
        ValueError: If normal_world has zero length.
    """
    inverse_parent = parent_world.GetInverse()
    return ParentSpaceOrigin(
        position=inverse_parent.Transform(_as_vec3d(position_world)),
        normal=_normal_to_parent_space(_as_vec3d(normal_world), parent_world),
        heading=None if heading_world is None else inverse_parent.TransformDir(_as_vec3d(heading_world)),
    )


def compose_parent_space_transform(
    origin: ParentSpaceOrigin,
    parent_world: Gf.Matrix4d,
    settings: ScatterBrushSettings,
    rng: np.random.Generator,
    stage_up: UpAxis,
    asset_up: UpAxis,
    asset_scale_factor: float | None = 1.0,
    yaw_offset: float = 0.0,
    turn: Gf.Matrix4d | None = None,
) -> tuple[Gf.Vec3d, Gf.Vec3f, Gf.Vec3f]:
    """Compute the translate / rotateXYZ / scale ops of one placement in the space of its parent instance.

    The rotation applied to the asset is, first to last: the asset up-axis correction, tilt about X, tilt about Y,
    yaw about the stage up axis (random spin plus stroke heading), then the rotation taking the stage up axis onto
    the surface normal when conform_to_surface is set.

    The parent's own scale is divided out of both the authored scale and the vertical offset, so a placement comes
    out the same size wherever it is painted; this makes them world-space quantities like radius, padding and
    stamp_spacing. Only the average magnitude is compensated, so a non-uniform parent still stretches its placements
    along its own axes.

    Composing from a stored origin rather than from the sampled surface is what lets a placement that already exists
    be recomposed under new settings, with the same code that painted it.

    Args:
        origin: Surface the placement sits on, in the space of its parent instance.
        parent_world: World transform of the instance that parents the placement.
        settings: Brush settings providing offsets, rotation and scale ranges.
        rng: Random generator for the rotation and scale draws.
        stage_up: Up axis of the stage.
        asset_up: Up axis the asset was authored with.
        asset_scale_factor: Multiplier turning a sampled scale value into what is authored: 1.0 leaves the sample a
            plain multiplier, the reciprocal of the asset's height in stage units turns a sampled target height into
            one. None means the asset could not be measured, so the placement keeps its own size.
        yaw_offset: Degrees about the stage up axis to add to the sampled yaw, so a placement the move gizmo turned
            keeps that turn when it is composed again.
        turn: Whole turn the gizmo or the Scatter window's turn fields gave the placement, in parent space, applied
            last so that it aims the placement the composed one ends up at. None leaves it unturned. A yaw arrives
            through ``yaw_offset`` instead and keeps its old place in the chain, so scatter painted before turns
            were recorded composes to exactly what it did then.

    Returns:
        (translate, rotate_xyz_degrees, scale) such that authoring them with xformOpOrder
        [translate, rotateXYZ, scale] reproduces the composed transform.
    """
    parent_scale = parent_uniform_scale(parent_world)
    normal = origin.normal
    up = up_axis_vector(stage_up)
    # The axis both the offset and the yaw are measured about. Without conform this stays the plain axis: it is the
    # one the asset ends up standing on, and it is also the axis the yaw below is applied about before conform.
    effective_up = normal if settings.conform_to_surface else up
    # Only the magnitude needs the parent out of it; vertical_offset is a world distance like radius and padding.
    position = origin.position + effective_up * (settings.vertical_offset / parent_scale)

    rotate_x, rotate_y, rotate_z = sample_rotation_degrees(rng, settings)
    # Drawn even when the sample goes unused, so an unmeasured asset does not shift the stamp's random stream.
    sampled_scale = sample_scale(rng, settings)
    if asset_scale_factor is None:
        scale = (1.0 / parent_scale, 1.0 / parent_scale, 1.0 / parent_scale)
    else:
        factor = asset_scale_factor / parent_scale
        scale = (sampled_scale[0] * factor, sampled_scale[1] * factor, sampled_scale[2] * factor)

    conform = Gf.Matrix4d(1.0)
    if settings.conform_to_surface:
        conform = Gf.Matrix4d().SetRotate(Gf.Rotation(up, normal))
    yaw = rotate_z + yaw_offset
    if settings.align_to_stroke and origin.heading is not None:
        yaw += _heading_yaw_degrees(origin.heading, effective_up, conform)

    # Row-vector products: the first rotation applied to the asset is on the left
    rotation = (
        Gf.Matrix4d().SetRotate(asset_up_correction(asset_up, stage_up))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.XAxis(), rotate_x))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.YAxis(), rotate_y))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(up, yaw))
        * conform
    )
    if turn is not None:
        # Last, and after conform: the gizmo turns a placement that is already standing on its surface, so a turn
        # kept here has to be applied to the same thing the gizmo applied it to.
        rotation = rotation * turn
    return position, rotation_to_xyz_degrees(rotation), Gf.Vec3f(scale[0], scale[1], scale[2])


def turn_matrix(degrees: Sequence[float]) -> Gf.Matrix4d:
    """Return the rotation an X, Y, Z turn in degrees stands for, composed in that order.

    The same order ``rotation_to_xyz_degrees`` reads back, so a turn written to the Scatter window's fields and one
    read out of a gizmo drag mean the same thing.
    """
    return (
        Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.XAxis(), float(degrees[0])))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.YAxis(), float(degrees[1])))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d.ZAxis(), float(degrees[2])))
    )


def turn_to_degrees(turn: Gf.Matrix4d) -> tuple[float, float, float]:
    """Return the X, Y, Z degrees a rotation stands for, as ``turn_matrix`` would rebuild it."""
    degrees = rotation_to_xyz_degrees(turn)
    return (float(degrees[0]), float(degrees[1]), float(degrees[2]))


def twist_degrees(rotation: Gf.Rotation, axis: Gf.Vec3d) -> float:
    """Return how far a rotation turns about an axis, ignoring the part of it that tips the axis over.

    This is the swing-twist decomposition: the quaternion's vector part is projected onto the axis, and the twist is
    the rotation that projection describes. The move gizmo uses it to record the turn it applied about the stage up
    axis, which is the only part of a drag a later recomposition can reproduce through the yaw it samples.

    Args:
        rotation: Rotation to decompose.
        axis: Axis to measure the turn about; need not be normalized.

    Returns:
        The angle in degrees, in [-180, 180]. Zero when the axis has no length.
    """
    length = axis.GetLength()
    if length < _EPSILON:
        return 0.0
    unit_axis = axis / length
    quaternion = rotation.GetQuaternion()
    projection = unit_axis * Gf.Dot(quaternion.GetImaginary(), unit_axis)
    twist = Gf.Quatd(quaternion.GetReal(), projection)
    if twist.GetLength() < _EPSILON:
        # The rotation is a half turn about an axis perpendicular to this one, which carries no twist about it.
        return 0.0
    return Gf.Rotation(twist.GetNormalized()).GetAngle() * (1.0 if Gf.Dot(projection, unit_axis) >= 0.0 else -1.0)


class PaddingIndex:
    """Uniform hash grid answering "is any stored point closer than a distance" for padding checks.

    Points are bucketed by floor(point / cell_size); queries walk the neighbouring cells, or every bucket when the
    neighbourhood would be larger than the number of occupied cells.
    """

    def __init__(self, cell_size: float):
        self._cell_size = max(float(cell_size), _MIN_CELL_SIZE)
        self._cells: dict[tuple[int, int, int], list[np.ndarray]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(self, point) -> None:
        """Store one point (any 3-component indexable)."""
        stored = _as_array3(point)
        self._cells.setdefault(self._cell_of(stored), []).append(stored)
        self._count += 1

    def add_many(self, points: np.ndarray) -> None:
        """Store every row of a (K, 3) array."""
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return
        for point in array.reshape(-1, 3):
            self.add(point)

    def is_free(self, point, min_distance: float) -> bool:
        """Whether no stored point lies strictly closer than min_distance to the point."""
        if min_distance <= 0.0 or not self._cells:
            return True
        query = _as_array3(point)
        reach = math.ceil(min_distance / self._cell_size)
        limit = min_distance * min_distance
        for bucket in self._candidate_buckets(self._cell_of(query), reach):
            deltas = np.asarray(bucket) - query
            if bool(np.any(np.einsum("ij,ij->i", deltas, deltas) < limit)):
                return False
        return True

    def _cell_of(self, point: np.ndarray) -> tuple[int, int, int]:
        """Integer grid coordinates of the cell containing the point."""
        cell = np.floor(point / self._cell_size)
        return (int(cell[0]), int(cell[1]), int(cell[2]))

    def _candidate_buckets(self, cell: tuple[int, int, int], reach: int) -> Iterable[list[np.ndarray]]:
        """Buckets that can hold a point within reach cells of the given cell.

        Falls back to every bucket when the neighbourhood would be larger than the number of occupied cells, which
        happens when min_distance is much larger than the cell size.
        """
        if (2 * reach + 1) ** 3 > len(self._cells):
            return self._cells.values()
        keys = (
            (x, y, z)
            for x in range(cell[0] - reach, cell[0] + reach + 1)
            for y in range(cell[1] - reach, cell[1] + reach + 1)
            for z in range(cell[2] - reach, cell[2] + reach + 1)
        )
        return (self._cells[key] for key in keys if key in self._cells)
