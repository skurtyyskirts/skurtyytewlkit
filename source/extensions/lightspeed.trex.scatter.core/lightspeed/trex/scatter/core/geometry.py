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
    "MeshGeometry",
    "MeshSurfaceCache",
    "SurfaceSample",
    "area_weighted_triangle_samples",
    "build_mesh_geometry",
    "closest_point_on_triangles",
    "closest_points",
    "raycast",
    "triangulate_faces",
]

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import carb
import numpy as np
from omni.flux.utils.common.interactive_usd_notices import register_objects_changed_listener
from pxr import Gf, Sdf, Usd, UsdGeom

# Mesh properties whose change invalidates the cached triangles of that mesh (and only that mesh).
_GEOMETRY_PROPERTY_NAMES = frozenset(
    {
        UsdGeom.Tokens.points,
        UsdGeom.Tokens.faceVertexCounts,
        UsdGeom.Tokens.faceVertexIndices,
        UsdGeom.Tokens.normals,
        UsdGeom.Tokens.orientation,
        f"primvars:{UsdGeom.Tokens.normals}",
    }
)
# Matches every xformOp:* attribute as well as xformOpOrder.
_XFORM_PROPERTY_PREFIX = "xformOp"
_RAY_EPSILON = 1e-6
_PARALLEL_EPSILON = 1e-12
_BARYCENTRIC_EPSILON = 1e-9
_DEGENERATE_CROSS_LENGTH = 1e-12
# Upper bound on the (points x triangles) boolean mask evaluated in one numpy chunk during candidate search.
_PAIR_CHUNK_BUDGET = 1 << 22

_Vector3Like = Any


@dataclass
class MeshGeometry:
    """World-space triangle soup of one mesh prim with per-triangle acceleration data.

    Arrays are shared between cache users and must be treated as read-only.

    Attributes:
        vertices: (N, 3) float64 world-space vertex positions.
        triangles: (M, 3) int64 vertex indices per triangle.
        face_normals: (M, 3) float64 unit normals oriented per authored normals / orientation.
        tri_min: (M, 3) per-triangle bounding-box minimum.
        tri_max: (M, 3) per-triangle bounding-box maximum.
        bbox_min: (3,) mesh bounding-box minimum.
        bbox_max: (3,) mesh bounding-box maximum.
        tri_areas: (M,) per-triangle area.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    face_normals: np.ndarray
    tri_min: np.ndarray
    tri_max: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    tri_areas: np.ndarray

    @property
    def area(self) -> float:
        """Total surface area of all triangles."""
        return float(self.tri_areas.sum())

    @property
    def triangle_count(self) -> int:
        """Number of triangles in the geometry."""
        return int(self.triangles.shape[0])


@dataclass
class SurfaceSample:
    """One point on a mesh surface.

    Attributes:
        position: (3,) world-space position on the surface.
        normal: (3,) unit world-space normal of the triangle that owns the position.
        triangle_index: Index of that triangle in the owning ``MeshGeometry``.
        distance: Distance from the query point (or ray origin) to the position.
    """

    position: np.ndarray
    normal: np.ndarray
    triangle_index: int
    distance: float


def triangulate_faces(counts: Any, indices: Any) -> np.ndarray:
    """Fan-triangulate polygon faces into a triangle index array.

    Faces with fewer than three vertices and faces whose index range runs past the end of ``indices`` are skipped
    without shifting the offsets of the faces that follow them.

    Args:
        counts: Per-face vertex counts (``faceVertexCounts``).
        indices: Flattened face-vertex indices (``faceVertexIndices``).

    Returns:
        (M, 3) int64 array of vertex indices, one row per triangle, in face order.
    """
    face_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    vertex_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if face_counts.size == 0 or vertex_indices.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    face_ends = np.cumsum(face_counts)
    face_starts = face_ends - face_counts
    usable = (face_counts >= 3) & (face_ends <= vertex_indices.size) & (face_starts >= 0)
    triangles_per_face = np.where(usable, face_counts - 2, 0)
    total = int(triangles_per_face.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)
    face_of_triangle = np.repeat(np.arange(face_counts.size), triangles_per_face)
    triangle_starts = np.cumsum(triangles_per_face) - triangles_per_face
    fan_offset = np.arange(total) - triangle_starts[face_of_triangle] + 1
    first_slot = face_starts[face_of_triangle]
    return np.stack(
        (
            vertex_indices[first_slot],
            vertex_indices[first_slot + fan_offset],
            vertex_indices[first_slot + fan_offset + 1],
        ),
        axis=1,
    )


def build_mesh_geometry(mesh_prim: Usd.Prim | None, time_code: Usd.TimeCode | None = None) -> MeshGeometry | None:
    """Build world-space triangle data for a ``UsdGeom.Mesh`` prim.

    Face normals come from the triangle winding, flipped for ``leftHanded`` meshes and for mirroring world transforms;
    when the mesh authors ``normals`` (or ``primvars:normals``) the sign of each face normal is aligned with the mean
    authored normal of its corners.

    Args:
        mesh_prim: Prim to read. Anything that is not a valid ``UsdGeom.Mesh`` yields None.
        time_code: Time at which attributes and transforms are evaluated. Defaults to ``Usd.TimeCode.Default()``.

    Returns:
        The geometry, or None when the prim is not a mesh or has no usable points / faces.
    """
    if mesh_prim is None or not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        return None
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get(time_code)
    face_counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
    face_indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
    if not points or not face_counts or not face_indices:
        return None

    local_vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(face_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    # Triangulating the slot numbers yields, per corner, the face-vertex slot; the vertex ids and face ids follow.
    corner_slots = triangulate_faces(counts, np.arange(indices.size, dtype=np.int64))
    if corner_slots.shape[0] == 0:
        return None
    triangles = indices[corner_slots]
    face_of_triangle = np.repeat(np.arange(counts.size, dtype=np.int64), np.maximum(counts, 0))[corner_slots[:, 0]]
    in_range = np.all((triangles >= 0) & (triangles < local_vertices.shape[0]), axis=1)
    if not in_range.all():
        carb.log_verbose(
            f"[lightspeed.trex.scatter.core] {mesh_prim.GetPath()}: dropping "
            f"{int(np.count_nonzero(~in_range))} triangles with out-of-range vertex indices"
        )
        triangles = triangles[in_range]
        corner_slots = corner_slots[in_range]
        face_of_triangle = face_of_triangle[in_range]
    if triangles.shape[0] == 0:
        return None

    world_matrix = _matrix_to_numpy(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time_code))
    linear = world_matrix[:3, :3]
    vertices = local_vertices @ linear + world_matrix[3, :3]
    flip_faces = mesh.GetOrientationAttr().Get(time_code) == UsdGeom.Tokens.leftHanded
    # A mirroring transform reverses the winding seen in world space while the surface still faces the same way.
    if np.linalg.det(linear) < 0.0:
        flip_faces = not flip_faces

    corner_normals = _authored_corner_normals(
        mesh, time_code, triangles, corner_slots, face_of_triangle, local_vertices.shape[0], indices.size, counts.size
    )
    if corner_normals is not None:
        try:
            corner_normals = corner_normals @ np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            corner_normals = None
    return _assemble_geometry(vertices, triangles, flip_faces, corner_normals)


def closest_point_on_triangles(
    point: np.ndarray, geometry: MeshGeometry, max_distance: float | None = None
) -> SurfaceSample | None:
    """Find the closest point on the geometry to a query point.

    Only triangles whose bounding box, expanded by ``max_distance``, contains the point are considered.

    Args:
        point: (3,) world-space query point.
        geometry: Triangles to query.
        max_distance: Optional cut-off; results farther than this yield None. None considers every triangle.

    Returns:
        The nearest surface sample, or None when no triangle qualifies.
    """
    query = _as_vector3(point)[None, :]
    positions, normals, distances, triangle_ids, valid = _nearest_on_surface(query, geometry, max_distance)
    if not valid[0]:
        return None
    return SurfaceSample(positions[0].copy(), normals[0].copy(), int(triangle_ids[0]), float(distances[0]))


def closest_points(
    points: np.ndarray, geometry: MeshGeometry, max_distance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a batch of points onto the geometry.

    Args:
        points: (K, 3) world-space query points.
        geometry: Triangles to query.
        max_distance: Points farther than this from the surface are flagged invalid.

    Returns:
        (K, 3) positions, (K, 3) normals and a (K,) boolean validity mask. Rows flagged invalid hold zeros.
    """
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    positions, normals, _distances, _triangle_ids, valid = _nearest_on_surface(query, geometry, max_distance)
    return positions, normals, valid


def raycast(origin: np.ndarray, direction: np.ndarray, geometry: MeshGeometry) -> tuple[float, int] | None:
    """Intersect a ray with the geometry (Moller-Trumbore, both-sided).

    Args:
        origin: (3,) ray origin.
        direction: (3,) ray direction; ``t`` is expressed in units of this vector's length.
        geometry: Triangles to test.

    Returns:
        ``(t, triangle_index)`` of the nearest hit with ``t > 1e-6`` (in normalized-direction units), or None.
    """
    if geometry.triangle_count == 0:
        return None
    ray_origin = _as_vector3(origin)
    ray_direction = _as_vector3(direction)
    length = float(np.linalg.norm(ray_direction))
    if not np.isfinite(length) or length <= 0.0:
        return None
    unit_direction = ray_direction / length

    corners = geometry.vertices[geometry.triangles]
    first = corners[:, 0]
    edge_ab = corners[:, 1] - first
    edge_ac = corners[:, 2] - first
    perpendicular = np.cross(unit_direction, edge_ac)
    determinant = _row_dot(edge_ab, perpendicular)
    non_parallel = np.abs(determinant) > _PARALLEL_EPSILON
    inverse_determinant = _safe_divide(np.ones_like(determinant), determinant)
    to_origin = ray_origin - first
    u = _row_dot(to_origin, perpendicular) * inverse_determinant
    cross_origin = np.cross(to_origin, edge_ab)
    v = (cross_origin @ unit_direction) * inverse_determinant
    t = _row_dot(edge_ac, cross_origin) * inverse_determinant
    hits = (
        non_parallel
        & (u >= -_BARYCENTRIC_EPSILON)
        & (v >= -_BARYCENTRIC_EPSILON)
        & (u + v <= 1.0 + _BARYCENTRIC_EPSILON)
        & (t > _RAY_EPSILON)
    )
    if not hits.any():
        return None
    candidates = np.nonzero(hits)[0]
    best = candidates[np.argmin(t[candidates])]
    return float(t[best] / length), int(best)


def area_weighted_triangle_samples(
    rng: np.random.Generator, geometry: MeshGeometry, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw points uniformly distributed over the surface area of the geometry.

    Args:
        rng: Random generator used for every draw.
        geometry: Triangles to sample.
        count: Number of samples.

    Returns:
        (count, 3) positions and (count, 3) unit normals of the triangles they lie on.
    """
    sample_count = int(count)
    if sample_count <= 0 or geometry.triangle_count == 0 or geometry.area <= 0.0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    cumulative = np.cumsum(geometry.tri_areas)
    cumulative /= cumulative[-1]
    triangle_ids = np.minimum(
        np.searchsorted(cumulative, rng.random(sample_count), side="right"), geometry.triangle_count - 1
    )
    corners = geometry.vertices[geometry.triangles[triangle_ids]]
    sqrt_first = np.sqrt(rng.random(sample_count))
    second = rng.random(sample_count)
    weight_a = 1.0 - sqrt_first
    weight_b = sqrt_first * (1.0 - second)
    weight_c = sqrt_first * second
    positions = (
        weight_a[:, None] * corners[:, 0] + weight_b[:, None] * corners[:, 1] + weight_c[:, None] * corners[:, 2]
    )
    return positions, geometry.face_normals[triangle_ids].copy()


class MeshSurfaceCache:
    """LRU cache of ``MeshGeometry`` per mesh prim path, kept coherent with stage change notices.

    The stage is resolved lazily through ``stage_getter`` on every query; a different stage clears the cache and
    moves the change subscription over. Entries are dropped when the mesh, one of its geometry properties, or any
    ``xformOp*`` property on the mesh or an ancestor changes; new prims appearing next to the mesh (for example
    scatter containers and placements) leave the cache untouched. Not thread-safe; use from the main thread.
    """

    def __init__(self, stage_getter: Callable[[], Usd.Stage | None], max_entries: int = 8):
        """Create an empty cache.

        Args:
            stage_getter: Returns the stage that owns the meshes, or None when no stage is open.
            max_entries: Number of mesh geometries kept before the least recently used one is evicted.
        """
        self._stage_getter: Callable[[], Usd.Stage | None] | None = stage_getter
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[Sdf.Path, MeshGeometry] = OrderedDict()
        self._stage: Usd.Stage | None = None
        self._stage_key: tuple[str, str] | None = None
        self._subscription = None

    def get(self, mesh_path: Sdf.Path | str) -> MeshGeometry | None:
        """Return the cached geometry for a mesh, building it on first use.

        Args:
            mesh_path: Path of a ``UsdGeom.Mesh`` prim on the current stage.

        Returns:
            The geometry, or None when there is no stage or the prim is not a usable mesh.
        """
        stage = self._sync_stage()
        if stage is None:
            return None
        path = _as_path(mesh_path)
        cached = self._entries.get(path)
        if cached is not None:
            self._entries.move_to_end(path)
            return cached
        geometry = build_mesh_geometry(stage.GetPrimAtPath(path))
        if geometry is None:
            return None
        self._entries[path] = geometry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        carb.log_verbose(
            f"[lightspeed.trex.scatter.core] Cached {geometry.triangle_count} triangles for {path} "
            f"({len(self._entries)}/{self._max_entries} entries)"
        )
        return geometry

    def closest_point(
        self, mesh_path: Sdf.Path | str, point: Gf.Vec3d | np.ndarray, max_distance: float
    ) -> SurfaceSample | None:
        """Project one point onto a cached mesh.

        Args:
            mesh_path: Mesh prim path.
            point: World-space query point.
            max_distance: Results farther than this from the point yield None.

        Returns:
            The nearest surface sample, or None when the mesh is unavailable or too far away.
        """
        geometry = self.get(mesh_path)
        if geometry is None:
            return None
        return closest_point_on_triangles(_as_vector3(point), geometry, max_distance)

    def closest_points(
        self, mesh_path: Sdf.Path | str, points: np.ndarray, max_distance: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a batch of points onto a cached mesh.

        Args:
            mesh_path: Mesh prim path.
            points: (K, 3) world-space query points.
            max_distance: Points farther than this from the surface are flagged invalid.

        Returns:
            (K, 3) positions, (K, 3) normals and a (K,) validity mask; all invalid when the mesh is unavailable.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        geometry = self.get(mesh_path)
        if geometry is None:
            count = query.shape[0]
            return np.zeros((count, 3), dtype=np.float64), np.zeros((count, 3), dtype=np.float64), np.zeros(count, bool)
        return closest_points(query, geometry, max_distance)

    def raycast(self, mesh_path: Sdf.Path | str, origin: Gf.Vec3d, direction: Gf.Vec3d) -> SurfaceSample | None:
        """Intersect a ray with a cached mesh.

        Args:
            mesh_path: Mesh prim path.
            origin: World-space ray origin.
            direction: World-space ray direction.

        Returns:
            The sample at the nearest hit (``distance`` measured from the origin), or None on a miss.
        """
        geometry = self.get(mesh_path)
        if geometry is None:
            return None
        ray_origin = _as_vector3(origin)
        ray_direction = _as_vector3(direction)
        hit = raycast(ray_origin, ray_direction, geometry)
        if hit is None:
            return None
        t, triangle_index = hit
        position = ray_origin + ray_direction * t
        distance = float(t * np.linalg.norm(ray_direction))
        return SurfaceSample(position, geometry.face_normals[triangle_index].copy(), triangle_index, distance)

    def invalidate(self, paths: Iterable[Sdf.Path | str]) -> None:
        """Drop every entry whose mesh path equals or lies under any of the given prim paths.

        Args:
            paths: Prim (or property) paths; a property path invalidates as its owning prim.
        """
        if not self._entries:
            return
        targets = [_as_path(path).GetPrimPath() for path in paths]
        if not targets:
            return
        for key in list(self._entries):
            if any(key.HasPrefix(target) for target in targets):
                del self._entries[key]

    def clear(self) -> None:
        """Drop every cached geometry but keep listening to the current stage."""
        self._entries.clear()

    def destroy(self) -> None:
        """Release the stage subscription and all cached data; the cache returns None afterwards."""
        self._detach_stage()
        self._stage_getter = None

    def _sync_stage(self) -> Usd.Stage | None:
        """Resolve the current stage and (re)attach the change listener when it differs from the tracked one."""
        if self._stage_getter is None:
            return None
        stage = self._stage_getter()
        stage_key = _stage_key(stage)
        if stage_key is None:
            self._detach_stage()
            return None
        if stage_key != self._stage_key:
            self._detach_stage()
            self._stage = stage
            self._stage_key = stage_key
            self._subscription = register_objects_changed_listener(stage, self._on_objects_changed)
        return stage

    def _detach_stage(self) -> None:
        """Revoke the change listener and forget the tracked stage together with its entries."""
        if self._subscription is not None:
            self._subscription.revoke()
            self._subscription = None
        self._entries.clear()
        self._stage = None
        self._stage_key = None

    def _on_objects_changed(self, notice: Any, _stage: Usd.Stage) -> None:
        """Drop entries affected by a live or aggregated ObjectsChanged notice."""
        if not self._entries:
            return
        subtree_roots: set[Sdf.Path] = set()
        exact_meshes: set[Sdf.Path] = set()
        for path in notice.GetResyncedPaths():
            if path.IsAbsoluteRootPath():
                self._entries.clear()
                return
            if path.IsPropertyPath():
                self._classify_property_change(path, subtree_roots, exact_meshes)
            else:
                # A resynced prim is recomposed together with its whole subtree; a new sibling or child prim only
                # reports its own path and therefore never matches a cached mesh or one of its ancestors.
                subtree_roots.add(path.GetPrimPath())
        for path in notice.GetChangedInfoOnlyPaths():
            if path.IsPropertyPath():
                self._classify_property_change(path, subtree_roots, exact_meshes)
        for mesh_path in exact_meshes:
            self._entries.pop(mesh_path, None)
        if subtree_roots:
            self.invalidate(subtree_roots)

    def _classify_property_change(
        self, property_path: Sdf.Path, subtree_roots: set[Sdf.Path], exact_meshes: set[Sdf.Path]
    ) -> None:
        """Sort one changed property into transform changes (affect the subtree) or mesh data changes."""
        name = property_path.name
        prim_path = property_path.GetPrimPath()
        if name.startswith(_XFORM_PROPERTY_PREFIX):
            subtree_roots.add(prim_path)
        elif name in _GEOMETRY_PROPERTY_NAMES and prim_path in self._entries:
            exact_meshes.add(prim_path)


def _assemble_geometry(
    vertices: np.ndarray, triangles: np.ndarray, flip_faces: bool, corner_normals: np.ndarray | None
) -> MeshGeometry | None:
    """Compute normals, areas and bounds for world-space vertices, dropping zero-area triangles."""
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    cross_lengths = np.linalg.norm(cross, axis=1)
    keep = cross_lengths > _DEGENERATE_CROSS_LENGTH
    if not keep.all():
        triangles = triangles[keep]
        corners = corners[keep]
        cross = cross[keep]
        cross_lengths = cross_lengths[keep]
        if corner_normals is not None:
            corner_normals = corner_normals[keep]
    if triangles.shape[0] == 0:
        return None
    face_normals = cross / cross_lengths[:, None]
    if flip_faces:
        face_normals = -face_normals
    if corner_normals is not None:
        alignment = _row_dot(corner_normals.mean(axis=1), face_normals)
        face_normals[alignment < 0.0] *= -1.0
    tri_min = corners.min(axis=1)
    tri_max = corners.max(axis=1)
    return MeshGeometry(
        vertices=vertices,
        triangles=triangles,
        face_normals=face_normals,
        tri_min=tri_min,
        tri_max=tri_max,
        bbox_min=tri_min.min(axis=0),
        bbox_max=tri_max.max(axis=0),
        tri_areas=0.5 * cross_lengths,
    )


def _authored_corner_normals(
    mesh: UsdGeom.Mesh,
    time_code: Usd.TimeCode,
    triangles: np.ndarray,
    corner_slots: np.ndarray,
    face_of_triangle: np.ndarray,
    vertex_count: int,
    slot_count: int,
    face_count: int,
) -> np.ndarray | None:
    """Return the authored (local-space) normal at each triangle corner as (M, 3, 3), or None when unauthored."""
    normals = None
    interpolation = None
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar(UsdGeom.Tokens.normals)
    if primvar and primvar.HasAuthoredValue():
        normals = primvar.ComputeFlattened(time_code)
        interpolation = primvar.GetInterpolation()
    else:
        normals_attr = mesh.GetNormalsAttr()
        if normals_attr.HasAuthoredValue():
            normals = normals_attr.Get(time_code)
            interpolation = mesh.GetNormalsInterpolation()
    if not normals:
        return None
    values = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    triangle_count = triangles.shape[0]
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        required = vertex_count
        gather = triangles
    elif interpolation == UsdGeom.Tokens.faceVarying:
        required = slot_count
        gather = corner_slots
    elif interpolation == UsdGeom.Tokens.uniform:
        required = face_count
        gather = np.broadcast_to(face_of_triangle[:, None], (triangle_count, 3))
    elif interpolation == UsdGeom.Tokens.constant:
        required = 1
        gather = np.zeros((triangle_count, 3), dtype=np.int64)
    else:
        return None
    if values.shape[0] < required:
        carb.log_verbose(
            f"[lightspeed.trex.scatter.core] {mesh.GetPath()}: ignoring {interpolation} normals with "
            f"{values.shape[0]} values (expected at least {required})"
        )
        return None
    return values[gather]


def _nearest_on_surface(
    points: np.ndarray, geometry: MeshGeometry, max_distance: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised closest-point query returning positions, normals, distances, triangle ids and validity."""
    count = points.shape[0]
    positions = np.zeros((count, 3), dtype=np.float64)
    normals = np.zeros((count, 3), dtype=np.float64)
    distances = np.full(count, np.inf, dtype=np.float64)
    triangle_ids = np.full(count, -1, dtype=np.int64)
    valid = np.zeros(count, dtype=bool)
    if count == 0 or geometry.triangle_count == 0:
        return positions, normals, distances, triangle_ids, valid
    point_ids, candidate_ids = _candidate_pairs(points, geometry, max_distance)
    if point_ids.size == 0:
        return positions, normals, distances, triangle_ids, valid
    corners = geometry.vertices[geometry.triangles[candidate_ids]]
    queries = points[point_ids]
    closest = _closest_points_on_triangle_pairs(queries, corners[:, 0], corners[:, 1], corners[:, 2])
    offsets = closest - queries
    squared = _row_dot(offsets, offsets)
    # Sort by (point, distance) so the first row of every point run is its nearest candidate.
    order = np.lexsort((squared, point_ids))
    sorted_points = point_ids[order]
    run_start = np.ones(order.size, dtype=bool)
    run_start[1:] = sorted_points[1:] != sorted_points[:-1]
    best = order[run_start]
    winners = point_ids[best]
    positions[winners] = closest[best]
    normals[winners] = geometry.face_normals[candidate_ids[best]]
    distances[winners] = np.sqrt(squared[best])
    triangle_ids[winners] = candidate_ids[best]
    if max_distance is None:
        valid[winners] = True
    else:
        valid[winners] = distances[winners] <= float(max_distance)
        rejected = winners[~valid[winners]]
        positions[rejected] = 0.0
        normals[rejected] = 0.0
    return positions, normals, distances, triangle_ids, valid


def _candidate_pairs(
    points: np.ndarray, geometry: MeshGeometry, max_distance: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (point index, triangle index) pairs whose expanded triangle bounds contain the point."""
    triangle_count = geometry.triangle_count
    point_count = points.shape[0]
    if max_distance is None:
        return (
            np.repeat(np.arange(point_count, dtype=np.int64), triangle_count),
            np.tile(np.arange(triangle_count, dtype=np.int64), point_count),
        )
    expand = abs(float(max_distance))
    query_min = points.min(axis=0) - expand
    query_max = points.max(axis=0) + expand
    triangle_ids = np.nonzero(
        np.all(geometry.tri_max >= query_min, axis=1) & np.all(geometry.tri_min <= query_max, axis=1)
    )[0]
    if triangle_ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lower = geometry.tri_min[triangle_ids] - expand
    upper = geometry.tri_max[triangle_ids] + expand
    chunk = max(1, _PAIR_CHUNK_BUDGET // triangle_ids.size)
    point_chunks = []
    triangle_chunks = []
    for start in range(0, point_count, chunk):
        block = points[start : start + chunk]
        inside = np.all(block[:, None, :] >= lower[None, :, :], axis=2)
        inside &= np.all(block[:, None, :] <= upper[None, :, :], axis=2)
        block_points, block_triangles = np.nonzero(inside)
        point_chunks.append(block_points + start)
        triangle_chunks.append(triangle_ids[block_triangles])
    return np.concatenate(point_chunks), np.concatenate(triangle_chunks)


def _closest_points_on_triangle_pairs(
    points: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray, corner_c: np.ndarray
) -> np.ndarray:
    """Closest point on each triangle to its paired query point (Ericson, Real-Time Collision Detection 5.1.5)."""
    edge_ab = corner_b - corner_a
    edge_ac = corner_c - corner_a
    to_point_a = points - corner_a
    d1 = _row_dot(edge_ab, to_point_a)
    d2 = _row_dot(edge_ac, to_point_a)
    to_point_b = points - corner_b
    d3 = _row_dot(edge_ab, to_point_b)
    d4 = _row_dot(edge_ac, to_point_b)
    to_point_c = points - corner_c
    d5 = _row_dot(edge_ab, to_point_c)
    d6 = _row_dot(edge_ac, to_point_c)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    result = np.empty_like(points)
    assigned = np.zeros(points.shape[0], dtype=bool)

    def assign(mask: np.ndarray, value: np.ndarray) -> None:
        select = mask & ~assigned
        result[select] = value[select]
        assigned[select] = True

    assign((d1 <= 0.0) & (d2 <= 0.0), corner_a)
    assign((d3 >= 0.0) & (d4 <= d3), corner_b)
    assign((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), corner_a + edge_ab * _safe_divide(d1, d1 - d3)[:, None])
    assign((d6 >= 0.0) & (d5 <= d6), corner_c)
    assign((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), corner_a + edge_ac * _safe_divide(d2, d2 - d6)[:, None])
    edge_bc_parameter = _safe_divide(d4 - d3, (d4 - d3) + (d5 - d6))
    assign(
        (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0),
        corner_b + (corner_c - corner_b) * edge_bc_parameter[:, None],
    )
    denominator = va + vb + vc
    v = _safe_divide(vb, denominator)
    w = _safe_divide(vc, denominator)
    assign(~assigned, corner_a + edge_ab * v[:, None] + edge_ac * w[:, None])
    return result


def _row_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise dot product of two (N, 3) arrays."""
    return np.einsum("ij,ij->i", left, right)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that yields 0 where the denominator is 0."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0.0)


def _matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    """Copy a Gf.Matrix4d into a (4, 4) float64 array (row-vector convention preserved)."""
    return np.array([[matrix[row][column] for column in range(4)] for row in range(4)], dtype=np.float64)


def _as_vector3(value: _Vector3Like) -> np.ndarray:
    """Return a (3,) float64 array for a Gf vector, sequence or numpy array."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(3)
    return np.array((float(value[0]), float(value[1]), float(value[2])), dtype=np.float64)


def _as_path(value: Sdf.Path | str) -> Sdf.Path:
    """Return an Sdf.Path for a path or string."""
    return value if isinstance(value, Sdf.Path) else Sdf.Path(str(value))


def _stage_key(stage: Usd.Stage | None) -> tuple[str, str] | None:
    """Identify a stage by its root and session layer identifiers; None for a missing or expired stage."""
    if stage is None:
        return None
    try:
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()
    except RuntimeError:
        # Tf.ErrorException derives from RuntimeError; an expired stage wrapper raises when accessed.
        return None
    if not root_layer:
        return None
    return root_layer.identifier, session_layer.identifier if session_layer else ""
