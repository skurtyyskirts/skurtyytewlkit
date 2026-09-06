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

__all__ = [
    "TestGeometryFunctions",
    "TestMeshGeometry",
    "TestMeshSurfaceCache",
]

from collections.abc import Sequence
from unittest import mock

import numpy as np
import omni.kit.test
from lightspeed.trex.scatter.core import geometry as geometry_module
from lightspeed.trex.scatter.core.geometry import (
    MeshSurfaceCache,
    SurfaceSample,
    area_weighted_triangle_samples,
    build_mesh_geometry,
    closest_point_on_triangles,
    closest_points,
    raycast,
    triangulate_faces,
)
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

# Counter-clockwise when viewed from +Z, so the geometric face normal is +Z.
_QUAD_POINTS = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0))
_MESH_PATH = "/World/Xform/Mesh"
_XFORM_PATH = "/World/Xform"


class _FakeNotice:
    """Minimal stand-in mirroring the ObjectsChanged / AggregatedObjectsChangedNotice read API."""

    def __init__(self, resynced: Sequence[str] = (), changed_info_only: Sequence[str] = ()):
        self._resynced = tuple(Sdf.Path(path) for path in resynced)
        self._changed_info_only = tuple(Sdf.Path(path) for path in changed_info_only)

    def get_resynced_paths(self) -> tuple[Sdf.Path, ...]:
        return self._resynced

    def get_changed_info_only_paths(self) -> tuple[Sdf.Path, ...]:
        return self._changed_info_only

    GetResyncedPaths = get_resynced_paths
    GetChangedInfoOnlyPaths = get_changed_info_only_paths


def _define_mesh(
    stage: Usd.Stage,
    path: str,
    points: Sequence[Sequence[float]],
    counts: Sequence[int],
    indices: Sequence[int],
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(list(counts)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(list(indices)))
    return mesh


def _define_quad_mesh(
    stage: Usd.Stage, path: str = _MESH_PATH, points: Sequence[Sequence[float]] = _QUAD_POINTS
) -> UsdGeom.Mesh:
    return _define_mesh(stage, path, points, [4], [0, 1, 2, 3])


def _define_stacked_quads_mesh(stage: Usd.Stage, path: str = _MESH_PATH) -> UsdGeom.Mesh:
    """Two horizontal 10x10 quads, one at z=0 (triangles 0, 1) and one at z=5 (triangles 2, 3)."""
    upper = [(x, y, 5.0) for x, y, _ in _QUAD_POINTS]
    return _define_mesh(stage, path, [*_QUAD_POINTS, *upper], [4, 4], [0, 1, 2, 3, 4, 5, 6, 7])


def _define_translated_xform(stage: Usd.Stage, path: str, translate: Sequence[float]) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return xform


def _translated_quad_points(offset: Sequence[float]) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in np.add(point, offset)) for point in _QUAD_POINTS]


def _constant_stage_getter(stage: Usd.Stage):
    """Bind the stage as a parameter so loop bodies can build getters without capturing the loop variable."""
    return lambda: stage


class TestMeshGeometry(omni.kit.test.AsyncTestCase):
    async def test_area_and_triangle_count_return_totals_of_triangulated_quad(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        mesh = _define_quad_mesh(stage)

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        self.assertEqual(geometry.triangle_count, 2)
        self.assertAlmostEqual(geometry.area, 100.0, places=6)
        np.testing.assert_allclose(geometry.tri_areas, [50.0, 50.0])


class TestGeometryFunctions(omni.kit.test.AsyncTestCase):
    async def test_triangulate_faces_with_quad_returns_two_fan_triangles(self):
        # Arrange
        counts = [4]
        indices = [0, 1, 2, 3]

        # Act
        triangles = triangulate_faces(counts, indices)

        # Assert
        self.assertEqual(triangles.dtype, np.int64)
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3]])

    async def test_triangulate_faces_with_degenerate_face_skips_it_and_keeps_offsets(self):
        # Arrange
        counts = [4, 2, 3]
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Act
        triangles = triangulate_faces(counts, indices)

        # Assert
        np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3], [6, 7, 8]])

    async def test_triangulate_faces_with_truncated_indices_drops_incomplete_face(self):
        # Arrange
        counts = [3, 4]
        indices = [0, 1, 2, 3, 4]

        # Act
        triangles = triangulate_faces(counts, indices)

        # Assert
        np.testing.assert_array_equal(triangles, [[0, 1, 2]])

    async def test_triangulate_faces_with_empty_input_returns_empty_array(self):
        # Arrange
        counts = Vt.IntArray()
        indices = Vt.IntArray()

        # Act
        triangles = triangulate_faces(counts, indices)

        # Assert
        self.assertEqual(triangles.shape, (0, 3))
        self.assertEqual(triangles.dtype, np.int64)

    async def test_build_mesh_geometry_with_translated_parent_applies_world_transform(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_translated_xform(stage, _XFORM_PATH, (10.0, 20.0, 30.0))
        mesh = _define_quad_mesh(stage)

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        np.testing.assert_allclose(geometry.vertices, np.asarray(_QUAD_POINTS) + [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(geometry.triangles, [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_allclose(geometry.bbox_min, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(geometry.bbox_max, [20.0, 30.0, 30.0])
        np.testing.assert_allclose(geometry.tri_min, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        np.testing.assert_allclose(geometry.tri_max, [[20.0, 30.0, 30.0], [20.0, 30.0, 30.0]])
        np.testing.assert_allclose(geometry.face_normals, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    async def test_build_mesh_geometry_with_orientation_uses_winding_and_flips_left_handed(self):
        cases = [
            (UsdGeom.Tokens.rightHanded, 1.0),
            (UsdGeom.Tokens.leftHanded, -1.0),
        ]
        for orientation, expected_z in cases:
            with self.subTest(title=f"orientation={orientation}"):
                # Arrange
                stage = Usd.Stage.CreateInMemory()
                mesh = _define_quad_mesh(stage)
                mesh.GetOrientationAttr().Set(orientation)

                # Act
                geometry = build_mesh_geometry(mesh.GetPrim())

                # Assert
                np.testing.assert_allclose(geometry.face_normals, [[0.0, 0.0, expected_z]] * 2, atol=1e-12)

    async def test_build_mesh_geometry_with_authored_vertex_normals_aligns_face_normal_sign(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        mesh = _define_quad_mesh(stage)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, -1.0)] * 4))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        np.testing.assert_allclose(geometry.face_normals, [[0.0, 0.0, -1.0]] * 2, atol=1e-12)

    async def test_build_mesh_geometry_with_authored_face_varying_normals_aligns_per_face(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        mesh = _define_stacked_quads_mesh(stage)
        lower_normals = [Gf.Vec3f(0.0, 0.0, -1.0)] * 4
        upper_normals = [Gf.Vec3f(0.0, 0.0, 1.0)] * 4
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray(lower_normals + upper_normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        np.testing.assert_allclose(
            geometry.face_normals,
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )

    async def test_build_mesh_geometry_with_mirrored_transform_keeps_outward_normals(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, _XFORM_PATH)
        xform.AddScaleOp().Set(Gf.Vec3f(-1.0, 1.0, 1.0))
        mesh = _define_quad_mesh(stage)

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        np.testing.assert_allclose(geometry.vertices[:, 0], [0.0, -10.0, -10.0, 0.0])
        np.testing.assert_allclose(geometry.face_normals, [[0.0, 0.0, 1.0]] * 2, atol=1e-12)

    async def test_build_mesh_geometry_with_zero_area_face_drops_it(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        points = [*_QUAD_POINTS, (20.0, 0.0, 0.0), (30.0, 0.0, 0.0), (40.0, 0.0, 0.0)]
        mesh = _define_mesh(stage, _MESH_PATH, points, [4, 3], [0, 1, 2, 3, 4, 5, 6])

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        self.assertEqual(geometry.triangle_count, 2)
        np.testing.assert_array_equal(geometry.triangles, [[0, 1, 2], [0, 2, 3]])

    async def test_build_mesh_geometry_with_invalid_prim_returns_none(self):
        stage = Usd.Stage.CreateInMemory()
        cases = [
            ("none_prim", None),
            ("xform_prim", UsdGeom.Xform.Define(stage, "/World/Xform").GetPrim()),
            ("mesh_without_points", UsdGeom.Mesh.Define(stage, "/World/Empty").GetPrim()),
            ("invalid_prim", stage.GetPrimAtPath("/World/DoesNotExist")),
        ]
        for title, prim in cases:
            with self.subTest(title=title):
                # Arrange
                mesh_prim = prim

                # Act
                geometry = build_mesh_geometry(mesh_prim)

                # Assert
                self.assertIsNone(geometry)

    async def test_build_mesh_geometry_with_out_of_range_indices_drops_those_faces(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        mesh = _define_mesh(stage, _MESH_PATH, _QUAD_POINTS, [3, 3], [0, 1, 2, 0, 2, 9])

        # Act
        geometry = build_mesh_geometry(mesh.GetPrim())

        # Assert
        np.testing.assert_array_equal(geometry.triangles, [[0, 1, 2]])

    async def test_closest_point_on_triangles_with_point_above_triangle_returns_projection_and_normal(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        point = np.array([3.0, 2.0, 5.0])

        # Act
        sample = closest_point_on_triangles(point, geometry)

        # Assert
        self.assertIsInstance(sample, SurfaceSample)
        np.testing.assert_allclose(sample.position, [3.0, 2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(sample.normal, [0.0, 0.0, 1.0], atol=1e-12)
        self.assertEqual(sample.triangle_index, 0)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_closest_point_on_triangles_beyond_max_distance_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        point = np.array([3.0, 2.0, 5.0])

        # Act
        sample = closest_point_on_triangles(point, geometry, max_distance=4.9)

        # Assert
        self.assertIsNone(sample)

    async def test_closest_point_on_triangles_within_max_distance_returns_sample(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        point = np.array([3.0, 2.0, 5.0])

        # Act
        sample = closest_point_on_triangles(point, geometry, max_distance=5.1)

        # Assert
        np.testing.assert_allclose(sample.position, [3.0, 2.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_closest_point_on_triangles_with_point_past_edge_returns_edge_point(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        point = np.array([15.0, 5.0, 0.0])

        # Act
        sample = closest_point_on_triangles(point, geometry)

        # Assert
        np.testing.assert_allclose(sample.position, [10.0, 5.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_closest_point_on_triangles_with_point_past_corner_returns_corner(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        point = np.array([-3.0, -4.0, 0.0])

        # Act
        sample = closest_point_on_triangles(point, geometry)

        # Assert
        np.testing.assert_allclose(sample.position, [0.0, 0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_closest_points_with_batch_matches_per_point_scalar_results(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_stacked_quads_mesh(stage).GetPrim())
        points = np.array(
            [
                [3.0, 2.0, 1.0],
                [15.0, 5.0, 4.0],
                [-2.0, -3.0, 6.0],
                [7.0, 8.0, -4.0],
                [2.0, 9.0, 2.6],
            ]
        )
        expected = [closest_point_on_triangles(point, geometry, max_distance=10.0) for point in points]

        # Act
        positions, normals, valid = closest_points(points, geometry, max_distance=10.0)

        # Assert
        self.assertEqual(positions.shape, (5, 3))
        self.assertEqual(normals.shape, (5, 3))
        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(positions, [sample.position for sample in expected], atol=1e-9)
        np.testing.assert_allclose(normals, [sample.normal for sample in expected], atol=1e-12)

    async def test_closest_points_with_far_point_marks_it_invalid(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        points = np.array([[3.0, 2.0, 1.0], [50.0, 50.0, 50.0]])

        # Act
        positions, _normals, valid = closest_points(points, geometry, max_distance=2.0)

        # Assert
        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_allclose(positions[0], [3.0, 2.0, 0.0], atol=1e-9)

    async def test_closest_points_with_empty_input_returns_empty_arrays(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        points = np.empty((0, 3))

        # Act
        positions, normals, valid = closest_points(points, geometry, max_distance=2.0)

        # Assert
        self.assertEqual(positions.shape, (0, 3))
        self.assertEqual(normals.shape, (0, 3))
        self.assertEqual(valid.shape, (0,))

    async def test_raycast_with_stacked_quads_returns_nearest_triangle_and_t(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_stacked_quads_mesh(stage).GetPrim())
        origin = np.array([2.0, 2.0, 10.0])
        direction = np.array([0.0, 0.0, -1.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        self.assertIsNotNone(hit)
        t, triangle_index = hit
        self.assertAlmostEqual(t, 5.0, places=9)
        self.assertEqual(triangle_index, 2)

    async def test_raycast_with_unnormalized_direction_returns_t_in_direction_units(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        origin = np.array([2.0, 2.0, 10.0])
        direction = np.array([0.0, 0.0, -2.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        t, triangle_index = hit
        self.assertAlmostEqual(t, 5.0, places=9)
        self.assertEqual(triangle_index, 0)

    async def test_raycast_with_parallel_ray_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        origin = np.array([2.0, 2.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        self.assertIsNone(hit)

    async def test_raycast_with_ray_missing_mesh_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        origin = np.array([50.0, 50.0, 10.0])
        direction = np.array([0.0, 0.0, -1.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        self.assertIsNone(hit)

    async def test_raycast_from_behind_hits_back_face(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        origin = np.array([2.0, 2.0, -10.0])
        direction = np.array([0.0, 0.0, 1.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        t, triangle_index = hit
        self.assertAlmostEqual(t, 10.0, places=9)
        self.assertEqual(triangle_index, 0)

    async def test_raycast_with_origin_on_surface_ignores_that_surface(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_stacked_quads_mesh(stage).GetPrim())
        origin = np.array([2.0, 2.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])

        # Act
        hit = raycast(origin, direction, geometry)

        # Assert
        t, triangle_index = hit
        self.assertAlmostEqual(t, 5.0, places=9)
        self.assertEqual(triangle_index, 2)

    async def test_area_weighted_triangle_samples_prefers_larger_triangle(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (10.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
        ]
        geometry = build_mesh_geometry(_define_mesh(stage, _MESH_PATH, points, [3, 3], [0, 1, 2, 3, 4, 5]).GetPrim())
        rng = np.random.default_rng(7)

        # Act
        positions, normals = area_weighted_triangle_samples(rng, geometry, 2000)

        # Assert
        self.assertEqual(positions.shape, (2000, 3))
        self.assertEqual(normals.shape, (2000, 3))
        in_large_triangle = np.count_nonzero(positions[:, 0] >= 10.0)
        self.assertGreater(in_large_triangle / 2000.0, 0.9)
        np.testing.assert_allclose(positions[:, 2], 0.0, atol=1e-12)
        self.assertTrue(np.all(positions[:, 0] >= 0.0))
        self.assertTrue(np.all(positions[:, 0] <= 20.0))
        self.assertTrue(np.all(positions[:, 1] >= 0.0))
        self.assertTrue(np.all(positions[:, 0] + positions[:, 1] <= 20.0 + 1e-9))
        np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (2000, 1)), atol=1e-12)

    async def test_area_weighted_triangle_samples_with_same_seed_is_deterministic(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        first_positions, _ = area_weighted_triangle_samples(np.random.default_rng(42), geometry, 16)

        # Act
        second_positions, _ = area_weighted_triangle_samples(np.random.default_rng(42), geometry, 16)

        # Assert
        np.testing.assert_array_equal(second_positions, first_positions)

    async def test_area_weighted_triangle_samples_with_zero_count_returns_empty_arrays(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        geometry = build_mesh_geometry(_define_quad_mesh(stage).GetPrim())
        rng = np.random.default_rng(1)

        # Act
        positions, normals = area_weighted_triangle_samples(rng, geometry, 0)

        # Assert
        self.assertEqual(positions.shape, (0, 3))
        self.assertEqual(normals.shape, (0, 3))


class TestMeshSurfaceCache(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._register_patch = mock.patch.object(geometry_module, "register_objects_changed_listener")
        self._register_mock = self._register_patch.start()
        self.addCleanup(self._register_patch.stop)

    def _notify(self, notice: _FakeNotice, stage: Usd.Stage, call_index: int = -1) -> None:
        """Drive the callback captured by the patched listener registration."""
        callback = self._register_mock.call_args_list[call_index].args[1]
        callback(notice, stage)

    async def test_get_with_same_path_twice_returns_cached_geometry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)

        # Act
        with mock.patch.object(geometry_module, "build_mesh_geometry") as build_mock:
            second = cache.get(Sdf.Path(_MESH_PATH))

        # Assert
        self.assertIs(second, first)
        build_mock.assert_not_called()

    async def test_get_with_missing_prim_returns_none_and_caches_nothing(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        cache = MeshSurfaceCache(lambda: stage)

        # Act
        geometry = cache.get("/World/DoesNotExist")

        # Assert
        self.assertIsNone(geometry)
        self.assertEqual(len(cache._entries), 0)

    async def test_get_with_stage_getter_returning_none_returns_none_without_subscribing(self):
        # Arrange
        cache = MeshSurfaceCache(lambda: None)

        # Act
        geometry = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNone(geometry)
        self._register_mock.assert_not_called()

    async def test_get_with_more_entries_than_max_evicts_least_recently_used(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        for name in ("A", "B", "C"):
            _define_quad_mesh(stage, f"/World/{name}")
        cache = MeshSurfaceCache(lambda: stage, max_entries=2)
        cache.get("/World/A")
        cache.get("/World/B")
        cache.get("/World/A")

        # Act
        cache.get("/World/C")

        # Assert
        self.assertEqual(list(cache._entries.keys()), [Sdf.Path("/World/A"), Sdf.Path("/World/C")])

    async def test_get_subscribes_objects_changed_listener_once_per_stage(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage, "/World/A")
        _define_quad_mesh(stage, "/World/B")
        cache = MeshSurfaceCache(lambda: stage)
        cache.get("/World/A")

        # Act
        cache.get("/World/B")

        # Assert
        self._register_mock.assert_called_once()
        registered_stage, callback = self._register_mock.call_args.args
        self.assertIs(registered_stage, stage)
        self.assertTrue(callable(callback))

    async def test_get_after_points_change_notice_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        mesh = _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in _translated_quad_points((0, 0, 5))]))
        self._notify(_FakeNotice(changed_info_only=(f"{_MESH_PATH}.points",)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)
        np.testing.assert_allclose(rebuilt.vertices[:, 2], 5.0)

    async def test_get_after_topology_property_resync_rebuilds_entry(self):
        cases = ["faceVertexCounts", "faceVertexIndices", "normals", "orientation", "primvars:normals"]
        for property_name in cases:
            with self.subTest(title=f"property={property_name}"):
                # Arrange
                self._register_mock.reset_mock()
                stage = Usd.Stage.CreateInMemory()
                _define_quad_mesh(stage)
                cache = MeshSurfaceCache(_constant_stage_getter(stage))
                first = cache.get(_MESH_PATH)
                self._notify(_FakeNotice(resynced=(f"{_MESH_PATH}.{property_name}",)), stage)

                # Act
                rebuilt = cache.get(_MESH_PATH)

                # Assert
                self.assertIsNot(rebuilt, first)

    async def test_get_after_ancestor_xform_change_notice_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        xform = _define_translated_xform(stage, _XFORM_PATH, (0.0, 0.0, 0.0))
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, 7.0))
        self._notify(_FakeNotice(changed_info_only=(f"{_XFORM_PATH}.xformOp:translate",)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)
        np.testing.assert_allclose(rebuilt.vertices[:, 2], 7.0)

    async def test_get_after_ancestor_xform_op_order_resync_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _XFORM_PATH)
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        self._notify(_FakeNotice(resynced=(f"{_XFORM_PATH}.xformOpOrder",)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)

    async def test_get_after_mesh_prim_resync_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        self._notify(_FakeNotice(resynced=(_MESH_PATH,)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)

    async def test_get_after_ancestor_prim_resync_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _XFORM_PATH)
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        self._notify(_FakeNotice(resynced=(_XFORM_PATH,)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)

    async def test_get_after_new_child_prim_resync_under_ancestor_keeps_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _XFORM_PATH)
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        self._notify(
            _FakeNotice(
                resynced=(f"{_XFORM_PATH}/scatter_default", f"{_XFORM_PATH}/scatter_default/s_0123456789ab"),
            ),
            stage,
        )

        # Act
        again = cache.get(_MESH_PATH)

        # Assert
        self.assertIs(again, first)

    async def test_get_after_sibling_placement_xform_change_keeps_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _XFORM_PATH)
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        placement = f"{_XFORM_PATH}/scatter_default/s_0123456789ab"
        self._notify(
            _FakeNotice(
                resynced=(f"{placement}.xformOp:translate", f"{placement}.xformOpOrder"),
                changed_info_only=(f"{placement}.xformOp:scale", f"{_XFORM_PATH}/Other.points"),
            ),
            stage,
        )

        # Act
        again = cache.get(_MESH_PATH)

        # Assert
        self.assertIs(again, first)

    async def test_get_after_pseudo_root_resync_rebuilds_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        first = cache.get(_MESH_PATH)
        self._notify(_FakeNotice(resynced=("/",)), stage)

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)

    async def test_get_after_stage_change_clears_entries_and_resubscribes(self):
        # Arrange
        first_subscription = mock.MagicMock()
        second_subscription = mock.MagicMock()
        self._register_mock.side_effect = [first_subscription, second_subscription]
        first_stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(first_stage)
        second_stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(second_stage, points=_translated_quad_points((0, 0, 9)))
        holder = {"stage": first_stage}
        cache = MeshSurfaceCache(lambda: holder["stage"])
        first = cache.get(_MESH_PATH)
        holder["stage"] = second_stage

        # Act
        rebuilt = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNot(rebuilt, first)
        np.testing.assert_allclose(rebuilt.vertices[:, 2], 9.0)
        self.assertEqual(self._register_mock.call_count, 2)
        self.assertIs(self._register_mock.call_args_list[1].args[0], second_stage)
        first_subscription.revoke.assert_called_once()
        second_subscription.revoke.assert_not_called()
        self.assertEqual(list(cache._entries.keys()), [Sdf.Path(_MESH_PATH)])

    async def test_get_after_stage_getter_returns_none_clears_entries_and_revokes(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        holder = {"stage": stage}
        cache = MeshSurfaceCache(lambda: holder["stage"])
        cache.get(_MESH_PATH)
        holder["stage"] = None

        # Act
        geometry = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNone(geometry)
        self.assertEqual(len(cache._entries), 0)
        self._register_mock.return_value.revoke.assert_called_once()

    async def test_clear_removes_all_entries_and_keeps_subscription(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage, "/World/A")
        _define_quad_mesh(stage, "/World/B")
        cache = MeshSurfaceCache(lambda: stage)
        cache.get("/World/A")
        cache.get("/World/B")

        # Act
        cache.clear()

        # Assert
        self.assertEqual(len(cache._entries), 0)
        self._register_mock.return_value.revoke.assert_not_called()

    async def test_invalidate_with_ancestor_path_drops_entries_under_it_only(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage, "/World/Xform/A")
        _define_quad_mesh(stage, "/World/Xform/B")
        _define_quad_mesh(stage, "/World/Other/C")
        cache = MeshSurfaceCache(lambda: stage)
        for path in ("/World/Xform/A", "/World/Xform/B", "/World/Other/C"):
            cache.get(path)

        # Act
        cache.invalidate(["/World/Xform", Sdf.Path("/World/Missing")])

        # Assert
        self.assertEqual(list(cache._entries.keys()), [Sdf.Path("/World/Other/C")])

    async def test_invalidate_with_exact_mesh_path_drops_that_entry(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage, "/World/A")
        _define_quad_mesh(stage, "/World/B")
        cache = MeshSurfaceCache(lambda: stage)
        cache.get("/World/A")
        cache.get("/World/B")

        # Act
        cache.invalidate([Sdf.Path("/World/A")])

        # Assert
        self.assertEqual(list(cache._entries.keys()), [Sdf.Path("/World/B")])

    async def test_destroy_revokes_subscription_and_clears_entries(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        cache.get(_MESH_PATH)

        # Act
        cache.destroy()

        # Assert
        self.assertEqual(len(cache._entries), 0)
        self._register_mock.return_value.revoke.assert_called_once()

    async def test_get_after_destroy_returns_none_without_resubscribing(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        cache.get(_MESH_PATH)
        cache.destroy()

        # Act
        geometry = cache.get(_MESH_PATH)

        # Assert
        self.assertIsNone(geometry)
        self._register_mock.assert_called_once()

    async def test_closest_point_with_gf_vec3d_returns_surface_sample(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)

        # Act
        sample = cache.closest_point(_MESH_PATH, Gf.Vec3d(3.0, 2.0, 5.0), max_distance=10.0)

        # Assert
        self.assertIsInstance(sample, SurfaceSample)
        np.testing.assert_allclose(sample.position, [3.0, 2.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(sample.normal, [0.0, 0.0, 1.0], atol=1e-12)
        self.assertEqual(sample.triangle_index, 0)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_closest_point_with_missing_mesh_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        cache = MeshSurfaceCache(lambda: stage)

        # Act
        sample = cache.closest_point("/World/Missing", Gf.Vec3d(0.0, 0.0, 0.0), max_distance=10.0)

        # Assert
        self.assertIsNone(sample)

    async def test_closest_points_returns_projected_positions_and_validity(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)
        points = np.array([[1.0, 1.0, 2.0], [8.0, 9.0, -3.0], [80.0, 80.0, 0.0]])

        # Act
        positions, normals, valid = cache.closest_points(_MESH_PATH, points, max_distance=5.0)

        # Assert
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(positions[:2], [[1.0, 1.0, 0.0], [8.0, 9.0, 0.0]], atol=1e-9)
        np.testing.assert_allclose(normals[:2], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1e-12)

    async def test_closest_points_with_missing_mesh_returns_all_invalid(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        cache = MeshSurfaceCache(lambda: stage)
        points = np.array([[1.0, 1.0, 2.0], [8.0, 9.0, -3.0]])

        # Act
        positions, normals, valid = cache.closest_points("/World/Missing", points, max_distance=5.0)

        # Assert
        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(normals.shape, (2, 3))
        np.testing.assert_array_equal(valid, [False, False])

    async def test_raycast_returns_sample_at_hit(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_stacked_quads_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)

        # Act
        sample = cache.raycast(_MESH_PATH, Gf.Vec3d(2.0, 2.0, 10.0), Gf.Vec3d(0.0, 0.0, -1.0))

        # Assert
        self.assertIsInstance(sample, SurfaceSample)
        np.testing.assert_allclose(sample.position, [2.0, 2.0, 5.0], atol=1e-9)
        np.testing.assert_allclose(sample.normal, [0.0, 0.0, 1.0], atol=1e-12)
        self.assertEqual(sample.triangle_index, 2)
        self.assertAlmostEqual(sample.distance, 5.0, places=9)

    async def test_raycast_with_miss_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_quad_mesh(stage)
        cache = MeshSurfaceCache(lambda: stage)

        # Act
        sample = cache.raycast(_MESH_PATH, Gf.Vec3d(50.0, 50.0, 10.0), Gf.Vec3d(0.0, 0.0, -1.0))

        # Assert
        self.assertIsNone(sample)
