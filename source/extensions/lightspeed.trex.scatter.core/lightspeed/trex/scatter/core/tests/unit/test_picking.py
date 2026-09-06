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
    "TestCameraRayFromNdc",
    "TestCpuRaySurfacePicker",
    "TestCreateSurfacePicker",
    "TestHdRemixSurfacePicker",
]

import asyncio
import threading
from unittest import mock

import carb
from lightspeed.hydra.remix.core import RemixSupport
from lightspeed.trex.scatter.core import constants, picking
from lightspeed.trex.scatter.core.geometry import MeshSurfaceCache
from omni.kit.test import AsyncTestCase
from pxr import Gf, Sdf, Usd, UsdGeom

_HASH = "0123456789ABCDEF"
_INSTANCE_ROOT = f"/RootNode/instances/inst_{_HASH}_0"
_FLOOR_PATH = f"{_INSTANCE_ROOT}/floor"
_CAMERA_HEIGHT = 50.0
_NEAR = 0.5
_FAR = 1000.0


def _top_down_frustum(height: float = _CAMERA_HEIGHT) -> Gf.Frustum:
    """Perspective frustum sitting at (0, height, 0) and looking straight down -Y."""
    frustum = Gf.Frustum()
    frustum.SetPosition(Gf.Vec3d(0.0, height, 0.0))
    frustum.SetRotation(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), -90.0))
    frustum.SetNearFar(Gf.Range1d(_NEAR, _FAR))
    return frustum


def _make_viewport(stage: Usd.Stage | None = None, frustum: Gf.Frustum | None = None) -> mock.MagicMock:
    """Viewport API stand-in exposing real camera matrices and the given stage."""
    frustum = frustum or _top_down_frustum()
    world_to_ndc = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
    viewport_api = mock.MagicMock()
    viewport_api.ndc_to_world = world_to_ndc.GetInverse()
    viewport_api.world_to_ndc = world_to_ndc
    viewport_api.transform = frustum.ComputeViewInverse()
    viewport_api.stage = stage
    return viewport_api


def _define_floor(stage: Usd.Stage, path: str, height: float = 0.0) -> UsdGeom.Mesh:
    """Author a horizontal quad (two triangles after fan triangulation) at the given Y height.

    The quad spans x in [-25, 175] and z in [-100, 100] so that the top-down test ray at (x, z) = (0, 0) lands inside
    one triangle instead of exactly on the shared diagonal.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-25.0, height, -100.0),
            Gf.Vec3f(175.0, height, -100.0),
            Gf.Vec3f(175.0, height, 100.0),
            Gf.Vec3f(-25.0, height, 100.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    return mesh


def _assert_vec3_close(test: AsyncTestCase, actual, expected, tolerance: float = 1e-4) -> None:
    for index in range(3):
        test.assertAlmostEqual(float(actual[index]), float(expected[index]), delta=tolerance, msg=f"component {index}")


class TestCameraRayFromNdc(AsyncTestCase):
    async def test_camera_ray_from_ndc_at_center_returns_near_origin_and_unit_view_direction(self):
        # Arrange
        viewport_api = _make_viewport()

        # Act
        origin, direction = picking.camera_ray_from_ndc(viewport_api, (0.0, 0.0))

        # Assert
        _assert_vec3_close(self, origin, (0.0, _CAMERA_HEIGHT - _NEAR, 0.0))
        _assert_vec3_close(self, direction, (0.0, -1.0, 0.0))
        self.assertAlmostEqual(direction.GetLength(), 1.0, places=9)

    async def test_camera_ray_from_ndc_off_center_points_from_near_to_far_unprojection(self):
        # Arrange
        viewport_api = _make_viewport()
        ndc = (0.5, -0.25)
        near = viewport_api.ndc_to_world.Transform(Gf.Vec3d(ndc[0], ndc[1], -1.0))
        far = viewport_api.ndc_to_world.Transform(Gf.Vec3d(ndc[0], ndc[1], 1.0))

        # Act
        origin, direction = picking.camera_ray_from_ndc(viewport_api, ndc)

        # Assert
        _assert_vec3_close(self, origin, near, tolerance=1e-6)
        _assert_vec3_close(self, direction, (far - near).GetNormalized(), tolerance=1e-9)
        self.assertGreater(direction[0], 0.0)
        self.assertGreater(Gf.Dot(direction, far - near), 0.0)
        self.assertAlmostEqual(direction.GetLength(), 1.0, places=9)


class TestCpuRaySurfacePicker(AsyncTestCase):
    def _make_cache(self, stage: Usd.Stage) -> MeshSurfaceCache:
        cache = MeshSurfaceCache(lambda: stage)
        self.addCleanup(cache.destroy)
        return cache

    async def test_pick_center_ray_hitting_floor_under_instance_returns_hit_path_and_position(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        _define_floor(stage, _FLOOR_PATH)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        issued = picker.pick((0.0, 0.0), callback)

        # Assert
        self.assertTrue(issued)
        callback.assert_called_once()
        hit = callback.call_args.args[0]
        self.assertIsInstance(hit, picking.SurfaceHit)
        self.assertEqual(hit.path, Sdf.Path(_FLOOR_PATH))
        _assert_vec3_close(self, hit.world_position, (0.0, 0.0, 0.0))

    async def test_pick_with_two_floors_returns_nearest_hit(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        _define_floor(stage, f"{_INSTANCE_ROOT}/lower_floor", height=-10.0)
        _define_floor(stage, f"{_INSTANCE_ROOT}/upper_floor", height=0.0)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(f"{_INSTANCE_ROOT}/upper_floor"))
        _assert_vec3_close(self, hit.world_position, (0.0, 0.0, 0.0))

    async def test_pick_without_any_mesh_calls_callback_with_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        issued = picker.pick((0.0, 0.0), callback)

        # Assert
        self.assertTrue(issued)
        callback.assert_called_once_with(None)

    async def test_pick_when_ray_misses_all_geometry_calls_callback_with_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        _define_floor(stage, _FLOOR_PATH, height=_CAMERA_HEIGHT + 10.0)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        callback.assert_called_once_with(None)

    async def test_pick_skips_invisible_prims_and_calls_callback_with_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        floor = _define_floor(stage, _FLOOR_PATH)
        UsdGeom.Imageable(floor.GetPrim()).MakeInvisible()
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        callback.assert_called_once_with(None)

    async def test_pick_skips_mesh_under_invisible_ancestor_and_returns_visible_hit(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        hidden_root = UsdGeom.Xform.Define(stage, f"/RootNode/instances/inst_{_HASH}_1")
        UsdGeom.Imageable(hidden_root.GetPrim()).MakeInvisible()
        _define_floor(stage, f"{hidden_root.GetPath()}/floor", height=10.0)
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        _define_floor(stage, _FLOOR_PATH, height=0.0)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(_FLOOR_PATH))

    async def test_pick_without_root_paths_on_stage_falls_back_to_whole_stage(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_floor(stage, "/World/floor")
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path("/World/floor"))
        _assert_vec3_close(self, hit.world_position, (0.0, 0.0, 0.0))

    async def test_pick_with_existing_root_paths_ignores_meshes_outside_roots(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        _define_floor(stage, _FLOOR_PATH, height=-20.0)
        _define_floor(stage, "/World/closer_floor", height=0.0)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(_FLOOR_PATH))
        _assert_vec3_close(self, hit.world_position, (0.0, -20.0, 0.0))

    async def test_pick_with_custom_root_paths_searches_those_roots(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_floor(stage, "/Custom/floor")
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage), root_paths=("/Custom",))
        callback = mock.Mock()

        # Act
        picker.pick((0.0, 0.0), callback)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path("/Custom/floor"))

    async def test_pick_with_ndc_outside_range_returns_false_without_callback(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_floor(stage, _FLOOR_PATH)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        callback = mock.Mock()

        # Act
        issued = picker.pick((1.5, 0.0), callback)

        # Assert
        self.assertFalse(issued)
        callback.assert_not_called()

    async def test_pick_without_stage_returns_false_without_callback(self):
        # Arrange
        viewport_api = _make_viewport(stage=None)
        picker = picking.CpuRaySurfacePicker(viewport_api, MeshSurfaceCache(lambda: None))
        callback = mock.Mock()

        # Act
        issued = picker.pick((0.0, 0.0), callback)

        # Assert
        self.assertFalse(issued)
        callback.assert_not_called()

    async def test_in_flight_after_pick_is_false(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_floor(stage, _FLOOR_PATH)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        picker.pick((0.0, 0.0), mock.Mock())

        # Act
        in_flight = picker.in_flight

        # Assert
        self.assertFalse(in_flight)

    async def test_cancel_leaves_picker_usable_for_next_pick(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        _define_floor(stage, _FLOOR_PATH)
        viewport_api = _make_viewport(stage)
        picker = picking.CpuRaySurfacePicker(viewport_api, self._make_cache(stage))
        picker.cancel()
        callback = mock.Mock()

        # Act
        issued = picker.pick((0.0, 0.0), callback)

        # Assert
        self.assertTrue(issued)
        self.assertFalse(picker.in_flight)
        self.assertEqual(callback.call_args.args[0].path, Sdf.Path(_FLOOR_PATH))


class TestCreateSurfacePicker(AsyncTestCase):
    def _patch_settings(self, force_cpu: bool | None) -> mock.MagicMock:
        settings = mock.MagicMock()
        settings.get.side_effect = lambda path: force_cpu if path == constants.FORCE_CPU_PICKER_SETTING else None
        patcher = mock.patch.object(picking.carb.settings, "get_settings", return_value=settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        return settings

    async def test_create_surface_picker_when_remix_not_supported_returns_cpu_picker(self):
        # Arrange
        self._patch_settings(force_cpu=False)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)

        # Act
        with mock.patch.object(
            picking, "is_remix_supported", return_value=(RemixSupport.NOT_SUPPORTED, "no dll")
        ) as is_supported_mock:
            picker = picking.create_surface_picker(viewport_api, cache)

        # Assert
        self.assertIsInstance(picker, picking.CpuRaySurfacePicker)
        is_supported_mock.assert_called_once_with()

    async def test_create_surface_picker_when_remix_waiting_for_init_returns_cpu_picker(self):
        # Arrange
        self._patch_settings(force_cpu=False)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)

        # Act
        with mock.patch.object(picking, "is_remix_supported", return_value=(RemixSupport.WAITING_FOR_INIT, "wait")):
            picker = picking.create_surface_picker(viewport_api, cache)

        # Assert
        self.assertIsInstance(picker, picking.CpuRaySurfacePicker)

    async def test_create_surface_picker_when_force_setting_true_returns_cpu_picker(self):
        # Arrange
        settings = self._patch_settings(force_cpu=True)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)

        # Act
        with mock.patch.object(picking, "is_remix_supported", return_value=(RemixSupport.SUPPORTED, "Success")):
            picker = picking.create_surface_picker(viewport_api, cache)

        # Assert
        self.assertIsInstance(picker, picking.CpuRaySurfacePicker)
        settings.get.assert_any_call(constants.FORCE_CPU_PICKER_SETTING)

    async def test_create_surface_picker_when_supported_and_not_forced_returns_hdremix_picker(self):
        # Arrange
        self._patch_settings(force_cpu=False)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)

        # Act
        with mock.patch.object(picking, "is_remix_supported", return_value=(RemixSupport.SUPPORTED, "Success")):
            picker = picking.create_surface_picker(viewport_api, cache)

        # Assert
        self.assertIsInstance(picker, picking.HdRemixSurfacePicker)

    async def test_create_surface_picker_when_setting_missing_and_supported_returns_hdremix_picker(self):
        # Arrange
        self._patch_settings(force_cpu=None)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)

        # Act
        with mock.patch.object(picking, "is_remix_supported", return_value=(RemixSupport.SUPPORTED, "Success")):
            picker = picking.create_surface_picker(viewport_api, cache)

        # Assert
        self.assertIsInstance(picker, picking.HdRemixSurfacePicker)

    async def test_create_surface_picker_with_factory_returns_factory_result_without_probing_remix(self):
        # Arrange
        self._patch_settings(force_cpu=False)
        viewport_api = mock.MagicMock()
        cache = mock.MagicMock(spec=MeshSurfaceCache)
        custom_picker = mock.MagicMock()
        factory = mock.Mock(return_value=custom_picker)

        # Act
        with mock.patch.object(picking, "is_remix_supported") as is_supported_mock:
            picker = picking.create_surface_picker(viewport_api, cache, picker_factory=factory)

        # Assert
        self.assertIs(picker, custom_picker)
        factory.assert_called_once_with(viewport_api, cache)
        is_supported_mock.assert_not_called()


class TestHdRemixSurfacePicker(AsyncTestCase):
    _PIXEL = (10, 20)
    _NDC = (0.25, -0.5)
    _HIT_PATH = f"{_INSTANCE_ROOT}/mesh"

    def _make_hdremix_viewport(self, camera_position=(0.0, 0.0, 0.0), pixel_valid: bool = True) -> mock.MagicMock:
        viewport_api = mock.MagicMock()
        viewport_api.map_ndc_to_texture_pixel.return_value = (self._PIXEL, viewport_api if pixel_valid else None)
        viewport_api.transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*camera_position))
        return viewport_api

    def _patch_request(self) -> mock.MagicMock:
        patcher = mock.patch.object(picking, "viewport_api_request_query_hdremix")
        request_mock = patcher.start()
        self.addCleanup(patcher.stop)
        return request_mock

    def _issue_pick(self, picker: picking.HdRemixSurfacePicker, request_mock: mock.MagicMock, callback):
        """Issue one pick and return the HdRemix completion callback captured from the request."""
        picker.pick(self._NDC, callback)
        return request_mock.call_args.kwargs["callback"]

    async def test_pick_with_valid_ndc_converts_to_texture_pixel_and_issues_one_request(self):
        # Arrange
        request_mock = self._patch_request()
        viewport_api = self._make_hdremix_viewport()
        picker = picking.HdRemixSurfacePicker(viewport_api)

        # Act
        issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertTrue(issued)
        self.assertTrue(picker.in_flight)
        viewport_api.map_ndc_to_texture_pixel.assert_called_once_with(self._NDC)
        request_mock.assert_called_once()
        pixel_arg = request_mock.call_args.args[0]
        self.assertIsInstance(pixel_arg, carb.Uint2)
        self.assertEqual((pixel_arg[0], pixel_arg[1]), self._PIXEL)
        self.assertEqual(
            request_mock.call_args.kwargs["request_query_type"], picking.RemixRequestQueryType.PATH_AND_WORLDPOS
        )
        self.assertTrue(callable(request_mock.call_args.kwargs["callback"]))

    async def test_pick_with_invalid_texture_pixel_returns_false_without_request(self):
        # Arrange
        request_mock = self._patch_request()
        viewport_api = self._make_hdremix_viewport(pixel_valid=False)
        picker = picking.HdRemixSurfacePicker(viewport_api)

        # Act
        issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertFalse(issued)
        self.assertFalse(picker.in_flight)
        request_mock.assert_not_called()

    async def test_pick_with_ndc_outside_range_returns_false_without_request(self):
        # Arrange
        request_mock = self._patch_request()
        viewport_api = self._make_hdremix_viewport()
        picker = picking.HdRemixSurfacePicker(viewport_api)

        # Act
        issued = picker.pick((0.0, -2.0), mock.Mock())

        # Assert
        self.assertFalse(issued)
        viewport_api.map_ndc_to_texture_pixel.assert_not_called()
        request_mock.assert_not_called()

    async def test_pick_while_request_in_flight_returns_false_and_issues_no_second_request(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        self._issue_pick(picker, request_mock, mock.Mock())

        # Act
        issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertFalse(issued)
        self.assertEqual(request_mock.call_count, 1)
        self.assertTrue(picker.in_flight)

    async def test_pick_when_request_raises_runtime_error_returns_false_and_clears_in_flight(self):
        # Arrange
        request_mock = self._patch_request()
        request_mock.side_effect = RuntimeError("HdRemix extern is unavailable")
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        callback = mock.Mock()

        # Act
        issued = picker.pick(self._NDC, callback)

        # Assert
        self.assertFalse(issued)
        self.assertFalse(picker.in_flight)
        callback.assert_not_called()

    async def test_hdremix_callback_with_valid_path_delivers_surface_hit(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        callback.assert_called_once()
        hit = callback.call_args.args[0]
        self.assertIsInstance(hit, picking.SurfaceHit)
        self.assertEqual(hit.path, Sdf.Path(self._HIT_PATH))
        self.assertIsInstance(hit.world_position, Gf.Vec3d)
        _assert_vec3_close(self, hit.world_position, (1.0, 2.0, 3.0), tolerance=1e-9)
        self.assertFalse(picker.in_flight)

    async def test_hdremix_callback_is_deferred_to_the_event_loop(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))

        # Assert
        callback.assert_not_called()
        self.assertTrue(picker.in_flight)

    async def test_hdremix_callback_with_empty_path_delivers_none(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback("", None, carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        callback.assert_called_once_with(None)
        self.assertFalse(picker.in_flight)

    async def test_hdremix_callback_with_path_but_no_world_position_delivers_none(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback(self._HIT_PATH, None, carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        callback.assert_called_once_with(None)

    async def test_hdremix_callback_with_hit_farther_than_max_distance_delivers_none(self):
        # Arrange
        request_mock = self._patch_request()
        viewport_api = self._make_hdremix_viewport(camera_position=(0.0, 0.0, 100.0))
        picker = picking.HdRemixSurfacePicker(viewport_api, max_distance_getter=lambda: 10.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback(self._HIT_PATH, carb.Double3(0.0, 0.0, 0.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        callback.assert_called_once_with(None)
        self.assertFalse(picker.in_flight)

    async def test_hdremix_callback_with_hit_within_max_distance_of_offset_camera_delivers_hit(self):
        # Arrange
        request_mock = self._patch_request()
        viewport_api = self._make_hdremix_viewport(camera_position=(0.0, 0.0, 100.0))
        picker = picking.HdRemixSurfacePicker(viewport_api, max_distance_getter=lambda: 10.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        hd_callback(self._HIT_PATH, carb.Double3(0.0, 0.0, 95.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(self._HIT_PATH))
        _assert_vec3_close(self, hit.world_position, (0.0, 0.0, 95.0), tolerance=1e-9)

    async def test_hdremix_callback_uses_carb_max_distance_setting_when_no_getter_given(self):
        # Arrange
        request_mock = self._patch_request()
        settings = mock.MagicMock()
        settings.get.side_effect = lambda path: 10.0 if path == constants.MAX_PICK_DISTANCE_SETTING else None
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        with mock.patch.object(picking.carb.settings, "get_settings", return_value=settings):
            hd_callback(self._HIT_PATH, carb.Double3(0.0, 0.0, 50.0), carb.Uint2(*self._PIXEL))
            await asyncio.sleep(0)

        # Assert
        callback.assert_called_once_with(None)
        settings.get.assert_any_call(constants.MAX_PICK_DISTANCE_SETTING)

    async def test_hdremix_callback_uses_default_max_distance_when_setting_unset(self):
        # Arrange
        request_mock = self._patch_request()
        settings = mock.MagicMock()
        settings.get.return_value = None
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)
        within_default = constants.DEFAULT_MAX_PICK_DISTANCE * 0.5

        # Act
        with mock.patch.object(picking.carb.settings, "get_settings", return_value=settings):
            hd_callback(self._HIT_PATH, carb.Double3(0.0, 0.0, within_default), carb.Uint2(*self._PIXEL))
            await asyncio.sleep(0)

        # Assert
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(self._HIT_PATH))

    async def test_hdremix_callback_arriving_after_cancel_is_dropped(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock()
        hd_callback = self._issue_pick(picker, request_mock, callback)
        picker.cancel()

        # Act
        hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        callback.assert_not_called()
        self.assertFalse(picker.in_flight)

    async def test_stale_callback_from_previous_pick_is_dropped_after_new_pick(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        first_callback = mock.Mock()
        stale_hd_callback = self._issue_pick(picker, request_mock, first_callback)
        picker.cancel()
        second_callback = mock.Mock()
        self._issue_pick(picker, request_mock, second_callback)

        # Act
        stale_hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        first_callback.assert_not_called()
        second_callback.assert_not_called()
        self.assertTrue(picker.in_flight)

    async def test_cancel_clears_in_flight_and_allows_new_pick(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        self._issue_pick(picker, request_mock, mock.Mock())
        picker.cancel()

        # Act
        issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertTrue(issued)
        self.assertEqual(request_mock.call_count, 2)

    async def test_hdremix_callback_when_user_callback_raises_logs_error_and_clears_in_flight(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        callback = mock.Mock(side_effect=ValueError("boom"))
        hd_callback = self._issue_pick(picker, request_mock, callback)

        # Act
        with mock.patch.object(picking.carb, "log_error") as log_error_mock:
            hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))
            await asyncio.sleep(0)

        # Assert
        callback.assert_called_once()
        log_error_mock.assert_called_once()
        self.assertFalse(picker.in_flight)

    async def test_in_flight_before_any_pick_is_false(self):
        # Arrange
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())

        # Act
        in_flight = picker.in_flight

        # Assert
        self.assertFalse(in_flight)

    async def test_hdremix_callback_when_scheduling_fails_logs_error_and_clears_in_flight(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        callback = mock.Mock()
        closed_loop = mock.Mock()
        closed_loop.call_soon_threadsafe.side_effect = RuntimeError("Event loop is closed")
        with mock.patch.object(picking, "_current_event_loop", return_value=closed_loop):
            hd_callback = self._issue_pick(picker, request_mock, callback)

        with mock.patch.object(picking.carb, "log_error") as log_error_mock:
            # Act
            hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))

        # Assert
        log_error_mock.assert_called_once()
        self.assertFalse(picker.in_flight)
        callback.assert_not_called()

    async def test_hdremix_callback_from_worker_thread_delivers_hit_on_the_main_thread(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        delivery_threads: list[str] = []
        callback = mock.Mock(side_effect=lambda _hit: delivery_threads.append(threading.current_thread().name))
        hd_callback = self._issue_pick(picker, request_mock, callback)
        worker = threading.Thread(
            target=hd_callback,
            args=(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL)),
            name="hdremix-worker",
        )

        # Act
        worker.start()
        worker.join()
        await asyncio.sleep(0)

        # Assert
        callback.assert_called_once()
        self.assertEqual(delivery_threads, [threading.main_thread().name])
        hit = callback.call_args.args[0]
        self.assertEqual(hit.path, Sdf.Path(self._HIT_PATH))
        _assert_vec3_close(self, hit.world_position, (1.0, 2.0, 3.0), tolerance=1e-9)
        self.assertFalse(picker.in_flight)

    async def test_pick_without_event_loop_returns_false_without_request(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        callback = mock.Mock()

        # Act
        with mock.patch.object(picking, "_current_event_loop", return_value=None):
            result = picker.pick(self._NDC, callback)

        # Assert
        self.assertFalse(result)
        request_mock.assert_not_called()
        self.assertFalse(picker.in_flight)
        callback.assert_not_called()

    async def test_pick_after_stale_request_timeout_drops_it_and_issues_new_request(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        with mock.patch.object(picking.time, "monotonic", return_value=0.0):
            self._issue_pick(picker, request_mock, mock.Mock())

        with (
            mock.patch.object(picking.time, "monotonic", return_value=picking._STALE_PICK_TIMEOUT + 1.0),
            mock.patch.object(picking.carb, "log_warn") as log_warn_mock,
        ):
            # Act
            issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertTrue(issued)
        self.assertEqual(request_mock.call_count, 2)
        self.assertTrue(picker.in_flight)
        log_warn_mock.assert_called_once()

    async def test_hdremix_callback_of_timed_out_request_is_dropped_after_new_pick(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport(), max_distance_getter=lambda: 1000.0)
        first_callback = mock.Mock()
        with mock.patch.object(picking.time, "monotonic", return_value=0.0):
            stale_hd_callback = self._issue_pick(picker, request_mock, first_callback)
        second_callback = mock.Mock()
        with mock.patch.object(picking.time, "monotonic", return_value=picking._STALE_PICK_TIMEOUT + 1.0):
            self._issue_pick(picker, request_mock, second_callback)

        # Act
        stale_hd_callback(self._HIT_PATH, carb.Double3(1.0, 2.0, 3.0), carb.Uint2(*self._PIXEL))
        await asyncio.sleep(0)

        # Assert
        first_callback.assert_not_called()
        second_callback.assert_not_called()
        self.assertTrue(picker.in_flight)

    async def test_pick_within_stale_timeout_while_in_flight_returns_false(self):
        # Arrange
        request_mock = self._patch_request()
        picker = picking.HdRemixSurfacePicker(self._make_hdremix_viewport())
        with mock.patch.object(picking.time, "monotonic", return_value=0.0):
            self._issue_pick(picker, request_mock, mock.Mock())

        with mock.patch.object(picking.time, "monotonic", return_value=picking._STALE_PICK_TIMEOUT - 0.5):
            # Act
            issued = picker.pick(self._NDC, mock.Mock())

        # Assert
        self.assertFalse(issued)
        self.assertEqual(request_mock.call_count, 1)
