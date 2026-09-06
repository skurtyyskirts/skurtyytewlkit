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

from collections.abc import Sequence
from unittest import mock

import carb.input
import omni.kit.undo
import omni.ui as ui
import omni.usd
from carb.input import KeyboardEventType, KeyboardInput, MouseEventType
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from lightspeed.hydra.remix.core import RemixSupport
from lightspeed.trex.contexts.extension import get_instance as _get_context_manager
from lightspeed.trex.contexts.setup import Contexts as _TrexContext
from lightspeed.trex.scatter.core import ScatterBrushSettings, ScatterMode, get_scatter_brush_controller
from lightspeed.trex.scatter.core.constants import IS_REMIX_SCATTER_ATTR
from lightspeed.trex.viewports.manipulators import global_selection as _global_selection
from lightspeed.trex.viewports.shared.widget import create_instance as _create_viewport_instance
from omni.flux.utils.widget.resources import get_test_data as _get_test_data
from omni.kit import ui_test
from omni.kit.ui_test import Vec2
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.ui.tests.test_base import OmniUiTest
from pxr import Gf, Sdf, Usd, UsdGeom

_WINDOW_WIDTH = 1436
_WINDOW_HEIGHT = 1000
_CONTEXT_NAME = "TestScatterBrushContext"
_PROJECT_STAGE = "usd/project_example/combined.usda"
_INGESTED_CUBE = "usd/project_example/assets/ingested/cube.usda"
_REPLACEMENT_LAYER_NAME = "replacements.usda"
_MESHES_ROOT = "/RootNode/meshes"
_FLOOR_PROTOTYPE = f"{_MESHES_ROOT}/mesh_0AB745B8BEE1F16B"
_FLOOR_INSTANCE = "/RootNode/instances/inst_0AB745B8BEE1F16B_0"
_WALL_PROTOTYPE = f"{_MESHES_ROOT}/mesh_CED45075A077A49A"
_TOOLBAR_BUTTON_IDENTIFIER = "scatter_brush"
_TOOL_LAYER_NAME = "Scatter Brush"
_TOOL_LAYER_CATEGORY = "tools"
_XFORM_OP_ORDER = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
_DRAG_STEPS = 18
# The brush is deterministic (fixed seed) and the cubes it places are scaled down so the CPU picker used in headless
# tests keeps hitting the captured surface, not the cubes painted by the previous stroke of the same test.
_BRUSH_SETTINGS = {
    "radius": 60.0,
    "density": 6.0,
    "padding": 0.0,
    "stamp_spacing": 20.0,
    "randomize_seed": False,
    "seed": 1,
    "conform_to_surface": True,
    "scale_min": 0.05,
    "scale_max": 0.05,
}
# Erase strokes use a wider brush than the paint stroke so clumps generated at the rim of a paint stamp are covered
# even when the erase picks land a few units away from the paint picks.
_ERASE_RADIUS = 90.0
# mesh_0AB745B8BEE1F16B is Z-up capture geometry whose only solid, walkable surface is a deck at z = 0 covering
# x[-1152, -752] y[1528, 1664]; the stroke below stays inside it with the brush radius to spare.
_FLOOR_CAMERA_POSITION = Gf.Vec3d(-955.0, 1000.0, 700.0)
_FLOOR_CAMERA_TARGET = Gf.Vec3d(-955.0, 1596.0, 0.0)
_FLOOR_STROKE = (Gf.Vec3d(-1080.0, 1596.0, 0.0), Gf.Vec3d(-830.0, 1596.0, 0.0))
_FLOOR_SURFACE_Z = 0.0
_SURFACE_Z_TOLERANCE = 24.0
# mesh_CED45075A077A49A exposes a flat top strip at z = 0 covering x[640, 768] y[1088, 1104].
_WALL_STROKE = (Gf.Vec3d(660.0, 1096.0, 0.0), Gf.Vec3d(748.0, 1096.0, 0.0))
# High vantage point from which both the deck and the wall strip are in view, so one stroke can cross from one to
# the other.
_WIDE_CAMERA_POSITION = Gf.Vec3d(-100.0, 600.0, 1800.0)
_WIDE_CAMERA_TARGET = Gf.Vec3d(-85.0, 1346.0, 0.0)
# One wheel notch upward; omni.kit.ui_test normalizes the delta by the application window size before it reaches the
# scene view, so a large value is needed to represent a single notch.
_WHEEL_UP_ONE_NOTCH = Vec2(0, 1200)


class TestScatterBrushViewportE2E(OmniUiTest):
    async def setUp(self):
        await super().setUp()
        self._usd_context = omni.usd.get_context(_CONTEXT_NAME) or omni.usd.create_context(_CONTEXT_NAME)
        await self._usd_context.open_stage_async(_get_test_data(_PROJECT_STAGE))
        await self.__wait_stage_loading()

        self._window = ui.Window("TestScatterBrushViewportUI", width=_WINDOW_WIDTH, height=_WINDOW_HEIGHT)
        with self._window.frame:
            self._widget = _create_viewport_instance(_CONTEXT_NAME)
        await self.__wait_for_viewport()
        self._viewport_api = self._widget.viewport_api
        self._stage = self._usd_context.get_stage()
        self.assertIsNotNone(
            self._widget.viewport_layers.find_viewport_layer(_TOOL_LAYER_NAME, _TOOL_LAYER_CATEGORY),
            "The scatter brush scene layer was not created for the viewport",
        )

        # The brush only listens to the active viewport, so click into it the way a user would before painting.
        viewport = ui_test.find(f"{self._window.title}//Frame/**/.identifier == 'viewport'")
        await viewport.click()
        await ui_test.human_delay()
        self.assertTrue(self._widget.is_active())

        # The Toolkit authors every replacement in the project's replacement layer, so the brush must too.
        self._edit_layer = self.__set_edit_target_to_replacement_layer()

        # The brush palette is process-wide and persisted; start every test from the same known brush and put the
        # previous brush back afterwards.
        self._controller = get_scatter_brush_controller()
        self._saved_settings = self._controller.settings
        self._saved_assets = self._controller.assets
        for entry in self._saved_assets:
            self._controller.remove_asset(entry.path)
        self._controller.replace_settings(ScatterBrushSettings(**_BRUSH_SETTINGS))
        self._controller.add_asset(_get_test_data(_INGESTED_CUBE))

        await self.__aim_camera(_FLOOR_CAMERA_POSITION, _FLOOR_CAMERA_TARGET)
        omni.kit.undo.clear_stack()

    async def tearDown(self):
        # The controller is the process-wide singleton the extension's toolbar button subscribed to at startup, so it
        # is reset rather than destroyed: destroying it would detach that button for every later test in the process.
        self._controller.set_mode(ScatterMode.OFF)
        await ui_test.human_delay()
        self._widget.destroy()
        self._window.destroy()
        self.__restore_brush()
        await self.__release_hydra_engines()
        await self._usd_context.close_stage_async()
        # The test instance outlives the test and a destroyed ui.Window stays registered (and visible to ui_test) for
        # as long as a Python reference keeps it alive, so drop everything that points at the window and the widget.
        self._widget = None
        self._window = None
        self._viewport_api = None
        self._stage = None
        self._edit_layer = None
        self._controller = None
        self._saved_settings = None
        self._saved_assets = None
        await ui_test.wait_n_updates()
        await super().tearDown()

    async def test_paint_stroke_places_scatter_prims_under_prototype(self):
        # The user turns the brush on from the viewport toolbar.
        button = ui_test.find(
            f"{self._window.title}//Frame/**/ToolButton[*].identifier=='{_TOOLBAR_BUTTON_IDENTIFIER}'"
        )
        self.assertIsNotNone(button, "The scatter brush toolbar button is missing from the viewport toolbar")
        await button.click()
        await ui_test.human_delay(human_delay_speed=4)
        self.assertEqual(ScatterMode.PAINT, self._controller.mode)

        # One left-button drag across the deck of the captured floor paints a stroke.
        await self.__drag([self.__screen_point(point) for point in _FLOOR_STROKE])

        # Every clump is a reference prim authored under the captured prototype, sitting on the deck surface.
        placements = self.__scatter_placements(_FLOOR_PROTOTYPE)
        self.assertGreaterEqual(len(placements), 1)
        for placement in placements:
            self.assertTrue(placement.GetAttribute(IS_REMIX_REF_ATTR).Get())
            self.assertTrue(placement.HasAuthoredReferences())
            self.assertEqual(_XFORM_OP_ORDER, list(placement.GetAttribute("xformOpOrder").Get()))
            translate = placement.GetAttribute("xformOp:translate").Get()
            self.assertAlmostEqual(_FLOOR_SURFACE_Z, translate[2], delta=_SURFACE_Z_TOLERANCE, msg=str(placement))

            # The prototype is what the game renders through its instances, so the same clump composes under the
            # instance (which sits at identity) and its cube resolves through the relative asset reference.
            instance_placement = self._stage.GetPrimAtPath(self.__instance_path(placement))
            self.assertTrue(instance_placement.IsValid(), f"{placement.GetPath()} does not compose under the instance")
            self.assertTrue(instance_placement.GetChild("Toto").IsA(UsdGeom.Mesh))

    async def test_ctrl_b_toggles_brush_and_toolbar_button_follows(self):
        button = ui_test.find(
            f"{self._window.title}//Frame/**/ToolButton[*].identifier=='{_TOOLBAR_BUTTON_IDENTIFIER}'"
        )
        self.assertIsNotNone(button, "The scatter brush toolbar button is missing from the viewport toolbar")
        self.assertFalse(button.model.as_bool)
        viewport = ui_test.find(f"{self._window.title}//Frame/**/.identifier == 'viewport'")
        await ui_test.input.emulate_mouse_move(viewport.center)
        await ui_test.human_delay()
        # The hotkey manager routes every press through the current Trex context, which the Toolkit sets when it
        # opens its first workspace; this bare viewport harness has to provide one the same way.
        _get_context_manager().get_usd_context(_TrexContext.STAGE_CRAFT)

        # Ctrl+B over the active viewport turns the brush on without touching the toolbar, and the toolbar toggle
        # still lights up because it follows the process-wide brush controller.
        await ui_test.emulate_keyboard_press(KeyboardInput.B, carb.input.KEYBOARD_MODIFIER_FLAG_CONTROL)
        await ui_test.human_delay(human_delay_speed=4)
        self.assertEqual(ScatterMode.PAINT, self._controller.mode)
        self.assertTrue(button.model.as_bool)

        # A second Ctrl+B turns the brush off again and the toggle goes dark with it.
        await ui_test.emulate_keyboard_press(KeyboardInput.B, carb.input.KEYBOARD_MODIFIER_FLAG_CONTROL)
        await ui_test.human_delay(human_delay_speed=4)
        self.assertEqual(ScatterMode.OFF, self._controller.mode)
        self.assertFalse(button.model.as_bool)

    async def test_undo_after_stroke_removes_all_placements_and_redo_restores(self):
        self._controller.set_mode(ScatterMode.PAINT)
        await ui_test.human_delay()
        await self.__drag([self.__screen_point(point) for point in _FLOOR_STROKE])
        painted_count = len(self.__scatter_placements(_FLOOR_PROTOTYPE))
        self.assertGreaterEqual(painted_count, 1)

        # A whole stroke is one undo step: undoing it removes every clump it painted.
        omni.kit.undo.undo()
        await ui_test.human_delay()
        self.assertEqual(0, len(self.__scatter_placements(_FLOOR_PROTOTYPE)))

        # Redo brings the same stroke back.
        omni.kit.undo.redo()
        await ui_test.human_delay()
        self.assertEqual(painted_count, len(self.__scatter_placements(_FLOOR_PROTOTYPE)))

    async def test_shift_drag_erases_previous_placements(self):
        self._controller.set_mode(ScatterMode.PAINT)
        await ui_test.human_delay()
        stroke = [self.__screen_point(point) for point in _FLOOR_STROKE]
        await self.__drag(stroke)
        self.assertGreaterEqual(len(self.__scatter_placements(_FLOOR_PROTOTYPE)), 1)

        # The user widens the brush a little and drags back over the same path while holding Shift to erase.
        self.assertTrue(self._controller.update_settings(radius=_ERASE_RADIUS))
        await ui_test.input.emulate_keyboard(KeyboardEventType.KEY_PRESS, KeyboardInput.LEFT_SHIFT)
        await ui_test.human_delay()
        try:
            await self.__drag(stroke)
        finally:
            await ui_test.input.emulate_keyboard(KeyboardEventType.KEY_RELEASE, KeyboardInput.LEFT_SHIFT)
            await ui_test.human_delay()

        self.assertEqual(0, len(self.__scatter_placements(_FLOOR_PROTOTYPE)))

    async def test_stroke_across_two_meshes_places_under_both_prototypes(self):
        self._controller.set_mode(ScatterMode.PAINT)
        await ui_test.human_delay()

        # From high above, both the floor deck and the top of the second captured mesh are in view, so one drag
        # crosses the deck, the gap where nothing is under the cursor, and then the wall top.
        await self.__aim_camera(_WIDE_CAMERA_POSITION, _WIDE_CAMERA_TARGET)
        world_points = [*_FLOOR_STROKE, *_WALL_STROKE]
        self.assertTrue(
            all(self.__is_in_view(point) for point in world_points),
            "The wide camera no longer frames both captured surfaces, so a single stroke cannot cross them",
        )
        await self.__drag([self.__screen_point(point) for point in world_points])

        # Each surface got its own container under its own prototype.
        self.assertGreaterEqual(len(self.__scatter_placements(_FLOOR_PROTOTYPE)), 1)
        self.assertGreaterEqual(len(self.__scatter_placements(_WALL_PROTOTYPE)), 1)

    async def test_b_hold_wheel_changes_radius_without_moving_camera(self):
        self._controller.set_mode(ScatterMode.PAINT)
        await ui_test.human_delay()
        viewport = ui_test.find(f"{self._window.title}//Frame/**/.identifier == 'viewport'")
        self.assertIsNotNone(viewport)
        await ui_test.input.emulate_mouse_move(viewport.center)
        radius_before = self._controller.settings.radius
        camera_before = self.__camera_matrix()

        # Holding B turns the wheel into a brush-size control instead of a camera zoom.
        await ui_test.input.emulate_keyboard(KeyboardEventType.KEY_PRESS, KeyboardInput.B)
        await ui_test.human_delay()
        try:
            await ui_test.input.emulate_mouse_scroll(_WHEEL_UP_ONE_NOTCH)
            await ui_test.human_delay(human_delay_speed=4)
        finally:
            await ui_test.input.emulate_keyboard(KeyboardEventType.KEY_RELEASE, KeyboardInput.B)
            await ui_test.human_delay()

        self.assertGreater(self._controller.settings.radius, radius_before)
        self.assertEqual(camera_before, self.__camera_matrix())

    async def test_brush_off_leaves_selection_drag_working(self):
        # With the brush off, the same drag is an ordinary viewport interaction: the selection tool owns it and asks
        # the renderer what lies inside the dragged rectangle. Headless tests render with Storm, so the HdRemix
        # picking seam answers with the floor instance in place of the renderer.
        self.assertEqual(ScatterMode.OFF, self._controller.mode)
        selection = self._usd_context.get_selection()
        selection.set_selected_prim_paths([], False)
        picking_requests = []

        def fake_objectpicking_request(x0, y0, x1, y1, callback):
            picking_requests.append((x0, y0, x1, y1))
            callback([_FLOOR_INSTANCE])

        with (
            mock.patch.object(_global_selection, "is_remix_supported", return_value=(RemixSupport.SUPPORTED, "")),
            mock.patch.object(
                _global_selection, "hdremix_objectpicking_request", side_effect=fake_objectpicking_request
            ),
            mock.patch.object(_global_selection, "hdremix_highlight_paths"),
        ):
            await self.__drag([self.__screen_point(point) for point in _FLOOR_STROKE])

        # The selection tool received the rectangle and applied the answer; the brush never wrote anything.
        self.assertEqual(1, len(picking_requests))
        self.assertEqual([_FLOOR_INSTANCE], selection.get_selected_prim_paths())
        self.assertEqual(ScatterMode.OFF, self._controller.mode)
        self.assertEqual([], self.__scatter_placements())

    async def test_hover_in_paint_mode_authors_nothing(self):
        self._controller.set_mode(ScatterMode.PAINT)
        await ui_test.human_delay()

        # Moving the brush over the floor only shows the cursor; nothing is written until the user presses the button.
        await ui_test.input.emulate_mouse_move(self.__screen_point(_FLOOR_STROKE[0]))
        await ui_test.human_delay(human_delay_speed=30)

        tool_layer = self._widget.viewport_layers.find_viewport_layer(_TOOL_LAYER_NAME, _TOOL_LAYER_CATEGORY)
        self.assertTrue(tool_layer.layer.enabled, "The brush should be listening to the mouse while in paint mode")
        self.assertEqual([], self.__scatter_placements())

    async def __wait_stage_loading(self, wait_frames: int = 2, timeout: int = 1000):
        for _ in range(timeout):
            _, files_loaded, total_files = self._usd_context.get_stage_loading_status()
            if not files_loaded and not total_files:
                break
            await ui_test.wait_n_updates()
        else:
            self.fail(f"Timed out waiting for stage loading in context {_CONTEXT_NAME!r}")
        self._usd_context.reset_renderer_accumulation()
        await ui_test.wait_n_updates(wait_frames)

    async def __wait_for_viewport(self, timeout: int = 120):
        for _ in range(timeout):
            viewports = ui_test.find_all(f"{self._window.title}//Frame/**/.identifier == 'viewport'")
            viewport_api = self._widget.viewport_api
            frame = self._widget.viewport_frame()
            if viewports and viewport_api and str(viewport_api.camera_path) and frame.computed_width > 0:
                return
            await ui_test.wait_n_updates()
        self.fail(f"The viewport in {self._window.title!r} did not come up")

    async def __release_hydra_engines(self):
        # Copied from the shared viewport harness (itself from omni/kit/widget/viewport/tests/test_ray_query.py).
        await self.wait_n_updates(10)
        omni.usd.release_all_hydra_engines(self._usd_context)
        await self.wait_n_updates(10)

    def __restore_brush(self):
        # Put back the palette and settings that were in the controller before the test replaced them.
        for entry in self._controller.assets:
            self._controller.remove_asset(entry.path)
        for entry in self._saved_assets:
            if not self._controller.add_asset(entry.path, up_axis=entry.up_axis):
                continue
            self._controller.set_asset_enabled(entry.path, entry.enabled)
            self._controller.set_asset_weight(entry.path, entry.weight)
        self._controller.replace_settings(self._saved_settings)

    def __set_edit_target_to_replacement_layer(self) -> Sdf.Layer:
        root_layer = self._stage.GetRootLayer()
        sublayer_path = next(path for path in root_layer.subLayerPaths if path.endswith(_REPLACEMENT_LAYER_NAME))
        layer = Sdf.Layer.FindOrOpen(root_layer.ComputeAbsolutePath(sublayer_path))
        self.assertIsNotNone(layer)
        self._stage.SetEditTarget(Usd.EditTarget(layer))
        self.assertEqual(layer.identifier, self._stage.GetEditTarget().GetLayer().identifier)
        return layer

    async def __aim_camera(self, position: Gf.Vec3d, target: Gf.Vec3d):
        camera_state = ViewportCameraState(str(self._viewport_api.camera_path), self._viewport_api)
        camera_state.set_position_world(position, False)
        camera_state.set_target_world(target, True)
        await ui_test.wait_n_updates(5)

    def __camera_matrix(self) -> tuple[float, ...]:
        camera_prim = self._stage.GetPrimAtPath(self._viewport_api.camera_path)
        self.assertTrue(camera_prim.IsValid())
        matrix = UsdGeom.Xformable(camera_prim).GetLocalTransformation(self._viewport_api.time)
        return tuple(component for row in matrix for component in row)

    def __is_in_view(self, world_point: Gf.Vec3d) -> bool:
        ndc = self._viewport_api.world_to_ndc.Transform(world_point)
        return -1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0 and ndc[2] <= 1.0

    def __screen_point(self, world_point: Gf.Vec3d) -> Vec2:
        self.assertTrue(self.__is_in_view(world_point), f"{world_point} projects outside the viewport")
        ndc = self._viewport_api.world_to_ndc.Transform(world_point)
        # Same mapping as the prim transform manipulator drag tests: viewport NDC -> render texture pixel -> the
        # on-screen frame the texture fills.
        pixel, viewport_api = self._viewport_api.map_ndc_to_texture_pixel((ndc[0], ndc[1]))
        self.assertIsNotNone(viewport_api)
        frame = self._widget.viewport_frame()
        resolution = self._viewport_api.resolution
        return Vec2(
            frame.screen_position_x + pixel[0] * frame.computed_width / resolution[0],
            frame.screen_position_y + pixel[1] * frame.computed_height / resolution[1],
        )

    async def __drag(self, points: Sequence[Vec2]):
        # Park the cursor first so the brush has picked the surface under it before the button goes down.
        await ui_test.input.emulate_mouse(MouseEventType.MOVE, points[0])
        await ui_test.human_delay(human_delay_speed=4)
        await ui_test.input.emulate_mouse(MouseEventType.LEFT_BUTTON_DOWN, points[0])
        await ui_test.human_delay()
        for start, end in zip(points, points[1:]):
            await ui_test.input.emulate_mouse_slow_move(start, end, num_steps=_DRAG_STEPS, human_delay_speed=2)
        await ui_test.human_delay()
        await ui_test.input.emulate_mouse(MouseEventType.LEFT_BUTTON_UP, points[-1])
        await ui_test.human_delay(human_delay_speed=30)

    def __scatter_placements(self, prototype_path: str | None = None) -> list[Usd.Prim]:
        # Scattered clumps are the referenced children of the marked containers directly under a prototype.
        if prototype_path is None:
            prototypes = list(self._stage.GetPrimAtPath(_MESHES_ROOT).GetChildren())
        else:
            prototypes = [self._stage.GetPrimAtPath(prototype_path)]
        placements = []
        for prototype in prototypes:
            for container in prototype.GetChildren():
                if not container.HasAttribute(IS_REMIX_SCATTER_ATTR):
                    continue
                placements.extend(
                    child
                    for child in container.GetChildren()
                    if child.HasAttribute(IS_REMIX_SCATTER_ATTR) and child.HasAuthoredReferences()
                )
        return placements

    @staticmethod
    def __instance_path(placement: Usd.Prim) -> Sdf.Path:
        relative_path = placement.GetPath().MakeRelativePath(Sdf.Path(_FLOOR_PROTOTYPE))
        return Sdf.Path(_FLOOR_INSTANCE).AppendPath(relative_path)
