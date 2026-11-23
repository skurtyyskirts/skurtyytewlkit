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

__all__ = ["TestBrushCursorMatrix", "TestScatterBrushButtonGroup", "TestScatterBrushTool"]

import contextlib
import itertools
import math
import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock

import carb
import carb.input
import numpy as np
import omni.kit.test
import omni.kit.undo
import omni.ui as ui
import omni.usd
from lightspeed.common.constants import INSTANCE_PATH, ROOTNODE_INSTANCES, ROOTNODE_MESHES
from lightspeed.trex.scatter.core import (
    ApplyTo,
    Falloff,
    PrimBudgetStatus,
    ScatterBrushSettings,
    ScatterMode,
    SurfaceHit,
    UpAxis,
)
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core import (
    destroy_scatter_brush_controller,
    get_scatter_brush_controller,
)
from lightspeed.trex.scatter.core import stroke as stroke_module
from lightspeed.trex.scatter.core import (
    tangent_basis,
)
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX
from lightspeed.trex.viewports.shared.widget.scene.utils import flatten_matrix
from lightspeed.trex.viewports.shared.widget.tools import scatter_brush as scatter_brush_module
from lightspeed.trex.viewports.shared.widget.tools.scatter_brush import (
    ScatterBrushButtonGroup,
    ScatterBrushTool,
    brush_cursor_matrix,
    create_button_instance,
    delete_button_instance,
    scatter_brush_factory,
)
from omni.kit.notification_manager import NotificationStatus
from omni.ui import scene as sc
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_CONTEXT_NAME = "scatter_brush_unit_test"
_HASH = "0AB745B8BEE1F16B"
_PROTOTYPE = Sdf.Path(f"{ROOTNODE_MESHES}/mesh_{_HASH}")
_INSTANCE_0 = Sdf.Path(f"{INSTANCE_PATH}{_HASH}_0")
_MESH_PATH = _INSTANCE_0.AppendChild("mesh")
_CONTAINER_PATH = _PROTOTYPE.AppendChild(f"{CONTAINER_PREFIX}default")
_HALF_SIZE = 500.0
_HIT_POSITION = Gf.Vec3d(5.0, 5.0, 0.0)
_RADIUS = 50.0
# Far enough from the hit that walking to it takes several stamps at the default spacing.
_FAR_POSITION = Gf.Vec3d(205.0, 5.0, 0.0)
_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{
    def Cube "Cube"
    {
    }
}
"""


def _author_instance(stage: Usd.Stage, index: int) -> None:
    """Author ``inst_<HASH>_<index>`` referencing the prototype, like a Remix capture does."""
    instance = UsdGeom.Xform.Define(stage, Sdf.Path(f"{INSTANCE_PATH}{_HASH}_{index}"))
    instance.GetPrim().GetReferences().AddInternalReference(_PROTOTYPE)


def _build_capture_stage(stage: Usd.Stage) -> None:
    """Mirror the capture topology: a prototype holding a flat quad in the XY plane and one instance referencing it."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/RootNode")
    stage.DefinePrim(ROOTNODE_MESHES)
    stage.DefinePrim(ROOTNODE_INSTANCES)
    prototype = UsdGeom.Xform.Define(stage, _PROTOTYPE)
    prototype.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh = UsdGeom.Mesh.Define(stage, _PROTOTYPE.AppendChild("mesh"))
    points = [
        Gf.Vec3f(-_HALF_SIZE, -_HALF_SIZE, 0.0),
        Gf.Vec3f(_HALF_SIZE, -_HALF_SIZE, 0.0),
        Gf.Vec3f(_HALF_SIZE, _HALF_SIZE, 0.0),
        Gf.Vec3f(-_HALF_SIZE, _HALF_SIZE, 0.0),
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateExtentAttr(Vt.Vec3fArray([points[0], points[2]]))
    _author_instance(stage, 0)


class _FakePicker:
    """Scriptable surface picker that records picks and delivers hits synchronously or on demand.

    Like the real pickers it refuses a pick while one is pending; with ``stale_pending`` set it drops the pending
    request first instead, the way ``HdRemixSurfacePicker`` recovers from a completion that never arrived.
    """

    def __init__(self):
        self.hit: SurfaceHit | None = None
        self.deliver_immediately = True
        self.stale_pending = False
        self.pick_attempts: list[tuple[float, float]] = []
        self.picks: list[tuple[float, float]] = []
        self.cancel_calls = 0
        self._pending = None

    @property
    def in_flight(self) -> bool:
        return self._pending is not None

    def pick(self, ndc, callback) -> bool:
        self.pick_attempts.append(ndc)
        if self._pending is not None:
            if not self.stale_pending:
                return False
            self.cancel()
        self.picks.append(ndc)
        if self.deliver_immediately:
            callback(self.hit)
        else:
            self._pending = callback
        return True

    def deliver(self, hit: SurfaceHit | None) -> None:
        callback, self._pending = self._pending, None
        callback(hit)

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._pending = None


def _clock_out_of_time_after(reads: int):
    """Return a stroke clock that stays at 0 for ``reads`` reads and then has passed every time budget."""
    remaining = iter(range(reads))
    return lambda: 0.0 if next(remaining, None) is not None else 1.0


def _patch_brush_persistence(enter) -> None:
    """Keep the process-wide persisted brush settings and assets out of the controller under test."""
    enter(mock.patch.object(ScatterBrushSettings, "save_to_carb"))
    enter(mock.patch.object(ScatterBrushSettings, "load_from_carb", return_value=ScatterBrushSettings()))
    enter(mock.patch.object(controller_module, "save_assets_to_carb"))
    enter(mock.patch.object(controller_module, "load_assets_from_carb", return_value=[]))


def _detach_process_controller() -> controller_module.ScatterBrushController | None:
    """Set the process-wide controller aside so the test creates its own, and return the one set aside.

    The extension's toolbar button subscribed to the process-wide controller at startup; destroying that controller
    would detach the button for every later test in the process.
    """
    previous = controller_module._CONTROLLER_INSTANCE
    controller_module._CONTROLLER_INSTANCE = None
    return previous


def _restore_process_controller(previous: controller_module.ScatterBrushController | None) -> None:
    """Destroy the controller the test created and hand the process-wide one back."""
    destroy_scatter_brush_controller()
    controller_module._CONTROLLER_INSTANCE = previous


class TestScatterBrushButtonGroup(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._stack = contextlib.ExitStack()
        enter = self._stack.enter_context
        self._previous_controller = _detach_process_controller()
        _patch_brush_persistence(enter)
        self._in_context = enter(mock.patch.object(ScatterBrushButtonGroup, "_is_in_context", return_value=True))
        self._controller = get_scatter_brush_controller()
        self._group = ScatterBrushButtonGroup()
        self._window = ui.Window("ScatterBrushButtonGroupTest", width=200, height=100)

    async def tearDown(self):
        self._group.clean()
        self._group = None
        self._window.destroy()
        self._window = None
        _restore_process_controller(self._previous_controller)
        self._stack.close()

    def _create_button(self) -> ui.Widget:
        with self._window.frame:
            widgets = self._group.create(ui.Pixel(24))
        return widgets[ScatterBrushButtonGroup.name]

    async def test_create_returns_tool_button_with_identifier(self):
        # Arrange
        # (setUp)

        # Act
        button = self._create_button()

        # Assert
        self.assertIsInstance(button, ui.ToolButton)
        self.assertEqual(button.identifier, "scatter_brush")
        self.assertEqual(button.name, "scatter_brush")

    async def test_get_style_with_defined_entries_includes_checked_key(self):
        # Arrange
        base_key = "Button.Image::scatter_brush"
        fake_style = SimpleNamespace(default={base_key: {"image_url": "a.svg"}, f"{base_key}:checked": {"color": 1}})

        # Act
        with mock.patch.object(scatter_brush_module, "style", fake_style):
            result = self._group.get_style()

        # Assert
        self.assertEqual(set(result), {base_key, f"{base_key}:checked"})

    async def test_set_model_true_in_context_switches_controller_to_paint(self):
        # Arrange
        self._create_button()

        # Act
        self._group._model.set_value(True)

        # Assert
        self.assertEqual(self._controller.mode, ScatterMode.PAINT)

    async def test_set_model_false_in_context_switches_controller_to_off(self):
        # Arrange
        self._create_button()
        self._controller.set_mode(ScatterMode.ERASE)

        # Act
        self._group._model.set_value(False)

        # Assert
        self.assertEqual(self._controller.mode, ScatterMode.OFF)

    async def test_set_model_out_of_context_reverts_model_and_keeps_mode(self):
        # Arrange
        self._create_button()
        self._in_context.return_value = False

        # Act
        self._group._model.set_value(True)

        # Assert
        self.assertEqual(self._controller.mode, ScatterMode.OFF)
        self.assertFalse(self._group._model.as_bool)

    async def test_controller_mode_change_updates_model_value(self):
        # Arrange
        self._create_button()

        # Act
        self._controller.set_mode(ScatterMode.ERASE)

        # Assert
        self.assertTrue(self._group._model.as_bool)

    async def test_clean_drops_button_and_model(self):
        # Arrange
        self._create_button()

        # Act
        self._group.clean()

        # Assert
        self.assertIsNone(self._group._button)
        self.assertIsNone(self._group._model)

    async def test_create_button_instance_registers_module_singleton(self):
        # Arrange
        self.addCleanup(delete_button_instance)

        # Act
        group = create_button_instance()
        self.addCleanup(group.clean)

        # Assert
        self.assertIsInstance(group, ScatterBrushButtonGroup)
        self.assertIs(scatter_brush_module._scatter_button_group, group)


class TestScatterBrushTool(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._stack = contextlib.ExitStack()
        enter = self._stack.enter_context
        self._previous_controller = _detach_process_controller()
        _patch_brush_persistence(enter)
        self._temp_dir = tempfile.TemporaryDirectory()
        asset_path = pathlib.Path(self._temp_dir.name) / "assets" / "cube.usda"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(_ASSET_USDA, encoding="utf-8")
        self._asset_path = asset_path.as_posix()

        self._keys_down: set[carb.input.KeyboardInput] = set()
        self._ndc: tuple[float, float] | None = None
        enter(mock.patch.object(scatter_brush_module, "_is_key_down", side_effect=lambda key: key in self._keys_down))
        enter(mock.patch.object(scatter_brush_module, "_mouse_ndc", side_effect=lambda _frame: self._ndc))
        self._post_notification = enter(mock.patch.object(scatter_brush_module, "post_notification"))

        self._usd_context = omni.usd.get_context(_CONTEXT_NAME) or omni.usd.create_context(_CONTEXT_NAME)
        await self._usd_context.new_stage_async()
        self._stage = self._usd_context.get_stage()
        _build_capture_stage(self._stage)

        self._controller = get_scatter_brush_controller()
        # Painting on whatever the picker reports, so a test only has to script the picker; the Selected mode cases
        # select a mesh and switch the setting themselves.
        self._controller.replace_settings(
            ScatterBrushSettings(
                radius=_RADIUS,
                density=8.0,
                strength=1.0,
                padding=0.0,
                falloff=Falloff.CONSTANT,
                randomize_seed=False,
                seed=3,
                apply_to=ApplyTo.ALL,
            )
        )
        self._controller.add_asset(self._asset_path, up_axis=UpAxis.Z)
        self._picker = _FakePicker()
        self._picker.hit = SurfaceHit(_MESH_PATH, _HIT_POSITION)
        self._controller.set_picker_factory(lambda _viewport_api, _cache: self._picker)
        self._committed: list[tuple[int, bool]] = []
        self._committed_sub = self._controller.subscribe_stroke_committed(
            lambda count, erase: self._committed.append((count, erase))
        )

        self._viewport_api = mock.MagicMock(name="viewport_api")
        self._viewport_api.stage = self._stage
        self._viewport_api.usd_context_name = _CONTEXT_NAME
        self._viewport_api.updates_enabled = True
        self._viewport_api.transform = Gf.Matrix4d(1.0)
        self._viewport_api.ndc_to_world = Gf.Matrix4d(1.0)
        self._viewport_api.world_to_ndc = Gf.Matrix4d(1.0)
        self._selection_layer = mock.MagicMock(name="selection_layer")
        self._selection_layer.visible = True
        self._layer_provider = mock.MagicMock(name="layer_provider")
        self._layer_provider.find_viewport_layer.return_value = self._selection_layer
        self._active = True

        self._window = ui.Window("ScatterBrushToolTest", width=400, height=300)
        with self._window.frame:
            self._scene_view = sc.SceneView()
            with self._scene_view.scene:
                self._tool = ScatterBrushTool(
                    self._viewport_api, self._layer_provider, lambda: self._active, _CONTEXT_NAME
                )

    async def tearDown(self):
        self._tool.destroy()
        self._tool = None
        self._committed_sub = None
        self._scene_view = None
        self._window.destroy()
        self._window = None
        _restore_process_controller(self._previous_controller)
        omni.kit.undo.clear_stack()
        await self._usd_context.close_stage_async()
        self._stack.close()
        self._temp_dir.cleanup()

    def _hover(self, ndc: tuple[float, float] = (0.0, 0.0)) -> None:
        """Turn the brush on and run one hover poll at ``ndc`` so the fake picker delivers its hit."""
        self._controller.set_mode(ScatterMode.PAINT)
        self._ndc = ndc
        self._tool._on_update(None)

    def _begin_stroke(self) -> None:
        """Hover, then press the left mouse button as the drag gesture would report it."""
        self._hover()
        self._tool._on_drag_began(self._tool._drag)

    def _placements(self) -> list[Usd.Prim]:
        container = self._stage.GetPrimAtPath(_CONTAINER_PATH)
        return list(container.GetChildren()) if container else []

    async def test_set_mode_paint_shows_screen_and_starts_hover_poll(self):
        # Arrange
        # (setUp)

        # Act
        self._controller.set_mode(ScatterMode.PAINT)

        # Assert
        self.assertTrue(self._tool.enabled)
        self.assertTrue(self._tool._screen.visible)
        self.assertIsNotNone(self._tool._update_subscription)
        self.assertIs(self._tool._picker, self._picker)

    async def test_set_mode_off_hides_screen_and_stops_hover_poll(self):
        # Arrange
        self._hover()

        # Act
        self._controller.set_mode(ScatterMode.OFF)

        # Assert
        self.assertFalse(self._tool.enabled)
        self.assertFalse(self._tool._screen.visible)
        self.assertIsNone(self._tool._update_subscription)
        self.assertFalse(self._tool._cursor.visible)
        self.assertEqual(self._picker.cancel_calls, 1)

    async def test_on_update_with_hit_issues_one_pick_and_places_cursor(self):
        # Arrange
        self._controller.set_mode(ScatterMode.PAINT)
        self._ndc = (0.25, -0.5)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._picker.picks, [(0.25, -0.5)])
        self.assertTrue(self._tool._cursor.visible)
        # omni.ui.scene stores single-precision matrices, hence the loose tolerances.
        transform = self._tool._cursor.transform
        self.assertTrue(Gf.IsClose(Gf.Vec3d(transform[12], transform[13], transform[14]), _HIT_POSITION, 1e-4))
        self.assertAlmostEqual(math.hypot(transform[0], transform[1], transform[2]), _RADIUS, places=3)
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._PAINT_COLOR)
        self.assertIsNotNone(self._tool._last_hover)

    async def test_on_update_with_unchanged_mouse_and_camera_does_not_pick_again(self):
        # Arrange
        self._hover()

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(len(self._picker.picks), 1)

    async def test_on_update_while_pick_in_flight_lets_picker_refuse_second_pick(self):
        # Arrange
        self._picker.deliver_immediately = False
        self._hover((0.1, 0.1))
        self._ndc = (0.2, 0.2)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._picker.pick_attempts, [(0.1, 0.1), (0.2, 0.2)])
        self.assertEqual(self._picker.picks, [(0.1, 0.1)])

    async def test_on_update_with_stale_in_flight_pick_lets_picker_drop_it_and_pick_again(self):
        # Arrange
        self._picker.deliver_immediately = False
        self._hover((0.1, 0.1))
        self._picker.stale_pending = True
        self._ndc = (0.2, 0.2)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._picker.picks, [(0.1, 0.1), (0.2, 0.2)])
        self.assertEqual(self._picker.cancel_calls, 1)
        self.assertEqual(self._tool._last_pick_ndc, (0.2, 0.2))

    async def test_on_update_after_deferred_delivery_picks_latest_ndc(self):
        # Arrange
        self._picker.deliver_immediately = False
        self._hover((0.1, 0.1))
        self._ndc = (0.2, 0.2)
        self._tool._on_update(None)
        self._picker.deliver(self._picker.hit)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._picker.picks, [(0.1, 0.1), (0.2, 0.2)])

    async def test_on_update_with_mouse_outside_viewport_hides_cursor_without_picking(self):
        # Arrange
        self._hover()
        self._ndc = (1.5, 0.0)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertFalse(self._tool._cursor.visible)
        self.assertEqual(len(self._picker.picks), 1)

    async def test_on_update_with_mouse_outside_viewport_clears_hover(self):
        # Arrange
        self._hover()
        self._ndc = (1.5, 0.0)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertIsNone(self._tool._last_hover)

    async def test_on_update_on_inactive_viewport_hides_cursor_and_clears_hover(self):
        # Arrange
        self._hover()
        self._active = False

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertFalse(self._tool._cursor.visible)
        self.assertIsNone(self._tool._last_hover)

    async def test_on_hit_with_miss_hides_cursor_and_clears_hover(self):
        # Arrange
        self._hover()

        # Act
        self._tool._on_hit(None)

        # Assert
        self.assertFalse(self._tool._cursor.visible)
        self.assertIsNone(self._tool._last_hover)

    async def test_on_hit_when_target_resolution_raises_logs_error_and_hides_cursor(self):
        # Arrange
        self._hover()
        enter = self._stack.enter_context
        enter(mock.patch.object(scatter_brush_module, "resolve_target", side_effect=RuntimeError("boom")))
        log_error = enter(mock.patch.object(carb, "log_error"))

        # Act
        self._tool._on_hit(self._picker.hit)

        # Assert
        self.assertFalse(self._tool._cursor.visible)
        self.assertIsNone(self._tool._last_hover)
        log_error.assert_called_once()
        self.assertIn("boom", log_error.call_args.args[0])

    async def test_stage_for_with_open_context_returns_its_stage(self):
        # Arrange
        # (setUp)

        # Act
        stage = scatter_brush_module._stage_for(_CONTEXT_NAME)

        # Assert
        self.assertEqual(stage, self._stage)

    async def test_stage_for_with_unknown_context_returns_none(self):
        # Arrange
        context_name = f"{_CONTEXT_NAME}_missing"

        # Act
        stage = scatter_brush_module._stage_for(context_name)

        # Assert
        self.assertIsNone(stage)

    async def test_on_hit_in_erase_mode_colours_cursor_red(self):
        # Arrange
        self._controller.set_mode(ScatterMode.ERASE)
        self._ndc = (0.0, 0.0)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._ERASE_COLOR)

    async def test_on_update_when_shift_is_pressed_with_the_mouse_at_rest_colours_cursor_red(self):
        # Arrange
        self._hover()
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._ERASE_COLOR)

    async def test_on_update_when_shift_is_released_with_the_mouse_at_rest_colours_cursor_green_again(self):
        # Arrange
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}
        self._hover()
        self._keys_down = set()

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._PAINT_COLOR)

    async def test_on_hit_apply_to_selected_without_selection_shows_grey_cursor_and_no_hover(self):
        # Arrange
        self._controller.update_settings(apply_to=ApplyTo.SELECTED)
        self._controller.set_mode(ScatterMode.PAINT)
        self._ndc = (0.0, 0.0)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertTrue(self._tool._cursor.visible)
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._NOT_APPLICABLE_COLOR)
        self.assertIsNone(self._tool._last_hover)

    def _paint_on_selection(self) -> None:
        """Select the capture instance and switch the brush to painting on the selection only."""
        self._usd_context.get_selection().set_selected_prim_paths([str(_INSTANCE_0)], False)
        self._controller.update_settings(apply_to=ApplyTo.SELECTED)

    async def test_on_update_apply_to_selected_paints_the_selected_mesh_through_whatever_is_in_front(self):
        # Arrange - the picker reports a prop standing in front of the ground
        self._paint_on_selection()
        self._picker.hit = SurfaceHit(Sdf.Path("/World/FoliageCard"), Gf.Vec3d(0.0, 0.0, -0.5))

        # Act
        self._hover((0.25, -0.5))

        # Assert
        self.assertEqual(self._picker.picks, [])
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._PAINT_COLOR)
        hit, target, _sample = self._tool._last_hover
        self.assertEqual(target.mesh_path, _MESH_PATH)
        self.assertTrue(Gf.IsClose(hit.world_position, Gf.Vec3d(0.25, -0.5, 0.0), 1e-6))

    async def test_on_update_apply_to_selected_drops_a_pick_still_in_flight(self):
        # Arrange - a slower renderer pick must not move the cursor back onto what stands in front
        self._picker.deliver_immediately = False
        self._hover((0.1, 0.1))
        self._paint_on_selection()
        self._ndc = (0.2, 0.2)

        # Act
        self._tool._on_update(None)

        # Assert
        self.assertFalse(self._picker.in_flight)
        self.assertEqual(self._picker.cancel_calls, 1)

    async def test_on_update_apply_to_selected_off_the_selected_mesh_falls_back_to_the_picker(self):
        # Arrange - the camera ray passes beside the quad, so the picker decides what the grey cursor sits on
        self._paint_on_selection()
        self._viewport_api.ndc_to_world = Gf.Matrix4d().SetScale(Gf.Vec3d(1000.0, 1000.0, 1.0))
        self._picker.hit = SurfaceHit(Sdf.Path("/World/FoliageCard"), Gf.Vec3d(900.0, 900.0, 0.0))

        # Act
        self._hover((0.9, 0.9))

        # Assert
        self.assertEqual(self._picker.picks, [(0.9, 0.9)])
        self.assertIsNone(self._tool._last_hover)

    async def test_on_hit_turns_the_cursor_normal_toward_the_camera(self):
        # Arrange - seen from below, the quad's upward normal points away from the camera
        self._viewport_api.transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, -100.0))

        # Act
        self._hover()

        # Assert
        self.assertLess(self._tool._cursor.transform[10], 0.0)

    async def test_drag_began_apply_to_selected_without_a_selection_posts_status_and_starts_nothing(self):
        # Arrange
        self._controller.update_settings(apply_to=ApplyTo.SELECTED)
        status = mock.Mock()
        self._status_sub = self._controller.subscribe_status_message(status)
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNone(self._tool._session)
        self.assertEqual(status.call_args.args, (scatter_brush_module._NO_SELECTION_MESSAGE, True))

    async def test_drag_began_hands_the_stroke_the_camera_position(self):
        # Arrange
        self._viewport_api.transform = Gf.Matrix4d().SetTranslate(Gf.Vec3d(1.0, 2.0, 300.0))
        session_class = self._stack.enter_context(
            mock.patch.object(scatter_brush_module, "StrokeSession", wraps=scatter_brush_module.StrokeSession)
        )
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertEqual(session_class.call_args.kwargs["view_point"], Gf.Vec3d(1.0, 2.0, 300.0))

    async def test_drag_began_with_hover_starts_paint_session(self):
        # Arrange
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNotNone(self._tool._session)
        self.assertTrue(self._tool._session.active)
        self.assertFalse(self._tool._session.erase)

    async def test_drag_began_with_shift_held_starts_erase_session(self):
        # Arrange
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNotNone(self._tool._session)
        self.assertTrue(self._tool._session.erase)
        self.assertTrue(self._tool._session.active)

    async def test_drag_began_with_alt_held_starts_nothing(self):
        # Arrange
        self._keys_down = {carb.input.KeyboardInput.LEFT_ALT}
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNone(self._tool._session)
        self.assertTrue(self._selection_layer.visible)

    async def test_drag_began_on_inactive_viewport_starts_nothing(self):
        # Arrange
        self._hover()
        self._active = False

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNone(self._tool._session)

    async def test_drag_began_after_mouse_left_viewport_does_not_stamp_at_stale_hover(self):
        # Arrange
        self._hover()
        self._ndc = (1.5, 0.0)
        self._tool._on_update(None)

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNotNone(self._tool._session)
        self.assertFalse(self._tool._session.active)
        self.assertEqual(self._placements(), [])

    async def test_drag_began_after_viewport_reactivated_does_not_stamp_at_stale_hover(self):
        # Arrange
        self._hover()
        self._active = False
        self._tool._on_update(None)
        self._active = True

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNotNone(self._tool._session)
        self.assertFalse(self._tool._session.active)
        self.assertEqual(self._placements(), [])

    async def test_drag_began_without_enabled_assets_posts_status_and_starts_nothing(self):
        # Arrange
        self._controller.remove_asset(self._asset_path)
        statuses = []
        self._status_sub = self._controller.subscribe_status_message(
            lambda text, is_error: statuses.append((text, is_error))
        )
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertIsNone(self._tool._session)
        self.assertEqual(statuses, [("Add at least one ingested asset to the scatter brush", True)])

    async def test_drag_began_hides_selection_layer(self):
        # Arrange
        self._hover()

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertFalse(self._selection_layer.visible)
        self.assertIsNotNone(self._tool._layer_guard)

    async def test_drag_ended_restores_selection_layer(self):
        # Arrange
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertTrue(self._selection_layer.visible)
        self.assertIsNone(self._tool._layer_guard)

    async def test_drag_ended_commits_placements_and_fires_stroke_committed(self):
        # Arrange
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        placements = self._placements()
        self.assertGreater(len(placements), 0)
        self.assertEqual(self._committed, [(len(placements), False)])
        self.assertIsNone(self._tool._session)
        self.assertTrue(all(prim.GetPath().name.startswith("s_") for prim in placements))

    async def test_drag_ended_makes_the_painted_stroke_the_preview(self):
        # Arrange
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert - live edits now go to exactly what this stroke painted
        preview = self._controller.preview_scope(_CONTEXT_NAME)
        self.assertIsNotNone(preview)
        self.assertEqual(preview.placement_count, len(self._placements()))

    async def test_drag_ended_erase_does_not_start_a_preview(self):
        # Arrange
        self._begin_stroke()
        self._tool._on_drag_ended(self._tool._drag)
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert - the erase is now on top of the undo stack, so the stroke before it is no longer the preview
        self.assertIsNone(self._controller.preview_scope(_CONTEXT_NAME))

    async def test_drag_ended_without_session_does_nothing(self):
        # Arrange
        self._hover()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertEqual(self._committed, [])
        self.assertEqual(self._placements(), [])

    async def test_drag_ended_on_replicated_prototype_posts_one_warning(self):
        # Arrange
        _author_instance(self._stage, 1)
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self._post_notification.assert_called_once()
        text = self._post_notification.call_args.args[0]
        self.assertIn(f"mesh_{_HASH}", text)
        self.assertIn("2 instances", text)
        self.assertEqual(self._post_notification.call_args.kwargs["status"], NotificationStatus.WARNING)

    async def test_drag_ended_twice_on_replicated_prototype_warns_only_once(self):
        # Arrange
        _author_instance(self._stage, 1)
        self._begin_stroke()
        self._tool._on_drag_ended(self._tool._drag)
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertEqual(self._post_notification.call_count, 1)

    async def test_drag_ended_on_single_instance_prototype_posts_no_warning(self):
        # Arrange
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self._post_notification.assert_not_called()

    async def test_drag_ended_erase_on_replicated_prototype_posts_no_warning(self):
        # Arrange
        self._begin_stroke()
        self._tool._on_drag_ended(self._tool._drag)
        _author_instance(self._stage, 1)
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self._post_notification.assert_not_called()
        erased_count, erase = self._committed[-1]
        self.assertTrue(erase)
        self.assertGreater(erased_count, 0)

    async def test_drag_ended_with_nothing_placed_on_replicated_prototype_posts_no_warning(self):
        # Arrange
        _author_instance(self._stage, 1)
        self._controller.update_settings(strength=0.0)
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self._post_notification.assert_not_called()
        self.assertEqual(self._committed, [(0, False)])

    async def test_on_wheel_with_b_held_scales_radius_and_consumes_event(self):
        # Arrange
        self._controller.set_mode(ScatterMode.PAINT)
        self._keys_down = {carb.input.KeyboardInput.B}

        # Act
        consumed = self._tool._on_wheel(self._viewport_api, 0.0, 1.0, 0)

        # Assert
        self.assertTrue(consumed)
        self.assertAlmostEqual(self._controller.settings.radius, _RADIUS * 1.1)

    async def test_on_wheel_down_with_b_held_shrinks_radius(self):
        # Arrange
        self._controller.set_mode(ScatterMode.PAINT)
        self._keys_down = {carb.input.KeyboardInput.B}

        # Act
        consumed = self._tool._on_wheel(self._viewport_api, 0.0, -1.0, 0)

        # Assert
        self.assertTrue(consumed)
        self.assertAlmostEqual(self._controller.settings.radius, _RADIUS / 1.1)

    async def test_on_wheel_without_b_held_is_not_consumed(self):
        # Arrange
        self._controller.set_mode(ScatterMode.PAINT)

        # Act
        consumed = self._tool._on_wheel(self._viewport_api, 0.0, 1.0, 0)

        # Assert
        self.assertFalse(consumed)
        self.assertEqual(self._controller.settings.radius, _RADIUS)

    async def test_on_wheel_while_brush_off_is_not_consumed(self):
        # Arrange
        self._keys_down = {carb.input.KeyboardInput.B}

        # Act
        consumed = self._tool._on_wheel(self._viewport_api, 0.0, 1.0, 0)

        # Assert
        self.assertFalse(consumed)
        self.assertEqual(self._controller.settings.radius, _RADIUS)

    async def test_on_wheel_from_other_viewport_is_not_consumed(self):
        # Arrange
        self._controller.set_mode(ScatterMode.PAINT)
        self._keys_down = {carb.input.KeyboardInput.B}

        # Act
        consumed = self._tool._on_wheel(mock.MagicMock(name="other_viewport_api"), 0.0, 1.0, 0)

        # Assert
        self.assertFalse(consumed)

    async def test_toggle_hotkey_switches_controller_to_paint(self):
        # Arrange
        # (setUp)

        # Act
        self._tool._on_toggle_hotkey()

        # Assert
        self.assertEqual(self._controller.mode, ScatterMode.PAINT)

    async def test_visible_false_hides_cursor_and_ends_stroke(self):
        # Arrange
        self._begin_stroke()
        session = self._tool._session

        # Act
        self._tool.visible = False

        # Assert
        self.assertFalse(self._tool.visible)
        self.assertFalse(self._tool.enabled)
        self.assertFalse(self._tool._screen.visible)
        self.assertFalse(self._tool._cursor.visible)
        self.assertIsNone(self._tool._session)
        self.assertFalse(session.active)
        self.assertTrue(self._selection_layer.visible)

    async def test_visible_true_after_hidden_re_enables_brush_while_mode_is_on(self):
        # Arrange
        self._hover()
        self._tool.visible = False

        # Act
        self._tool.visible = True

        # Assert
        self.assertTrue(self._tool.enabled)
        self.assertTrue(self._tool._screen.visible)

    async def test_destroy_mid_stroke_ends_stroke_and_releases_everything(self):
        # Arrange
        self._begin_stroke()
        session = self._tool._session

        # Act
        self._tool.destroy()

        # Assert
        self.assertFalse(session.active)
        self.assertIsNone(self._tool._session)
        self.assertTrue(self._selection_layer.visible)
        self.assertEqual(self._picker.cancel_calls, 1)
        self.assertIsNone(self._tool._cursor)
        self.assertIsNone(self._tool._screen)
        self.assertIsNone(self._tool._update_subscription)
        self.assertEqual(self._committed, [(len(self._placements()), False)])

    async def test_scatter_brush_factory_builds_tool_from_desc(self):
        # Arrange
        desc = {
            "viewport_api": self._viewport_api,
            "layer_provider": self._layer_provider,
            "is_active_fn": lambda: True,
            "usd_context_name": _CONTEXT_NAME,
        }

        # Act
        with self._window.frame, self._scene_view.scene:
            tool = scatter_brush_factory(desc)

        # Assert
        self.addCleanup(tool.destroy)
        self.assertIsInstance(tool, ScatterBrushTool)
        self.assertEqual(tool.name, "Scatter Brush")
        self.assertEqual(tool.categories, ["tools"])
        self.assertTrue(tool.visible)

    async def test_begin_stroke_shows_counter_with_rendered_prim_total(self):
        # Arrange
        # (setUp authors a single instance, so rendered prims equal the placements authored)

        # Act
        self._begin_stroke()

        # Assert
        self.assertTrue(self._tool._counter_label.visible)
        self.assertEqual(self._tool._counter_label.text, f"{len(self._placements()):,} prims")

    async def test_begin_stroke_on_replicated_prototype_shows_placed_and_rendered_totals(self):
        # Arrange
        _author_instance(self._stage, 1)
        _author_instance(self._stage, 2)

        # Act
        self._begin_stroke()

        # Assert
        placed = len(self._placements())
        self.assertEqual(self._tool._counter_label.text, f"{placed:,} placed  ->  {placed * 3:,} prims")

    async def test_begin_stroke_within_prim_budget_keeps_counter_status_ok(self):
        # Arrange
        self._controller.update_settings(prim_budget=10000)

        # Act
        self._begin_stroke()

        # Assert
        self.assertEqual(self._tool._counter_status, PrimBudgetStatus.OK)

    async def test_begin_stroke_past_alert_multiple_of_budget_sets_counter_status_alert(self):
        # Arrange
        self._controller.update_settings(prim_budget=1)

        # Act
        self._begin_stroke()

        # Assert
        self.assertEqual(self._tool._counter_status, PrimBudgetStatus.ALERT)

    async def test_counter_while_stroke_trails_the_cursor_says_it_is_catching_up(self):
        # Arrange
        self._begin_stroke()

        with mock.patch.object(stroke_module, "_clock", side_effect=_clock_out_of_time_after(1)):
            # Act
            self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))

        # Assert
        self.assertTrue(self._tool._session.pending)
        self.assertTrue(self._tool._counter_label.text.endswith(scatter_brush_module._CATCHING_UP_SUFFIX))

    async def test_on_update_while_stroke_trails_the_cursor_catches_it_up_without_picking(self):
        # Arrange
        self._begin_stroke()
        with mock.patch.object(stroke_module, "_clock", side_effect=_clock_out_of_time_after(1)):
            self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))
        self.assertTrue(self._tool._session.pending)
        self._tool._force_pick = False
        picks_before = len(self._picker.picks)

        with mock.patch.object(stroke_module, "_clock", return_value=0.0):
            # Act
            self._tool._on_update(None)

        # Assert
        self.assertFalse(self._tool._session.pending)
        self.assertEqual(len(self._picker.picks), picks_before)
        self.assertFalse(self._tool._counter_label.text.endswith(scatter_brush_module._CATCHING_UP_SUFFIX))

    async def test_on_hit_during_stroke_draws_cursor_at_the_radius_the_stroke_stamps(self):
        # Arrange
        self._begin_stroke()
        self._controller.update_settings(radius=_RADIUS * 2.0)

        # Act
        self._tool._on_hit(SurfaceHit(_MESH_PATH, _HIT_POSITION))

        # Assert
        transform = self._tool._cursor.transform
        self.assertAlmostEqual(math.hypot(transform[0], transform[1], transform[2]), _RADIUS, places=3)

    async def test_drag_ended_hides_counter(self):
        # Arrange
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertFalse(self._tool._counter_label.visible)

    async def test_begin_erase_stroke_leaves_counter_hidden(self):
        # Arrange
        self._controller.set_mode(ScatterMode.ERASE)
        self._ndc = (0.0, 0.0)
        self._tool._on_update(None)

        # Act
        self._tool._on_drag_began(self._tool._drag)

        # Assert
        self.assertFalse(self._tool._counter_label.visible)

    def _collect_statuses(self) -> list[tuple[str, bool]]:
        """Return a list that receives every status message the controller posts from now on."""
        statuses: list[tuple[str, bool]] = []
        self._status_sub = self._controller.subscribe_status_message(
            lambda text, is_error: statuses.append((text, is_error))
        )
        return statuses

    async def test_begin_stroke_at_the_prim_limit_trims_the_stamp_and_says_so_on_the_counter(self):
        # Arrange - a budget of 1 limits a stroke to 5 rendered prims, fewer than the 8 one stamp draws
        self._controller.update_settings(prim_budget=1)

        # Act
        self._begin_stroke()

        # Assert
        self.assertEqual(len(self._placements()), 5)
        self.assertTrue(self._tool._counter_label.text.endswith(scatter_brush_module._LIMIT_REACHED_SUFFIX))
        self.assertEqual(self._tool._counter_status, PrimBudgetStatus.ALERT)

    async def test_on_hit_after_the_prim_limit_was_reached_greys_the_cursor_and_paints_nothing_more(self):
        # Arrange
        self._controller.update_settings(prim_budget=1)
        self._begin_stroke()

        # Act
        self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))

        # Assert
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._NOT_APPLICABLE_COLOR)
        self.assertEqual(len(self._placements()), 5)

    async def test_drag_ended_at_the_prim_limit_posts_status_and_a_notification(self):
        # Arrange
        self._controller.update_settings(prim_budget=1)
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert - what the stroke did paint, then why it stopped
        expected = scatter_brush_module._LIMIT_REACHED_MESSAGE.format(limit=5)
        self.assertIn((f"Painted 5 placements, rendering 5 prims. {expected}", True), statuses)
        self._post_notification.assert_called_once()
        self.assertIn(expected, self._post_notification.call_args.args[0])
        self.assertEqual(self._committed, [(5, False)])

    async def test_drag_ended_twice_at_the_prim_limit_notifies_only_once(self):
        # Arrange
        self._controller.update_settings(prim_budget=1)
        self._begin_stroke()
        self._tool._on_drag_ended(self._tool._drag)
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertTrue(statuses[-1][0].endswith(scatter_brush_module._LIMIT_REACHED_MESSAGE.format(limit=5)))
        self.assertTrue(statuses[-1][1])
        self.assertEqual(self._post_notification.call_count, 1)

    async def test_begin_stroke_on_a_mesh_with_more_instances_than_the_prim_limit_shows_an_empty_counter(self):
        # Arrange - six instances, so one placement alone renders more than the limit of 5
        for index in range(1, 6):
            _author_instance(self._stage, index)
        self._controller.update_settings(prim_budget=1)

        # Act
        self._begin_stroke()

        # Assert
        self.assertEqual(self._placements(), [])
        self.assertTrue(self._tool._counter_label.visible)
        self.assertEqual(self._tool._counter_label.text, f"0 prims  {scatter_brush_module._LIMIT_BLOCKED_SUFFIX}")
        self.assertEqual(self._tool._counter_status, PrimBudgetStatus.ALERT)

    async def test_on_hit_over_a_mesh_with_too_many_instances_greys_the_cursor_and_keeps_the_stroke_open(self):
        # Arrange
        for index in range(1, 6):
            _author_instance(self._stage, index)
        self._controller.update_settings(prim_budget=1)
        self._begin_stroke()

        # Act
        self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))

        # Assert - only this mesh is out of reach, so the stroke itself is not at its limit
        self.assertEqual(self._tool._cursor_color, scatter_brush_module._NOT_APPLICABLE_COLOR)
        self.assertFalse(self._tool._session.limit_reached)

    async def test_drag_ended_on_a_mesh_with_more_instances_than_the_prim_limit_says_nothing_was_painted(self):
        # Arrange
        for index in range(1, 6):
            _author_instance(self._stage, index)
        self._controller.update_settings(prim_budget=1)
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        expected = scatter_brush_module._LIMIT_BLOCKED_MESSAGE.format(name=f"mesh_{_HASH}", count=6, limit=5)
        self.assertIn((expected, True), statuses)
        self.assertEqual(self._committed, [(0, False)])

    async def test_drag_ended_before_the_stroke_caught_up_reports_the_path_left_unpainted(self):
        # Arrange
        statuses = self._collect_statuses()
        self._begin_stroke()
        with mock.patch.object(stroke_module, "_clock", side_effect=_clock_out_of_time_after(1)):
            self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))
        self.assertTrue(self._tool._session.pending)
        # The end reads the clock once for its deadline; every later read finds its whole budget spent.
        out_of_time = itertools.chain([0.0], itertools.repeat(stroke_module._END_DRAIN_BUDGET))

        with mock.patch.object(stroke_module, "_clock", side_effect=out_of_time):
            # Act
            self._tool._on_drag_ended(self._tool._drag)

        # Assert
        text, is_error = statuses[-1]
        self.assertTrue(text.endswith(scatter_brush_module._CUT_SHORT_MESSAGE))
        self.assertTrue(is_error)
        self._post_notification.assert_called_once()
        self.assertIsNone(self._tool._session)

    async def test_drag_ended_when_catching_up_fails_reports_the_error_and_keeps_what_was_painted(self):
        # Arrange - the stroke is behind the cursor, and the stamp the end catches up with fails
        statuses = self._collect_statuses()
        self._begin_stroke()
        with mock.patch.object(stroke_module, "_clock", side_effect=_clock_out_of_time_after(1)):
            self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))
        self.assertTrue(self._tool._session.pending)
        painted = len(self._placements())

        with (
            mock.patch.object(stroke_module, "generate_stamp", side_effect=RuntimeError("stamp failed")),
            mock.patch.object(stroke_module.carb, "log_error"),
        ):
            # Act
            self._tool._on_drag_ended(self._tool._drag)

        # Assert
        text, is_error = statuses[-1]
        self.assertTrue(text.endswith(scatter_brush_module._FAILED_MESSAGE))
        self.assertTrue(is_error)
        self.assertEqual(self._committed, [(painted, False)])

    async def test_drag_ended_after_the_stroke_caught_up_reports_nothing(self):
        # Arrange
        statuses = self._collect_statuses()
        self._begin_stroke()
        self._tool._on_hit(SurfaceHit(_MESH_PATH, _FAR_POSITION))

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertNotIn(scatter_brush_module._CUT_SHORT_MESSAGE, [text for text, _is_error in statuses])
        self._post_notification.assert_not_called()

    async def test_drag_ended_posts_what_the_stroke_painted(self):
        # Arrange
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert - one instance of the mesh, so every placement renders once
        count = len(self._placements())
        self.assertGreater(count, 1)
        self.assertEqual(statuses[-1], (f"Painted {count} placements, rendering {count} prims", False))

    async def test_drag_ended_keeps_the_up_axis_warning_the_stroke_started_with_in_front_of_what_it_painted(self):
        # Arrange - the asset's file says Y-up while the palette says Z-up, which a stroke reports as it starts
        statuses = self._collect_statuses()
        with mock.patch.object(controller_module, "read_authored_asset_up_axis", return_value=UpAxis.Y):
            self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert - the summary would otherwise take the warning off the status line as soon as the mouse is released
        count = len(self._placements())
        text, is_error = statuses[-1]
        self.assertIn("is authored Y-up but the brush is set to Z-up", text)
        self.assertTrue(text.endswith(f". Painted {count} placements, rendering {count} prims"))
        self.assertFalse(is_error)

    async def test_drag_ended_on_a_replicated_prototype_posts_the_prims_every_instance_renders(self):
        # Arrange
        _author_instance(self._stage, 1)
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        count = len(self._placements())
        self.assertEqual(statuses[-1], (f"Painted {count} placements, rendering {count * 2} prims", False))

    async def test_drag_ended_erase_posts_how_many_placements_it_erased(self):
        # Arrange
        self._begin_stroke()
        self._tool._on_drag_ended(self._tool._drag)
        self._keys_down = {carb.input.KeyboardInput.LEFT_SHIFT}
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        erased_count, _erase = self._committed[-1]
        self.assertGreater(erased_count, 1)
        self.assertEqual(statuses[-1], (f"Erased {erased_count} placements", False))

    async def test_drag_ended_with_nothing_placed_posts_what_can_keep_a_stroke_from_painting(self):
        # Arrange
        self._controller.update_settings(strength=0.0)
        statuses = self._collect_statuses()
        self._begin_stroke()

        # Act
        self._tool._on_drag_ended(self._tool._drag)

        # Assert
        self.assertEqual(statuses[-1], (scatter_brush_module._NOTHING_PAINTED_MESSAGE, False))


class TestBrushCursorMatrix(omni.kit.test.AsyncTestCase):
    async def test_brush_cursor_matrix_maps_local_x_to_position_plus_tangent_radius(self):
        # Arrange
        position = Gf.Vec3d(10.0, -4.0, 2.5)
        normal = np.array([0.0, 0.0, 2.0])
        tangent, _ = tangent_basis(normal)

        # Act
        matrix = brush_cursor_matrix(position, normal, _RADIUS)

        # Assert
        expected = position + Gf.Vec3d(*(tangent * _RADIUS))
        self.assertTrue(Gf.IsClose(matrix.Transform(Gf.Vec3d(1.0, 0.0, 0.0)), expected, 1e-9))

    async def test_brush_cursor_matrix_maps_local_z_to_position_plus_unit_normal_radius(self):
        # Arrange
        position = Gf.Vec3d(10.0, -4.0, 2.5)
        normal = np.array([0.0, 0.0, 2.0])

        # Act
        matrix = brush_cursor_matrix(position, normal, _RADIUS)

        # Assert
        self.assertTrue(
            Gf.IsClose(matrix.Transform(Gf.Vec3d(0.0, 0.0, 1.0)), position + Gf.Vec3d(0.0, 0.0, _RADIUS), 1e-9)
        )

    async def test_brush_cursor_matrix_flattened_keeps_translation_in_last_row_for_scene_matrix(self):
        # Arrange
        position = Gf.Vec3d(10.0, -4.0, 2.5)
        normal = np.array([0.0, 1.0, 0.0])
        tangent, _ = tangent_basis(normal)
        flat = flatten_matrix(brush_cursor_matrix(position, normal, _RADIUS))

        # Act
        scene_matrix = sc.Matrix44(*flat)

        # Assert
        # omni.ui.scene stores single-precision matrices, hence the loose tolerances.
        self.assertTrue(np.allclose([scene_matrix[index] for index in range(16)], flat, atol=1e-4))
        translation = Gf.Vec3d(scene_matrix[12], scene_matrix[13], scene_matrix[14])
        local_x = Gf.Vec3d(scene_matrix[0], scene_matrix[1], scene_matrix[2])
        self.assertTrue(Gf.IsClose(translation, position, 1e-4))
        self.assertTrue(Gf.IsClose(local_x, Gf.Vec3d(*(tangent * _RADIUS)), 1e-4))
