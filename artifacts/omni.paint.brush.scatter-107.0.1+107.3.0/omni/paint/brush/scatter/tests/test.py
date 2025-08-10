from pathlib import Path
import carb.input
import carb.settings
import omni.appwindow
import omni.kit.app
import omni.kit.test
import omni.ui as ui
import omni.usd
import omni.paint.system.ui as pu
import omni.paint.system.core as pc
import omni.kit.ui_test as ui_test
from omni.kit.ui_test.vec2 import Vec2
from omni.ui.tests.test_base import OmniUiTest
from omni.paint.system.core import PAINT_MODES
from pxr import Gf, UsdGeom
from ..brush import get_paint_pointInstancer_path
from .img_compare import compare_img


def get_result_count(stage, asset_url):
    pointinstancer_path = get_paint_pointInstancer_path(stage, asset_url)
    prim = stage.GetPrimAtPath(pointinstancer_path)
    if prim:
        pointinstancer = UsdGeom.PointInstancer(prim)
        if pointinstancer:
            indices = pointinstancer.GetProtoIndicesAttr().Get()
            return len(indices)

    return -1


class ScatterBrushTests(OmniUiTest):
    async def setUp(self):
        await super().setUp()
        extension_root_folder = Path(
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        )
        self._golden_img = extension_root_folder.joinpath("test_data/golden_img")
        self._test_files = extension_root_folder.joinpath("test_data/usd")

        # Load the USD
        self._usd_context = omni.usd.get_context()
        test_file_path = self._test_files.joinpath("plane.usda").absolute()
        await self._usd_context.open_stage_async(str(test_file_path))

        await omni.kit.app.get_app().next_update_async()
        self._stage = self._usd_context.get_stage()

        # set brush
        omni.kit.commands.execute("SetPaintBrushCommand", brush_name="scatter")

        brush = pc.get_instance().get_brush_manager().current_brush
        asset0 = brush["assets"][0]
        self._asset_url = asset0["path"]

    async def tearDown(self):
        self._usd_context = None
        self._stage = None
        await super().tearDown()

    async def _emulate_keyboard(self, event_type: carb.input.KeyboardEventType, key: carb.input.KeyboardInput, modifier: carb.input.KeyboardInput=0):
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        input = carb.input.acquire_input_provider()
        input.buffer_keyboard_key_event(keyboard, event_type, key, modifier)

    """
    The first test case load
    """
    async def test_paint_basic(self):
        # show paint tool window
        ui.Workspace.show_window("Paint")

        await omni.kit.app.get_app().next_update_async()
        # dock the window
        await self.docked_test_window(
            window=ui.Workspace.get_window("Paint"),
            width=500,
            height=1000,
            block_devices=False)
        # wait async thumbnail loading
        await ui_test.human_delay(500)

        # remove asset
        await ui_test.emulate_mouse_move_and_click(Vec2(435, 260))
        await ui_test.human_delay(100)

        # add asset
        omni.usd.get_context().get_selection().set_selected_prim_paths(["/World/Sphere"], False)
        await ui_test.emulate_mouse_move_and_click(Vec2(300, 225))
        await ui_test.human_delay(100)

        # undo
        omni.kit.undo.undo()
        omni.kit.undo.undo()
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
        await ui_test.emulate_mouse_move(Vec2(0, 0))
        await ui_test.human_delay(100)

        # capture the golden image, set threshold=1000 to make it always pass the compare
        await self.finalize_test(threshold=1000, golden_img_dir=self._golden_img, golden_img_name="scatter.png")
        # the asset url path in brush is different in different environments
        # do image compare which filter out the asset path part
        # OM-91523: also filter out thumbnail area
        filter_rect = [30, 240, 400, 350]
        diff = compare_img("scatter.png", self._golden_img, filter_rect)
        print("...diff", diff)
        self.assertTrue(
            (diff is not None and diff < 200),
            msg=f"The image for test PaintBundelTests doesn't match the golden one. Difference of {diff}.",
        )

        paint_tool = pu.get_instance().get_paint_tool()
        paint_tool.set_paint_mode(PAINT_MODES.PAINT)
        await ui_test.human_delay(20)

        # hide ui
        settings = carb.settings.get_settings()
        settings.set_bool("/app/window/hideUi", True)

        # test stamp on viewport
        await ui_test.human_delay(20)
        await ui_test.emulate_mouse_move_and_click(Vec2(500, 500))
        await ui_test.human_delay(100)
        # check result
        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num > 0)

        # shift to toggle paint mode
        await self._emulate_keyboard(carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardInput.LEFT_SHIFT)
        await ui_test.human_delay(2)
        await ui_test.emulate_mouse_move_and_click(Vec2(500, 500))
        await ui_test.human_delay(10)
        await self._emulate_keyboard(carb.input.KeyboardEventType.KEY_RELEASE, carb.input.KeyboardInput.LEFT_SHIFT)
        await ui_test.human_delay(20)

        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num == 0)

        # stamp command
        omni.kit.commands.execute("DoPaintStamp", position=Gf.Vec3d(0), radius=100.0, target_meshes=["/World/Plane"])
        await ui_test.human_delay(3)

        # check result
        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num > 0)

        # erase
        omni.kit.commands.execute("DoPaintErase", position=Gf.Vec3d(0))
        await omni.kit.app.get_app().next_update_async()
        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num == 0)

        # flood paint
        omni.kit.commands.execute("DoPaintFlood", flood_mode=pc.PAINT_MODES.PAINT)
        await omni.kit.app.get_app().next_update_async()
        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num > 10)

        # flood erase
        omni.kit.commands.execute("DoPaintFlood", flood_mode=pc.PAINT_MODES.ERASE)
        await omni.kit.app.get_app().next_update_async()
        num = get_result_count(self._stage, self._asset_url)
        self.assertTrue(num == 0)
