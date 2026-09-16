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

import json
import tempfile
from pathlib import Path

import carb.settings
import numpy as np
import omni.kit.undo
import omni.ui as ui
import omni.usd
from carb.input import KeyboardInput
from lightspeed.common.constants import WindowNames
from lightspeed.trex.contexts.setup import Contexts as _TrexContexts
from lightspeed.trex.scatter.core import (
    Falloff,
    MeshSurfaceCache,
    PaddingIndex,
    ScatterAssetEntry,
    ScatterBrushSettings,
    ScatterMode,
    ScatterTarget,
    UpAxis,
    author_placements,
    destroy_scatter_brush_controller,
    generate_stamp,
    get_scatter_brush_controller,
)
from lightspeed.trex.scatter.core.constants import BRUSH_SETTINGS_PATH, CONTAINER_PREFIX, PRESETS_DIR_SETTING
from omni.kit import ui_test
from omni.kit.test import AsyncTestCase
from pxr import Gf, Sdf, UsdGeom

_WINDOW_TITLE = WindowNames.SCATTER.value
_SAVE_AS_PROMPT_TITLE = "Save Preset As"
_WINDOW_WIDTH = 520
_WINDOW_HEIGHT = 680
_PRESET_NAME = "E2E Grass"
_CONTEXT_NAME = _TrexContexts.STAGE_CRAFT.value
_HASH = "0123456789ABCDEF"
_PROTOTYPE_ROOT = f"/RootNode/meshes/mesh_{_HASH}"
_INSTANCE_ROOT = f"/RootNode/instances/inst_{_HASH}_0"
_MESH_PATH = f"{_INSTANCE_ROOT}/mesh"
_GLASS_CONTAINER = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}glass"
_SCATTER_ASSET = "C:/assets/blade.usda"
_QUAD_HALF_SIZE = 500.0


def _control_query(window_title: str, widget_type: str, identifier: str) -> str:
    """Build the ui_test query for a control of a window by its identifier."""
    return f"{window_title}//Frame/**/{widget_type}[*].identifier=='{identifier}'"


class TestScatterWindowE2E(AsyncTestCase):
    """Drives the docked Scatter window like a user and checks that the shared brush controller follows.

    Every end-to-end test of this window belongs in this class. The pane binds to the brush controller once, when the
    window first builds it, and is never rebuilt; ``setUpClass`` below relies on that by dropping the controller
    before any test has shown the window. A second class that showed the window first would leave this one's
    ``setUpClass`` destroying the controller the pane is bound to, and every test here would then drive a pane wired
    to a controller nothing else can see.
    """

    @classmethod
    def setUpClass(cls):
        # The pane binds to the controller when the window first builds it, and the controller reads the presets
        # directory once when it is created. Point the setting at a scratch directory and drop the controller here,
        # before any test shows the window, so the pane binds to a controller that stores presets in the scratch
        # directory and starts from default brush settings.
        settings = carb.settings.get_settings()
        cls._previous_presets_dir = settings.get(PRESETS_DIR_SETTING)
        cls._previous_brush_settings = settings.get(BRUSH_SETTINGS_PATH)
        cls._temp_presets_dir = tempfile.TemporaryDirectory()
        settings.set(PRESETS_DIR_SETTING, cls._temp_presets_dir.name)
        settings.destroy_item(BRUSH_SETTINGS_PATH)
        destroy_scatter_brush_controller()

    @classmethod
    def tearDownClass(cls):
        # The pane the extension built while these tests ran stays bound to the controller created for this class for
        # the rest of the process, and it is never rebuilt; destroying that controller would leave the window wired
        # to dead events. Only the settings are restored. The preset store tolerates the scratch directory going
        # away: it lists nothing and recreates the directory on the next save.
        settings = carb.settings.get_settings()
        settings.set(PRESETS_DIR_SETTING, cls._previous_presets_dir or "")
        if cls._previous_brush_settings is None:
            settings.destroy_item(BRUSH_SETTINGS_PATH)
        else:
            settings.set(BRUSH_SETTINGS_PATH, cls._previous_brush_settings)
        cls._temp_presets_dir.cleanup()

    async def setUp(self):
        self._settings = carb.settings.get_settings()
        self._painted_stage = False
        self._controller = get_scatter_brush_controller()
        self._controller.set_mode(ScatterMode.OFF)
        self._controller.replace_settings(ScatterBrushSettings())

        # Float the window at a size that fits the test app so emulated clicks land on the controls.
        ui.Workspace.show_window(_WINDOW_TITLE, True)
        window = ui.Workspace.get_window(_WINDOW_TITLE)
        self.assertIsNotNone(window, "The Scatter extension did not create its window at startup")
        window.undock()
        window.position_x = 0
        window.position_y = 0
        window.width = _WINDOW_WIDTH
        window.height = _WINDOW_HEIGHT
        window.focus()
        await ui_test.human_delay(10)

    async def tearDown(self):
        # A failed preset test can leave the modal name prompt open; cancel it so it cannot block later tests.
        prompt = ui.Workspace.get_window(_SAVE_AS_PROMPT_TITLE)
        if prompt is not None and prompt.visible:
            cancel_button = ui_test.find(_control_query(_SAVE_AS_PROMPT_TITLE, "Button", "scatter_preset_name_cancel"))
            if cancel_button is not None:
                await cancel_button.click()
        self._controller.delete_preset(_PRESET_NAME)
        self._controller.set_mode(ScatterMode.OFF)
        ui.Workspace.show_window(_WINDOW_TITLE, False)
        await ui_test.human_delay(2)
        if self._painted_stage:
            usd_context = omni.usd.get_context(_CONTEXT_NAME)
            usd_context.get_selection().clear_selected_prim_paths()
            await usd_context.new_stage_async()

    async def _find_control(self, widget_type: str, identifier: str):
        """Find a pane control by identifier and scroll it into view so emulated input lands on it."""
        control = ui_test.find(_control_query(_WINDOW_TITLE, widget_type, identifier))
        self.assertIsNotNone(control, f"{identifier} is missing from the Scatter window")
        control.widget.scroll_here_y(0.5)
        await ui_test.human_delay(5)
        return control

    async def test_radius_field_input_updates_controller_and_persists_setting(self):
        # The user types a new brush radius into the BRUSH section and confirms it with Enter.
        radius_field = await self._find_control("FloatBoundedDrag", "scatter_radius")
        self.assertNotEqual(self._controller.settings.radius, 75.0)
        await radius_field.input("75", end_key=KeyboardInput.ENTER)
        await ui_test.human_delay(10)

        # The controller adopted the value, the field shows it and the persistent brush settings carry it.
        self.assertEqual(self._controller.settings.radius, 75.0)
        self.assertEqual(radius_field.widget.model.get_value_as_float(), 75.0)
        persisted = json.loads(self._settings.get(BRUSH_SETTINGS_PATH))
        self.assertEqual(persisted["radius"], 75.0)

    async def test_paint_button_toggles_paint_mode_on_and_off(self):
        # The user arms the brush with the Paint button in the window header.
        paint_button = await self._find_control("ToolButton", "scatter_mode_paint")
        await paint_button.click()
        await ui_test.human_delay(5)
        self.assertEqual(self._controller.mode, ScatterMode.PAINT)
        self.assertTrue(paint_button.widget.model.get_value_as_bool())

        # Clicking the same button again disarms the brush.
        await paint_button.click()
        await ui_test.human_delay(5)
        self.assertEqual(self._controller.mode, ScatterMode.OFF)
        self.assertFalse(paint_button.widget.model.get_value_as_bool())

    async def test_conform_checkbox_click_flips_conform_to_surface(self):
        # The user toggles "Conform to Surface" in the PLACEMENT section.
        conform_checkbox = await self._find_control("CheckBox", "scatter_conform")
        conform_before = self._controller.settings.conform_to_surface
        self.assertEqual(conform_checkbox.widget.model.get_value_as_bool(), conform_before)
        await conform_checkbox.click()
        await ui_test.human_delay(5)

        # The brush follows the checkbox.
        self.assertEqual(self._controller.settings.conform_to_surface, not conform_before)
        self.assertEqual(conform_checkbox.widget.model.get_value_as_bool(), not conform_before)

    async def test_assign_textures_checkbox_click_flips_assign_textures(self):
        # The user turns off "Assign Textures" in the ASSETS section to paint an asset without its maps.
        assign_checkbox = await self._find_control("CheckBox", "scatter_assign_textures")
        assign_before = self._controller.settings.assign_textures
        self.assertEqual(assign_checkbox.widget.model.get_value_as_bool(), assign_before)
        await assign_checkbox.click()
        await ui_test.human_delay(5)

        # The brush follows the checkbox.
        self.assertEqual(self._controller.settings.assign_textures, not assign_before)
        self.assertEqual(assign_checkbox.widget.model.get_value_as_bool(), not assign_before)

    async def test_falloff_combo_selection_updates_falloff(self):
        # The user picks another falloff curve in the BRUSH section; the combo lists the curves in enum order.
        falloff_combo = await self._find_control("ComboBox", "scatter_falloff")
        falloff_values = list(Falloff)
        selected_index = falloff_combo.widget.model.get_item_value_model().as_int
        self.assertEqual(falloff_values[selected_index], self._controller.settings.falloff)
        target = Falloff.LINEAR
        self.assertNotEqual(self._controller.settings.falloff, target)
        falloff_combo.widget.model.get_item_value_model().set_value(falloff_values.index(target))
        await ui_test.human_delay(5)

        # The brush uses the selected curve.
        self.assertEqual(self._controller.settings.falloff, target)

    async def test_preset_save_as_with_new_name_stores_preset(self):
        # The user saves the current brush under a new preset name from the PRESETS section.
        self.assertNotIn(_PRESET_NAME, self._controller.preset_names())
        save_as_button = await self._find_control("Button", "scatter_preset_save_as")
        await save_as_button.click()
        await ui_test.human_delay(10)

        # The name prompt opens pre-filled with the current preset name; the user replaces it and confirms.
        name_field = ui_test.find(_control_query(_SAVE_AS_PROMPT_TITLE, "StringField", "scatter_preset_name_field"))
        self.assertIsNotNone(name_field, "Save As did not open the preset name prompt")
        await name_field.input(_PRESET_NAME, end_key=KeyboardInput.ENTER, clear_before_input=True)
        await ui_test.human_delay(5)
        ok_button = ui_test.find(_control_query(_SAVE_AS_PROMPT_TITLE, "Button", "scatter_preset_name_ok"))
        self.assertIsNotNone(ok_button, "The preset name prompt has no OK button")
        await ok_button.click()
        await ui_test.human_delay(10)

        # The preset is stored in the scratch directory, became the current preset and the combo shows it.
        self.assertIn(_PRESET_NAME, self._controller.preset_names())
        self.assertTrue((Path(self._temp_presets_dir.name) / f"{_PRESET_NAME}.json").is_file())
        self.assertEqual(self._controller.settings.preset_name, _PRESET_NAME)
        preset_combo = ui_test.find(_control_query(_WINDOW_TITLE, "ComboBox", "scatter_preset_combo"))
        self.assertIsNotNone(preset_combo)
        self.assertEqual(preset_combo.widget.model.current_choice, _PRESET_NAME)
        prompt = ui.Workspace.get_window(_SAVE_AS_PROMPT_TITLE)
        self.assertTrue(prompt is None or not prompt.visible, "The preset name prompt stayed open after OK")

    async def _paint_two_patches(self) -> tuple[list[str], list[str]]:
        """Open a capture-like stage and author a grass patch and a glass patch on the same mesh.

        Returns:
            The placement paths of the grass patch and of the glass patch.
        """
        usd_context = omni.usd.get_context(_CONTEXT_NAME)
        await usd_context.new_stage_async()
        self._painted_stage = True
        stage = usd_context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
        UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
        mesh = UsdGeom.Mesh.Define(stage, _MESH_PATH)
        mesh.GetPointsAttr().Set(
            [
                (-_QUAD_HALF_SIZE, -_QUAD_HALF_SIZE, 0.0),
                (_QUAD_HALF_SIZE, -_QUAD_HALF_SIZE, 0.0),
                (_QUAD_HALF_SIZE, _QUAD_HALF_SIZE, 0.0),
                (-_QUAD_HALF_SIZE, _QUAD_HALF_SIZE, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        self._controller.replace_settings(ScatterBrushSettings(randomize_seed=False, size_min=10.0, size_max=20.0))
        return (self._paint_patch(stage, "Grass", 7), self._paint_patch(stage, "Glass", 99))

    def _paint_patch(self, stage, preset_name: str, seed: int) -> list[str]:
        """Author one stamp of a preset onto the quad the way a brush stroke does, and return its placements."""
        target = ScatterTarget(
            prototype_root=Sdf.Path(_PROTOTYPE_ROOT),
            parent_instance_root=Sdf.Path(_INSTANCE_ROOT),
            mesh_path=Sdf.Path(_MESH_PATH),
            parent_world=Gf.Matrix4d(1.0),
            instance_count=1,
        )
        settings = ScatterBrushSettings(
            preset_name=preset_name, seed=seed, randomize_seed=False, padding=0.0, size_min=10.0, size_max=20.0
        )
        asset = ScatterAssetEntry(path=_SCATTER_ASSET, up_axis=UpAxis.Z, natural_extent=(1.0, 1.0, 2.0))
        cache = MeshSurfaceCache(lambda: stage)
        sample = cache.closest_point(_MESH_PATH, Gf.Vec3d(0.0, 0.0, 0.0), 10.0)
        records = generate_stamp(
            cache,
            target,
            sample,
            settings,
            [asset],
            np.random.default_rng(1),
            PaddingIndex(1.0),
            None,
            UpAxis.Z,
            1.0,
            stage.GetRootLayer(),
        )
        created = author_placements(stage.GetRootLayer(), records)
        cache.destroy()
        self.assertTrue(created, f"The {preset_name} patch authored no placement")
        return created

    def _scales(self, paths: list[str]) -> list[tuple]:
        """Return the authored scale of every placement, as the edit layer holds it."""
        layer = omni.usd.get_context(_CONTEXT_NAME).get_stage().GetRootLayer()
        return [tuple(layer.GetPrimAtPath(Sdf.Path(path)).attributes["xformOp:scale"].default) for path in paths]

    async def test_editing_size_resizes_the_selected_patch_and_leaves_the_other_one_alone(self):
        # The user has painted grass and glass on one mesh, and selects the glass patch.
        grass, glass = await self._paint_two_patches()
        omni.usd.get_context(_CONTEXT_NAME).get_selection().set_selected_prim_paths([_GLASS_CONTAINER], True)
        await ui_test.human_delay(10)

        # The window tells them what is in scope before they change anything.
        scope_label = await self._find_control("Label", "scatter_live_edit_scope")
        self.assertIn("glass", scope_label.widget.text)
        grass_before = self._scales(grass)
        glass_before = self._scales(glass)

        # They raise the smallest size the brush places.
        size_min = await self._find_control("FloatBoundedDrag", "scatter_size_min")
        await size_min.input("60", end_key=KeyboardInput.ENTER)
        await ui_test.human_delay(20)

        # The selected glass patch followed the setting and the unselected grass patch is exactly as it was.
        self.assertEqual(self._controller.settings.size_min, 60.0)
        self.assertNotEqual(self._scales(glass), glass_before)
        self.assertEqual(self._scales(grass), grass_before)

    async def test_undoing_a_size_edit_puts_the_whole_selected_patch_back(self):
        # The user resizes the selected glass patch.
        _grass, glass = await self._paint_two_patches()
        omni.usd.get_context(_CONTEXT_NAME).get_selection().set_selected_prim_paths([_GLASS_CONTAINER], True)
        await ui_test.human_delay(10)
        glass_before = self._scales(glass)
        size_min = await self._find_control("FloatBoundedDrag", "scatter_size_min")
        await size_min.input("60", end_key=KeyboardInput.ENTER)
        await ui_test.human_delay(20)
        self.assertNotEqual(self._scales(glass), glass_before)

        # They change their mind and undo once.
        omni.kit.undo.undo()
        await ui_test.human_delay(10)

        # The whole patch came back in that single undo, not one placement at a time.
        self.assertEqual(self._scales(glass), glass_before)

    async def test_editing_size_with_nothing_selected_leaves_every_painted_patch_alone(self):
        # Nothing is selected, so the user is only setting the brush up for their next stroke.
        grass, glass = await self._paint_two_patches()
        omni.usd.get_context(_CONTEXT_NAME).get_selection().clear_selected_prim_paths()
        await ui_test.human_delay(10)
        grass_before = self._scales(grass)
        glass_before = self._scales(glass)

        # They raise the smallest size the brush places.
        size_min = await self._find_control("FloatBoundedDrag", "scatter_size_min")
        await size_min.input("60", end_key=KeyboardInput.ENTER)
        await ui_test.human_delay(20)

        # The brush took the value and nothing already painted moved.
        self.assertEqual(self._controller.settings.size_min, 60.0)
        self.assertEqual(self._scales(grass), grass_before)
        self.assertEqual(self._scales(glass), glass_before)

    async def test_turning_off_live_editing_leaves_the_selected_patch_alone(self):
        # The user turns "Edit Selection Live" off before selecting anything.
        _grass, glass = await self._paint_two_patches()
        live_edit = await self._find_control("CheckBox", "scatter_edit_selection_live")
        self.assertTrue(live_edit.widget.model.get_value_as_bool())
        await live_edit.click()
        await ui_test.human_delay(20)
        self.assertFalse(self._controller.settings.edit_selection_live)

        # They then select the glass patch, which would otherwise be what a settings change updates.
        omni.usd.get_context(_CONTEXT_NAME).get_selection().set_selected_prim_paths([_GLASS_CONTAINER], True)
        await ui_test.human_delay(20)
        glass_before = self._scales(glass)

        # They raise the smallest size the brush places.
        size_min = await self._find_control("FloatBoundedDrag", "scatter_size_min")
        await size_min.input("60", end_key=KeyboardInput.ENTER)
        await ui_test.human_delay(20)

        # The setting changed for the next stroke only; the selected patch did not move.
        self.assertEqual(self._controller.settings.size_min, 60.0)
        self.assertEqual(self._scales(glass), glass_before)
