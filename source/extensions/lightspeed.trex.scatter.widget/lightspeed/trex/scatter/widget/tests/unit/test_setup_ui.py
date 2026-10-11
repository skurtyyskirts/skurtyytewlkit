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

__all__ = ["TestScatterPane"]

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import omni.kit.app
import omni.kit.commands
import omni.kit.test
import omni.kit.undo
import omni.usd
from lightspeed.trex.scatter.core import controller as _controller_module
from lightspeed.trex.scatter.core.controller import ScatterMode
from lightspeed.trex.scatter.core.placement import PlacementRecord
from lightspeed.trex.scatter.core.settings import (
    Falloff,
    ScaleMode,
    ScatterBrushSettings,
    TargetMode,
    UpAxis,
)
from lightspeed.trex.scatter.widget import setup_ui as _setup_ui_module
from omni import ui
from omni.kit import ui_test
from pxr import Sdf, Usd, UsdGeom

# The StageCraft context always exists in the test app; the pane only reads its stage for the instance warning.
_CONTEXT_NAME = ""
_ANCHOR_HASH = "0123456789ABCDEF"
_ANCHOR_PROTOTYPE = f"/RootNode/meshes/mesh_{_ANCHOR_HASH}"
_STATUS_STYLE = "PropertiesPaneSectionTreeItem"
_STATUS_ERROR_STYLE = "PropertiesPaneSectionTreeItemError"

# (widget class name, identifier) of every control the pane builds itself; the asset list and the presets widget
# are replaced by mocks in these tests and carry their own identifiers.
_EXPECTED_CONTROLS = (
    ("ToolButton", "scatter_mode_paint"),
    ("ToolButton", "scatter_mode_erase"),
    ("Button", "scatter_flood"),
    ("Label", "scatter_status"),
    ("FloatBoundedDrag", "scatter_radius"),
    ("ComboBox", "scatter_falloff"),
    ("FloatBoundedDrag", "scatter_density"),
    ("FloatBoundedDrag", "scatter_strength"),
    ("FloatBoundedDrag", "scatter_spacing"),
    ("FloatBoundedDrag", "scatter_padding"),
    ("FloatBoundedDrag", "scatter_max_surface_angle"),
    ("FloatBoundedDrag", "scatter_turn_x"),
    ("FloatBoundedDrag", "scatter_turn_y"),
    ("FloatBoundedDrag", "scatter_turn_z"),
    ("FloatBoundedDrag", "scatter_vertical_offset"),
    ("CheckBox", "scatter_conform"),
    ("CheckBox", "scatter_align_stroke"),
    ("FloatBoundedDrag", "scatter_base_rotation_x"),
    ("FloatBoundedDrag", "scatter_base_rotation_y"),
    ("FloatBoundedDrag", "scatter_base_rotation_z"),
    ("FloatBoundedDrag", "scatter_rot_z_min"),
    ("FloatBoundedDrag", "scatter_rot_z_max"),
    ("FloatBoundedDrag", "scatter_rot_x_min"),
    ("FloatBoundedDrag", "scatter_rot_x_max"),
    ("FloatBoundedDrag", "scatter_rot_y_min"),
    ("FloatBoundedDrag", "scatter_rot_y_max"),
    ("ComboBox", "scatter_scale_mode"),
    ("CheckBox", "scatter_scale_enabled"),
    ("FloatBoundedDrag", "scatter_size_min"),
    ("FloatBoundedDrag", "scatter_size_max"),
    ("CheckBox", "scatter_scale_uniform"),
    ("FloatBoundedDrag", "scatter_scale_min"),
    ("FloatBoundedDrag", "scatter_scale_max"),
    ("FloatBoundedDrag", "scatter_scale_bias"),
    ("FloatBoundedDrag", "scatter_scale_weight"),
    ("FloatBoundedDrag", "scatter_scale_x_min"),
    ("FloatBoundedDrag", "scatter_scale_x_max"),
    ("FloatBoundedDrag", "scatter_scale_y_min"),
    ("FloatBoundedDrag", "scatter_scale_y_max"),
    ("FloatBoundedDrag", "scatter_scale_z_min"),
    ("FloatBoundedDrag", "scatter_scale_z_max"),
    ("IntBoundedDrag", "scatter_seed"),
    ("Button", "scatter_reroll"),
    ("CheckBox", "scatter_randomize_seed"),
    ("ComboBox", "scatter_apply_to"),
    ("ComboBox", "scatter_target_mode"),
    ("Label", "scatter_anchor_path"),
    ("Button", "scatter_anchor_use_selection"),
    ("ComboBox", "scatter_erase_scope"),
    ("IntBoundedDrag", "scatter_flood_cap"),
    ("Label", "scatter_flood_estimate"),
    ("IntBoundedDrag", "scatter_prim_budget"),
    ("Label", "scatter_instance_warning"),
    ("CheckBox", "scatter_assign_textures"),
)


def _patch_controller_environment(stack: contextlib.ExitStack) -> None:
    """Keep the controller away from carb persistence and the user preset folder for the duration of a test."""
    settings_class = _controller_module.ScatterBrushSettings
    stack.enter_context(mock.patch.object(settings_class, "load_from_carb", return_value=ScatterBrushSettings()))
    stack.enter_context(mock.patch.object(_controller_module.ScatterBrushSettings, "save_to_carb"))
    stack.enter_context(mock.patch.object(_controller_module, "load_assets_from_carb", return_value=[]))
    stack.enter_context(mock.patch.object(_controller_module, "save_assets_to_carb"))
    preset_store_class = stack.enter_context(mock.patch.object(_controller_module, "PresetStore"))
    preset_store_class.return_value.list_names.return_value = []
    stack.enter_context(
        mock.patch.object(_controller_module, "get_default_presets_directory", return_value=Path("presets"))
    )


def _make_anchor_stage(instance_total: int) -> Usd.Stage:
    """Build an in-memory stage holding the anchor prototype and ``instance_total`` instances of its hash."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, _ANCHOR_PROTOTYPE)
    for index in range(instance_total):
        UsdGeom.Xform.Define(stage, f"/RootNode/instances/inst_{_ANCHOR_HASH}_{index}")
    return stage


class TestScatterPane(omni.kit.test.AsyncTestCase):
    """Tests the Scatter pane against a fresh brush controller inside a real window.

    Header toggles are exercised through their bool models, which is what a ``ui.ToolButton`` flips on a click.
    """

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        _patch_controller_environment(self._stack)
        self.asset_list_class = self._stack.enter_context(mock.patch.object(_setup_ui_module, "ScatterAssetListWidget"))
        self.presets_class = self._stack.enter_context(mock.patch.object(_setup_ui_module, "PresetsWidget"))
        self._previous_instance = _controller_module._CONTROLLER_INSTANCE
        _controller_module._CONTROLLER_INSTANCE = None
        self.controller = _controller_module.get_scatter_brush_controller()
        self.window_title = f"ScatterPaneTest_{self._testMethodName}"
        self.window = ui.Window(self.window_title, width=520, height=900)
        with self.window.frame:
            self.pane = _setup_ui_module.ScatterPane(_CONTEXT_NAME)
        await omni.kit.app.get_app().next_update_async()
        self.pane.show(True)

    async def tearDown(self):
        self.pane.destroy()
        self.window.destroy()
        self.window = None
        await omni.kit.app.get_app().next_update_async()
        self.controller.destroy()
        _controller_module._CONTROLLER_INSTANCE = self._previous_instance
        self._stack.close()

    def _find(self, widget_type: str, identifier: str):
        """Return the pane widget of the given class name and identifier, or None."""
        return ui_test.find(f"{self.window_title}//Frame/**/{widget_type}[*].identifier=='{identifier}'")

    def _model(self, field_name: str) -> ui.AbstractValueModel:
        """Return the value model bound to a settings field."""
        return self.pane._bindings[field_name].model

    async def test_build_pane_creates_every_identified_control(self):
        # Arrange
        expected = _EXPECTED_CONTROLS

        # Act
        missing = [identifier for widget_type, identifier in expected if self._find(widget_type, identifier) is None]

        # Assert
        self.assertEqual(missing, [])

    async def test_build_pane_hosts_asset_list_and_presets_widgets(self):
        # Arrange
        asset_list_class = self.asset_list_class
        presets_class = self.presets_class

        # Act
        hosted = (asset_list_class.call_args, presets_class.call_args)

        # Assert
        self.assertEqual(hosted, (mock.call(_CONTEXT_NAME), mock.call()))

    async def test_build_pane_seeds_fields_from_controller_settings(self):
        # Arrange
        settings = self.controller.settings

        # Act
        radius = self._model("radius").as_float

        # Assert
        self.assertEqual(radius, settings.radius)

    async def test_edit_radius_model_updates_controller_settings(self):
        # Arrange
        model = self._model("radius")

        # Act
        model.set_value(75.0)

        # Assert
        self.assertEqual(self.controller.settings.radius, 75.0)

    async def test_edit_base_rotation_x_model_updates_controller_settings(self):
        # Arrange
        model = self._model("base_rotation_x")

        # Act
        model.set_value(90.0)

        # Assert
        self.assertEqual(self.controller.settings.base_rotation_x, 90.0)

    async def test_update_settings_base_rotation_z_pushes_value_into_model(self):
        # Arrange
        model = self._model("base_rotation_z")

        # Act
        self.controller.update_settings(base_rotation_z=-45.0)

        # Assert
        self.assertEqual(model.as_float, -45.0)

    async def test_edit_max_surface_angle_model_updates_controller_settings(self):
        # Arrange
        model = self._model("max_surface_angle")

        # Act
        model.set_value(45.0)

        # Assert
        self.assertEqual(self.controller.settings.max_surface_angle, 45.0)

    async def test_update_settings_radius_pushes_value_into_model_without_reentry(self):
        # Arrange
        model = self._model("radius")

        with mock.patch.object(self.controller, "update_settings", wraps=self.controller.update_settings) as spy:
            # Act
            self.controller.update_settings(radius=120.0)

        # Assert
        self.assertEqual(model.as_float, 120.0)
        self.assertEqual(spy.call_count, 1)

    async def test_edit_model_with_unchanged_value_does_not_call_update_settings(self):
        # Arrange
        model = self._model("radius")

        with mock.patch.object(self.controller, "update_settings", wraps=self.controller.update_settings) as spy:
            # Act
            model.set_value(self.controller.settings.radius)

        # Assert
        spy.assert_not_called()

    async def test_drag_rotation_z_min_past_its_max_carries_the_max_along(self):
        # A drag field reports every value it crosses, so the min has to be able to pass the max rather than be
        # refused at it.
        # Arrange
        self.controller.update_settings(rotation_z_max=90.0)
        model = self._model("rotation_z_min")

        # Act
        model.set_value(100.0)

        # Assert
        self.assertEqual(self.controller.settings.rotation_z_min, 100.0)
        self.assertEqual(self.controller.settings.rotation_z_max, 100.0)
        self.assertEqual(model.as_float, 100.0)

    async def test_an_edit_the_controller_rejects_restores_the_field_from_the_settings(self):
        # The drag fields clamp to their own bounds before the controller sees a value, so the rollback path is
        # driven by a refusing controller rather than by an out-of-bounds number.
        # Arrange
        model = self._model("radius")
        unchanged = self.controller.settings.radius

        # Act
        with mock.patch.object(self.controller, "update_settings", return_value=False):
            model.set_value(75.0)

        # Assert
        self.assertEqual(model.as_float, unchanged)
        self.assertEqual(self.controller.settings.radius, unchanged)

    async def test_select_falloff_combo_index_updates_controller_settings(self):
        # Arrange
        model = self._model("falloff")

        # Act
        model.set_value(list(Falloff).index(Falloff.GAUSSIAN))

        # Assert
        self.assertEqual(self.controller.settings.falloff, Falloff.GAUSSIAN)

    async def test_update_settings_target_mode_pushes_combo_index(self):
        # Arrange
        model = self._model("target_mode")

        # Act
        self.controller.update_settings(target_mode=TargetMode.ANCHOR)

        # Assert
        self.assertEqual(model.as_int, list(TargetMode).index(TargetMode.ANCHOR))

    async def test_uncheck_conform_checkbox_updates_controller_settings(self):
        # Arrange
        model = self._model("conform_to_surface")

        # Act
        model.set_value(False)

        # Assert
        self.assertFalse(self.controller.settings.conform_to_surface)

    async def test_uncheck_assign_textures_checkbox_updates_controller_settings(self):
        # Arrange
        model = self._model("assign_textures")

        # Act
        model.set_value(False)

        # Assert
        self.assertFalse(self.controller.settings.assign_textures)

    async def test_edit_flood_cap_model_updates_controller_settings(self):
        # Arrange
        model = self._model("flood_max_instances")

        # Act
        model.set_value(42)

        # Assert
        self.assertEqual(self.controller.settings.flood_max_instances, 42)

    async def test_toggle_paint_button_when_off_sets_paint_mode(self):
        # Arrange
        paint_model = self.pane._paint_model

        # Act
        paint_model.set_value(True)

        # Assert
        self.assertEqual(self.controller.mode, ScatterMode.PAINT)
        self.assertFalse(self.pane._erase_model.as_bool)

    async def test_toggle_paint_button_when_painting_sets_mode_off(self):
        # Arrange
        self.controller.set_mode(ScatterMode.PAINT)
        paint_model = self.pane._paint_model

        # Act
        paint_model.set_value(False)

        # Assert
        self.assertEqual(self.controller.mode, ScatterMode.OFF)

    async def test_toggle_erase_button_when_painting_sets_erase_mode(self):
        # Arrange
        self.controller.set_mode(ScatterMode.PAINT)
        erase_model = self.pane._erase_model

        # Act
        erase_model.set_value(True)

        # Assert
        self.assertEqual(self.controller.mode, ScatterMode.ERASE)
        self.assertFalse(self.pane._paint_model.as_bool)

    async def test_set_mode_from_controller_updates_toggle_buttons(self):
        # Arrange
        controller = self.controller

        # Act
        controller.set_mode(ScatterMode.ERASE)

        # Assert
        self.assertTrue(self.pane._erase_model.as_bool)
        self.assertFalse(self.pane._paint_model.as_bool)

    async def test_click_flood_at_default_cap_floods_with_pane_context_and_cache(self):
        # Arrange
        with mock.patch.object(self.controller, "flood", return_value=0) as flood_mock:
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        flood_mock.assert_called_once_with(_CONTEXT_NAME, self.pane._surface_cache)

    async def test_click_flood_that_reaches_the_prim_budget_asks_for_confirmation_before_flooding(self):
        # Arrange
        self.controller.update_settings(flood_max_instances=1000, prim_budget=500)

        with (
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        dialog_class.assert_called_once()
        self.assertIn("1000 prims", dialog_class.call_args.kwargs["message"])
        flood_mock.assert_not_called()

    async def test_click_flood_above_default_cap_within_the_prim_budget_floods_without_asking(self):
        # Arrange
        self.controller.update_settings(flood_max_instances=1000, prim_budget=10000)

        with (
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        dialog_class.assert_not_called()
        flood_mock.assert_called_once_with(_CONTEXT_NAME, self.pane._surface_cache)

    async def test_click_flood_on_a_replicated_anchor_asks_for_confirmation_with_the_rendered_prims(self):
        # Arrange - the default cap on an anchor shown by 40 instances renders 12000 prims, over the default budget
        stage = _make_anchor_stage(40)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        cap = self.controller.settings.flood_max_instances
        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

        with (
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        dialog_class.assert_called_once()
        self.assertIn(f"{cap * 40} prims", dialog_class.call_args.kwargs["message"])
        flood_mock.assert_not_called()

    async def test_click_flood_on_a_replicated_selection_asks_for_confirmation_with_the_rendered_prims(self):
        # Arrange
        stage = _make_anchor_stage(40)
        usd_context = mock.MagicMock(name="usd_context")
        usd_context.get_stage.return_value = stage
        usd_context.get_selection.return_value.get_selected_prim_paths.return_value = [_ANCHOR_PROTOTYPE]
        cap = self.controller.settings.flood_max_instances

        with (
            mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context),
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        dialog_class.assert_called_once()
        self.assertIn(f"{cap * 40} prims", dialog_class.call_args.kwargs["message"])
        flood_mock.assert_not_called()

    async def test_confirm_flood_dialog_floods_with_pane_context_and_cache(self):
        # Arrange
        self.controller.update_settings(flood_max_instances=1000, prim_budget=500)

        with (
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            self.pane._flood_button.call_clicked_fn()
            ok_handler = dialog_class.call_args.kwargs["ok_handler"]

            # Act
            ok_handler()

        # Assert
        flood_mock.assert_called_once_with(_CONTEXT_NAME, self.pane._surface_cache)

    async def test_build_pane_in_absolute_size_mode_shows_size_rows_and_hides_the_multiplier_rows(self):
        # Arrange - Absolute Size is the default, so this is what a freshly built pane shows.

        # Act
        size_visible = self.pane._size_stack.visible
        multiplier_visible = self.pane._multiplier_scale_stack.visible

        # Assert
        self.assertTrue(size_visible)
        self.assertFalse(multiplier_visible)

    async def test_select_multiplier_scale_mode_swaps_the_size_rows_for_the_multiplier_rows(self):
        # Arrange
        model = self._model("scale_mode")

        # Act
        model.set_value(list(ScaleMode).index(ScaleMode.MULTIPLIER))

        # Assert
        self.assertEqual(self.controller.settings.scale_mode, ScaleMode.MULTIPLIER)
        self.assertFalse(self.pane._size_stack.visible)
        self.assertTrue(self.pane._multiplier_scale_stack.visible)

    async def test_update_settings_back_to_absolute_size_restores_the_size_rows(self):
        # Arrange
        self.controller.update_settings(scale_mode=ScaleMode.MULTIPLIER)

        # Act
        self.controller.update_settings(scale_mode=ScaleMode.ABSOLUTE_SIZE)

        # Assert
        self.assertTrue(self.pane._size_stack.visible)
        self.assertFalse(self.pane._multiplier_scale_stack.visible)

    async def test_build_pane_with_uniform_scale_hides_per_axis_rows(self):
        # Arrange
        stack = self.pane._per_axis_scale_stack

        # Act
        visible = stack.visible

        # Assert
        self.assertFalse(visible)

    async def test_uncheck_scale_uniform_shows_per_axis_rows(self):
        # Arrange
        model = self._model("scale_uniform")

        # Act
        model.set_value(False)

        # Assert
        self.assertFalse(self.controller.settings.scale_uniform)
        self.assertTrue(self.pane._per_axis_scale_stack.visible)

    async def test_click_anchor_use_selection_forwards_pane_context(self):
        # Arrange
        with mock.patch.object(self.controller, "set_anchor_from_selection", return_value=True) as anchor_mock:
            # Act
            self.pane._anchor_use_selection_button.call_clicked_fn()

        # Assert
        anchor_mock.assert_called_once_with(_CONTEXT_NAME)

    async def test_build_pane_without_anchor_shows_none_placeholder(self):
        # Arrange
        label = self.pane._anchor_path_label

        # Act
        text = label.text

        # Assert
        self.assertEqual(text, "<none>")

    async def test_update_anchor_path_shows_path_in_label(self):
        # Arrange
        label = self.pane._anchor_path_label

        # Act
        self.controller.update_settings(anchor_prototype_path=_ANCHOR_PROTOTYPE)

        # Assert
        self.assertEqual(label.text, _ANCHOR_PROTOTYPE)

    async def test_update_flood_cap_refreshes_flood_estimate_label(self):
        # Arrange
        label = self.pane._flood_estimate_label

        # Act
        self.controller.update_settings(flood_max_instances=42)

        # Assert
        self.assertEqual(label.text, "Up to 42 prims per instance")

    async def test_select_anchor_mode_with_replicated_anchor_multiplies_flood_estimate_by_instances(self):
        # Arrange
        stage = _make_anchor_stage(3)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        cap = self.controller.settings.flood_max_instances

        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            # Act
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

        # Assert
        self.assertEqual(self.pane._flood_estimate_label.text, f"Up to {cap * 3} prims")

    async def test_flood_estimate_within_the_prim_budget_is_shown_plainly(self):
        # Arrange
        label = self.pane._flood_estimate_label

        # Act
        self.controller.update_settings(flood_max_instances=300, prim_budget=10000)

        # Assert
        self.assertEqual(label.name, "")

    async def test_flood_estimate_that_reaches_the_prim_budget_is_shown_as_a_warning(self):
        # Arrange
        label = self.pane._flood_estimate_label

        # Act
        self.controller.update_settings(flood_max_instances=300, prim_budget=300)

        # Assert
        self.assertEqual(label.name, "Warning")

    async def test_flood_estimate_at_the_alert_multiple_of_the_prim_budget_is_shown_as_an_error(self):
        # Arrange
        label = self.pane._flood_estimate_label

        # Act
        self.controller.update_settings(flood_max_instances=300, prim_budget=60)

        # Assert
        self.assertEqual(label.name, "PropertiesWidgetLabelWarning")

    async def test_flood_estimate_on_a_replicated_anchor_is_judged_by_the_prims_it_renders(self):
        # Arrange - 300 placements are within the budget, 300 on each of 40 instances are not
        stage = _make_anchor_stage(40)
        usd_context = SimpleNamespace(get_stage=lambda: stage)

        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            # Act
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

        # Assert
        self.assertEqual(self.pane._flood_estimate_label.name, "Warning")

    async def test_opening_a_project_counts_the_anchor_instances_again(self):
        # Arrange - the anchor path is the same, but the project that just opened shows it on more instances
        cap = self.controller.settings.flood_max_instances
        one_instance = _make_anchor_stage(1)
        with mock.patch.object(
            _setup_ui_module.omni.usd, "get_context", return_value=SimpleNamespace(get_stage=lambda: one_instance)
        ):
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)
        three_instances = _make_anchor_stage(3)

        with mock.patch.object(
            _setup_ui_module.omni.usd, "get_context", return_value=SimpleNamespace(get_stage=lambda: three_instances)
        ):
            # Act
            self.pane._on_stage_event(SimpleNamespace(type=int(omni.usd.StageEventType.OPENED)))

        # Assert
        self.assertEqual(self.pane._flood_estimate_label.text, f"Up to {cap * 3} prims")

    async def test_select_hit_surface_mode_after_replicated_anchor_resets_flood_estimate_to_cap(self):
        # Arrange
        stage = _make_anchor_stage(3)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        cap = self.controller.settings.flood_max_instances
        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

            # Act
            self.controller.update_settings(target_mode=TargetMode.HIT_SURFACE)

        # Assert
        self.assertEqual(self.pane._flood_estimate_label.text, f"Up to {cap} prims per instance")

    async def test_select_anchor_mode_with_replicated_anchor_shows_instance_warning(self):
        # Arrange
        stage = _make_anchor_stage(3)
        usd_context = SimpleNamespace(get_stage=lambda: stage)

        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            # Act
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

        # Assert
        self.assertTrue(self.pane._instance_warning_label.visible)
        self.assertEqual(
            self.pane._instance_warning_label.text, f"Placements replicate onto 3 instances of mesh_{_ANCHOR_HASH}"
        )

    async def test_select_anchor_mode_with_single_instance_hides_instance_warning(self):
        # Arrange
        stage = _make_anchor_stage(1)
        usd_context = SimpleNamespace(get_stage=lambda: stage)

        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            # Act
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

        # Assert
        self.assertFalse(self.pane._instance_warning_label.visible)

    async def test_select_hit_surface_mode_hides_instance_warning_for_replicated_anchor(self):
        # Arrange
        stage = _make_anchor_stage(3)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

            # Act
            self.controller.update_settings(target_mode=TargetMode.HIT_SURFACE)

        # Assert
        self.assertFalse(self.pane._instance_warning_label.visible)

    async def test_asset_palette_change_does_not_recount_anchor_instances(self):
        # Arrange
        stage = _make_anchor_stage(3)
        usd_context = SimpleNamespace(get_stage=lambda: stage)
        with mock.patch.object(_setup_ui_module.omni.usd, "get_context", return_value=usd_context):
            self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)

            with mock.patch.object(
                _setup_ui_module, "instance_count", wraps=_setup_ui_module.instance_count
            ) as count_mock:
                # Act
                self.controller.add_asset("C:/project/assets/ingested/rock.usd", up_axis=UpAxis.Z)

        # Assert
        count_mock.assert_not_called()
        self.assertTrue(self.pane._instance_warning_label.visible)
        self.assertEqual(
            self.pane._instance_warning_label.text, f"Placements replicate onto 3 instances of mesh_{_ANCHOR_HASH}"
        )

    async def test_click_reroll_sets_seed_from_fresh_random_bits(self):
        # Arrange
        self.controller.update_settings(randomize_seed=False)

        with mock.patch.object(_setup_ui_module.secrets, "randbits", return_value=12345):
            # Act
            self.pane._reroll_button.call_clicked_fn()

        # Assert
        self.assertEqual(self.controller.settings.seed, 12345)
        self.assertEqual(self._model("seed").as_int, 12345)

    async def test_build_pane_with_randomized_seed_disables_seed_controls(self):
        # Arrange
        seed_field = self.pane._seed_field
        reroll_button = self.pane._reroll_button

        # Act
        enabled = (seed_field.enabled, reroll_button.enabled)

        # Assert
        self.assertEqual(enabled, (False, False))

    async def test_uncheck_randomize_seed_enables_seed_controls(self):
        # Arrange
        model = self._model("randomize_seed")

        # Act
        model.set_value(False)

        # Assert
        self.assertTrue(self.pane._seed_field.enabled)
        self.assertTrue(self.pane._reroll_button.enabled)

    async def test_post_status_shows_text_in_status_label(self):
        # Arrange
        label = self.pane._status_label

        # Act
        self.controller.post_status("Placed 12 assets")

        # Assert
        self.assertEqual(label.text, "Placed 12 assets")
        self.assertEqual(label.style_type_name_override, _STATUS_STYLE)

    async def test_post_error_status_uses_error_style(self):
        # Arrange
        label = self.pane._status_label

        # Act
        self.controller.post_status("Open a project before flooding", is_error=True)

        # Assert
        self.assertEqual(label.text, "Open a project before flooding")
        self.assertEqual(label.style_type_name_override, _STATUS_ERROR_STYLE)

    async def test_update_settings_while_hidden_leaves_fields_unchanged(self):
        # Arrange
        self.pane.show(False)
        model = self._model("radius")

        # Act
        self.controller.update_settings(radius=200.0)

        # Assert
        self.assertEqual(model.as_float, 50.0)

    async def test_show_after_hidden_update_refreshes_fields(self):
        # Arrange
        self.pane.show(False)
        self.controller.update_settings(radius=200.0)
        model = self._model("radius")

        # Act
        self.pane.show(True)

        # Assert
        self.assertEqual(model.as_float, 200.0)

    async def test_set_mode_while_hidden_then_show_refreshes_toggle_buttons(self):
        # Arrange
        self.pane.show(False)
        self.controller.set_mode(ScatterMode.PAINT)

        # Act
        self.pane.show(True)

        # Assert
        self.assertTrue(self.pane._paint_model.as_bool)

    async def test_destroy_pane_releases_subscriptions_cache_and_child_widgets(self):
        # Arrange
        pane = self.pane

        # Act
        pane.destroy()

        # Assert
        self.assertTrue(pane.destroyed)
        self.assertIsNone(pane._settings_changed_sub)
        self.assertIsNone(pane._mode_changed_sub)
        self.assertIsNone(pane._assets_changed_sub)
        self.assertIsNone(pane._status_message_sub)
        self.assertIsNone(pane._surface_cache)
        self.asset_list_class.return_value.destroy.assert_called_once_with()
        self.presets_class.return_value.destroy.assert_called_once_with()

    async def test_update_settings_after_destroy_leaves_models_unchanged(self):
        # Arrange
        model = self._model("radius")
        self.pane.destroy()

        # Act
        self.controller.update_settings(radius=200.0)

        # Assert
        self.assertEqual(model.as_float, 50.0)


class TestScatterPaneTurnFields(TestScatterPane):
    """The turn fields show what the viewport gizmo has done to the selection, and drive the same move themselves.

    They are not bound to a brush setting: a brush setting decides what the next stroke paints, while a turn is an
    edit to scatter that is already painted.
    """

    def _turn(self) -> tuple:
        """Return the X, Y, Z degrees the three fields are showing."""
        return tuple(model.as_float for model in self.pane._turn_models)  # noqa: PLW0212

    async def test_the_fields_start_at_no_turn(self):
        # Assert
        self.assertEqual(self._turn(), (0.0, 0.0, 0.0))

    async def test_a_turn_the_gizmo_reports_is_shown_on_the_fields(self):
        # Arrange
        scope = self.pane._reapply_driver.scope  # noqa: PLW0212

        # Act
        self.controller.set_selection_turn(scope, (10.0, 20.0, 30.0))

        # Assert
        self.assertEqual(self._turn(), (10.0, 20.0, 30.0))

    async def test_a_turn_given_to_something_else_is_not_shown(self):
        # Arrange
        other = _setup_ui_module.ReapplyScope()
        other_scope = other if other != self.pane._reapply_driver.scope else None  # noqa: PLW0212

        # Act
        if other_scope is not None:
            self.controller.set_selection_turn(other_scope, (10.0, 20.0, 30.0))

        # Assert
        self.assertEqual(self._turn(), (0.0, 0.0, 0.0))

    async def test_the_fields_are_disabled_while_nothing_scattered_is_selected(self):
        # Assert - the pane's scope is empty in the test stage, so there is nothing a turn could be applied to
        self.assertTrue(self.pane._reapply_driver.scope.is_empty)  # noqa: PLW0212
        for field in self.pane._turn_fields:  # noqa: PLW0212
            self.assertFalse(field.enabled)

    async def test_typing_a_turn_with_nothing_selected_writes_nothing(self):
        # Act
        self.pane._turn_models[2].set_value(45.0)  # noqa: PLW0212

        # Assert - no scope means no session was ever opened
        self.assertFalse(self.pane._turn_session.active)  # noqa: PLW0212


class TestScatterPaneStrokePreview(TestScatterPane):
    """Right after a stroke the pane edits that stroke, and says so."""

    async def setUp(self):
        await super().setUp()
        omni.kit.undo.clear_stack()
        self._layer = Sdf.Layer.CreateAnonymous("scatter_preview_test")
        record = PlacementRecord(
            container_path=f"{_ANCHOR_PROTOTYPE}/scatter_grass",
            prim_name="s_000000000001",
            asset_rel_path="blade.usda",
            asset_abs_path="C:/assets/blade.usda",
            translate=(0.0, 0.0, 0.0),
            rotate_xyz=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            brush_id="Grass",
        )
        # Committed the way a stroke is: already authored live, so the command's first do() writes nothing.
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name=_CONTEXT_NAME,
            layer_identifier=self._layer.identifier,
            kind="PLACE",
            records=[record],
            already_applied=True,
        )

    async def tearDown(self):
        self.controller.end_preview()
        omni.kit.undo.clear_stack()
        await super().tearDown()

    async def test_begin_preview_shows_the_stroke_as_the_live_edit_scope(self):
        # Act
        self.controller.begin_preview(_CONTEXT_NAME)

        # Assert
        self.assertTrue(self.pane._reapply_driver.previewing)  # noqa: PLW0212
        self.assertTrue(self.pane._live_edit_label.text.startswith("Last stroke: 1 placement"))  # noqa: PLW0212

    async def test_begin_preview_disables_the_turn_fields(self):
        # Act
        self.controller.begin_preview(_CONTEXT_NAME)

        # Assert - the stroke is tuned through the brush settings instead, so later strokes follow
        for field in self.pane._turn_fields:  # noqa: PLW0212
            self.assertFalse(field.enabled)

    async def test_end_preview_returns_the_scope_to_the_selection(self):
        # Arrange
        self.controller.begin_preview(_CONTEXT_NAME)

        # Act
        self.controller.end_preview()

        # Assert
        self.assertFalse(self.pane._reapply_driver.previewing)  # noqa: PLW0212
        self.assertTrue(self.pane._reapply_driver.scope.is_empty)  # noqa: PLW0212
