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
import omni.kit.test
from lightspeed.trex.scatter.core import controller as _controller_module
from lightspeed.trex.scatter.core.controller import ScatterMode
from lightspeed.trex.scatter.core.settings import Falloff, ScatterBrushSettings, TargetMode, UpAxis
from lightspeed.trex.scatter.widget import setup_ui as _setup_ui_module
from omni import ui
from omni.kit import ui_test
from pxr import Usd, UsdGeom

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
    ("FloatBoundedDrag", "scatter_vertical_offset"),
    ("CheckBox", "scatter_conform"),
    ("CheckBox", "scatter_align_stroke"),
    ("FloatBoundedDrag", "scatter_rot_z_min"),
    ("FloatBoundedDrag", "scatter_rot_z_max"),
    ("FloatBoundedDrag", "scatter_rot_x_min"),
    ("FloatBoundedDrag", "scatter_rot_x_max"),
    ("FloatBoundedDrag", "scatter_rot_y_min"),
    ("FloatBoundedDrag", "scatter_rot_y_max"),
    ("CheckBox", "scatter_scale_enabled"),
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
    ("Label", "scatter_instance_warning"),
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

    async def test_edit_rotation_z_min_above_max_restores_field_from_settings(self):
        # Arrange
        self.controller.update_settings(rotation_z_max=90.0)
        model = self._model("rotation_z_min")

        # Act
        model.set_value(100.0)

        # Assert
        self.assertEqual(self.controller.settings.rotation_z_min, 0.0)
        self.assertEqual(model.as_float, 0.0)

    async def test_edit_rotation_z_min_above_max_shows_error_status(self):
        # Arrange
        self.controller.update_settings(rotation_z_max=90.0)
        model = self._model("rotation_z_min")

        # Act
        model.set_value(100.0)

        # Assert
        self.assertTrue(self.pane._status_label.text.startswith("Invalid brush setting"))
        self.assertEqual(self.pane._status_label.style_type_name_override, _STATUS_ERROR_STYLE)

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

    async def test_click_flood_above_default_cap_asks_for_confirmation_before_flooding(self):
        # Arrange
        self.controller.update_settings(flood_max_instances=1000)

        with (
            mock.patch.object(self.controller, "flood", return_value=0) as flood_mock,
            mock.patch.object(_setup_ui_module, "_TrexMessageDialog") as dialog_class,
        ):
            # Act
            self.pane._flood_button.call_clicked_fn()

        # Assert
        dialog_class.assert_called_once()
        flood_mock.assert_not_called()

    async def test_confirm_flood_dialog_floods_with_pane_context_and_cache(self):
        # Arrange
        self.controller.update_settings(flood_max_instances=1000)

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
        self.assertEqual(label.text, "Up to 42 prims")

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
        self.assertEqual(self.pane._flood_estimate_label.text, f"Up to {cap} prims")

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
