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

__all__ = ["TestScatterAssetDelegate", "TestScatterAssetItem", "TestScatterAssetListWidget", "TestScatterAssetModel"]

import contextlib
import tempfile
from pathlib import Path
from unittest import mock

import carb.settings
import omni.kit.app
import omni.kit.test
import omni.usd
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core import settings as settings_module
from lightspeed.trex.scatter.core.settings import ScatterAssetEntry, UpAxis
from lightspeed.trex.scatter.widget import asset_list as asset_list_module
from omni import ui
from omni.flux.utils.widget.drag_field import FloatBoundedDrag
from omni.kit import ui_test

_CONTEXT_NAME = ""
_ROCK = "C:/project/assets/ingested/rock.usd"
_GRASS = "C:/project/assets/ingested/grass.usd"
_BROWSE_PLACEHOLDER = "Browse ingested models..."
_TEST_SETTINGS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/asset_list/brushSettings"
_TEST_ASSETS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/asset_list/assets"


def _accept_immediately(asset_path, _layer, _asset_core, _context, accept_handler, **_kwargs) -> bool:
    """Stand-in for the validation helper that accepts every path without a dialog."""
    accept_handler(asset_path)
    return True


def _never_accept(*_args, **_kwargs) -> bool:
    """Stand-in for the validation helper that leaves a dialog pending and never accepts."""
    return False


def _isolated_controller(stack: contextlib.ExitStack, presets_dir: Path) -> controller_module.ScatterBrushController:
    """Build a controller that persists to throwaway carb paths and a temporary preset directory."""
    stack.enter_context(mock.patch.object(settings_module, "BRUSH_SETTINGS_PATH", _TEST_SETTINGS_PATH))
    stack.enter_context(mock.patch.object(settings_module, "ASSETS_SETTING_PATH", _TEST_ASSETS_PATH))
    stack.enter_context(mock.patch.object(controller_module, "get_default_presets_directory", return_value=presets_dir))
    settings = carb.settings.get_settings()
    stack.callback(settings.destroy_item, _TEST_SETTINGS_PATH)
    stack.callback(settings.destroy_item, _TEST_ASSETS_PATH)
    return controller_module.ScatterBrushController()


class _AssetListTestCase(omni.kit.test.AsyncTestCase):
    """Shared fixture: the asset palette built inside a real window and bound to an isolated controller."""

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        temp_dir = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.controller = _isolated_controller(self._stack, Path(temp_dir) / "presets")
        self._stack.enter_context(
            mock.patch.object(asset_list_module, "get_scatter_brush_controller", return_value=self.controller)
        )
        self._stack.enter_context(mock.patch.object(controller_module, "read_asset_up_axis", return_value=UpAxis.Z))
        self.accept_mock = self._stack.enter_context(
            mock.patch.object(
                asset_list_module, "accept_asset_if_valid_for_replacement", side_effect=_accept_immediately
            )
        )
        self.crate_mock = self._stack.enter_context(
            mock.patch.object(asset_list_module, "asset_crate_supported", return_value=True)
        )
        self.list_models_mock = self._stack.enter_context(
            mock.patch.object(asset_list_module, "list_ingested_models", return_value=[])
        )
        self.usd_context = omni.usd.get_context(_CONTEXT_NAME)
        await self.usd_context.new_stage_async()
        self.window = ui.Window(f"ScatterAssetList_{self._testMethodName}", width=500, height=400)
        with self.window.frame:
            self.widget = asset_list_module.ScatterAssetListWidget(_CONTEXT_NAME)
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self):
        self.widget.destroy()
        self.window.destroy()
        self.controller.destroy()
        self._stack.close()
        await omni.kit.app.get_app().next_update_async()

    def _asset_paths(self) -> list[str]:
        return [entry.path for entry in self.controller.assets]

    async def _add_assets(self, *paths: str) -> list[asset_list_module.ScatterAssetItem]:
        """Add assets through the controller and return the rows built for them."""
        for path in paths:
            self.controller.add_asset(path)
        await omni.kit.app.get_app().next_update_async()
        return self.widget.model.get_item_children(None)


class TestScatterAssetListWidget(_AssetListTestCase):
    """Tests adding, browsing and removing assets through the palette widget, and its cleanup."""

    async def test_open_file_picker_when_asset_accepted_appends_row_with_file_name(self):
        # Arrange
        def pick(_title, _asset_type, callback, _callback_cancel, **_kwargs):
            callback(_ROCK)

        with mock.patch.object(asset_list_module, "open_asset_file_picker", side_effect=pick) as picker_mock:
            # Act
            self.widget._open_file_picker()
            await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(self._asset_paths(), [_ROCK])
        rows = self.widget.model.get_item_children(None)
        self.assertEqual([row.path for row in rows], [_ROCK])
        self.assertEqual(rows[0].display_name, "rock.usd")
        self.assertEqual(picker_mock.call_args.args[1], asset_list_module.ReplacementAssetType.MESH)

    async def test_accept_when_validation_pending_adds_nothing(self):
        # Arrange
        self.accept_mock.side_effect = _never_accept

        # Act
        self.widget._accept(_ROCK)
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(self._asset_paths(), [])
        self.assertEqual(self.widget.model.get_item_children(None), [])

    async def test_accept_passes_edit_layer_core_and_context_to_validation(self):
        # Arrange
        edit_layer = self.usd_context.get_stage().GetEditTarget().GetLayer()

        # Act
        self.widget._accept(_ROCK)

        # Assert
        self.accept_mock.assert_called_once()
        args, kwargs = self.accept_mock.call_args
        self.assertEqual(args[0], _ROCK)
        self.assertEqual(args[1], edit_layer)
        self.assertIs(args[2], self.widget._asset_core)
        self.assertIs(args[3], self.usd_context)
        self.assertEqual(kwargs["accept_handler"], self.widget._add_accepted_asset)

    async def test_accept_when_no_stage_posts_error_without_validation(self):
        # Arrange
        status = mock.Mock()
        status_sub = self.controller.subscribe_status_message(status)
        stageless_context = mock.Mock()
        stageless_context.get_stage.return_value = None

        # Act
        with mock.patch.object(asset_list_module.omni.usd, "get_context", return_value=stageless_context):
            self.widget._accept(_ROCK)

        # Assert
        self.accept_mock.assert_not_called()
        status.assert_called_once_with("Open a project before adding assets to the brush", True)
        self.assertEqual(self._asset_paths(), [])
        self.assertIsNotNone(status_sub)

    async def test_accept_when_crate_version_unsupported_posts_error_and_adds_nothing(self):
        # Arrange
        self.crate_mock.return_value = False
        status = mock.Mock()
        status_sub = self.controller.subscribe_status_message(status)

        # Act
        self.widget._accept(_ROCK)
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.crate_mock.assert_called_once_with(_ROCK)
        status.assert_called_once_with(f"{_ROCK} was saved with a USD crate version this build cannot read", True)
        self.assertEqual(self._asset_paths(), [])
        self.assertEqual(self.widget.model.get_item_children(None), [])
        self.assertIsNotNone(status_sub)

    async def test_remove_selected_removes_only_selected_rows(self):
        # Arrange
        rows = await self._add_assets(_ROCK, _GRASS)
        self.widget._tree_view.selection = [rows[0]]

        # Act
        self.widget._remove_selected()
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(self._asset_paths(), [_GRASS])
        self.assertEqual([row.path for row in self.widget.model.get_item_children(None)], [_GRASS])

    async def test_selection_change_enables_remove_button_only_with_selection(self):
        # Arrange
        rows = await self._add_assets(_ROCK)
        cases = [([], False), (rows, True)]
        for selection, expected in cases:
            with self.subTest(title=f"selection={len(selection)}"):
                # Arrange
                self.widget._remove_button.enabled = not expected

                # Act
                self.widget._on_selection_changed(selection)

                # Assert
                self.assertEqual(self.widget._remove_button.enabled, expected)

    async def test_add_from_selection_when_reference_resolved_accepts_resolved_path(self):
        # Arrange
        with mock.patch.object(
            asset_list_module, "resolve_reference_asset_from_selection", return_value=_GRASS
        ) as resolve_mock:
            # Act
            self.widget._add_from_selection()
            await omni.kit.app.get_app().next_update_async()

        # Assert
        resolve_mock.assert_called_once_with(self.usd_context)
        self.assertEqual(self._asset_paths(), [_GRASS])

    async def test_add_from_selection_when_nothing_resolved_posts_error_status(self):
        # Arrange
        status = mock.Mock()
        status_sub = self.controller.subscribe_status_message(status)

        with mock.patch.object(asset_list_module, "resolve_reference_asset_from_selection", return_value=None):
            # Act
            self.widget._add_from_selection()

        # Assert
        status.assert_called_once_with("Select a prim that references an ingested asset", True)
        self.accept_mock.assert_not_called()
        self.assertEqual(self._asset_paths(), [])
        self.assertIsNotNone(status_sub)

    async def test_refresh_ingested_models_lists_placeholder_then_file_names(self):
        # Arrange
        self.list_models_mock.return_value = [_ROCK, _GRASS]

        # Act
        self.widget.refresh_ingested_models()

        # Assert
        self.list_models_mock.assert_called_with(_CONTEXT_NAME)
        self.assertEqual(self.widget._browse_model.choices, [_BROWSE_PLACEHOLDER, "rock.usd", "grass.usd"])
        self.assertEqual(self.widget._browse_model.index_model.as_int, 0)
        self.assertEqual(self._asset_paths(), [])

    async def test_refresh_ingested_models_with_unchanged_listing_keeps_combo_entries(self):
        # Arrange
        self.list_models_mock.return_value = [_ROCK]
        self.widget.refresh_ingested_models()
        set_choices = mock.Mock(wraps=self.widget._browse_model.set_choices)

        with mock.patch.object(self.widget._browse_model, "set_choices", set_choices):
            # Act
            self.widget.refresh_ingested_models()

        # Assert
        set_choices.assert_not_called()
        self.assertEqual(self.widget._browse_model.choices, [_BROWSE_PLACEHOLDER, "rock.usd"])

    async def test_browse_combo_press_defers_listing_until_next_frame(self):
        # Arrange
        self.list_models_mock.reset_mock()
        self.list_models_mock.return_value = [_ROCK]

        # Act
        self.widget._browse_combo.call_mouse_pressed_fn(0.0, 0.0, 0, 0)
        listed_during_press = self.list_models_mock.called
        # The re-list runs from a task that itself waits one update, so let two updates pass.
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertFalse(listed_during_press)
        self.list_models_mock.assert_called_once_with(_CONTEXT_NAME)
        self.assertEqual(self.widget._browse_model.choices, [_BROWSE_PLACEHOLDER, "rock.usd"])

    async def test_browse_combo_pressed_twice_before_next_frame_lists_once(self):
        # Arrange
        self.list_models_mock.reset_mock()
        self.widget._browse_combo.call_mouse_pressed_fn(0.0, 0.0, 0, 0)

        # Act
        self.widget._browse_combo.call_mouse_pressed_fn(0.0, 0.0, 0, 0)
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(self.list_models_mock.call_count, 1)

    async def test_browse_selection_adds_chosen_model_and_returns_to_placeholder(self):
        # Arrange
        self.list_models_mock.return_value = [_ROCK, _GRASS]
        self.widget.refresh_ingested_models()

        # Act
        self.widget._browse_model.index_model.set_value(2)
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(self._asset_paths(), [_GRASS])
        self.assertEqual(self.widget._browse_model.index_model.as_int, 0)

    async def test_stage_opened_event_refreshes_ingested_models(self):
        # Arrange
        self.list_models_mock.return_value = [_ROCK]
        self.list_models_mock.reset_mock()

        # Act
        await self.usd_context.new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.list_models_mock.assert_called_with(_CONTEXT_NAME)
        self.assertEqual(self.widget._browse_model.choices, [_BROWSE_PLACEHOLDER, "rock.usd"])

    async def test_destroy_drops_subscriptions_rows_and_delegate(self):
        # Arrange
        await self._add_assets(_ROCK)
        model = self.widget.model

        # Act
        self.widget.destroy()

        # Assert
        self.assertIsNone(model._assets_changed_sub)
        self.assertEqual(model.get_item_children(None), [])
        self.assertIsNone(self.widget._stage_event_sub)
        self.assertIsNone(self.widget._tree_view)
        self.assertIsNone(self.widget._delegate)

    async def test_destroy_with_pending_browse_refresh_cancels_it(self):
        # Arrange
        self.widget._browse_combo.call_mouse_pressed_fn(0.0, 0.0, 0, 0)
        task = self.widget._browse_refresh_task
        self.list_models_mock.reset_mock()

        # Act
        self.widget.destroy()
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertTrue(task.cancelled())
        self.list_models_mock.assert_not_called()
        self.assertIsNone(self.widget._browse_refresh_task)


class TestScatterAssetModel(_AssetListTestCase):
    """Tests the tree model's write-back to the controller and the way it mirrors palette changes."""

    async def test_weight_change_updates_controller_weight(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        # Act
        row.weight_model.set_value(4.5)

        # Assert
        self.assertEqual(self.controller.assets[0].weight, 4.5)
        self.assertEqual(row.weight_model.as_float, 4.5)

    async def test_enabled_toggle_updates_controller(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        # Act
        row.enabled_model.set_value(False)

        # Assert
        self.assertFalse(self.controller.assets[0].enabled)

    async def test_up_axis_selection_updates_controller(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        # Act
        row.up_axis_model.index_model.set_value(list(UpAxis).index(UpAxis.Y))

        # Assert
        self.assertEqual(self.controller.assets[0].up_axis, UpAxis.Y)

    async def test_row_edit_when_controller_rejects_restores_controller_value(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]
        self.controller.set_asset_enabled = mock.Mock(return_value=False)

        # Act
        row.enabled_model.set_value(False)

        # Assert
        self.assertTrue(row.enabled_model.as_bool)
        self.controller.set_asset_enabled.assert_called_once_with(_ROCK, False)

    async def test_refresh_when_controller_values_change_updates_rows_in_place(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        # Act
        self.controller.set_asset_weight(_ROCK, 7.0)

        # Assert
        self.assertIs(self.widget.model.get_item_children(None)[0], row)
        self.assertEqual(row.weight_model.as_float, 7.0)
        self.assertEqual(row.entry.weight, 7.0)

    async def test_refresh_when_asset_added_rebuilds_rows_in_palette_order(self):
        # Arrange
        await self._add_assets(_ROCK)

        # Act
        self.controller.add_asset(_GRASS)

        # Assert
        rows = self.widget.model.get_item_children(None)
        self.assertEqual([row.path for row in rows], [_ROCK, _GRASS])
        self.assertEqual(rows[1].display_name, "grass.usd")

    async def test_refresh_when_rows_rebuilt_destroys_previous_rows(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        with mock.patch.object(row, "destroy", wraps=row.destroy) as destroy_mock:
            # Act
            self.controller.add_asset(_GRASS)

        # Assert
        destroy_mock.assert_called_once_with()
        self.assertNotIn(row, self.widget.model.get_item_children(None))

    async def test_get_item_value_model_returns_row_models_per_column(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]
        model = self.widget.model

        # Act
        column_models = [model.get_item_value_model(row, column) for column in range(4)]

        # Assert
        self.assertEqual(model.get_item_value_model_count(None), 4)
        self.assertIs(column_models[0], row.enabled_model)
        self.assertEqual(column_models[1].as_string, _ROCK)
        self.assertIs(column_models[2], row.weight_model)
        self.assertIs(column_models[3], row.up_axis_model.index_model)
        self.assertIsNone(model.get_item_value_model(None, 0))

    async def test_destroy_destroys_rows_and_drops_controller_subscription(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]
        model = self.widget.model

        with mock.patch.object(row, "destroy", wraps=row.destroy) as destroy_mock:
            # Act
            model.destroy()

        # Assert
        destroy_mock.assert_called_once_with()
        self.assertIsNone(model._assets_changed_sub)
        self.assertEqual(model.get_item_children(None), [])


class TestScatterAssetDelegate(_AssetListTestCase):
    """Tests the row cells the delegate builds inside the tree."""

    async def test_build_rows_gives_each_cell_an_identifier_indexed_by_row(self):
        # Arrange
        await self._add_assets(_ROCK, _GRASS)
        await omni.kit.app.get_app().next_update_async()
        expected = [
            ("CheckBox", "scatter_asset_row_0_enabled"),
            ("Label", "scatter_asset_row_0_path"),
            ("FloatBoundedDrag", "scatter_asset_row_0_weight"),
            ("ComboBox", "scatter_asset_row_0_up_axis"),
            ("Label", "scatter_asset_row_1_path"),
        ]

        # Act
        missing = [
            identifier
            for widget_type, identifier in expected
            if ui_test.find(f"{self.window.title}//Frame/**/{widget_type}[*].identifier=='{identifier}'") is None
        ]

        # Assert
        self.assertEqual(missing, [])

    async def test_build_weight_cell_disables_batch_edit_and_hands_field_to_row(self):
        # Arrange
        drag_class = mock.Mock(side_effect=FloatBoundedDrag)

        with mock.patch.object(asset_list_module, "FloatBoundedDrag", drag_class):
            # Act
            self.controller.add_asset(_ROCK)
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()

        # Assert
        row = self.widget.model.get_item_children(None)[0]
        drag_class.assert_called_once()
        self.assertIs(drag_class.call_args.kwargs["enable_batch_edit"], False)
        self.assertIs(drag_class.call_args.kwargs["model"], row.weight_model)
        self.assertIsInstance(row._weight_field, FloatBoundedDrag)

    async def test_weight_change_beyond_hard_bound_is_clamped_by_field(self):
        # Arrange
        row = (await self._add_assets(_ROCK))[0]

        # Act
        row.weight_model.set_value(500.0)

        # Assert
        self.assertEqual(row.weight_model.as_float, 100.0)
        self.assertEqual(self.controller.assets[0].weight, 100.0)


class TestScatterAssetItem(omni.kit.test.AsyncTestCase):
    """Tests the row item: model seeding, in-place updates and ownership of the weight drag field."""

    async def test_init_seeds_models_from_entry(self):
        # Arrange
        entry = ScatterAssetEntry(path=_ROCK, enabled=False, weight=2.5, up_axis=UpAxis.Y)

        # Act
        item = asset_list_module.ScatterAssetItem(entry)

        # Assert
        self.assertFalse(item.enabled_model.as_bool)
        self.assertEqual(item.weight_model.as_float, 2.5)
        self.assertEqual(item.up_axis_model.current_choice, UpAxis.Y.value)
        self.assertEqual(item.display_name, "rock.usd")

    async def test_update_with_changed_values_pushes_them_into_models(self):
        # Arrange
        item = asset_list_module.ScatterAssetItem(ScatterAssetEntry(path=_ROCK))

        # Act
        item.update(ScatterAssetEntry(path=_ROCK, enabled=False, weight=3.0, up_axis=UpAxis.Y))

        # Assert
        self.assertFalse(item.enabled_model.as_bool)
        self.assertEqual(item.weight_model.as_float, 3.0)
        self.assertEqual(item.up_axis_model.index_model.as_int, list(UpAxis).index(UpAxis.Y))
        self.assertEqual(item.entry.weight, 3.0)

    async def test_update_with_same_values_does_not_notify_models(self):
        # Arrange
        entry = ScatterAssetEntry(path=_ROCK, weight=3.0)
        item = asset_list_module.ScatterAssetItem(entry)
        weight_changed = mock.Mock()
        subscription = item.weight_model.subscribe_value_changed_fn(weight_changed)

        # Act
        item.update(entry.model_copy())

        # Assert
        weight_changed.assert_not_called()
        self.assertIsNotNone(subscription)

    async def test_attach_weight_field_twice_destroys_previous_field(self):
        # Arrange
        item = asset_list_module.ScatterAssetItem(ScatterAssetEntry(path=_ROCK))
        first_field = mock.Mock()
        item.attach_weight_field(first_field)

        # Act
        item.attach_weight_field(mock.Mock())

        # Assert
        first_field.destroy.assert_called_once_with()

    async def test_destroy_destroys_attached_weight_field(self):
        # Arrange
        item = asset_list_module.ScatterAssetItem(ScatterAssetEntry(path=_ROCK))
        field = mock.Mock()
        item.attach_weight_field(field)

        # Act
        item.destroy()

        # Assert
        field.destroy.assert_called_once_with()
        self.assertIsNone(item._weight_field)
