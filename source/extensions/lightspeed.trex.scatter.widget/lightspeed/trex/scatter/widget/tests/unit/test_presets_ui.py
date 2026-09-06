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

__all__ = ["TestPresetsWidget"]

import contextlib
import tempfile
from pathlib import Path
from unittest import mock

import carb.settings
import omni.kit.app
import omni.kit.test
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core import settings as settings_module
from lightspeed.trex.scatter.widget import presets_ui as presets_ui_module
from omni import ui

_PLACEHOLDER_NAME = "Default"
_TEST_SETTINGS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/presets_ui/brushSettings"
_TEST_ASSETS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/presets_ui/assets"


def _isolated_controller(stack: contextlib.ExitStack, presets_dir: Path) -> controller_module.ScatterBrushController:
    """Build a controller that persists to throwaway carb paths and a temporary preset directory."""
    stack.enter_context(mock.patch.object(settings_module, "BRUSH_SETTINGS_PATH", _TEST_SETTINGS_PATH))
    stack.enter_context(mock.patch.object(settings_module, "ASSETS_SETTING_PATH", _TEST_ASSETS_PATH))
    stack.enter_context(mock.patch.object(controller_module, "get_default_presets_directory", return_value=presets_dir))
    settings = carb.settings.get_settings()
    stack.callback(settings.destroy_item, _TEST_SETTINGS_PATH)
    stack.callback(settings.destroy_item, _TEST_ASSETS_PATH)
    return controller_module.ScatterBrushController()


class TestPresetsWidget(omni.kit.test.AsyncTestCase):
    """Tests the preset picker and its actions against a real window and a temporary preset store."""

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        temp_dir = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.presets_dir = Path(temp_dir) / "presets"
        self.controller = _isolated_controller(self._stack, self.presets_dir)
        self._stack.enter_context(
            mock.patch.object(presets_ui_module, "get_scatter_brush_controller", return_value=self.controller)
        )
        self.window = ui.Window(f"ScatterPresets_{self._testMethodName}", width=400, height=200)
        self.widget: presets_ui_module.PresetsWidget | None = None

    async def tearDown(self):
        if self.widget is not None:
            self.widget.destroy()
        self.window.destroy()
        self.controller.destroy()
        self._stack.close()
        await omni.kit.app.get_app().next_update_async()

    async def _build_widget(self) -> presets_ui_module.PresetsWidget:
        """Build the widget inside the test window and let it render once."""
        with self.window.frame:
            self.widget = presets_ui_module.PresetsWidget()
        await omni.kit.app.get_app().next_update_async()
        return self.widget

    def _store_presets(self, *names: str) -> None:
        """Save the current settings under each name; the last one becomes the current preset."""
        for name in names:
            self.controller.save_preset(name)

    async def test_build_lists_stored_presets_with_current_selected(self):
        # Arrange
        self._store_presets("Rocks", "Grass")

        # Act
        widget = await self._build_widget()

        # Assert
        self.assertEqual(widget.names_model.choices, ["Grass", "Rocks"])
        self.assertEqual(widget.names_model.current_choice, "Grass")
        self.assertTrue(widget._delete_button.enabled)

    async def test_build_when_current_preset_not_stored_shows_it_first_and_disables_file_actions(self):
        # Arrange
        self._store_presets("Rocks")
        self.controller.update_settings(preset_name="Scratch")

        # Act
        widget = await self._build_widget()

        # Assert
        self.assertEqual(widget.names_model.choices, ["Scratch", "Rocks"])
        self.assertEqual(widget.names_model.current_choice, "Scratch")
        self.assertFalse(widget._rename_button.enabled)
        self.assertFalse(widget._clone_button.enabled)
        self.assertFalse(widget._delete_button.enabled)

    async def test_build_without_presets_shows_default_name(self):
        # Arrange

        # Act
        widget = await self._build_widget()

        # Assert
        self.assertEqual(widget.names_model.choices, [_PLACEHOLDER_NAME])
        self.assertEqual(widget.names_model.current_choice, _PLACEHOLDER_NAME)

    async def test_select_preset_applies_its_settings(self):
        # Arrange
        self.controller.update_settings(radius=77.0)
        self._store_presets("Rocks")
        self.controller.update_settings(radius=10.0)
        self._store_presets("Grass")
        widget = await self._build_widget()

        # Act
        widget.names_model.index_model.set_value(widget.names_model.choices.index("Rocks"))

        # Assert
        self.assertEqual(self.controller.settings.preset_name, "Rocks")
        self.assertEqual(self.controller.settings.radius, 77.0)
        self.assertEqual(widget.names_model.current_choice, "Rocks")

    async def test_select_preset_when_load_fails_restores_current_selection(self):
        # Arrange
        self.presets_dir.mkdir(parents=True)
        (self.presets_dir / "Broken.json").write_text("not json", encoding="utf-8")
        widget = await self._build_widget()
        status = mock.Mock()
        status_sub = self.controller.subscribe_status_message(status)

        # Act
        widget.names_model.index_model.set_value(widget.names_model.choices.index("Broken"))

        # Assert
        self.assertEqual(self.controller.settings.preset_name, _PLACEHOLDER_NAME)
        self.assertEqual(widget.names_model.current_choice, _PLACEHOLDER_NAME)
        self.assertTrue(status.call_args.args[1])
        self.assertIsNotNone(status_sub)

    async def test_save_clicked_stores_current_preset_and_enables_file_actions(self):
        # Arrange
        widget = await self._build_widget()

        # Act
        widget._save()

        # Assert
        self.assertEqual(self.controller.preset_names(), [_PLACEHOLDER_NAME])
        self.assertEqual(widget.names_model.choices, [_PLACEHOLDER_NAME])
        self.assertTrue(widget._rename_button.enabled)
        self.assertTrue(widget._delete_button.enabled)

    async def test_save_as_opens_prompt_prefilled_with_current_name(self):
        # Arrange
        widget = await self._build_widget()

        # Act
        widget._save_as()
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual(widget._name_prompt.name, _PLACEHOLDER_NAME)
        self.assertTrue(widget._name_prompt._ok_button.enabled)
        self.assertEqual(widget._name_prompt._field.identifier, "scatter_preset_name_field")

    async def test_save_as_prompt_accepted_stores_settings_under_typed_name(self):
        # Arrange
        widget = await self._build_widget()
        widget._save_as()
        await omni.kit.app.get_app().next_update_async()
        prompt = widget._name_prompt
        prompt._field.model.set_value("  Pebbles ")

        # Act
        prompt._accept()

        # Assert
        self.assertEqual(self.controller.preset_names(), ["Pebbles"])
        self.assertEqual(self.controller.settings.preset_name, "Pebbles")
        self.assertEqual(widget.names_model.current_choice, "Pebbles")
        self.assertIsNone(prompt._window)

    async def test_save_as_prompt_accepted_hides_window_now_and_destroys_it_next_frame(self):
        # Arrange
        widget = await self._build_widget()
        widget._save_as()
        await omni.kit.app.get_app().next_update_async()
        self.assertIsNotNone(ui.Workspace.get_window("Save Preset As"))

        # Act
        widget._name_prompt._accept()
        hidden_after_accept = not ui.Workspace.get_window("Save Preset As").visible
        # The window is destroyed by a task that itself waits one update, so let two updates pass.
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertTrue(hidden_after_accept)
        self.assertIsNone(ui.Workspace.get_window("Save Preset As"))

    async def test_save_as_prompt_when_name_blank_keeps_prompt_open_and_disables_ok(self):
        # Arrange
        widget = await self._build_widget()
        widget._save_as()
        await omni.kit.app.get_app().next_update_async()
        prompt = widget._name_prompt
        prompt._field.model.set_value("   ")

        # Act
        prompt._accept()

        # Assert
        self.assertFalse(prompt._ok_button.enabled)
        self.assertIsNotNone(prompt._window)
        self.assertEqual(self.controller.preset_names(), [])

    async def test_prompt_opened_twice_closes_previous_prompt(self):
        # Arrange
        widget = await self._build_widget()
        widget._save_as()
        first_prompt = widget._name_prompt

        # Act
        widget._rename()

        # Assert
        self.assertIsNone(first_prompt._window)
        self.assertIsNot(widget._name_prompt, first_prompt)

    async def test_rename_prompt_accepted_renames_current_preset(self):
        # Arrange
        self._store_presets("Rocks")
        widget = await self._build_widget()
        widget._rename()
        prompt = widget._name_prompt
        prompt._field.model.set_value("Stones")

        # Act
        prompt._accept()

        # Assert
        self.assertEqual(self.controller.preset_names(), ["Stones"])
        self.assertEqual(self.controller.settings.preset_name, "Stones")
        self.assertEqual(widget.names_model.choices, ["Stones"])

    async def test_clone_prompt_accepted_copies_current_preset_and_keeps_it_selected(self):
        # Arrange
        self._store_presets("Rocks")
        widget = await self._build_widget()
        widget._clone()
        prompt = widget._name_prompt

        # Act
        prompt._accept()

        # Assert
        self.assertEqual(self.controller.preset_names(), ["Rocks", "Rocks Copy"])
        self.assertEqual(self.controller.settings.preset_name, "Rocks")
        self.assertEqual(widget.names_model.choices, ["Rocks", "Rocks Copy"])
        self.assertEqual(widget.names_model.current_choice, "Rocks")

    async def test_delete_clicked_asks_for_confirmation_before_deleting(self):
        # Arrange
        self._store_presets("Rocks")
        widget = await self._build_widget()

        with mock.patch.object(presets_ui_module, "TrexMessageDialog") as dialog_mock:
            # Act
            widget._delete()

        # Assert
        dialog_mock.assert_called_once()
        self.assertEqual(dialog_mock.call_args.kwargs["title"], "Delete Preset")
        self.assertIn("Rocks", dialog_mock.call_args.kwargs["message"])
        self.assertEqual(dialog_mock.call_args.kwargs["ok_handler"], widget._delete_current)
        self.assertEqual(self.controller.preset_names(), ["Rocks"])

    async def test_delete_confirmed_removes_preset_file_and_disables_file_actions(self):
        # Arrange
        self._store_presets("Rocks")
        widget = await self._build_widget()

        # Act
        widget._delete_current()

        # Assert
        self.assertEqual(self.controller.preset_names(), [])
        self.assertEqual(widget.names_model.choices, ["Rocks"])
        self.assertFalse(widget._delete_button.enabled)

    async def test_refresh_when_settings_preset_name_changes_selects_it(self):
        # Arrange
        self._store_presets("Rocks", "Grass")
        widget = await self._build_widget()

        # Act
        self.controller.update_settings(preset_name="Rocks")

        # Assert
        self.assertEqual(widget.names_model.current_choice, "Rocks")

    async def test_settings_change_keeping_preset_name_does_not_rebuild_choices(self):
        # Arrange
        self._store_presets("Rocks")
        widget = await self._build_widget()
        set_choices = mock.Mock(wraps=widget.names_model.set_choices)

        with mock.patch.object(widget.names_model, "set_choices", set_choices):
            # Act
            self.controller.update_settings(radius=42.0)

        # Assert
        set_choices.assert_not_called()
        self.assertEqual(widget.names_model.current_choice, "Rocks")

    async def test_destroy_drops_subscriptions_and_prompt(self):
        # Arrange
        widget = await self._build_widget()
        widget._save_as()
        prompt = widget._name_prompt

        # Act
        widget.destroy()

        # Assert
        self.assertIsNone(widget._settings_sub)
        self.assertIsNone(widget._index_sub)
        self.assertIsNone(widget._name_prompt)
        self.assertIsNone(prompt._window)
