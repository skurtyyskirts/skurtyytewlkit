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

__all__ = ["TestAssetUpAxisWidget"]

import contextlib
import tempfile
from pathlib import Path
from unittest import mock

import carb.settings
import omni.kit.app
import omni.kit.test
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core import settings as settings_module
from lightspeed.trex.scatter.core.settings import UpAxis
from lightspeed.trex.scatter.widget import asset_up_axis as asset_up_axis_module
from omni import ui
from omni.kit import ui_test

_ROCK = "C:/project/assets/ingested/rock.usd"
_GRASS = "C:/project/assets/ingested/grass.usd"
_TEST_SETTINGS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/asset_up_axis/brushSettings"
_TEST_ASSETS_PATH = "/exts/lightspeed.trex.scatter.widget.tests/asset_up_axis/assets"
# What each asset's file authors; None is a file that says nothing about its up axis.
_FILE_AXES = {_ROCK: UpAxis.Z, _GRASS: None}


def _isolated_controller(stack: contextlib.ExitStack, presets_dir: Path) -> controller_module.ScatterBrushController:
    """Build a controller that persists to throwaway carb paths and a temporary preset directory."""
    stack.enter_context(mock.patch.object(settings_module, "BRUSH_SETTINGS_PATH", _TEST_SETTINGS_PATH))
    stack.enter_context(mock.patch.object(settings_module, "ASSETS_SETTING_PATH", _TEST_ASSETS_PATH))
    stack.enter_context(mock.patch.object(controller_module, "get_default_presets_directory", return_value=presets_dir))
    settings = carb.settings.get_settings()
    stack.callback(settings.destroy_item, _TEST_SETTINGS_PATH)
    stack.callback(settings.destroy_item, _TEST_ASSETS_PATH)
    return controller_module.ScatterBrushController()


class TestAssetUpAxisWidget(omni.kit.test.AsyncTestCase):
    """The Asset Up Axis rows built inside a real window and bound to an isolated controller."""

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        temp_dir = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.controller = _isolated_controller(self._stack, Path(temp_dir) / "presets")
        self._stack.enter_context(
            mock.patch.object(asset_up_axis_module, "get_scatter_brush_controller", return_value=self.controller)
        )
        self.read_file_axis = self._stack.enter_context(
            mock.patch.object(asset_up_axis_module, "read_authored_asset_up_axis", side_effect=_FILE_AXES.get)
        )
        self.window_title = f"AssetUpAxis_{self._testMethodName}"
        self.window = ui.Window(self.window_title, width=500, height=300)
        with self.window.frame:
            self.widget = asset_up_axis_module.AssetUpAxisWidget(label_width=ui.Pixel(160))
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self):
        self.widget.destroy()
        self.window.destroy()
        self.controller.destroy()
        self._stack.close()

    def _find(self, widget_type: str, identifier: str):
        """Return the widget of the given class name and identifier, or None."""
        return ui_test.find(f"{self.window_title}//Frame/**/{widget_type}[*].identifier=='{identifier}'")

    def _file_label(self, row: int) -> ui.Label:
        return self._find("Label", f"scatter_up_axis_row_{row}_file").widget

    def _axis_model(self, row: int) -> ui.AbstractValueModel:
        return self._find("ComboBox", f"scatter_up_axis_row_{row}_axis").widget.model.get_item_value_model()

    async def test_build_with_an_empty_palette_shows_the_hint_and_no_rows(self):
        # Arrange
        # (setUp: the palette is empty)

        # Act
        hint = self._find("Label", "scatter_up_axis_empty")

        # Assert
        self.assertTrue(hint.widget.visible)
        self.assertIsNone(self._find("ComboBox", "scatter_up_axis_row_0_axis"))

    async def test_adding_assets_shows_what_each_file_authors(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        self.controller.add_asset(_GRASS, up_axis=UpAxis.Y)
        await omni.kit.app.get_app().next_update_async()

        # Assert
        self.assertEqual((self._file_label(0).text, self._file_label(1).text), ("File: Z-up", "File: not set"))
        self.assertFalse(self._find("Label", "scatter_up_axis_empty").widget.visible)

    async def test_a_row_disagreeing_with_its_file_says_the_asset_lies_on_its_side(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        await omni.kit.app.get_app().next_update_async()

        # Act
        self.controller.set_asset_up_axis(_ROCK, UpAxis.Y)

        # Assert
        label = self._file_label(0)
        self.assertEqual(label.text, "File: Z-up - set Z or it lies on its side")
        self.assertEqual(label.name, "Warning")

    async def test_a_row_matching_its_file_is_not_flagged(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Y)
        await omni.kit.app.get_app().next_update_async()

        # Act
        self.controller.set_asset_up_axis(_ROCK, UpAxis.Z)

        # Assert
        self.assertNotEqual(self._file_label(0).name, "Warning")

    async def test_choosing_an_axis_in_a_row_sets_the_palette_entry(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Y)
        await omni.kit.app.get_app().next_update_async()

        # Act
        self._axis_model(0).set_value(list(UpAxis).index(UpAxis.Z))

        # Assert
        self.assertEqual(self.controller.assets[0].up_axis, UpAxis.Z)

    async def test_setting_the_axis_from_the_palette_moves_the_row_choice(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        await omni.kit.app.get_app().next_update_async()

        # Act
        self.controller.set_asset_up_axis(_ROCK, UpAxis.Y)

        # Assert
        self.assertEqual(self._axis_model(0).as_int, list(UpAxis).index(UpAxis.Y))

    async def test_changing_a_weight_reads_no_file_again(self):
        # Arrange - a weight drag changes the palette at every step, so the rows must not reopen asset files for it
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        await omni.kit.app.get_app().next_update_async()
        self.read_file_axis.reset_mock()

        # Act
        self.controller.set_asset_weight(_ROCK, 3.0)

        # Assert
        self.read_file_axis.assert_not_called()
