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

__all__ = ["TestGetDefaultPresetsDirectory", "TestPresetStore"]

import contextlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import omni.kit.test

from lightspeed.trex.scatter.core import constants as constants_module
from lightspeed.trex.scatter.core import presets as presets_module
from lightspeed.trex.scatter.core.settings import ScatterBrushSettings


class TestPresetStore(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp_dir.name) / "presets"
        self.store = presets_module.PresetStore(self.directory)

    async def tearDown(self):
        self._temp_dir.cleanup()

    async def test_list_names_when_directory_missing_returns_empty_list(self):
        # Arrange
        store = presets_module.PresetStore(self.directory / "missing")

        # Act
        names = store.list_names()

        # Assert
        self.assertEqual(names, [])

    async def test_save_then_list_names_returns_sorted_saved_names(self):
        # Arrange
        self.store.save("Rocks", ScatterBrushSettings())
        self.store.save("Grass", ScatterBrushSettings())

        # Act
        names = self.store.list_names()

        # Assert
        self.assertEqual(names, ["Grass", "Rocks"])

    async def test_save_creates_directory_and_returns_json_path(self):
        # Arrange
        settings = ScatterBrushSettings(radius=77.0)

        # Act
        path = self.store.save("Grass", settings)

        # Assert
        self.assertEqual(path, self.directory / "Grass.json")
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["preset_name"], "Grass")

    async def test_exists_after_save_returns_true(self):
        # Arrange
        self.store.save("Grass", ScatterBrushSettings())

        # Act
        result = self.store.exists("Grass")

        # Assert
        self.assertTrue(result)

    async def test_exists_without_save_returns_false(self):
        # Arrange
        name = "Nothing"

        # Act
        result = self.store.exists(name)

        # Assert
        self.assertFalse(result)

    async def test_load_after_save_returns_saved_values(self):
        # Arrange
        self.store.save("Grass", ScatterBrushSettings(radius=77.0, density=2.5))

        # Act
        loaded = self.store.load("Grass")

        # Assert
        self.assertEqual((loaded.radius, loaded.density, loaded.preset_name), (77.0, 2.5, "Grass"))

    async def test_load_overwrites_preset_name_with_file_name(self):
        # Arrange
        self.directory.mkdir(parents=True)
        data = ScatterBrushSettings(preset_name="Other Name", radius=33.0).to_json_dict()
        (self.directory / "Grass.json").write_text(json.dumps(data), encoding="utf-8")

        # Act
        loaded = self.store.load("Grass")

        # Assert
        self.assertEqual((loaded.preset_name, loaded.radius), ("Grass", 33.0))

    async def test_load_with_corrupt_json_raises_value_error(self):
        # Arrange
        self.directory.mkdir(parents=True)
        (self.directory / "Bad.json").write_text("{not valid json", encoding="utf-8")

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.load("Bad")

    async def test_load_with_non_object_json_raises_value_error(self):
        # Arrange
        self.directory.mkdir(parents=True)
        (self.directory / "Bad.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.load("Bad")

    async def test_load_with_out_of_range_value_raises_value_error(self):
        # Arrange
        self.directory.mkdir(parents=True)
        (self.directory / "Bad.json").write_text(json.dumps({"radius": -5}), encoding="utf-8")

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.load("Bad")

    async def test_load_with_missing_preset_raises_value_error(self):
        # Arrange
        name = "Missing"

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.load(name)

    async def test_rename_moves_file_to_new_name(self):
        # Arrange
        self.store.save("Old", ScatterBrushSettings(radius=44.0))

        # Act
        self.store.rename("Old", "New")

        # Assert
        self.assertEqual(self.store.list_names(), ["New"])
        self.assertFalse((self.directory / "Old.json").exists())
        self.assertEqual(self.store.load("New").radius, 44.0)

    async def test_rename_with_missing_source_raises_value_error(self):
        # Arrange
        names = ("Missing", "New")

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.rename(*names)

    async def test_rename_onto_existing_preset_raises_value_error(self):
        # Arrange
        self.store.save("A", ScatterBrushSettings())
        self.store.save("B", ScatterBrushSettings())

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.rename("A", "B")

    async def test_clone_copies_settings_to_new_name(self):
        # Arrange
        self.store.save("Source", ScatterBrushSettings(radius=66.0))

        # Act
        self.store.clone("Source", "Copy")

        # Assert
        self.assertEqual(self.store.list_names(), ["Copy", "Source"])
        cloned = self.store.load("Copy")
        self.assertEqual((cloned.radius, cloned.preset_name), (66.0, "Copy"))

    async def test_clone_with_missing_source_raises_value_error(self):
        # Arrange
        names = ("Missing", "Copy")

        # Act / Assert
        with self.assertRaises(ValueError):
            self.store.clone(*names)

    async def test_delete_existing_preset_removes_file(self):
        # Arrange
        self.store.save("Grass", ScatterBrushSettings())

        # Act
        self.store.delete("Grass")

        # Assert
        self.assertEqual(self.store.list_names(), [])

    async def test_delete_missing_preset_is_noop(self):
        # Arrange
        self.store.save("Keep", ScatterBrushSettings())

        # Act
        self.store.delete("Missing")

        # Assert
        self.assertEqual(self.store.list_names(), ["Keep"])

    async def test_save_with_invalid_name_raises_value_error(self):
        for name in ("", "   ", "a/b", "a\\b", "..", ".", "bad:name"):
            with self.subTest(title=f"name={name!r}"):
                # Arrange
                settings = ScatterBrushSettings()

                # Act / Assert
                with self.assertRaises(ValueError):
                    self.store.save(name, settings)

    async def test_save_with_invalid_name_does_not_create_files(self):
        # Arrange
        settings = ScatterBrushSettings()
        for name in ("", "a/b", ".."):
            with contextlib.suppress(ValueError):
                self.store.save(name, settings)

        # Act
        names = self.store.list_names()

        # Assert
        self.assertEqual(names, [])


class TestGetDefaultPresetsDirectory(omni.kit.test.AsyncTestCase):
    async def test_get_default_presets_directory_with_setting_returns_setting_path(self):
        # Arrange
        settings_interface = mock.MagicMock()
        settings_interface.get.return_value = "C:/custom/presets"
        tokens = mock.MagicMock()

        with (
            mock.patch.object(presets_module.carb.settings, "get_settings", return_value=settings_interface),
            mock.patch.object(presets_module.carb.tokens, "get_tokens_interface", return_value=tokens),
        ):
            # Act
            result = presets_module.get_default_presets_directory()

        # Assert
        self.assertEqual(result, Path("C:/custom/presets"))
        settings_interface.get.assert_called_once_with(constants_module.PRESETS_DIR_SETTING)
        tokens.resolve.assert_not_called()

    async def test_get_default_presets_directory_without_setting_uses_app_documents_token(self):
        # Arrange
        settings_interface = mock.MagicMock()
        settings_interface.get.return_value = ""
        tokens = mock.MagicMock()
        tokens.resolve.return_value = "C:/Users/someone/Documents"

        with (
            mock.patch.object(presets_module.carb.settings, "get_settings", return_value=settings_interface),
            mock.patch.object(presets_module.carb.tokens, "get_tokens_interface", return_value=tokens),
        ):
            # Act
            result = presets_module.get_default_presets_directory()

        # Assert
        self.assertEqual(result, Path("C:/Users/someone/Documents") / constants_module.PRESETS_SUBDIRECTORY)
        tokens.resolve.assert_called_once_with("${app_documents}")
