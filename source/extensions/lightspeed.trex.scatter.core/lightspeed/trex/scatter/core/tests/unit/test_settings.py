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

__all__ = ["TestAssetsCarbPersistence", "TestScatterAssetEntry", "TestScatterBrushSettings"]

import json
from unittest import mock

import omni.kit.test
from pydantic import ValidationError

from lightspeed.trex.scatter.core import constants as constants_module
from lightspeed.trex.scatter.core import settings as settings_module


def _make_settings_interface(initial: dict[str, str] | None = None) -> tuple[mock.MagicMock, dict[str, str]]:
    """Build a carb settings stand-in backed by a plain dictionary."""
    store: dict[str, str] = dict(initial or {})
    interface = mock.MagicMock()

    def _set(path: str, value: str) -> None:
        store[path] = value

    interface.set.side_effect = _set
    interface.get.side_effect = store.get
    return interface, store


class TestScatterBrushSettings(omni.kit.test.AsyncTestCase):
    async def test_init_without_arguments_uses_contract_defaults(self):
        # Arrange
        expected = {
            "preset_name": "Default",
            "radius": 50.0,
            "falloff": settings_module.Falloff.SMOOTH,
            "density": 8.0,
            "strength": 1.0,
            "stamp_spacing": 25.0,
            "padding": 10.0,
            "vertical_offset": 0.0,
            "conform_to_surface": True,
            "align_to_stroke": False,
            "rotation_x_min": 0.0,
            "rotation_x_max": 0.0,
            "rotation_y_min": 0.0,
            "rotation_y_max": 0.0,
            "rotation_z_min": 0.0,
            "rotation_z_max": 360.0,
            "scale_enabled": True,
            "scale_uniform": True,
            "scale_min": 0.8,
            "scale_max": 1.2,
            "scale_x_min": 0.8,
            "scale_x_max": 1.2,
            "scale_y_min": 0.8,
            "scale_y_max": 1.2,
            "scale_z_min": 0.8,
            "scale_z_max": 1.2,
            "scale_bias": 0.0,
            "scale_weight": 1.0,
            "seed": 0,
            "randomize_seed": True,
            "apply_to": settings_module.ApplyTo.ALL,
            "target_mode": settings_module.TargetMode.HIT_SURFACE,
            "anchor_prototype_path": "",
            "erase_scope": settings_module.EraseScope.ALL_SCATTERED,
            "flood_max_instances": 300,
        }

        # Act
        settings = settings_module.ScatterBrushSettings()

        # Assert
        self.assertEqual({name: getattr(settings, name) for name in expected}, expected)

    async def test_init_with_radius_out_of_range_raises_validation_error(self):
        for radius in (0.5, 10001.0):
            with self.subTest(title=f"radius={radius}"):
                # Arrange
                kwargs = {"radius": radius}

                # Act / Assert
                with self.assertRaises(ValidationError):
                    settings_module.ScatterBrushSettings(**kwargs)

    async def test_init_with_min_greater_than_max_raises_validation_error(self):
        # Arrange
        kwargs = {"scale_min": 2.0, "scale_max": 1.0}

        # Act / Assert
        with self.assertRaises(ValidationError):
            settings_module.ScatterBrushSettings(**kwargs)

    async def test_assign_min_greater_than_max_raises_validation_error(self):
        # Arrange
        settings = settings_module.ScatterBrushSettings(rotation_z_min=10.0, rotation_z_max=20.0)

        # Act / Assert
        with self.assertRaises(ValidationError):
            settings.rotation_z_min = 30.0

    async def test_from_json_dict_with_to_json_dict_output_round_trips(self):
        # Arrange
        original = settings_module.ScatterBrushSettings(
            preset_name="Grass",
            radius=75.0,
            falloff=settings_module.Falloff.LINEAR,
            density=3.5,
            rotation_z_max=180.0,
            apply_to=settings_module.ApplyTo.SELECTED,
            erase_scope=settings_module.EraseScope.BRUSH_ASSETS,
            seed=42,
        )
        data = original.to_json_dict()

        # Act
        restored = settings_module.ScatterBrushSettings.from_json_dict(data)

        # Assert
        self.assertEqual(restored, original)
        self.assertEqual(data["falloff"], "LINEAR")
        self.assertEqual(data["apply_to"], "SELECTED")

    async def test_from_json_dict_with_unknown_keys_ignores_them_and_fills_defaults(self):
        # Arrange
        data = {"radius": 12.0, "bogus_key": 1, "another": "value"}

        # Act
        settings = settings_module.ScatterBrushSettings.from_json_dict(data)

        # Assert
        self.assertEqual(settings.radius, 12.0)
        self.assertEqual(settings.density, 8.0)
        self.assertEqual(settings.preset_name, "Default")
        self.assertFalse(hasattr(settings, "bogus_key"))

    async def test_slug_sanitizes_preset_name_for_each_input(self):
        cases = [
            ("My Grass!", "my_grass"),
            ("", "default"),
            ("1x", "p_1x"),
            ("Rocks - Large", "rocks_large"),
        ]
        for preset_name, expected in cases:
            with self.subTest(title=f"preset_name={preset_name!r}"):
                # Arrange
                settings = settings_module.ScatterBrushSettings(preset_name=preset_name)

                # Act
                result = settings.slug()

                # Assert
                self.assertEqual(result, expected)

    async def test_save_to_carb_writes_json_to_brush_settings_path(self):
        # Arrange
        interface, store = _make_settings_interface()
        settings = settings_module.ScatterBrushSettings(radius=99.0)

        # Act
        settings.save_to_carb(interface)

        # Assert
        interface.set.assert_called_once()
        self.assertEqual(json.loads(store[constants_module.BRUSH_SETTINGS_PATH]), settings.to_json_dict())

    async def test_load_from_carb_after_save_to_carb_round_trips(self):
        # Arrange
        interface, _ = _make_settings_interface()
        original = settings_module.ScatterBrushSettings(radius=99.0, density=2.0, preset_name="Round Trip")
        original.save_to_carb(interface)

        # Act
        loaded = settings_module.ScatterBrushSettings.load_from_carb(interface)

        # Assert
        self.assertEqual(loaded, original)

    async def test_load_from_carb_with_invalid_json_returns_defaults(self):
        # Arrange
        interface, _ = _make_settings_interface({constants_module.BRUSH_SETTINGS_PATH: "{not valid json"})

        # Act
        loaded = settings_module.ScatterBrushSettings.load_from_carb(interface)

        # Assert
        self.assertEqual(loaded, settings_module.ScatterBrushSettings())

    async def test_load_from_carb_with_out_of_range_value_returns_defaults(self):
        # Arrange
        interface, _ = _make_settings_interface({constants_module.BRUSH_SETTINGS_PATH: json.dumps({"radius": -1})})

        # Act
        loaded = settings_module.ScatterBrushSettings.load_from_carb(interface)

        # Assert
        self.assertEqual(loaded.radius, 50.0)

    async def test_load_from_carb_with_missing_value_returns_defaults(self):
        # Arrange
        interface, _ = _make_settings_interface()

        # Act
        loaded = settings_module.ScatterBrushSettings.load_from_carb(interface)

        # Assert
        self.assertEqual(loaded, settings_module.ScatterBrushSettings())


class TestScatterAssetEntry(omni.kit.test.AsyncTestCase):
    async def test_init_with_path_only_uses_defaults(self):
        # Arrange
        path = "C:/project/assets/ingested/rock.usd"

        # Act
        entry = settings_module.ScatterAssetEntry(path=path)

        # Assert
        self.assertEqual(
            (entry.path, entry.enabled, entry.weight, entry.up_axis),
            (path, True, 1.0, settings_module.UpAxis.Z),
        )

    async def test_init_with_weight_out_of_range_raises_validation_error(self):
        for weight in (-0.1, 100.5):
            with self.subTest(title=f"weight={weight}"):
                # Arrange
                kwargs = {"path": "rock.usd", "weight": weight}

                # Act / Assert
                with self.assertRaises(ValidationError):
                    settings_module.ScatterAssetEntry(**kwargs)

    async def test_init_with_empty_path_raises_validation_error(self):
        # Arrange
        kwargs = {"path": ""}

        # Act / Assert
        with self.assertRaises(ValidationError):
            settings_module.ScatterAssetEntry(**kwargs)


class TestAssetsCarbPersistence(omni.kit.test.AsyncTestCase):
    async def test_load_assets_from_carb_after_save_assets_to_carb_round_trips(self):
        # Arrange
        interface, _ = _make_settings_interface()
        assets = [
            settings_module.ScatterAssetEntry(path="rock.usd", weight=2.0, up_axis=settings_module.UpAxis.Y),
            settings_module.ScatterAssetEntry(path="grass.usd", enabled=False),
        ]
        settings_module.save_assets_to_carb(assets, interface)

        # Act
        loaded = settings_module.load_assets_from_carb(interface)

        # Assert
        self.assertEqual(loaded, assets)

    async def test_save_assets_to_carb_writes_json_list_to_assets_path(self):
        # Arrange
        interface, store = _make_settings_interface()
        assets = [settings_module.ScatterAssetEntry(path="rock.usd")]

        # Act
        settings_module.save_assets_to_carb(assets, interface)

        # Assert
        self.assertEqual(
            json.loads(store[constants_module.ASSETS_SETTING_PATH]),
            [{"path": "rock.usd", "enabled": True, "weight": 1.0, "up_axis": "Z"}],
        )

    async def test_load_assets_from_carb_with_invalid_json_returns_empty_list(self):
        # Arrange
        interface, _ = _make_settings_interface({constants_module.ASSETS_SETTING_PATH: "[not valid json"})

        # Act
        loaded = settings_module.load_assets_from_carb(interface)

        # Assert
        self.assertEqual(loaded, [])

    async def test_load_assets_from_carb_with_invalid_entry_returns_empty_list(self):
        # Arrange
        interface, _ = _make_settings_interface({constants_module.ASSETS_SETTING_PATH: json.dumps([{"weight": 1}])})

        # Act
        loaded = settings_module.load_assets_from_carb(interface)

        # Assert
        self.assertEqual(loaded, [])

    async def test_load_assets_from_carb_with_missing_value_returns_empty_list(self):
        # Arrange
        interface, _ = _make_settings_interface()

        # Act
        loaded = settings_module.load_assets_from_carb(interface)

        # Assert
        self.assertEqual(loaded, [])
