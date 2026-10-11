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

__all__ = ["TestScatterBrushController", "TestScatterBrushControllerPreview", "TestScatterBrushControllerSingleton"]

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import omni.kit.commands
import omni.kit.test
import omni.kit.undo
from pxr import Gf, Sdf, Usd, UsdGeom

from lightspeed.trex.scatter.core import commands as commands_module
from lightspeed.trex.scatter.core import constants as constants_module
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core.commands import ScatterStrokeKind
from lightspeed.trex.scatter.core.placement import PlacementRecord
from lightspeed.trex.scatter.core.settings import (
    Falloff,
    ScatterAssetEntry,
    ScatterBrushSettings,
    ScatterMaterialOverride,
    ScatterTextureSet,
    TargetMode,
    UpAxis,
)
from lightspeed.trex.scatter.core.reapply import ReapplyScope, ScopeEntry, scope_of_placements
from lightspeed.trex.scatter.core.targets import ScatterTarget

_ROCK = "C:/project/assets/ingested/rock.usd"
_GRASS = "C:/project/assets/ingested/grass.usd"
_ANCHOR_HASH = "0123456789ABCDEF"
_ANCHOR_PROTOTYPE = f"/RootNode/meshes/mesh_{_ANCHOR_HASH}"


def _make_settings_interface() -> tuple[mock.MagicMock, dict[str, str]]:
    """Build a carb settings stand-in backed by a plain dictionary."""
    store: dict[str, str] = {}
    interface = mock.MagicMock()

    def _set(path: str, value: str) -> None:
        store[path] = value

    interface.set.side_effect = _set
    interface.get.side_effect = store.get
    return interface, store


def _make_record(name: str) -> mock.Mock:
    """Build a PlacementRecord-like value whose to_dict output is recognisable."""
    record = mock.Mock(name=name)
    record.to_dict.return_value = {"prim_name": name}
    return record


def _make_target(prototype_name: str) -> SimpleNamespace:
    """Build a ScatterTarget-like value for a prototype hash name."""
    hash_name = prototype_name.removeprefix("mesh_")
    return SimpleNamespace(
        prototype_root=Sdf.Path(f"/RootNode/meshes/{prototype_name}"),
        parent_instance_root=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0"),
        mesh_path=Sdf.Path(f"/RootNode/instances/inst_{hash_name}_0/mesh"),
        parent_world=Gf.Matrix4d(1.0),
        instance_count=1,
    )


def _make_usd_context(stage: Usd.Stage | None) -> mock.MagicMock:
    """Build a UsdContext stand-in returning a fixed stage."""
    usd_context = mock.MagicMock(name="usd_context")
    usd_context.get_stage.return_value = stage
    return usd_context


def _patch_controller_environment(stack: contextlib.ExitStack) -> SimpleNamespace:
    """Patch carb settings and the preset store the controller depends on; returns the stand-ins for assertions."""
    settings_interface, settings_store = _make_settings_interface()
    stack.enter_context(
        mock.patch.object(controller_module.carb.settings, "get_settings", return_value=settings_interface)
    )
    preset_store_class = stack.enter_context(mock.patch.object(controller_module, "PresetStore"))
    preset_store_class.return_value.list_names.return_value = []
    stack.enter_context(
        mock.patch.object(controller_module, "get_default_presets_directory", return_value=Path("presets"))
    )
    return SimpleNamespace(
        settings_interface=settings_interface,
        settings_store=settings_store,
        preset_store_class=preset_store_class,
        preset_store=preset_store_class.return_value,
    )


class TestScatterBrushController(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._stack = contextlib.ExitStack()
        environment = _patch_controller_environment(self._stack)
        self.settings_interface = environment.settings_interface
        self.settings_store = environment.settings_store
        self.preset_store_class = environment.preset_store_class
        self.preset_store = environment.preset_store
        self._previous_instance = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = None
        self._subscriptions = []
        self.controller = controller_module.ScatterBrushController()

    async def tearDown(self):
        self.controller.destroy()
        controller_module._CONTROLLER_INSTANCE = self._previous_instance
        self._subscriptions.clear()
        self._stack.close()

    def _subscribe(self, subscribe: Callable) -> mock.Mock:
        """Subscribe a mock callback and keep the subscription alive for the test."""
        callback = mock.Mock()
        self._subscriptions.append(subscribe(callback))
        return callback

    def _persisted_settings(self) -> dict:
        return json.loads(self.settings_store[constants_module.BRUSH_SETTINGS_PATH])

    def _persisted_assets(self) -> list[dict]:
        return json.loads(self.settings_store[constants_module.ASSETS_SETTING_PATH])

    def _patch_flood_dependencies(self, stack: contextlib.ExitStack, stage: Usd.Stage, **overrides) -> dict:
        """Patch every collaborator of flood(); overrides replace the default patch keyword arguments by name."""
        specs = {
            "get_context": (controller_module.omni.usd, "get_context", {"return_value": _make_usd_context(stage)}),
            "group": (controller_module.omni.kit.undo, "group", {"return_value": contextlib.nullcontext()}),
            "execute": (controller_module.omni.kit.commands, "execute", {"return_value": (True, None)}),
            "generate_flood": (controller_module, "generate_flood", {"return_value": []}),
            "existing_placement_points": (
                controller_module,
                "existing_placement_points",
                {"return_value": np.empty((0, 3))},
            ),
            "stage_up_axis": (controller_module, "stage_up_axis", {"return_value": UpAxis.Z}),
            "PaddingIndex": (controller_module, "PaddingIndex", {}),
        }
        patched = {}
        for key, (target, attribute, kwargs) in specs.items():
            patched[key] = stack.enter_context(mock.patch.object(target, attribute, **overrides.get(key, kwargs)))
        return patched

    async def test_init_loads_persisted_settings_and_assets_from_carb(self):
        # Arrange
        ScatterBrushSettings(radius=123.0).save_to_carb(self.settings_interface)
        self.settings_store[constants_module.ASSETS_SETTING_PATH] = json.dumps([{"path": _ROCK}])

        # Act
        controller = controller_module.ScatterBrushController()

        # Assert
        self.assertEqual(controller.settings.radius, 123.0)
        self.assertEqual([asset.path for asset in controller.assets], [_ROCK])
        self.assertEqual(controller.mode, controller_module.ScatterMode.OFF)
        self.preset_store_class.assert_called_with(Path("presets"))

    async def test_toggle_paint_when_off_switches_to_paint_and_fires_mode_changed(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_mode_changed)

        # Act
        self.controller.toggle_paint()

        # Assert
        self.assertEqual(self.controller.mode, controller_module.ScatterMode.PAINT)
        callback.assert_called_once_with(controller_module.ScatterMode.PAINT)

    async def test_toggle_paint_when_paint_switches_to_off(self):
        # Arrange
        self.controller.set_mode(controller_module.ScatterMode.PAINT)

        # Act
        self.controller.toggle_paint()

        # Assert
        self.assertEqual(self.controller.mode, controller_module.ScatterMode.OFF)

    async def test_toggle_paint_when_erase_switches_to_off(self):
        # Arrange
        self.controller.set_mode(controller_module.ScatterMode.ERASE)
        callback = self._subscribe(self.controller.subscribe_mode_changed)

        # Act
        self.controller.toggle_paint()

        # Assert
        self.assertEqual(self.controller.mode, controller_module.ScatterMode.OFF)
        callback.assert_called_once_with(controller_module.ScatterMode.OFF)

    async def test_set_mode_with_same_mode_does_not_fire_mode_changed(self):
        # Arrange
        self.controller.set_mode(controller_module.ScatterMode.ERASE)
        callback = self._subscribe(self.controller.subscribe_mode_changed)

        # Act
        self.controller.set_mode(controller_module.ScatterMode.ERASE)

        # Assert
        callback.assert_not_called()

    async def test_update_settings_with_valid_value_persists_and_fires_settings_changed(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_settings_changed)

        # Act
        result = self.controller.update_settings(radius=80.0, falloff="LINEAR")

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.settings.radius, 80.0)
        self.assertEqual(self.controller.settings.falloff, Falloff.LINEAR)
        self.assertEqual(self._persisted_settings()["radius"], 80.0)
        callback.assert_called_once_with(self.controller.settings)

    async def test_update_settings_with_invalid_value_returns_false_and_posts_status_without_changing_settings(self):
        # Arrange
        settings_callback = self._subscribe(self.controller.subscribe_settings_changed)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.update_settings(radius=-5.0)

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.radius, 50.0)
        settings_callback.assert_not_called()
        status_callback.assert_called_once()
        self.assertTrue(status_callback.call_args.args[1])
        self.assertNotIn(constants_module.BRUSH_SETTINGS_PATH, self.settings_store)

    async def test_update_settings_with_an_out_of_bounds_value_returns_false(self):
        # Arrange
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.update_settings(radius=0.5)

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.radius, 50.0)
        status_callback.assert_called_once()

    async def test_update_settings_with_unknown_field_returns_false(self):
        # Arrange
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.update_settings(not_a_field=1)

        # Assert
        self.assertFalse(result)
        status_callback.assert_called_once()

    async def test_replace_settings_stores_copy_and_fires_settings_changed(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_settings_changed)
        new_settings = ScatterBrushSettings(radius=200.0, preset_name="Replaced")

        # Act
        self.controller.replace_settings(new_settings)

        # Assert
        self.assertEqual(self.controller.settings, new_settings)
        self.assertIsNot(self.controller.settings, new_settings)
        self.assertEqual(self._persisted_settings()["preset_name"], "Replaced")
        callback.assert_called_once_with(self.controller.settings)

    async def test_add_asset_reads_up_axis_and_appends_entry(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        with mock.patch.object(controller_module, "read_asset_up_axis", return_value=UpAxis.Y) as read_up_axis:
            # Act
            result = self.controller.add_asset(_ROCK)

        # Assert
        self.assertTrue(result)
        read_up_axis.assert_called_once_with(_ROCK)
        self.assertEqual(self.controller.assets, [ScatterAssetEntry(path=_ROCK, up_axis=UpAxis.Y)])
        self.assertEqual(self._persisted_assets()[0]["up_axis"], "Y")
        callback.assert_called_once_with(self.controller.assets)

    async def test_add_asset_stores_and_persists_the_assets_unit_scale_and_natural_extent(self):
        # Arrange
        with (
            mock.patch.object(controller_module, "read_asset_meters_per_unit", return_value=0.01) as read_unit,
            mock.patch.object(
                controller_module, "read_asset_natural_extent", return_value=(1.0, 2.0, 3.0)
            ) as read_size,
        ):
            # Act
            result = self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        self.assertTrue(result)
        read_unit.assert_called_once_with(_ROCK)
        read_size.assert_called_once_with(_ROCK)
        self.assertEqual(self.controller.assets[0].meters_per_unit, 0.01)
        self.assertEqual(self.controller.assets[0].natural_extent, (1.0, 2.0, 3.0))
        persisted = self._persisted_assets()[0]
        self.assertEqual(persisted["meters_per_unit"], 0.01)
        self.assertEqual(persisted["natural_extent"], [1.0, 2.0, 3.0])

    async def test_ensure_asset_measurements_backfills_entries_restored_from_carb(self):
        # Arrange - a palette restored from carb never went through add_asset, so it carries no measurement.
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        with (
            mock.patch.object(controller_module, "read_asset_meters_per_unit", return_value=1.0),
            mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(4.0, 5.0, 6.0)),
        ):
            # Act
            changed = self.controller.ensure_asset_measurements()

        # Assert
        self.assertTrue(changed)
        self.assertEqual(self.controller.assets[0].natural_extent, (4.0, 5.0, 6.0))
        callback.assert_called_once_with(self.controller.assets)

    async def test_ensure_asset_measurements_with_measured_entries_changes_nothing(self):
        # Arrange
        with (
            mock.patch.object(controller_module, "read_asset_meters_per_unit", return_value=1.0),
            mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(4.0, 5.0, 6.0)),
        ):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        with mock.patch.object(controller_module, "read_asset_natural_extent") as read_size:
            # Act
            changed = self.controller.ensure_asset_measurements()

        # Assert
        self.assertFalse(changed)
        read_size.assert_not_called()
        callback.assert_not_called()

    async def test_ensure_asset_measurements_with_unmeasurable_asset_stays_quiet_and_retries_later(self):
        # Arrange - an asset that cannot be read must not fire an event on every stroke that asks for it.
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        with mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(0.0, 0.0, 0.0)):
            # Act
            changed = [self.controller.ensure_asset_measurements() for _ in range(3)]

        # Assert
        self.assertEqual(changed, [False, False, False])
        callback.assert_not_called()

    async def test_report_mismatched_asset_up_axes_warns_once_per_entry(self):
        # Arrange - the file says Y, the palette says Z, so the up-axis correction would be skipped silently.
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_status_message)

        with mock.patch.object(controller_module, "read_authored_asset_up_axis", return_value=UpAxis.Y):
            # Act
            reported = [self.controller.report_mismatched_asset_up_axes() for _ in range(3)]

        # Assert
        self.assertEqual(reported, [[_ROCK], [], []])
        self.assertEqual(callback.call_count, 1)
        self.assertIn("Y-up", callback.call_args.args[0])

    async def test_report_mismatched_asset_up_axes_with_matching_or_unauthored_axis_says_nothing(self):
        cases = [("matches the file", UpAxis.Z), ("file authors nothing", None)]
        for title, authored in cases:
            with self.subTest(title=title):
                # Arrange
                controller = controller_module.ScatterBrushController()
                controller.add_asset(_ROCK, up_axis=UpAxis.Z)
                callback = self._subscribe(controller.subscribe_status_message)

                with mock.patch.object(controller_module, "read_authored_asset_up_axis", return_value=authored):
                    # Act
                    reported = controller.report_mismatched_asset_up_axes()

                # Assert
                self.assertEqual(reported, [])
                callback.assert_not_called()

    async def test_set_asset_up_axis_lets_a_still_wrong_axis_be_reported_again(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        with mock.patch.object(controller_module, "read_authored_asset_up_axis", return_value=UpAxis.Y):
            self.controller.report_mismatched_asset_up_axes()

            # Act
            self.controller.set_asset_up_axis(_ROCK, UpAxis.Z)
            reported = self.controller.report_mismatched_asset_up_axes()

        # Assert
        self.assertEqual(reported, [_ROCK])

    async def test_next_stroke_reports_a_mismatched_up_axis(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        with mock.patch.object(self.controller, "report_mismatched_asset_up_axes") as report:
            # Act
            self.controller.next_stroke()

        # Assert
        report.assert_called_once_with()

    async def test_next_stroke_resolves_textures_and_measurements_before_handing_out_a_seed(self):
        # Arrange - the viewport tool copies the palette straight after this call, so it has to be resolved by then.
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        with (
            mock.patch.object(self.controller, "ensure_asset_textures") as ensure_textures,
            mock.patch.object(self.controller, "ensure_asset_measurements") as ensure_measurements,
        ):
            # Act
            self.controller.next_stroke()

        # Assert
        ensure_textures.assert_called_once_with()
        ensure_measurements.assert_called_once_with()

    async def test_add_asset_with_explicit_up_axis_does_not_read_file(self):
        # Arrange
        with mock.patch.object(controller_module, "read_asset_up_axis") as read_up_axis:
            # Act
            result = self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        self.assertTrue(result)
        read_up_axis.assert_not_called()
        self.assertEqual(self.controller.assets[0].up_axis, UpAxis.Z)

    async def test_add_asset_with_duplicate_path_returns_false(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        # Act
        result = self.controller.add_asset(_ROCK.replace("/", "\\"), up_axis=UpAxis.Z)

        # Assert
        self.assertFalse(result)
        self.assertEqual(len(self.controller.assets), 1)
        callback.assert_not_called()

    async def test_add_asset_with_empty_path_returns_false(self):
        # Arrange
        path = "   "

        # Act
        result = self.controller.add_asset(path, up_axis=UpAxis.Z)

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.assets, [])

    async def test_remove_asset_with_known_path_removes_entry_and_fires_assets_changed(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.add_asset(_GRASS, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        # Act
        result = self.controller.remove_asset(_ROCK)

        # Assert
        self.assertTrue(result)
        self.assertEqual([asset.path for asset in self.controller.assets], [_GRASS])
        self.assertEqual([asset["path"] for asset in self._persisted_assets()], [_GRASS])
        callback.assert_called_once()

    async def test_remove_asset_with_unknown_path_returns_false(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        result = self.controller.remove_asset(_GRASS)

        # Assert
        self.assertFalse(result)
        self.assertEqual(len(self.controller.assets), 1)

    async def test_set_asset_enabled_updates_entry(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        # Act
        result = self.controller.set_asset_enabled(_ROCK, False)

        # Assert
        self.assertTrue(result)
        self.assertFalse(self.controller.assets[0].enabled)
        self.assertFalse(self._persisted_assets()[0]["enabled"])
        callback.assert_called_once()

    async def test_set_asset_weight_updates_entry(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        result = self.controller.set_asset_weight(_ROCK, 3.5)

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.assets[0].weight, 3.5)
        self.assertEqual(self._persisted_assets()[0]["weight"], 3.5)

    async def test_set_asset_weight_out_of_range_returns_false_and_keeps_weight(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.set_asset_weight(_ROCK, -1.0)

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.assets[0].weight, 1.0)
        status_callback.assert_called_once()

    async def test_set_asset_weight_with_unknown_path_returns_false(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        result = self.controller.set_asset_weight(_GRASS, 2.0)

        # Assert
        self.assertFalse(result)

    async def test_set_asset_up_axis_updates_entry(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        result = self.controller.set_asset_up_axis(_ROCK, UpAxis.Y)

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.assets[0].up_axis, UpAxis.Y)
        self.assertEqual(self._persisted_assets()[0]["up_axis"], "Y")

    async def test_enabled_assets_returns_only_enabled_entries(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.add_asset(_GRASS, up_axis=UpAxis.Z)
        self.controller.set_asset_enabled(_GRASS, False)

        # Act
        enabled = self.controller.enabled_assets()

        # Assert
        self.assertEqual([asset.path for asset in enabled], [_ROCK])

    async def test_assets_property_returns_copy_that_does_not_alias_internal_state(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Act
        assets = self.controller.assets

        # Assert
        assets.clear()
        self.assertEqual(len(self.controller.assets), 1)

    async def test_preset_names_delegates_to_store(self):
        # Arrange
        self.preset_store.list_names.return_value = ["Grass", "Rocks"]

        # Act
        names = self.controller.preset_names()

        # Assert
        self.assertEqual(names, ["Grass", "Rocks"])

    async def test_apply_preset_replaces_settings_and_fires_settings_changed(self):
        # Arrange
        self.preset_store.load.return_value = ScatterBrushSettings(preset_name="Grass", radius=123.0)
        callback = self._subscribe(self.controller.subscribe_settings_changed)

        # Act
        result = self.controller.apply_preset("Grass")

        # Assert
        self.assertTrue(result)
        self.preset_store.load.assert_called_once_with("Grass")
        self.assertEqual((self.controller.settings.preset_name, self.controller.settings.radius), ("Grass", 123.0))
        callback.assert_called_once_with(self.controller.settings)

    async def test_apply_preset_when_store_raises_returns_false_and_posts_status(self):
        # Arrange
        self.preset_store.load.side_effect = ValueError("corrupt")
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.apply_preset("Bad")

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.radius, 50.0)
        status_callback.assert_called_once()
        self.assertTrue(status_callback.call_args.args[1])

    async def test_save_preset_without_name_saves_current_settings_under_preset_name(self):
        # Arrange
        self.controller.update_settings(preset_name="Grass")
        self.preset_store.save.return_value = Path("presets/Grass.json")

        # Act
        result = self.controller.save_preset()

        # Assert
        self.assertTrue(result)
        self.preset_store.save.assert_called_once_with("Grass", self.controller.settings)

    async def test_save_preset_with_name_updates_preset_name_and_saves(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_settings_changed)
        self.preset_store.save.return_value = Path("presets/Rocks.json")

        # Act
        result = self.controller.save_preset("Rocks")

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.settings.preset_name, "Rocks")
        self.preset_store.save.assert_called_once_with("Rocks", self.controller.settings)
        callback.assert_called_once()

    async def test_save_preset_under_a_name_stored_in_another_case_adopts_the_stored_spelling(self):
        # Arrange - the store wrote over "Grass.json", so the combo must not list "grass" beside it
        self.preset_store.save.return_value = Path("presets/Grass.json")

        # Act
        result = self.controller.save_preset("grass")

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.settings.preset_name, "Grass")

    async def test_save_preset_when_store_raises_returns_false_and_posts_status(self):
        # Arrange
        self.preset_store.save.side_effect = ValueError("bad name")
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.save_preset("a/b")

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.preset_name, "Default")
        status_callback.assert_called_once()

    async def test_rename_preset_delegates_to_store(self):
        # Arrange
        names = ("Old", "New")

        # Act
        result = self.controller.rename_preset(*names)

        # Assert
        self.assertTrue(result)
        self.preset_store.rename.assert_called_once_with("Old", "New")

    async def test_rename_preset_of_current_preset_updates_preset_name(self):
        # Arrange
        self.controller.update_settings(preset_name="Old")
        self.preset_store.rename.return_value = Path("presets/New.json")

        # Act
        self.controller.rename_preset("Old", "New")

        # Assert
        self.assertEqual(self.controller.settings.preset_name, "New")

    async def test_rename_preset_of_current_preset_named_in_another_case_updates_preset_name(self):
        # Arrange - the store matches "old" to the stored "Old", which is the current preset
        self.controller.update_settings(preset_name="Old")
        self.preset_store.rename.return_value = Path("presets/New.json")

        # Act
        self.controller.rename_preset("old", "New")

        # Assert
        self.assertEqual(self.controller.settings.preset_name, "New")

    async def test_rename_preset_when_store_raises_returns_false(self):
        # Arrange
        self.preset_store.rename.side_effect = ValueError("missing")
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.rename_preset("Old", "New")

        # Assert
        self.assertFalse(result)
        status_callback.assert_called_once()

    async def test_clone_preset_delegates_to_store(self):
        # Arrange
        names = ("Source", "Copy")

        # Act
        result = self.controller.clone_preset(*names)

        # Assert
        self.assertTrue(result)
        self.preset_store.clone.assert_called_once_with("Source", "Copy")

    async def test_clone_preset_when_store_raises_returns_false(self):
        # Arrange
        self.preset_store.clone.side_effect = ValueError("missing")

        # Act
        result = self.controller.clone_preset("Source", "Copy")

        # Assert
        self.assertFalse(result)

    async def test_delete_preset_delegates_to_store(self):
        # Arrange
        name = "Grass"

        # Act
        result = self.controller.delete_preset(name)

        # Assert
        self.assertTrue(result)
        self.preset_store.delete.assert_called_once_with("Grass")

    async def test_delete_preset_when_store_raises_returns_false(self):
        # Arrange
        self.preset_store.delete.side_effect = OSError("locked")

        # Act
        result = self.controller.delete_preset("Grass")

        # Assert
        self.assertFalse(result)

    async def test_set_anchor_from_selection_stores_first_selected_prototype(self):
        # Arrange
        usd_context = mock.MagicMock(name="usd_context")
        prototypes = {Sdf.Path("/RootNode/meshes/mesh_B"), Sdf.Path("/RootNode/meshes/mesh_A")}

        with (
            mock.patch.object(controller_module.omni.usd, "get_context", return_value=usd_context) as get_context,
            mock.patch.object(controller_module, "selected_prototypes", return_value=prototypes) as selected,
        ):
            # Act
            result = self.controller.set_anchor_from_selection("scatter_context")

        # Assert
        self.assertTrue(result)
        get_context.assert_called_once_with("scatter_context")
        selected.assert_called_once_with(usd_context)
        self.assertEqual(self.controller.settings.anchor_prototype_path, "/RootNode/meshes/mesh_A")

    async def test_set_anchor_from_selection_without_prototype_returns_false(self):
        # Arrange
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with (
            mock.patch.object(controller_module.omni.usd, "get_context", return_value=mock.MagicMock()),
            mock.patch.object(controller_module, "selected_prototypes", return_value=set()),
        ):
            # Act
            result = self.controller.set_anchor_from_selection("")

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.anchor_prototype_path, "")
        status_callback.assert_called_once()

    async def test_next_stroke_with_randomize_seed_disabled_uses_settings_seed_and_increments_index(self):
        # Arrange
        self.controller.update_settings(randomize_seed=False, seed=1234)
        first = self.controller.next_stroke()

        # Act
        second = self.controller.next_stroke()

        # Assert
        self.assertEqual(first, (1234, 0))
        self.assertEqual(second, (1234, 1))

    async def test_next_stroke_with_randomize_seed_enabled_uses_random_seed(self):
        # Arrange
        self.controller.update_settings(randomize_seed=True, seed=1234)

        with mock.patch.object(controller_module.secrets, "randbits", return_value=99) as randbits:
            # Act
            seed, stroke_index = self.controller.next_stroke()

        # Assert
        self.assertEqual((seed, stroke_index), (99, 0))
        randbits.assert_called_once_with(31)

    async def test_create_picker_with_picker_factory_uses_factory(self):
        # Arrange
        picker = mock.Mock(name="picker")
        factory = mock.Mock(return_value=picker)
        viewport_api = mock.Mock(name="viewport_api")
        cache = mock.Mock(name="cache")
        self.controller.set_picker_factory(factory)

        with mock.patch.object(controller_module, "create_surface_picker") as create_surface_picker:
            # Act
            result = self.controller.create_picker(viewport_api, cache)

        # Assert
        self.assertIs(result, picker)
        factory.assert_called_once_with(viewport_api, cache)
        create_surface_picker.assert_not_called()

    async def test_create_picker_without_factory_uses_create_surface_picker(self):
        # Arrange
        picker = mock.Mock(name="picker")
        viewport_api = mock.Mock(name="viewport_api")
        cache = mock.Mock(name="cache")
        self.controller.set_picker_factory(None)

        with mock.patch.object(controller_module, "create_surface_picker", return_value=picker) as create_picker:
            # Act
            result = self.controller.create_picker(viewport_api, cache)

        # Assert
        self.assertIs(result, picker)
        create_picker.assert_called_once_with(viewport_api, cache)

    async def test_lowering_a_max_below_its_min_carries_the_min_down(self):
        # A drag field reports every value it passes through, so a max that cannot cross its min cannot be dragged
        # under it at all.
        # Arrange
        self.controller.update_settings(scale_min=0.8, scale_max=1.2)

        # Act
        updated = self.controller.update_settings(scale_max=0.01)

        # Assert
        self.assertTrue(updated)
        self.assertEqual(self.controller.settings.scale_max, 0.01)
        self.assertEqual(self.controller.settings.scale_min, 0.01)

    async def test_raising_a_min_above_its_max_carries_the_max_up(self):
        # Arrange
        self.controller.update_settings(scale_min=0.8, scale_max=1.2)

        # Act
        updated = self.controller.update_settings(scale_min=4.0)

        # Assert
        self.assertTrue(updated)
        self.assertEqual(self.controller.settings.scale_min, 4.0)
        self.assertEqual(self.controller.settings.scale_max, 4.0)

    async def test_setting_both_ends_of_a_range_in_conflict_is_still_rejected(self):
        # Arrange
        status = self._subscribe(self.controller.subscribe_status_message)

        # Act
        updated = self.controller.update_settings(scale_min=2.0, scale_max=1.0)

        # Assert
        self.assertFalse(updated)
        self.assertIn("scale_min", status.call_args.args[0])

    async def test_moving_one_range_leaves_the_other_ranges_alone(self):
        # Arrange
        self.controller.update_settings(rotation_z_min=10.0, rotation_z_max=200.0, scale_min=0.8, scale_max=1.2)

        # Act
        self.controller.update_settings(scale_max=0.5)

        # Assert
        self.assertEqual(self.controller.settings.rotation_z_min, 10.0)
        self.assertEqual(self.controller.settings.rotation_z_max, 200.0)
        self.assertEqual(self.controller.settings.scale_min, 0.5)

    async def test_add_asset_resolves_and_caches_the_texture_set(self):
        # Arrange
        texture_set = ScatterTextureSet(source_hash="abc", assigned=("inputs:diffuse_texture",))

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set) as resolve:
            # Act
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        resolve.assert_called_once_with(_ROCK)
        self.assertEqual(self.controller.assets[0].textures, texture_set)

    async def test_add_asset_whose_textures_cannot_be_resolved_posts_a_status_and_still_adds_it(self):
        # Arrange
        status = self._subscribe(self.controller.subscribe_status_message)
        texture_set = ScatterTextureSet(reason="the asset has no AperturePBR material")

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set):
            # Act
            added = self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        self.assertTrue(added)
        self.assertEqual([asset.path for asset in self.controller.assets], [_ROCK])
        self.assertIn("no AperturePBR material", status.call_args.args[0])

    async def test_add_asset_that_received_textures_posts_no_status(self):
        # A texture set that carries no height map is ordinary, not a fault worth announcing on every add.
        # Arrange
        status = self._subscribe(self.controller.subscribe_status_message)
        texture_set = ScatterTextureSet(
            overrides=(
                ScatterMaterialOverride(
                    shader_rel_path="XForms/Looks/rock/Shader",
                    inputs={"inputs:diffuse_texture": "rock.a.rtex.dds"},
                ),
            ),
            unresolved=("inputs:height_texture",),
        )

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set):
            # Act
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        status.assert_not_called()

    async def test_add_asset_that_supplies_its_own_textures_posts_no_status(self):
        # Arrange
        status = self._subscribe(self.controller.subscribe_status_message)
        texture_set = ScatterTextureSet(
            assigned=("inputs:diffuse_texture",),
            unresolved=("inputs:height_texture",),
            reason="no matching ingested texture set was found for this asset",
        )

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set):
            # Act
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        # Assert
        status.assert_not_called()

    async def test_ensure_asset_textures_backfills_an_entry_restored_from_carb(self):
        # Arrange
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=None):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        texture_set = ScatterTextureSet(source_hash="abc")

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set):
            # Act
            changed = self.controller.ensure_asset_textures()

        # Assert
        self.assertTrue(changed)
        self.assertEqual(self.controller.assets[0].textures, texture_set)

    async def test_ensure_asset_textures_re_resolves_when_the_asset_was_re_ingested(self):
        # Arrange
        stale = ScatterTextureSet(source_hash="old")
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=stale):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        fresh = ScatterTextureSet(source_hash="new")

        with mock.patch.object(controller_module, "asset_source_hash", return_value="new"):
            with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=fresh):
                # Act
                changed = self.controller.ensure_asset_textures()

        # Assert
        self.assertTrue(changed)
        self.assertEqual(self.controller.assets[0].textures, fresh)

    async def test_ensure_asset_textures_leaves_an_up_to_date_entry_alone(self):
        # Arrange
        texture_set = ScatterTextureSet(source_hash="same")
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)

        with mock.patch.object(controller_module, "asset_source_hash", return_value="same"):
            with mock.patch.object(controller_module, "resolve_asset_texture_set") as resolve:
                # Act
                changed = self.controller.ensure_asset_textures()

        # Assert
        self.assertFalse(changed)
        resolve.assert_not_called()

    async def test_ensure_asset_textures_with_an_unresolvable_asset_reports_no_change(self):
        # Arrange - every stroke asks, and a change is persisted and broadcast to the window and the live edit
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=None):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        callback = self._subscribe(self.controller.subscribe_assets_changed)

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=None):
            # Act
            changed = self.controller.ensure_asset_textures()

        # Assert
        self.assertFalse(changed)
        callback.assert_not_called()

    async def test_ensure_asset_textures_does_not_resolve_an_unchanged_failure_again(self):
        # Arrange
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=None):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
            self.controller.ensure_asset_textures()

        with mock.patch.object(controller_module, "resolve_asset_texture_set") as resolve:
            # Act
            self.controller.ensure_asset_textures()

        # Assert
        resolve.assert_not_called()

    async def test_ensure_asset_textures_resolves_a_failed_asset_again_once_it_was_re_ingested(self):
        # Arrange
        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=None):
            self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
            self.controller.ensure_asset_textures()
        texture_set = ScatterTextureSet(source_hash="new")

        with (
            mock.patch.object(controller_module, "asset_source_hash", return_value="new"),
            mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=texture_set),
        ):
            # Act
            changed = self.controller.ensure_asset_textures()

        # Assert
        self.assertTrue(changed)
        self.assertEqual(self.controller.assets[0].textures, texture_set)

    async def test_ensure_asset_measurements_does_not_measure_an_unchanged_failure_again(self):
        # Arrange - measuring opens the whole asset, and every stroke asks
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        with mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(0.0, 0.0, 0.0)):
            self.controller.ensure_asset_measurements()

        with mock.patch.object(controller_module, "read_asset_natural_extent") as read_size:
            # Act
            self.controller.ensure_asset_measurements()

        # Assert
        read_size.assert_not_called()

    async def test_ensure_asset_measurements_measures_a_failed_asset_again_once_it_was_re_ingested(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        with mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(0.0, 0.0, 0.0)):
            self.controller.ensure_asset_measurements()

        with (
            mock.patch.object(controller_module, "asset_source_hash", return_value="new"),
            mock.patch.object(controller_module, "read_asset_meters_per_unit", return_value=1.0),
            mock.patch.object(controller_module, "read_asset_natural_extent", return_value=(4.0, 5.0, 6.0)),
        ):
            # Act
            changed = self.controller.ensure_asset_measurements()

        # Assert
        self.assertTrue(changed)
        self.assertEqual(self.controller.assets[0].natural_extent, (4.0, 5.0, 6.0))

    async def test_set_asset_texture_source_re_resolves_with_the_picked_texture(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        picked = "C:/project/assets/ingested/SubUSDs/textures/rock.a.rtex.dds"
        resolved = ScatterTextureSet(source_hash="abc")

        with mock.patch.object(controller_module, "resolve_asset_texture_set", return_value=resolved) as resolve:
            # Act
            updated = self.controller.set_asset_texture_source(_ROCK, picked)

        # Assert
        self.assertTrue(updated)
        resolve.assert_called_once_with(_ROCK, picked)
        self.assertEqual(self.controller.assets[0].texture_source, picked)

    async def test_set_asset_texture_source_for_an_unknown_asset_returns_false(self):
        # Act / Assert
        self.assertFalse(self.controller.set_asset_texture_source(_GRASS, "whatever.dds"))

    async def test_materials_payload_is_empty_when_assign_textures_is_off(self):
        # Arrange
        self.controller.update_settings(assign_textures=False)
        layer = Usd.Stage.CreateInMemory().GetEditTarget().GetLayer()

        with mock.patch.object(controller_module, "build_materials_payload") as build:
            # Act
            payload = self.controller.materials_payload(layer)

        # Assert
        self.assertEqual(payload, {})
        build.assert_not_called()

    async def test_materials_payload_builds_from_the_enabled_assets(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        layer = Usd.Stage.CreateInMemory().GetEditTarget().GetLayer()

        with mock.patch.object(controller_module, "build_materials_payload", return_value={"a": []}) as build:
            # Act
            payload = self.controller.materials_payload(layer)

        # Assert
        self.assertEqual(payload, {"a": []})
        self.assertEqual([asset.path for asset in build.call_args.args[1]], [_ROCK])

    async def test_flood_with_explicit_target_executes_flood_command_inside_undo_group(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(randomize_seed=False, seed=5, flood_max_instances=100)
        target = _make_target("mesh_A")
        records = [_make_record("s_1"), _make_record("s_2")]
        cache = mock.Mock(name="cache")
        committed = self._subscribe(self.controller.subscribe_stroke_committed)
        events: list[str] = []

        group_context = mock.MagicMock(name="undo_group")
        group_context.__enter__.side_effect = lambda *args: events.append("enter")
        group_context.__exit__.side_effect = lambda *args: events.append("exit")

        def _execute(*args, **kwargs):
            events.append("execute")
            return True, None

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                group={"return_value": group_context},
                execute={"side_effect": _execute},
                generate_flood={"return_value": records},
                stage_up_axis={"return_value": UpAxis.Y},
            )

            # Act
            count = self.controller.flood("scatter_context", cache, targets=[target])

        # Assert
        self.assertEqual(count, 2)
        patched["group"].assert_called_once_with()
        self.assertEqual(events, ["enter", "execute", "exit"])
        patched["execute"].assert_called_once_with(
            "ScatterFloodCommand",
            context_name="scatter_context",
            layer_identifier=stage.GetEditTarget().GetLayer().identifier,
            records=records,
            materials={},
        )
        patched["generate_flood"].assert_called_once()
        flood_args = patched["generate_flood"].call_args.args
        self.assertIs(flood_args[0], cache)
        self.assertIs(flood_args[1], target)
        self.assertEqual(flood_args[2], self.controller.settings)
        self.assertEqual([asset.path for asset in flood_args[3]], [_ROCK])
        self.assertIs(flood_args[5], patched["PaddingIndex"].return_value)
        self.assertEqual(flood_args[6], UpAxis.Y)
        self.assertEqual(flood_args[7], 1.0)
        self.assertIs(flood_args[8], stage.GetEditTarget().GetLayer())
        self.assertEqual(flood_args[9], 100)
        committed.assert_called_once_with(2, False)

    async def test_flood_seeds_padding_index_with_existing_placements(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(padding=0.5)
        target = _make_target("mesh_A")
        existing = np.array([[1.0, 2.0, 3.0]])

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                existing_placement_points={"return_value": existing},
            )

            # Act
            self.controller.flood("", mock.Mock(), targets=[target])

        # Assert
        patched["PaddingIndex"].assert_called_once_with(1.0)
        patched["existing_placement_points"].assert_called_once_with(stage, target)
        patched["PaddingIndex"].return_value.add_many.assert_called_once_with(existing)

    async def test_flood_shares_max_instances_cap_across_targets(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(flood_max_instances=3)
        targets = [_make_target("mesh_A"), _make_target("mesh_B")]
        generate_flood_results = [[_make_record("s_1"), _make_record("s_2")], [_make_record("s_3")]]

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                generate_flood={"side_effect": generate_flood_results},
            )

            # Act
            count = self.controller.flood("", mock.Mock(), targets=targets)

        # Assert
        self.assertEqual(count, 3)
        self.assertEqual([call.args[9] for call in patched["generate_flood"].call_args_list], [3, 1])
        self.assertEqual(patched["execute"].call_count, 2)

    async def test_flood_stops_when_max_instances_cap_is_reached(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(flood_max_instances=2)
        targets = [_make_target("mesh_A"), _make_target("mesh_B")]

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                generate_flood={"return_value": [_make_record("s_1"), _make_record("s_2")]},
            )

            # Act
            count = self.controller.flood("", mock.Mock(), targets=targets)

        # Assert
        self.assertEqual(count, 2)
        patched["generate_flood"].assert_called_once()
        patched["execute"].assert_called_once()

    async def test_flood_without_targets_in_anchor_mode_builds_target_from_anchor(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim(_ANCHOR_PROTOTYPE, "Xform")
        instance = UsdGeom.Xform.Define(stage, f"/RootNode/instances/inst_{_ANCHOR_HASH}_0")
        instance.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)
        mesh_path = instance.GetPath().AppendChild("mesh")

        anchor_target = ScatterTarget(
            prototype_root=Sdf.Path(_ANCHOR_PROTOTYPE),
            parent_instance_root=instance.GetPath(),
            mesh_path=mesh_path,
            parent_world=Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, 20.0, 30.0)),
            instance_count=4,
        )
        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                generate_flood={"return_value": [_make_record("s_1")]},
            )
            build_target = stack.enter_context(
                mock.patch.object(controller_module, "build_prototype_target", return_value=anchor_target)
            )

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 1)
        self.assertEqual(build_target.call_args.args[1], Sdf.Path(_ANCHOR_PROTOTYPE))
        self.assertIs(patched["generate_flood"].call_args.args[1], anchor_target)

    async def test_flood_with_malformed_anchor_path_posts_error_and_places_nothing(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path="my anchor")
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)
            build_target = stack.enter_context(mock.patch.object(controller_module, "build_prototype_target"))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["generate_flood"].assert_not_called()
        patched["execute"].assert_not_called()
        build_target.assert_not_called()
        status_callback.assert_called_once()
        self.assertIn("anchor", status_callback.call_args.args[0])
        self.assertTrue(status_callback.call_args.args[1])

    async def test_flood_with_anchor_missing_from_stage_posts_error_and_places_nothing(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path=_ANCHOR_PROTOTYPE)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)
            build_target = stack.enter_context(mock.patch.object(controller_module, "build_prototype_target"))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        build_target.assert_not_called()
        status_callback.assert_called_once()

    async def test_flood_without_targets_in_hit_surface_mode_uses_selected_prototypes(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        prototypes = {Sdf.Path("/RootNode/meshes/mesh_B"), Sdf.Path("/RootNode/meshes/mesh_A")}

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                generate_flood={"return_value": [_make_record("s_1")]},
            )
            stack.enter_context(mock.patch.object(controller_module, "selected_prototypes", return_value=prototypes))
            stack.enter_context(
                mock.patch.object(
                    controller_module,
                    "build_prototype_target",
                    side_effect=lambda stage, prototype: _make_target(prototype.name),
                )
            )

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 2)
        self.assertEqual(
            [call.args[1].prototype_root for call in patched["generate_flood"].call_args_list],
            [Sdf.Path("/RootNode/meshes/mesh_A"), Sdf.Path("/RootNode/meshes/mesh_B")],
        )

    async def test_flood_skips_prototypes_without_capture_mesh(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)
            stack.enter_context(
                mock.patch.object(
                    controller_module, "selected_prototypes", return_value={Sdf.Path("/RootNode/meshes/mesh_A")}
                )
            )
            stack.enter_context(mock.patch.object(controller_module, "build_prototype_target", return_value=None))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["generate_flood"].assert_not_called()
        patched["execute"].assert_not_called()
        status_callback.assert_called_once()

    async def test_flood_without_enabled_assets_returns_zero_and_posts_status(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)

            # Act
            count = self.controller.flood("", mock.Mock(), targets=[_make_target("mesh_A")])

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        status_callback.assert_called_once()
        self.assertTrue(status_callback.call_args.args[1])

    async def test_flood_without_stage_returns_zero_and_posts_status(self):
        # Arrange
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, None)

            # Act
            count = self.controller.flood("", mock.Mock(), targets=[_make_target("mesh_A")])

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        status_callback.assert_called_once()

    async def test_flood_with_no_resolvable_target_returns_zero_and_posts_status(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)
            stack.enter_context(mock.patch.object(controller_module, "selected_prototypes", return_value=set()))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        status_callback.assert_called_once()

    async def test_flood_with_no_generated_records_executes_no_command(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        committed = self._subscribe(self.controller.subscribe_stroke_committed)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)

            # Act
            count = self.controller.flood("", mock.Mock(), targets=[_make_target("mesh_A")])

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        committed.assert_called_once_with(0, False)

    async def test_notify_stroke_committed_fires_stroke_committed(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_stroke_committed)

        # Act
        self.controller.notify_stroke_committed(12, True)

        # Assert
        callback.assert_called_once_with(12, True)

    async def test_post_status_fires_status_message(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        self.controller.post_status("Placed 12 assets")

        # Assert
        callback.assert_called_once_with("Placed 12 assets", False)

    async def test_destroy_drops_subscribers(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_mode_changed)
        self.controller.destroy()

        # Act
        self.controller.set_mode(controller_module.ScatterMode.PAINT)

        # Assert
        callback.assert_not_called()


class TestScatterBrushControllerPreview(omni.kit.test.AsyncTestCase):
    """The stroke just painted stays the preview live edits fold into until anything else reaches the undo stack."""

    _CONTEXT = "scatter_context"
    _CONTAINER = f"{_ANCHOR_PROTOTYPE}/scatter_default"

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        # Persistence is stubbed at the controller's own readers rather than by patching carb.settings: these tests
        # run real commands, and omni.kit.undo caches a carb setting the first time one runs.
        self._stack.enter_context(mock.patch.object(ScatterBrushSettings, "save_to_carb"))
        self._stack.enter_context(
            mock.patch.object(ScatterBrushSettings, "load_from_carb", return_value=ScatterBrushSettings())
        )
        self._stack.enter_context(mock.patch.object(controller_module, "save_assets_to_carb"))
        self._stack.enter_context(mock.patch.object(controller_module, "load_assets_from_carb", return_value=[]))
        # The extension registers these commands on startup; register them here too so the test is self-contained,
        # and restore that startup registration in tearDown so later tests in the same process still find them.
        self._commands_were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        self._registered = omni.kit.commands.register_all_commands_in_module(commands_module)
        omni.kit.undo.clear_stack()
        self.stage = Usd.Stage.CreateInMemory()
        self.layer = self.stage.GetRootLayer()
        self.controller = controller_module.ScatterBrushController()
        self.preview_changed = mock.Mock()
        self._subscription = self.controller.subscribe_preview_changed(self.preview_changed)

    async def tearDown(self):
        self._subscription = None
        self.controller.destroy()
        omni.kit.undo.clear_stack()
        omni.kit.commands.unregister_module_commands(self._registered)
        if self._commands_were_registered:
            omni.kit.commands.register_all_commands_in_module(commands_module)
        self.layer = None
        self.stage = None
        self._stack.close()

    def _paint(self, count: int = 2) -> list[str]:
        """Commit a placing stroke the way the brush does at mouse-up; returns the paths it created."""
        records = [
            PlacementRecord(
                container_path=self._CONTAINER,
                prim_name=f"s_00000000000{index}",
                asset_rel_path=_ROCK,
                asset_abs_path=_ROCK,
                translate=(float(index), 0.0, 0.0),
                rotate_xyz=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                brush_id="Default",
            )
            for index in range(count)
        ]
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name=self._CONTEXT,
            layer_identifier=self.layer.identifier,
            kind=ScatterStrokeKind.PLACE.value,
            records=records,
            already_applied=True,
        )
        return [record.prim_path for record in records]

    def _erase(self) -> None:
        """Commit an erasing stroke the way the brush does at mouse-up."""
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name=self._CONTEXT,
            layer_identifier=self.layer.identifier,
            kind=ScatterStrokeKind.ERASE.value,
            prim_paths=[f"{self._CONTAINER}/s_000000000000"],
            snapshot_layer=Sdf.Layer.CreateAnonymous(),
            already_applied=True,
        )

    def _execute_another_command(self) -> None:
        """Commit an edit that is not a stroke, as a live edit of a selection does."""
        omni.kit.commands.execute(
            "ScatterReapplyCommand",
            context_name=self._CONTEXT,
            layer_identifier=self.layer.identifier,
            before=[],
            after=[],
        )

    def _begin_preview_of_a_stroke(self) -> list[str]:
        """Paint a stroke, make it the preview and forget the notification that started it."""
        paths = self._paint()
        self.assertTrue(self.controller.begin_preview(self._CONTEXT))
        self.preview_changed.reset_mock()
        return paths

    async def test_begin_preview_with_a_placing_stroke_on_top_of_the_undo_stack_returns_true(self):
        # Arrange
        self._paint()

        # Act
        started = self.controller.begin_preview(self._CONTEXT)

        # Assert
        self.assertTrue(started)
        self.preview_changed.assert_called_once_with()

    async def test_preview_scope_of_the_previewed_stroke_is_the_placements_it_created(self):
        # Arrange
        paths = self._paint()
        self.controller.begin_preview(self._CONTEXT)

        # Act
        scope = self.controller.preview_scope(self._CONTEXT)

        # Assert
        self.assertEqual(scope, scope_of_placements(paths))
        self.assertEqual(scope.placement_count, 2)

    async def test_begin_preview_with_an_erasing_stroke_on_top_returns_false_and_previews_nothing(self):
        # Arrange
        self._erase()

        # Act
        started = self.controller.begin_preview(self._CONTEXT)

        # Assert
        self.assertFalse(started)
        self.assertIsNone(self.controller.preview_scope(self._CONTEXT))

    async def test_begin_preview_with_another_command_on_top_returns_false_and_previews_nothing(self):
        # Arrange
        self._paint()
        self._execute_another_command()

        # Act
        started = self.controller.begin_preview(self._CONTEXT)

        # Assert
        self.assertFalse(started)
        self.assertIsNone(self.controller.preview_scope(self._CONTEXT))

    async def test_begin_preview_with_a_stroke_that_placed_nothing_returns_false(self):
        # Arrange
        self._paint(count=0)

        # Act
        started = self.controller.begin_preview(self._CONTEXT)

        # Assert
        self.assertFalse(started)

    async def test_begin_preview_with_an_empty_undo_stack_returns_false(self):
        # Arrange
        omni.kit.undo.clear_stack()

        # Act
        started = self.controller.begin_preview(self._CONTEXT)

        # Assert
        self.assertFalse(started)

    async def test_preview_scope_for_another_usd_context_is_none(self):
        # Arrange
        self._begin_preview_of_a_stroke()

        # Act
        scope = self.controller.preview_scope("another_context")

        # Assert
        self.assertIsNone(scope)

    async def test_executing_another_command_ends_the_preview_and_tells_subscribers(self):
        # Arrange
        self._begin_preview_of_a_stroke()

        # Act
        self._execute_another_command()

        # Assert
        self.assertIsNone(self.controller.preview_scope(self._CONTEXT))
        self.preview_changed.assert_called_once_with()

    async def test_undoing_the_stroke_ends_the_preview_and_tells_subscribers(self):
        # Arrange
        self._begin_preview_of_a_stroke()

        # Act
        omni.kit.undo.undo()

        # Assert
        self.assertIsNone(self.controller.preview_scope(self._CONTEXT))
        self.preview_changed.assert_called_once_with()

    async def test_folds_into_preview_of_an_edit_of_the_previewed_stroke_returns_true(self):
        # Arrange
        paths = self._begin_preview_of_a_stroke()

        # Act
        folds = self.controller.folds_into_preview(self._CONTEXT, self.layer.identifier, scope_of_placements(paths))

        # Assert
        self.assertTrue(folds)

    async def test_folds_into_preview_of_an_edit_in_another_layer_returns_false(self):
        # Arrange
        paths = self._begin_preview_of_a_stroke()
        other_layer = Sdf.Layer.CreateAnonymous("other_edit_target")

        # Act
        folds = self.controller.folds_into_preview(self._CONTEXT, other_layer.identifier, scope_of_placements(paths))

        # Assert
        self.assertFalse(folds)

    async def test_folds_into_preview_of_an_edit_of_another_scope_returns_false(self):
        # Arrange - only part of the stroke, which is a selection rather than the stroke
        paths = self._begin_preview_of_a_stroke()

        # Act
        folds = self.controller.folds_into_preview(self._CONTEXT, self.layer.identifier, scope_of_placements(paths[:1]))

        # Assert
        self.assertFalse(folds)

    async def test_folds_into_preview_of_an_edit_in_another_usd_context_returns_false(self):
        # Arrange
        paths = self._begin_preview_of_a_stroke()

        # Act
        folds = self.controller.folds_into_preview("another_context", self.layer.identifier, scope_of_placements(paths))

        # Assert
        self.assertFalse(folds)

    async def test_folds_into_preview_after_the_stroke_was_undone_returns_false(self):
        # Arrange
        paths = self._begin_preview_of_a_stroke()
        omni.kit.undo.undo()

        # Act
        folds = self.controller.folds_into_preview(self._CONTEXT, self.layer.identifier, scope_of_placements(paths))

        # Assert
        self.assertFalse(folds)

    async def test_end_preview_ends_the_preview_and_tells_subscribers_once(self):
        # Arrange
        self._begin_preview_of_a_stroke()

        # Act
        self.controller.end_preview()

        # Assert
        self.assertIsNone(self.controller.preview_scope(self._CONTEXT))
        self.preview_changed.assert_called_once_with()

    async def test_end_preview_when_nothing_is_previewed_tells_nobody(self):
        # Arrange
        self._begin_preview_of_a_stroke()
        self.controller.end_preview()
        self.preview_changed.reset_mock()

        # Act
        self.controller.end_preview()

        # Assert
        self.preview_changed.assert_not_called()


class TestScatterBrushControllerSingleton(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._stack = contextlib.ExitStack()
        _patch_controller_environment(self._stack)
        self._previous_instance = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = None

    async def tearDown(self):
        controller_module.destroy_scatter_brush_controller()
        controller_module._CONTROLLER_INSTANCE = self._previous_instance
        self._stack.close()

    async def test_get_scatter_brush_controller_returns_same_instance(self):
        # Arrange
        first = controller_module.get_scatter_brush_controller()

        # Act
        second = controller_module.get_scatter_brush_controller()

        # Assert
        self.assertIs(second, first)
        self.assertIsInstance(first, controller_module.ScatterBrushController)

    async def test_destroy_scatter_brush_controller_resets_instance(self):
        # Arrange
        first = controller_module.get_scatter_brush_controller()

        # Act
        controller_module.destroy_scatter_brush_controller()

        # Assert
        self.assertIsNone(controller_module._CONTROLLER_INSTANCE)
        self.assertIsNot(controller_module.get_scatter_brush_controller(), first)

    async def test_destroy_scatter_brush_controller_without_instance_does_not_raise(self):
        # Arrange
        controller_module._CONTROLLER_INSTANCE = None

        # Act
        controller_module.destroy_scatter_brush_controller()

        # Assert
        self.assertIsNone(controller_module._CONTROLLER_INSTANCE)

    async def test_destroy_scatter_brush_controller_destroys_the_live_edit_drivers_bound_to_it(self):
        # Arrange
        controller_module.get_scatter_brush_controller()

        with mock.patch.object(controller_module, "destroy_reapply_drivers") as destroy_reapply_drivers:
            # Act
            controller_module.destroy_scatter_brush_controller()

        # Assert
        destroy_reapply_drivers.assert_called_once_with()


class TestControllerSelectionTurn(omni.kit.test.AsyncTestCase):
    """The turn a patch has been given is shared state: the viewport gizmo writes it and the Scatter window shows it.

    It is kept with the scope it was given to, because a different selection has not been turned by anything and has
    to read as untouched rather than inheriting the last patch's angles.
    """

    async def setUp(self):
        self._previous = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = None
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(mock.patch.object(ScatterBrushSettings, "save_to_carb"))
        self._stack.enter_context(
            mock.patch.object(ScatterBrushSettings, "load_from_carb", return_value=ScatterBrushSettings())
        )
        self._stack.enter_context(mock.patch.object(controller_module, "save_assets_to_carb"))
        self._stack.enter_context(mock.patch.object(controller_module, "load_assets_from_carb", return_value=[]))
        self.controller = controller_module.get_scatter_brush_controller()

    async def tearDown(self):
        self.controller.destroy()
        controller_module._CONTROLLER_INSTANCE = self._previous
        self._stack.close()

    def _scope(self, name: str) -> ReapplyScope:
        """A scope over one named container, standing in for one selection."""
        return ReapplyScope(
            entries=(
                ScopeEntry(
                    prototype_root=Sdf.Path("/RootNode/meshes/mesh_0123456789ABCDEF"),
                    container_names=(name,),
                    placement_paths=(Sdf.Path(f"/RootNode/meshes/mesh_0123456789ABCDEF/{name}/s_000000000001"),),
                ),
            )
        )

    async def test_a_scope_nothing_has_turned_reads_as_untouched(self):
        # Assert
        self.assertEqual(self.controller.selection_turn(self._scope("grass")), (0.0, 0.0, 0.0))

    async def test_a_turn_is_reported_back_for_the_scope_it_was_given_to(self):
        # Arrange
        scope = self._scope("grass")

        # Act
        self.controller.set_selection_turn(scope, (10.0, 20.0, 30.0))

        # Assert
        self.assertEqual(self.controller.selection_turn(scope), (10.0, 20.0, 30.0))

    async def test_selecting_something_else_reads_as_untouched_again(self):
        # Arrange
        self.controller.set_selection_turn(self._scope("grass"), (10.0, 20.0, 30.0))

        # Assert
        self.assertEqual(self.controller.selection_turn(self._scope("glass")), (0.0, 0.0, 0.0))

    async def test_subscribers_are_told_which_scope_was_turned_and_by_how_much(self):
        # Arrange
        seen = []
        subscription = self.controller.subscribe_selection_turn_changed(lambda scope, turn: seen.append((scope, turn)))
        scope = self._scope("grass")

        # Act
        self.controller.set_selection_turn(scope, (0.0, 0.0, 45.0))

        # Assert
        self.assertEqual(seen, [(scope, (0.0, 0.0, 45.0))])
        del subscription

    async def test_setting_the_same_turn_again_tells_nobody(self):
        # Arrange
        scope = self._scope("grass")
        self.controller.set_selection_turn(scope, (0.0, 0.0, 45.0))
        seen = []
        subscription = self.controller.subscribe_selection_turn_changed(lambda *args: seen.append(args))

        # Act
        self.controller.set_selection_turn(scope, (0.0, 0.0, 45.0))

        # Assert
        self.assertEqual(seen, [])
        del subscription


class TestReapplyDriverRegistry(omni.kit.test.AsyncTestCase):
    """One driver per USD context, following whichever brush controller is current."""

    _CONTEXT = "scatter_core_driver_registry_test"

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(mock.patch.object(ScatterBrushSettings, "save_to_carb"))
        self._stack.enter_context(
            mock.patch.object(ScatterBrushSettings, "load_from_carb", return_value=ScatterBrushSettings())
        )
        self._stack.enter_context(mock.patch.object(controller_module, "save_assets_to_carb"))
        self._stack.enter_context(mock.patch.object(controller_module, "load_assets_from_carb", return_value=[]))
        usd_context = mock.Mock()
        usd_context.get_selection.return_value.get_selected_prim_paths.return_value = []
        usd_context.get_stage.return_value = None
        self._stack.enter_context(
            mock.patch.object(controller_module.omni.usd, "get_context", return_value=usd_context)
        )
        self._previous_controller = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = None
        self._previous_drivers = dict(controller_module._DRIVERS)
        controller_module._DRIVERS.clear()

    async def tearDown(self):
        controller_module.destroy_reapply_drivers()
        if controller_module._CONTROLLER_INSTANCE is not None:
            controller_module._CONTROLLER_INSTANCE.destroy()
        controller_module._CONTROLLER_INSTANCE = self._previous_controller
        controller_module._DRIVERS.update(self._previous_drivers)
        self._stack.close()

    def replace_controller(self) -> None:
        """Swap in a new brush controller without telling the drivers, as reloading the extension can."""
        old = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = controller_module.ScatterBrushController()
        if old is not None:
            old.destroy()

    async def test_get_reapply_driver_twice_for_one_context_returns_the_same_driver(self):
        # Arrange
        first = controller_module.get_reapply_driver(self._CONTEXT)

        # Act
        second = controller_module.get_reapply_driver(self._CONTEXT)

        # Assert
        self.assertIs(second, first)

    async def test_get_reapply_driver_for_another_context_returns_another_driver(self):
        # Arrange
        first = controller_module.get_reapply_driver(self._CONTEXT)

        # Act
        other = controller_module.get_reapply_driver(f"{self._CONTEXT}_other")

        # Assert
        self.assertIsNot(other, first)

    async def test_get_reapply_driver_binds_the_driver_to_the_current_brush_controller(self):
        # Act
        driver = controller_module.get_reapply_driver(self._CONTEXT)

        # Assert
        self.assertIs(driver.controller, controller_module.get_scatter_brush_controller())

    async def test_get_reapply_driver_after_destroy_reapply_drivers_creates_a_new_driver(self):
        # Arrange
        first = controller_module.get_reapply_driver(self._CONTEXT)
        controller_module.destroy_reapply_drivers()

        # Act
        second = controller_module.get_reapply_driver(self._CONTEXT)

        # Assert
        self.assertIsNot(second, first)

    async def test_get_reapply_driver_after_the_controller_was_replaced_returns_a_driver_of_the_new_controller(self):
        # Arrange
        first = controller_module.get_reapply_driver(self._CONTEXT)
        self.replace_controller()

        # Act
        second = controller_module.get_reapply_driver(self._CONTEXT)

        # Assert
        self.assertIsNot(second, first)
        self.assertIs(second.controller, controller_module.get_scatter_brush_controller())

    async def test_get_reapply_driver_after_the_controller_was_replaced_destroys_the_old_driver(self):
        # Arrange
        first = controller_module.get_reapply_driver(self._CONTEXT)
        self.replace_controller()

        with mock.patch.object(first, "destroy", wraps=first.destroy) as destroy:
            # Act
            controller_module.get_reapply_driver(self._CONTEXT)

        # Assert
        destroy.assert_called_once()
