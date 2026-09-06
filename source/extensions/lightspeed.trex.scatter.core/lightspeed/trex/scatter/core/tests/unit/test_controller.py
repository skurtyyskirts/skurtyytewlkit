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

__all__ = ["TestScatterBrushController", "TestScatterBrushControllerSingleton"]

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import omni.kit.test
from pxr import Gf, Sdf, Usd, UsdGeom

from lightspeed.trex.scatter.core import constants as constants_module
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core.settings import (
    Falloff,
    ScatterAssetEntry,
    ScatterBrushSettings,
    TargetMode,
    UpAxis,
)

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

    async def test_update_settings_with_range_violation_returns_false(self):
        # Arrange
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        # Act
        result = self.controller.update_settings(scale_min=5.0)

        # Assert
        self.assertFalse(result)
        self.assertEqual(self.controller.settings.scale_min, 0.8)
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

        # Act
        result = self.controller.save_preset()

        # Assert
        self.assertTrue(result)
        self.preset_store.save.assert_called_once_with("Grass", self.controller.settings)

    async def test_save_preset_with_name_updates_preset_name_and_saves(self):
        # Arrange
        callback = self._subscribe(self.controller.subscribe_settings_changed)

        # Act
        result = self.controller.save_preset("Rocks")

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.controller.settings.preset_name, "Rocks")
        self.preset_store.save.assert_called_once_with("Rocks", self.controller.settings)
        callback.assert_called_once()

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

        # Act
        self.controller.rename_preset("Old", "New")

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
            records=[{"prim_name": "s_1"}, {"prim_name": "s_2"}],
        )
        patched["generate_flood"].assert_called_once()
        flood_args = patched["generate_flood"].call_args.args
        self.assertIs(flood_args[0], cache)
        self.assertIs(flood_args[1], target)
        self.assertEqual(flood_args[2], self.controller.settings)
        self.assertEqual([asset.path for asset in flood_args[3]], [_ROCK])
        self.assertIs(flood_args[5], patched["PaddingIndex"].return_value)
        self.assertEqual(flood_args[6], UpAxis.Y)
        self.assertIs(flood_args[7], stage.GetEditTarget().GetLayer())
        self.assertEqual(flood_args[8], 100)
        committed.assert_called_once_with(2, False)

    async def test_flood_seeds_padding_index_with_existing_placements(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(padding=0.0)
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
        self.assertEqual([call.args[8] for call in patched["generate_flood"].call_args_list], [3, 1])
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

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(
                stack,
                stage,
                generate_flood={"return_value": [_make_record("s_1")]},
            )
            stack.enter_context(
                mock.patch.object(controller_module, "canonical_instance", return_value=instance.GetPath())
            )
            stack.enter_context(mock.patch.object(controller_module, "find_capture_mesh", return_value=mesh_path))
            stack.enter_context(mock.patch.object(controller_module, "instance_count", return_value=4))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 1)
        target = patched["generate_flood"].call_args.args[1]
        self.assertEqual(target.prototype_root, Sdf.Path(_ANCHOR_PROTOTYPE))
        self.assertEqual(target.parent_instance_root, instance.GetPath())
        self.assertEqual(target.mesh_path, mesh_path)
        self.assertEqual(target.parent_world.ExtractTranslation(), Gf.Vec3d(10.0, 20.0, 30.0))
        self.assertEqual(target.instance_count, 4)

    async def test_flood_with_malformed_anchor_path_posts_error_and_places_nothing(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        self.controller.add_asset(_ROCK, up_axis=UpAxis.Z)
        self.controller.update_settings(target_mode=TargetMode.ANCHOR, anchor_prototype_path="my anchor")
        status_callback = self._subscribe(self.controller.subscribe_status_message)

        with contextlib.ExitStack() as stack:
            patched = self._patch_flood_dependencies(stack, stage)
            canonical_instance = stack.enter_context(mock.patch.object(controller_module, "canonical_instance"))
            find_capture_mesh = stack.enter_context(mock.patch.object(controller_module, "find_capture_mesh"))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["generate_flood"].assert_not_called()
        patched["execute"].assert_not_called()
        canonical_instance.assert_not_called()
        find_capture_mesh.assert_not_called()
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
            find_capture_mesh = stack.enter_context(mock.patch.object(controller_module, "find_capture_mesh"))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 0)
        patched["execute"].assert_not_called()
        find_capture_mesh.assert_not_called()
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
            stack.enter_context(mock.patch.object(controller_module, "canonical_instance", return_value=None))
            stack.enter_context(
                mock.patch.object(
                    controller_module, "find_capture_mesh", side_effect=lambda stage, root: root.AppendChild("mesh")
                )
            )
            stack.enter_context(mock.patch.object(controller_module, "instance_count", return_value=1))

            # Act
            count = self.controller.flood("", mock.Mock())

        # Assert
        self.assertEqual(count, 2)
        self.assertEqual(
            [call.args[1].prototype_root for call in patched["generate_flood"].call_args_list],
            [Sdf.Path("/RootNode/meshes/mesh_A"), Sdf.Path("/RootNode/meshes/mesh_B")],
        )
        self.assertEqual(patched["generate_flood"].call_args_list[0].args[1].parent_world, Gf.Matrix4d(1.0))
        self.assertIsNone(patched["generate_flood"].call_args_list[0].args[1].parent_instance_root)

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
            stack.enter_context(mock.patch.object(controller_module, "canonical_instance", return_value=None))
            stack.enter_context(mock.patch.object(controller_module, "find_capture_mesh", return_value=None))

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
