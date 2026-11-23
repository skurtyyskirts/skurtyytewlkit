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

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import omni.kit.test
from lightspeed.trex.control.stagecraft import light_rigs as _light_rigs_module
from lightspeed.trex.control.stagecraft.light_rigs import LightRigMenu
from omni.kit.viewport.menubar.lighting import actions as _lighting_actions
from omni.kit.viewport.menubar.lighting import menu_container as _lighting_menu_container
from pxr import Usd, UsdGeom, UsdLux

_EXTRA_DIRECTORIES_SETTING = "/exts/lightspeed.trex.control.stagecraft/lightRigs/extraDirectories"
_MOD_SKY_DIRECTORIES_SETTING = "/exts/lightspeed.trex.control.stagecraft/lightRigs/modSkyDirectories"
_RIGS_SETTING = "/exts/omni.kit.viewport.menubar.lighting/rigs"


class TestLightRigMenu(omni.kit.test.AsyncTestCase):
    """Test the extra and mod sky rigs added to the viewport Lighting menu."""

    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.extra_directory = Path(self._temp_dir.name)

    async def tearDown(self):
        self._temp_dir.cleanup()

    def patched_dependencies(self, settings_values: dict):
        """Patch the settings and the StageCraft context the Lighting menu rigs read."""
        settings = MagicMock()
        settings.get.side_effect = settings_values.get
        usd_context = MagicMock()
        usd_context.get_stage.return_value = None
        return (
            settings,
            patch.object(_light_rigs_module.carb.settings, "get_settings", return_value=settings),
            patch.object(_light_rigs_module.omni.usd, "get_context", return_value=usd_context),
        )

    async def test_list_light_rigs_extra_rigs_follow_menu_rigs_without_repeating_a_label(self):
        # Arrange
        (self.extra_directory / "default.usda").write_text("#usda 1.0\n", encoding="utf-8")
        (self.extra_directory / "grey_studio.usda").write_text("#usda 1.0\n", encoding="utf-8")
        menu_default = Path("C:/menu/default.usda")
        _, settings_patch, context_patch = self.patched_dependencies(
            {_EXTRA_DIRECTORIES_SETTING: [str(self.extra_directory)], _MOD_SKY_DIRECTORIES_SETTING: []}
        )
        with settings_patch, context_patch:
            menu = LightRigMenu("test")
            try:
                # Act
                rigs = menu.list_light_rigs(lambda _extension_id: [("Default", menu_default)], "lighting")
            finally:
                menu.destroy()

        # Assert
        self.assertEqual([("Default", menu_default), ("Grey Studio", self.extra_directory / "grey_studio.usda")], rigs)

    async def test_destroy_restores_lighting_menu_rig_listers(self):
        # Arrange
        original_listers = (
            vars(_lighting_actions)["_get_rig_names_and_paths"],
            vars(_lighting_menu_container)["_get_rig_names_and_paths"],
        )
        _, settings_patch, context_patch = self.patched_dependencies({})
        with settings_patch, context_patch:
            menu = LightRigMenu("test")

            # Act
            menu.destroy()

        # Assert
        self.assertEqual(
            original_listers,
            (
                vars(_lighting_actions)["_get_rig_names_and_paths"],
                vars(_lighting_menu_container)["_get_rig_names_and_paths"],
            ),
        )

    async def test_refresh_lighting_menus_switches_to_other_spelling_of_rig_directory(self):
        # Arrange
        settings, settings_patch, _ = self.patched_dependencies({_RIGS_SETTING: "C:/rigs"})
        with settings_patch:
            # Act
            _light_rigs_module._refresh_lighting_menus()

        # Assert
        settings.set.assert_called_once_with(_RIGS_SETTING, "C:/rigs/")

    async def test_is_light_kept_with_mod_sky_keeps_stage_lights_except_domes_and_rig_lights(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        prims = [
            UsdLux.SphereLight.Define(stage, "/World/Lamp").GetPrim(),
            UsdLux.DomeLight.Define(stage, "/World/CapturedSky").GetPrim(),
            UsdLux.DistantLight.Define(stage, "/OmniKit_Viewport_LightRig/Sun").GetPrim(),
            UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim(),
        ]

        # Act
        kept = [_light_rigs_module._is_light_kept_with_mod_sky(prim, prim.GetPath()) for prim in prims]

        # Assert
        self.assertEqual([True, False, False, False], kept)
