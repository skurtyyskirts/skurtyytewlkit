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

import shutil
import tempfile
from pathlib import Path

import carb.settings
import carb.tokens
import omni.kit.app
import omni.usd
from lightspeed.trex.contexts.setup import Contexts as _Contexts
from omni.kit.test import AsyncTestCase
from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode
from pxr import Usd, UsdGeom, UsdLux

_LIGHTING_EXTENSION_ID = "omni.kit.viewport.menubar.lighting"
_LIGHT_RIG_PRIM_PATH = "/OmniKit_Viewport_LightRig"
_RIGS_SETTING = f"/exts/{_LIGHTING_EXTENSION_ID}/rigs"
_FIXTURE_DOME_TEXTURE = (
    "${lightspeed.trex.app.resources}/data/tests/usd/project_example/assets/ingested/16px_Diffuse.dds"
)
_MOD_SKY_MANIFEST = """
[Skydomes]
Levels=1

[Level0]
Name=Test Valley
Texture=test_valley.dds
Radiance=0.25
Rotation=66
SunAzimuth=36.3
Sun=1.8,1.6,1.5
SunElevation=53.1
"""


class TestLightingRigSwap(AsyncTestCase):
    async def setUp(self):
        self._settings = carb.settings.get_settings()
        self._original_rigs = self._settings.get(_RIGS_SETTING)
        self._usd_context = omni.usd.get_context(_Contexts.STAGE_CRAFT.value)
        await self._usd_context.new_stage_async()

    async def tearDown(self):
        self._settings.set(_RIGS_SETTING, self._original_rigs)
        await self._usd_context.close_stage_async()

    async def test_set_lighting_mode_rig_invalid_usd_file_raises_without_authoring_reference(self):
        # Arrange
        stage = self._usd_context.get_stage()
        self.assertIsNotNone(stage)
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_rig_path = Path(temp_dir) / "invalid_rig.usda"
            invalid_rig_path.write_text("not a valid usd file", encoding="utf-8")
            self._settings.set(_RIGS_SETTING, temp_dir)

            # Act
            with self.assertRaisesRegex(RuntimeError, "Invalid light rig USD file"):
                _set_lighting_mode("Invalid Rig", usd_context=self._usd_context)

            # Assert
            light_rig_prim = stage.GetPrimAtPath(_LIGHT_RIG_PRIM_PATH)
            if light_rig_prim.IsValid():
                self.assertNotIn(str(invalid_rig_path), str(light_rig_prim.GetMetadata("references")))


class TestModSkyLightRigs(AsyncTestCase):
    async def setUp(self):
        self._usd_context = omni.usd.get_context(_Contexts.STAGE_CRAFT.value)
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

        # A saved project whose mod has one skydome level next to its dome texture.
        project_dir = Path(self._temp_dir.name)
        sky_dir = project_dir / "assets" / "skydomes"
        sky_dir.mkdir(parents=True)
        self._dome_texture = sky_dir / "test_valley.dds"
        shutil.copy(carb.tokens.get_tokens_interface().resolve(_FIXTURE_DOME_TEXTURE), self._dome_texture)
        (sky_dir / "skydomes.ini").write_text(_MOD_SKY_MANIFEST, encoding="utf-8")

        # The level's captured lights: an ordinary light and the capture's own sky dome.
        project_file = project_dir / "project.usda"
        stage = Usd.Stage.CreateNew(str(project_file))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdLux.SphereLight.Define(stage, "/RootNode/lights/LevelLight")
        UsdLux.DomeLight.Define(stage, "/RootNode/lights/SkyDome_Non_Replaceable")
        stage.Save()
        await self._usd_context.open_stage_async(str(project_file))
        # Stage events are delivered on the next updates, and the Lighting menu rigs follow the opened stage there.
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    async def tearDown(self):
        await self._usd_context.close_stage_async()
        self._temp_dir.cleanup()

    def _visibility(self, prim_path: str) -> str:
        return UsdGeom.Imageable(self._usd_context.get_stage().GetPrimAtPath(prim_path)).ComputeVisibility()

    async def test_set_lighting_mode_mod_sky_shows_game_dome_with_level_lights(self):
        # Apply the level's mod sky the way the Lighting menu applies a rig.
        applied, lighting_mode, _ = _set_lighting_mode("Mod Sky: Test Valley", usd_context=self._usd_context)
        await omni.kit.app.get_app().next_update_async()

        # The rig is the Remix API dome of the level: its texture, radiance and rotation, plus the dome's own sun.
        self.assertTrue(applied)
        self.assertEqual("Mod Sky: Test Valley", lighting_mode)
        stage = self._usd_context.get_stage()
        dome = UsdLux.DomeLight(stage.GetPrimAtPath(f"{_LIGHT_RIG_PRIM_PATH}/SkyDome"))
        self.assertTrue(dome)
        self.assertEqual(self._dome_texture.as_posix(), dome.GetTextureFileAttr().Get().path)
        self.assertAlmostEqual(0.25, dome.GetIntensityAttr().Get())
        self.assertAlmostEqual(66.0, dome.GetPrim().GetAttribute("xformOp:rotateZ").Get())
        sun = UsdLux.DistantLight(stage.GetPrimAtPath(f"{_LIGHT_RIG_PRIM_PATH}/Sun"))
        self.assertAlmostEqual(0.25 * 1.8, sun.GetIntensityAttr().Get(), places=6)

        # As in game, the level's lights stay lit and only the capture's sky dome gives way to the mod sky.
        self.assertEqual(UsdGeom.Tokens.inherited, self._visibility("/RootNode/lights/LevelLight"))
        self.assertEqual(UsdGeom.Tokens.invisible, self._visibility("/RootNode/lights/SkyDome_Non_Replaceable"))

    async def test_set_lighting_mode_studio_rig_after_mod_sky_replaces_level_lights(self):
        # Switch from the mod sky to Kit's stock studio rig, which the StageCraft Lighting menu still lists.
        _set_lighting_mode("Mod Sky: Test Valley", usd_context=self._usd_context)
        applied, _, _ = _set_lighting_mode("Default", usd_context=self._usd_context)
        await omni.kit.app.get_app().next_update_async()

        # Ordinary rigs keep replacing every stage light.
        self.assertTrue(applied)
        self.assertEqual(UsdGeom.Tokens.invisible, self._visibility("/RootNode/lights/LevelLight"))

    async def test_set_lighting_mode_lookdev_hdri_applies_in_stagecraft(self):
        # The StageCraft viewport lists the lookdev HDRI rigs without IngestCraft being loaded.
        applied, lighting_mode, _ = _set_lighting_mode("Bush Restaurant", usd_context=self._usd_context)

        # The HDRI rig is referenced onto the light rig prim.
        self.assertTrue(applied)
        self.assertEqual("Bush Restaurant", lighting_mode)
        rig_prim = self._usd_context.get_stage().GetPrimAtPath(_LIGHT_RIG_PRIM_PATH)
        self.assertIn("rtx-remix-lookdev-lighting", str(rig_prim.GetMetadata("references")))
