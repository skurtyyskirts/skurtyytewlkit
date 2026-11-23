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

import os
import tempfile
from pathlib import Path

import omni.kit.test
from lightspeed.trex.control.stagecraft.mod_skies import (
    MANIFEST_NAME,
    ModSky,
    ModSkyRigs,
    read_mod_skies,
    write_mod_sky_rig,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

_SUN_LEVEL = """
[Skydomes]
Levels=1

[Level0]
Name=Peru - The Lost Valley
Texture=valley.dds
Radiance=0.6375
Tint=1,0.5,0.25
Rotation=200
SunAzimuth=36.3
Sun=1.8,1.6,1.5
SunScale=2
SunElevation=53.1
SunAngle=0.6
"""


class _SkyFolderTestCase(omni.kit.test.AsyncTestCase):
    """Provide a temporary skydome folder for each test."""

    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.folder = Path(self._temp_dir.name)

    async def tearDown(self):
        self._temp_dir.cleanup()

    def write_manifest(self, text: str) -> None:
        """Write the folder's skydome manifest."""
        (self.folder / MANIFEST_NAME).write_text(text, encoding="utf-8")

    def write_texture(self, name: str) -> Path:
        """Write a placeholder dome texture into the folder."""
        texture = self.folder / name
        texture.write_bytes(b"DDS ")
        return texture


class TestModSky(omni.kit.test.AsyncTestCase):
    """Test the sun direction of a mod sky."""

    async def test_sun_travel_unturned_dome_sun_at_image_azimuth_90_shines_toward_minus_x(self):
        # Arrange
        sky = ModSky(name="Level", texture=Path("sky.dds"), sun=(1.0, 1.0, 1.0), sun_azimuth=90.0, sun_elevation=0.0)

        # Act
        travel = sky.sun_travel

        # Assert
        self.assertTrue(Gf.IsClose(Gf.Vec3d(*travel), Gf.Vec3d(-1.0, 0.0, 0.0), 1e-9))

    async def test_sun_travel_dome_turned_by_its_azimuth_shines_toward_minus_y(self):
        # Arrange
        sky = ModSky(
            name="Level",
            texture=Path("sky.dds"),
            rotation=90.0,
            sun=(1.0, 1.0, 1.0),
            sun_azimuth=90.0,
            sun_elevation=0.0,
        )

        # Act
        travel = sky.sun_travel

        # Assert
        self.assertTrue(Gf.IsClose(Gf.Vec3d(*travel), Gf.Vec3d(0.0, -1.0, 0.0), 1e-9))


class TestReadModSkies(_SkyFolderTestCase):
    """Test reading the skies of a mod's skydome folder."""

    async def test_read_mod_skies_manifest_level_reads_values_like_the_proxy(self):
        # Arrange
        texture = self.write_texture("valley.dds")
        self.write_manifest(_SUN_LEVEL)

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertEqual(1, len(skies))
        sky = skies[0]
        self.assertEqual("Peru - The Lost Valley", sky.name)
        self.assertEqual(texture, sky.texture)
        self.assertAlmostEqual(0.6375, sky.radiance)
        self.assertEqual((1.0, 0.5, 0.25), sky.tint)
        self.assertAlmostEqual(-160.0, sky.rotation)
        self.assertEqual((3.6, 3.2, 3.0), tuple(round(value, 6) for value in sky.sun))
        self.assertAlmostEqual(36.3, sky.sun_azimuth)
        self.assertAlmostEqual(53.1, sky.sun_elevation)
        self.assertAlmostEqual(0.6, sky.sun_angle)

    async def test_read_mod_skies_invalid_values_fall_back_to_proxy_defaults(self):
        # Arrange
        self.write_texture("valley.dds")
        self.write_manifest(
            "[Skydomes]\nLevels=1\n[Level0]\nName=Valley\nTexture=valley.dds\nRadiance=-1\nTint=1,2\n"
            "Rotation=north\nSunAzimuth=10\nSun=1,1,1\nSunScale=-3\nSunElevation=120\nSunAngle=0\n"
        )

        # Act
        sky = read_mod_skies(self.folder)[0]

        # Assert
        self.assertEqual(
            (1.0, (1.0, 1.0, 1.0), 0.0, None, 45.0, 0.53),
            (sky.radiance, sky.tint, sky.rotation, sky.sun, sky.sun_elevation, sky.sun_angle),
        )

    async def test_read_mod_skies_sun_without_azimuth_has_no_sun(self):
        # Arrange
        self.write_texture("valley.dds")
        self.write_manifest("[Skydomes]\nLevels=1\n[Level0]\nName=Valley\nTexture=valley.dds\nSun=1,1,1\n")

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertIsNone(skies[0].sun)

    async def test_read_mod_skies_missing_texture_skips_level(self):
        # Arrange
        self.write_manifest(_SUN_LEVEL)

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertEqual([], skies)

    async def test_read_mod_skies_level_count_limits_levels_read(self):
        # Arrange
        self.write_texture("valley.dds")
        self.write_manifest(
            "[Skydomes]\nLevels=1\n[Level0]\nName=First\nTexture=valley.dds\n[Level1]\nName=Second\nTexture=valley.dds\n"
        )

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertEqual(["First"], [sky.name for sky in skies])

    async def test_read_mod_skies_unused_textures_follow_levels_as_plain_skies(self):
        # Arrange
        self.write_texture("valley.dds")
        spare_texture = self.write_texture("kloppenheim_06_puresky.dds")
        self.write_manifest(_SUN_LEVEL)

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertEqual(["Peru - The Lost Valley", "Kloppenheim 06 Puresky"], [sky.name for sky in skies])
        self.assertEqual(
            (spare_texture, 1.0, 0.0, None), (skies[1].texture, skies[1].radiance, skies[1].rotation, skies[1].sun)
        )

    async def test_read_mod_skies_unreadable_manifest_lists_only_loose_textures(self):
        # Arrange
        self.write_texture("valley.dds")
        self.write_manifest("Levels=1\n")

        # Act
        skies = read_mod_skies(self.folder)

        # Assert
        self.assertEqual(["Valley"], [sky.name for sky in skies])

    async def test_read_mod_skies_missing_folder_returns_no_skies(self):
        # Arrange
        missing_folder = self.folder / "missing"

        # Act
        skies = read_mod_skies(missing_folder)

        # Assert
        self.assertEqual([], skies)


class TestWriteModSkyRig(_SkyFolderTestCase):
    """Test writing the viewport light rig of a mod sky."""

    async def test_write_mod_sky_rig_dome_matches_the_api_dome(self):
        # Arrange
        texture = self.write_texture("valley.dds")
        sky = ModSky(name="Valley", texture=texture, radiance=0.25, tint=(1.0, 0.5, 0.25), rotation=66.0)
        rig_path = self.folder / "rigs" / "valley.usda"

        # Act
        write_mod_sky_rig(sky, rig_path, UsdGeom.Tokens.z)

        # Assert
        stage = Usd.Stage.Open(str(rig_path))
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/SkyDome"))
        self.assertEqual(UsdGeom.Tokens.z, UsdGeom.GetStageUpAxis(stage))
        self.assertEqual("/World", str(stage.GetDefaultPrim().GetPath()))
        self.assertEqual(texture.as_posix(), dome.GetTextureFileAttr().Get().path)
        self.assertEqual(UsdLux.Tokens.latlong, dome.GetTextureFormatAttr().Get())
        self.assertAlmostEqual(0.25, dome.GetIntensityAttr().Get())
        self.assertEqual(Gf.Vec3f(1.0, 0.5, 0.25), dome.GetColorAttr().Get())
        self.assertAlmostEqual(66.0, dome.GetPrim().GetAttribute("xformOp:rotateZ").Get())
        self.assertFalse(stage.GetPrimAtPath("/World/Sun").IsValid())

    async def test_write_mod_sky_rig_sun_shines_along_its_travel_direction(self):
        # Arrange
        texture = self.write_texture("valley.dds")
        sky = ModSky(
            name="Valley",
            texture=texture,
            radiance=0.5,
            rotation=66.0,
            sun=(2.0, 1.0, 0.5),
            sun_azimuth=36.3,
            sun_elevation=53.1,
            sun_angle=0.6,
        )
        rig_path = self.folder / "valley.usda"

        # Act
        write_mod_sky_rig(sky, rig_path, UsdGeom.Tokens.z)

        # Assert
        stage = Usd.Stage.Open(str(rig_path))
        sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Sun"))
        world = UsdGeom.Xformable(sun).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        shine = world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        self.assertTrue(Gf.IsClose(shine, Gf.Vec3d(*sky.sun_travel), 1e-6))
        self.assertAlmostEqual(1.0, sun.GetIntensityAttr().Get())
        self.assertEqual(Gf.Vec3f(1.0, 0.5, 0.25), sun.GetColorAttr().Get())
        self.assertAlmostEqual(0.6, sun.GetAngleAttr().Get(), places=6)

    async def test_write_mod_sky_rig_open_layer_shows_rewritten_rig(self):
        # Arrange
        texture = self.write_texture("valley.dds")
        rig_path = self.folder / "valley.usda"
        write_mod_sky_rig(ModSky(name="Valley", texture=texture, radiance=1.0), rig_path, UsdGeom.Tokens.z)
        open_stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(rig_path)))

        # Act
        write_mod_sky_rig(ModSky(name="Valley", texture=texture, radiance=0.125), rig_path, UsdGeom.Tokens.z)

        # Assert
        dome = UsdLux.DomeLight(open_stage.GetPrimAtPath("/World/SkyDome"))
        self.assertAlmostEqual(0.125, dome.GetIntensityAttr().Get())


class TestModSkyRigs(_SkyFolderTestCase):
    """Test keeping the mod sky rigs of a project up to date."""

    async def setUp(self):
        await super().setUp()
        self.project_file = self.folder / "project.usda"
        self.sky_folder = self.folder / "assets" / "skydomes"
        self.sky_folder.mkdir(parents=True)
        self.rigs = ModSkyRigs(self.folder / "rigs", ["assets/skydomes"])

    def write_sky(self, name: str) -> None:
        """Write one loose dome texture into the project's skydome folder."""
        (self.sky_folder / name).write_bytes(b"DDS ")

    async def test_refresh_project_with_skies_writes_one_labelled_rig_per_sky(self):
        # Arrange
        self.write_sky("pizzo_pernice_puresky.dds")
        self.write_sky("qwantani_noon_puresky.dds")

        # Act
        changed = self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Assert
        self.assertTrue(changed)
        self.assertEqual(
            ["Mod Sky: Pizzo Pernice Puresky", "Mod Sky: Qwantani Noon Puresky"],
            [label for label, _ in self.rigs.rigs],
        )
        self.assertTrue(all(rig_path.is_file() for _, rig_path in self.rigs.rigs))

    async def test_refresh_unchanged_folder_reports_no_change(self):
        # Arrange
        self.write_sky("pizzo_pernice_puresky.dds")
        self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Act
        changed = self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Assert
        self.assertFalse(changed)

    async def test_refresh_edited_manifest_rewrites_the_rig(self):
        # Arrange
        self.write_sky("valley.dds")
        manifest = self.sky_folder / MANIFEST_NAME
        manifest.write_text(
            "[Skydomes]\nLevels=1\n[Level0]\nName=Valley\nTexture=valley.dds\nRadiance=1\n", encoding="utf-8"
        )
        self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)
        manifest.write_text(
            "[Skydomes]\nLevels=1\n[Level0]\nName=Valley\nTexture=valley.dds\nRadiance=0.5\n", encoding="utf-8"
        )
        os.utime(manifest, ns=(manifest.stat().st_atime_ns, manifest.stat().st_mtime_ns + 1_000_000_000))

        # Act
        changed = self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Assert
        self.assertTrue(changed)
        stage = Usd.Stage.Open(str(self.rigs.rigs[0][1]))
        self.assertAlmostEqual(0.5, UsdLux.DomeLight(stage.GetPrimAtPath("/World/SkyDome")).GetIntensityAttr().Get())

    async def test_refresh_without_project_clears_rigs(self):
        # Arrange
        self.write_sky("pizzo_pernice_puresky.dds")
        self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Act
        changed = self.rigs.refresh(None, UsdGeom.Tokens.z)

        # Assert
        self.assertTrue(changed)
        self.assertEqual(([], set()), (self.rigs.rigs, self.rigs.labels))

    async def test_refresh_levels_with_one_name_get_numbered_labels(self):
        # Arrange
        self.write_sky("valley.dds")
        (self.sky_folder / MANIFEST_NAME).write_text(
            "[Skydomes]\nLevels=2\n[Level0]\nName=Valley\nTexture=valley.dds\n[Level1]\nName=Valley\nTexture=valley.dds\n",
            encoding="utf-8",
        )

        # Act
        self.rigs.refresh(self.project_file, UsdGeom.Tokens.z)

        # Assert
        self.assertEqual(["Mod Sky: Valley", "Mod Sky: Valley (2)"], [label for label, _ in self.rigs.rigs])
