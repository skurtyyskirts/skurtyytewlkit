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

__all__ = ["MANIFEST_NAME", "MOD_SKY_LABEL_PREFIX", "ModSky", "ModSkyRigs", "read_mod_skies", "write_mod_sky_rig"]

import configparser
import hashlib
import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

MANIFEST_NAME = "skydomes.ini"
MOD_SKY_LABEL_PREFIX = "Mod Sky: "

# The Remix runtime loads dome textures from DDS files only.
_DOME_TEXTURE_SUFFIX = ".dds"
_DEFAULT_SUN_ELEVATION = 45.0
_DEFAULT_SUN_ANGLE = 0.53
# The proxy rejects any number this large as a typing slip in a manifest that is edited while the game runs.
_LARGEST_NUMBER = 1.0e6
_NUMBER_PATTERN = re.compile(r" *(-?)(\d*)(?:\.(\d*))?")
_SLUG_PATTERN = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class ModSky:
    """One sky a mod shows in game through the Remix API dome.

    Attributes:
        name: The level name, or the texture title for a texture no manifest level uses.
        texture: The lat-long DDS texture of the dome.
        radiance: The linear multiplier on the texture.
        tint: The linear color multiplier on the dome, on top of the radiance.
        rotation: Degrees counter-clockwise about +Z.
        sun: The radiance of the dome's own sun at a radiance of 1, or None when the sky has no sun.
        sun_azimuth: The image azimuth of the texture's sun, in degrees.
        sun_elevation: The elevation of the texture's sun, in degrees.
        sun_angle: The angular diameter of the sun disc, in degrees.
    """

    name: str
    texture: Path
    radiance: float = 1.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: float = 0.0
    sun: tuple[float, float, float] | None = None
    sun_azimuth: float = 0.0
    sun_elevation: float = _DEFAULT_SUN_ELEVATION
    sun_angle: float = _DEFAULT_SUN_ANGLE

    @property
    def sun_travel(self) -> tuple[float, float, float]:
        """The unit direction the sun's light travels in, for a dome turned by its rotation."""
        # The runtime shows a texel at image azimuth phi at world azimuth phi - rotation, measured as atan2(x, y).
        azimuth = math.radians(self.sun_azimuth - self.rotation)
        elevation = math.radians(self.sun_elevation)
        horizontal = math.cos(elevation)
        return (-math.sin(azimuth) * horizontal, -math.cos(azimuth) * horizontal, -math.sin(elevation))


def _parse_number(text: str | None, fallback: float | None) -> float | None:
    """Read a decimal the way the skydome proxy reads one, or return the fallback when it holds no digits."""
    whole_sign, whole, fraction = _NUMBER_PATTERN.match(text or "").groups()
    if not whole and not fraction:
        return fallback
    value = float(f"{whole or '0'}.{fraction or '0'}")
    if value >= _LARGEST_NUMBER:
        return fallback
    return -value if whole_sign else value


def _parse_color(text: str | None) -> tuple[float, float, float] | None:
    """Read three comma-separated non-negative decimals, or return None when any is missing."""
    fields = (text or "").split(",")
    if len(fields) < 3:
        return None
    values = tuple(_parse_number(field, -1.0) for field in fields[:3])
    if any(value < 0.0 for value in values):
        return None
    return values


def _wrap_degrees(degrees: float) -> float:
    """Fold an angle into [-180, 180), or return 0 for an angle too large to be a setting."""
    if not -1.0e7 < degrees < 1.0e7:
        return 0.0
    return (degrees + 180.0) % 360.0 - 180.0


def _texture_title(texture: Path) -> str:
    """Return the label of a texture, written the way the Lighting menu labels its rig files."""
    return string.capwords(texture.stem.replace("_", " "))


def _same_file_key(path: Path) -> str:
    """Return a key under which two spellings of one file path compare equal."""
    return os.path.normcase(os.path.abspath(path))


def _read_level(manifest: Path, section: configparser.SectionProxy) -> ModSky | None:
    """Read one level of a skydome manifest, or return None when its dome texture does not exist."""
    name = section.get("name") or "?"
    texture_name = section.get("texture") or ""
    texture = manifest.parent / texture_name
    if not texture_name or not texture.is_file():
        carb.log_warn(f"Skipping the {name} mod sky of {manifest}: its dome texture '{texture_name}' does not exist.")
        return None

    radiance = _parse_number(section.get("radiance"), 1.0)
    if radiance < 0.0:
        radiance = 1.0
    sun_azimuth = _parse_number(section.get("sunazimuth"), None)
    sun = _parse_color(section.get("sun"))
    sun_scale = max(_parse_number(section.get("sunscale"), 1.0), 0.0)
    sun_elevation = _parse_number(section.get("sunelevation"), _DEFAULT_SUN_ELEVATION)
    if not -90.0 <= sun_elevation <= 90.0:
        sun_elevation = _DEFAULT_SUN_ELEVATION
    sun_angle = _parse_number(section.get("sunangle"), _DEFAULT_SUN_ANGLE)
    # The runtime drops any other disc size.
    if not 0.0 < sun_angle < 180.0:
        sun_angle = _DEFAULT_SUN_ANGLE

    # Without the image azimuth of its sun, the proxy cannot place the sun and draws the dome without it.
    has_sun = sun is not None and sun_azimuth is not None and sun_scale > 0.0 and any(value > 0.0 for value in sun)
    return ModSky(
        name=name,
        texture=texture,
        radiance=radiance,
        tint=_parse_color(section.get("tint")) or (1.0, 1.0, 1.0),
        rotation=_wrap_degrees(_parse_number(section.get("rotation"), 0.0)),
        sun=tuple(value * sun_scale for value in sun) if has_sun else None,
        sun_azimuth=_wrap_degrees(sun_azimuth) if sun_azimuth is not None else 0.0,
        sun_elevation=sun_elevation,
        sun_angle=sun_angle,
    )


def _read_manifest(manifest: Path) -> list[ModSky]:
    """Read the levels of a skydome manifest in the order the skydome proxy numbers them."""
    # Windows INI files, which the proxy reads, tolerate repeated and value-less keys.
    parser = configparser.ConfigParser(interpolation=None, strict=False, allow_no_value=True)
    try:
        parser.read(manifest, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError) as exc:
        carb.log_warn(f"Skipping the mod skies of {manifest}: the manifest could not be read ({exc}).")
        return []

    # Windows INI files also ignore the case of section names.
    sections = {section_name.lower(): parser[section_name] for section_name in parser.sections()}
    summary = sections.get("skydomes")
    level_count = int(_parse_number(summary.get("levels") if summary else None, 0.0))
    skies = []
    for index in range(max(level_count, 0)):
        section = sections.get(f"level{index}")
        if section is None:
            continue
        sky = _read_level(manifest, section)
        if sky:
            skies.append(sky)
    return skies


def read_mod_skies(directory: Path) -> list[ModSky]:
    """Read the skies a mod's skydome folder shows in game.

    The levels of the folder's skydome manifest come first, read the way the skydome proxy reads them. Every other DDS
    texture in the folder follows as a sky of radiance 1 that no level uses yet.

    Args:
        directory: The skydome folder.

    Returns:
        The skies of the folder; empty when the folder does not exist.
    """
    if not directory.is_dir():
        return []
    manifest = directory / MANIFEST_NAME
    skies = _read_manifest(manifest) if manifest.is_file() else []
    used_textures = {_same_file_key(sky.texture) for sky in skies}
    loose_textures = [
        path
        for path in directory.iterdir()
        if path.suffix.lower() == _DOME_TEXTURE_SUFFIX and _same_file_key(path) not in used_textures
    ]
    for texture in sorted(loose_textures, key=lambda path: path.name.lower()):
        skies.append(ModSky(name=_texture_title(texture), texture=texture))
    return skies


def write_mod_sky_rig(sky: ModSky, rig_path: Path, up_axis: str) -> None:
    """Write a viewport light rig that shows a mod sky the way the Remix API dome shows it in game.

    The rig holds a dome light turned about +Z by the sky's rotation and, when the sky has a sun, a distant light
    along the sun's travel direction. The rig uses the stage's up axis so the Lighting menu adds no up-axis correction
    and the dome keeps the game's world orientation.

    Args:
        sky: The sky to show.
        rig_path: The light rig file to write; an open layer of that file is reloaded.
        up_axis: The up axis of the stage the rig is applied to.

    Raises:
        RuntimeError: If the rig file cannot be written.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up_axis)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
    dome.CreateTextureFileAttr(Sdf.AssetPath(sky.texture.as_posix()))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(sky.radiance)
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(*sky.tint))
    UsdGeom.Xformable(dome).AddRotateZOp().Set(sky.rotation)

    if sky.sun:
        peak = max(sky.sun)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(sky.radiance * peak)
        sun.CreateExposureAttr(0.0)
        sun.CreateColorAttr(Gf.Vec3f(*(value / peak for value in sky.sun)))
        sun.CreateAngleAttr(sky.sun_angle)
        # A distant light shines along its local -Z axis.
        travel = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*sky.sun_travel))
        UsdGeom.Xformable(sun).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(travel.GetQuat())

    layer = stage.GetRootLayer()
    layer.documentation = (
        f"RTX Remix viewport light rig for the {sky.name} mod sky, regenerated from {sky.texture.parent}."
    )
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(rig_path)):
        raise RuntimeError(f"Could not write the mod sky light rig {rig_path}")
    opened_layer = Sdf.Layer.Find(str(rig_path))
    if opened_layer:
        opened_layer.Reload(force=True)


class ModSkyRigs:
    """Keep viewport light rigs for the mod skies of the open project up to date."""

    def __init__(self, output_root: Path, sky_directories: list[str]):
        """Create an empty set of mod sky rigs.

        Args:
            output_root: The folder under which each project's rig files are written.
            sky_directories: The skydome folders to read, relative to the project file's folder.
        """
        self._output_root = output_root
        self._sky_directories = sky_directories
        self._signature = None
        self._rigs: list[tuple[str, Path]] = []

    @property
    def rigs(self) -> list[tuple[str, Path]]:
        """The menu label and rig file of every mod sky of the last refreshed project."""
        return list(self._rigs)

    @property
    def labels(self) -> set[str]:
        """The menu labels of the mod sky rigs."""
        return {label for label, _ in self._rigs}

    @staticmethod
    def __folder_signature(directory: Path) -> tuple:
        if not directory.is_dir():
            return ()
        return tuple(
            sorted((entry.name, entry.stat().st_mtime_ns, entry.stat().st_size) for entry in os.scandir(directory))
        )

    def refresh(self, project_file: Path | None, up_axis: str) -> bool:
        """Rewrite the rig files when the project, its skydome folders, or the stage up axis changed.

        Args:
            project_file: The open project file, or None when no saved project is open.
            up_axis: The up axis of the open stage.

        Returns:
            True when the menu labels or rig files changed.

        Raises:
            OSError: If a rig file cannot be removed or its folder cannot be created; the previous rigs are kept.
            RuntimeError: If a rig file cannot be written; the previous rigs are kept.
        """
        directories = [project_file.parent / directory for directory in self._sky_directories] if project_file else []
        signature = (up_axis, tuple((directory, self.__folder_signature(directory)) for directory in directories))
        if signature == self._signature and all(rig_path.is_file() for _, rig_path in self._rigs):
            return False

        rigs = []
        if project_file:
            project_key = hashlib.sha1(_same_file_key(project_file.parent).encode("utf-8")).hexdigest()[:12]
            output_dir = self._output_root / project_key
            if output_dir.is_dir():
                for stale_rig in output_dir.glob("*.usda"):
                    stale_rig.unlink()

            labels = set()
            skies = [sky for directory in directories for sky in read_mod_skies(directory)]
            for index, sky in enumerate(skies):
                label = f"{MOD_SKY_LABEL_PREFIX}{sky.name}"
                duplicate = 2
                while label in labels:
                    label = f"{MOD_SKY_LABEL_PREFIX}{sky.name} ({duplicate})"
                    duplicate += 1
                labels.add(label)
                slug = _SLUG_PATTERN.sub("_", sky.name).strip("_").lower() or "sky"
                rig_path = output_dir / f"{index:02d}_{slug}.usda"
                write_mod_sky_rig(sky, rig_path, up_axis)
                rigs.append((label, rig_path))

        changed = bool(rigs) or rigs != self._rigs
        self._signature = signature
        self._rigs = rigs
        return changed
