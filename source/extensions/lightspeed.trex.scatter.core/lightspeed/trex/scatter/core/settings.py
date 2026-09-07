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

__all__ = [
    "RANGE_PAIRS",
    "ApplyTo",
    "EraseScope",
    "Falloff",
    "ScatterAssetEntry",
    "ScatterBrushSettings",
    "ScatterMaterialOverride",
    "ScatterTextureSet",
    "TargetMode",
    "UpAxis",
    "load_assets_from_carb",
    "save_assets_to_carb",
]

import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import carb
import carb.settings
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .constants import ASSETS_SETTING_PATH, BRUSH_SETTINGS_PATH, DEFAULT_PRESET_NAME


class Falloff(StrEnum):
    """Acceptance curve applied to samples as a function of normalized distance from the brush center."""

    CONSTANT = "CONSTANT"
    LINEAR = "LINEAR"
    SMOOTH = "SMOOTH"
    SPHERE = "SPHERE"
    GAUSSIAN = "GAUSSIAN"


class ApplyTo(StrEnum):
    ALL = "ALL"
    SELECTED = "SELECTED"


class TargetMode(StrEnum):
    HIT_SURFACE = "HIT_SURFACE"
    ANCHOR = "ANCHOR"


class EraseScope(StrEnum):
    ALL_SCATTERED = "ALL_SCATTERED"
    BRUSH_ASSETS = "BRUSH_ASSETS"


class UpAxis(StrEnum):
    Y = "Y"
    Z = "Z"


class ScatterMaterialOverride(BaseModel):
    """The texture inputs the brush authors on one shader of a scattered asset.

    Frozen so it can be shared by the shallow ``model_copy`` the controller hands out for every palette read.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    shader_rel_path: str = Field(min_length=1)
    """Shader prim path relative to the asset default prim, e.g. ``XForms/grass/Looks/grass/Shader``.

    A reference without a ``primPath`` maps the referenced default prim's children onto the referencing prim, so the
    default prim name itself is not part of the path under a placement.
    """

    inputs: dict[str, str] = Field(default_factory=dict)
    """Shader input name (``inputs:normalmap_texture``) to texture path."""

    encoding: int | None = None
    """``inputs:encoding`` to author with a normal map, or None to leave the asset's own value alone."""


class ScatterTextureSet(BaseModel):
    """What ``resolve_asset_texture_set`` found for one palette asset."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source_hash: str = ""
    """The asset's ingestion ``base_hash`` when it was resolved, used to notice a re-ingested asset."""

    overrides: tuple[ScatterMaterialOverride, ...] = ()
    """Per-shader inputs the brush will author; empty when the asset needs nothing."""

    assigned: tuple[str, ...] = ()
    """Input names the asset already authors itself, which are never touched."""

    unresolved: tuple[str, ...] = ()
    """Input names that are still missing because no matching texture was found."""

    reason: str = ""
    """Why nothing could be resolved, or empty when the asset was inspected successfully."""

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ScatterAssetEntry(BaseModel):
    """One scatterable asset: an ingested USD file with a selection weight and its authored up axis."""

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    path: str = Field(min_length=1)
    enabled: bool = True
    weight: float = Field(1.0, ge=0.0, le=100.0)
    up_axis: UpAxis = UpAxis.Z
    texture_source: str = ""
    """A texture the user picked for this asset, whose siblings drive resolution instead of auto-detection."""

    textures: ScatterTextureSet | None = Field(default=None, exclude=True)
    """Resolved texture set, rebuilt from disk on demand and deliberately never persisted."""


RANGE_PAIRS = (
    ("rotation_x_min", "rotation_x_max"),
    ("rotation_y_min", "rotation_y_max"),
    ("rotation_z_min", "rotation_z_max"),
    ("scale_min", "scale_max"),
    ("scale_x_min", "scale_x_max"),
    ("scale_y_min", "scale_y_max"),
    ("scale_z_min", "scale_z_max"),
)
"""Settings fields that form a low/high pair and must stay ordered."""


class ScatterBrushSettings(BaseModel):
    """All brush parameters. Bounds are hard clamps; UIs use them for their drag fields."""

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    preset_name: str = DEFAULT_PRESET_NAME

    # Brush
    radius: float = Field(50.0, ge=1.0, le=10000.0)
    falloff: Falloff = Falloff.SMOOTH
    density: float = Field(8.0, ge=0.1, le=500.0)
    strength: float = Field(1.0, ge=0.0, le=1.0)
    stamp_spacing: float = Field(25.0, ge=1.0, le=10000.0)
    padding: float = Field(10.0, ge=0.0, le=5000.0)

    # Placement
    vertical_offset: float = Field(0.0, ge=-10000.0, le=10000.0)
    conform_to_surface: bool = True
    align_to_stroke: bool = False
    rotation_x_min: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_x_max: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_y_min: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_y_max: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_z_min: float = Field(0.0, ge=-360.0, le=360.0)
    rotation_z_max: float = Field(360.0, ge=-360.0, le=360.0)

    # Scale
    scale_enabled: bool = True
    scale_uniform: bool = True
    scale_min: float = Field(0.8, ge=0.01, le=1000.0)
    scale_max: float = Field(1.2, ge=0.01, le=1000.0)
    scale_x_min: float = Field(0.8, ge=0.01, le=1000.0)
    scale_x_max: float = Field(1.2, ge=0.01, le=1000.0)
    scale_y_min: float = Field(0.8, ge=0.01, le=1000.0)
    scale_y_max: float = Field(1.2, ge=0.01, le=1000.0)
    scale_z_min: float = Field(0.8, ge=0.01, le=1000.0)
    scale_z_max: float = Field(1.2, ge=0.01, le=1000.0)
    scale_bias: float = Field(0.0, ge=-1.0, le=1.0)
    scale_weight: float = Field(1.0, ge=0.1, le=10.0)

    # Randomness
    seed: int = Field(0, ge=0, le=2**31 - 1)
    randomize_seed: bool = True

    # Target
    apply_to: ApplyTo = ApplyTo.ALL
    target_mode: TargetMode = TargetMode.HIT_SURFACE
    anchor_prototype_path: str = ""
    erase_scope: EraseScope = EraseScope.ALL_SCATTERED
    flood_max_instances: int = Field(300, ge=1, le=100000)

    # Materials
    assign_textures: bool = True

    @model_validator(mode="after")
    def _check_ranges(self) -> ScatterBrushSettings:
        for low_name, high_name in RANGE_PAIRS:
            low, high = getattr(self, low_name), getattr(self, high_name)
            if low > high:
                raise ValueError(f"{low_name} ({low}) must be lower than or equal to {high_name} ({high})")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> ScatterBrushSettings:
        """Build settings from a JSON-like mapping. Unknown keys are ignored, missing keys use defaults."""
        return cls.model_validate(dict(data))

    def slug(self) -> str:
        """Prim-name-safe identifier for the preset, used to name the scatter container under a prototype."""
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", self.preset_name).strip("_")
        if not cleaned:
            cleaned = "default"
        if cleaned[0].isdigit():
            cleaned = f"p_{cleaned}"
        return cleaned.lower()

    def save_to_carb(self, settings: carb.settings.ISettings | None = None) -> None:
        settings = settings or carb.settings.get_settings()
        settings.set(BRUSH_SETTINGS_PATH, json.dumps(self.to_json_dict()))

    @classmethod
    def load_from_carb(cls, settings: carb.settings.ISettings | None = None) -> ScatterBrushSettings:
        settings = settings or carb.settings.get_settings()
        raw = settings.get(BRUSH_SETTINGS_PATH)
        if not raw or not isinstance(raw, str):
            return cls()
        try:
            return cls.from_json_dict(json.loads(raw))
        except (ValueError, TypeError, ValidationError) as exc:
            carb.log_warn(f"[lightspeed.trex.scatter.core] Ignoring invalid persisted brush settings: {exc}")
            return cls()


def save_assets_to_carb(assets: Sequence[ScatterAssetEntry], settings: carb.settings.ISettings | None = None) -> None:
    settings = settings or carb.settings.get_settings()
    settings.set(ASSETS_SETTING_PATH, json.dumps([asset.model_dump(mode="json") for asset in assets]))


def load_assets_from_carb(settings: carb.settings.ISettings | None = None) -> list[ScatterAssetEntry]:
    settings = settings or carb.settings.get_settings()
    raw = settings.get(ASSETS_SETTING_PATH)
    if not raw or not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
        return [ScatterAssetEntry.model_validate(entry) for entry in data]
    except (ValueError, TypeError, ValidationError) as exc:
        carb.log_warn(f"[lightspeed.trex.scatter.core] Ignoring invalid persisted brush assets: {exc}")
        return []
