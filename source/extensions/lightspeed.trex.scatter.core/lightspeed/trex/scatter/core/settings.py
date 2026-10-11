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
    "ALERT_BUDGET_MULTIPLE",
    "LIVE_EDIT_SOFT_LIMIT",
    "RANGE_PAIRS",
    "REAPPLIABLE_FIELDS",
    "ApplyTo",
    "EraseScope",
    "Falloff",
    "PrimBudgetStatus",
    "ScaleMode",
    "ScatterAssetEntry",
    "ScatterBrushSettings",
    "ScatterMaterialOverride",
    "ScatterTextureSet",
    "TargetMode",
    "UpAxis",
    "load_assets_from_carb",
    "prim_budget_status",
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

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"


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


class ScaleMode(StrEnum):
    """What the placement scale range means.

    ``ABSOLUTE_SIZE`` is the default because a multiplier is meaningless without knowing how large the asset already
    is: ingestion bakes the source file's unit scale into the asset, so the same multiplier gives wildly different
    sizes for two assets that look the same in a DCC.
    """

    ABSOLUTE_SIZE = "ABSOLUTE_SIZE"
    MULTIPLIER = "MULTIPLIER"


class UpAxis(StrEnum):
    Y = "Y"
    Z = "Z"


class PrimBudgetStatus(StrEnum):
    """How the prims a stroke would render compare with the brush prim budget."""

    OK = "OK"
    WARNING = "WARNING"
    ALERT = "ALERT"


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

    meters_per_unit: float = Field(1.0, gt=0.0)
    """Meters per unit of the asset file, so its size can be converted into the units of whatever stage it lands on."""

    natural_extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Size of the asset's default prim along each axis in the asset's own units, or zeros when it was not measured.

    All three components are stored rather than the height alone because the up axis can be overridden after the
    asset was measured, and the height has to follow without re-opening the file.
    """

    texture_source: str = ""
    """A texture the user picked for this asset, whose siblings drive resolution instead of auto-detection."""

    textures: ScatterTextureSet | None = Field(default=None, exclude=True)
    """Resolved texture set, rebuilt from disk on demand and deliberately never persisted."""

    @property
    def natural_height(self) -> float:
        """Size of the asset along its own up axis in its own units; 0.0 when it has not been measured."""
        return self.natural_extent[1 if UpAxis(self.up_axis) == UpAxis.Y else 2]


ALERT_BUDGET_MULTIPLE = 5
"""Multiple of the prim budget at which a stroke counter goes from a warning to an alert."""

LIVE_EDIT_SOFT_LIMIT = 2000
"""Placements above which a live edit waits for the edit to finish instead of following every drag step.

Like the prim budget this only changes how an edit is paced; it never stops one, because the count the artist is
working with is theirs to decide.
"""

RANGE_PAIRS = (
    ("rotation_x_min", "rotation_x_max"),
    ("rotation_y_min", "rotation_y_max"),
    ("rotation_z_min", "rotation_z_max"),
    ("size_min", "size_max"),
    ("scale_min", "scale_max"),
    ("scale_x_min", "scale_x_max"),
    ("scale_y_min", "scale_y_max"),
    ("scale_z_min", "scale_z_max"),
)
"""Settings fields that form a low/high pair and must stay ordered."""

REAPPLIABLE_FIELDS = frozenset(
    {
        "vertical_offset",
        "conform_to_surface",
        "align_to_stroke",
        "base_rotation_x",
        "base_rotation_y",
        "base_rotation_z",
        "rotation_x_min",
        "rotation_x_max",
        "rotation_y_min",
        "rotation_y_max",
        "rotation_z_min",
        "rotation_z_max",
        "scale_mode",
        "size_min",
        "size_max",
        "scale_enabled",
        "scale_uniform",
        "scale_min",
        "scale_max",
        "scale_x_min",
        "scale_x_max",
        "scale_y_min",
        "scale_y_max",
        "scale_z_min",
        "scale_z_max",
        "scale_bias",
        "scale_weight",
        "seed",
    }
)
"""Settings fields a live edit can re-apply to placements that already exist.

Every one of them is an input of ``compose_parent_space_transform``, so a placement that recorded the surface point
it was composed from can be recomposed without knowing anything about the stroke that painted it. The brush fields
(radius, density, strength, stamp_spacing, padding, max_surface_angle) are absent because they decide where stamps
land, which depends on the mouse path the stroke followed and is not recorded.
"""


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
    max_surface_angle: float = Field(180.0, ge=0.0, le=180.0)
    """Steepest surface, in degrees from the stage up axis, that a stamp or a flood still paints on.

    A stamp spreads its candidates over the plane of the surface under the cursor and snaps each one onto the closest
    surface, so a candidate that falls past the edge of a floor lands on the wall beside it and, conformed to that
    wall, stands sideways. Lowering this skips those candidates, and with them any stamp whose cursor landed on a
    wall or a bevel. 180 keeps every candidate, which is what painting on a wall or a ceiling needs.
    """

    # Placement
    vertical_offset: float = Field(0.0, ge=-10000.0, le=10000.0)
    conform_to_surface: bool = True
    align_to_stroke: bool = False
    base_rotation_x: float = Field(0.0, ge=-180.0, le=180.0)
    """Degrees every placement is turned about the stage X axis before its random rotation, about its own root.

    The three base rotations are how an asset that comes out lying down or leaning is stood up for good: unlike a
    turn of painted scatter, they are a brush setting, so they carry into every stroke painted afterwards.
    """
    base_rotation_y: float = Field(0.0, ge=-180.0, le=180.0)
    """Degrees every placement is turned about the stage Y axis before its random rotation; see ``base_rotation_x``."""
    base_rotation_z: float = Field(0.0, ge=-180.0, le=180.0)
    """Degrees every placement is turned about the stage Z axis before its random rotation; see ``base_rotation_x``."""
    rotation_x_min: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_x_max: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_y_min: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_y_max: float = Field(0.0, ge=-180.0, le=180.0)
    rotation_z_min: float = Field(0.0, ge=-360.0, le=360.0)
    rotation_z_max: float = Field(360.0, ge=-360.0, le=360.0)

    # Scale
    scale_mode: ScaleMode = ScaleMode.ABSOLUTE_SIZE
    size_min: float = Field(10.0, ge=0.001, le=100000.0)
    size_max: float = Field(15.0, ge=0.001, le=100000.0)
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
    """Placements one flood may author across every target it fills."""

    prim_budget: int = Field(10000, ge=1, le=10000000)
    """Prims a single stroke is expected to render before the brush flags it as expensive.

    Counted in rendered prims rather than authored placements: a placement under a prototype renders once per
    instance of that prototype's hash, so a stroke on a mesh with many instances costs far more than it authors. The
    budget only colours the stroke counter; it never stops or trims a stroke.
    """

    # Materials
    assign_textures: bool = True

    # Live editing
    edit_selection_live: bool = True
    """Whether changing a re-appliable setting also updates the placements the user has selected.

    Off, a settings change only affects the next stroke, which is how the brush behaved before live editing existed.
    """

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
        """Read the persisted brush settings, keeping every valid one when some of them are not.

        A value that fails validation is dropped on its own and takes its default, so one bad field cannot throw
        away every other setting the user had. Defaults are only used throughout when the persisted value is not a
        JSON object, or when what is left still fails validation.

        Args:
            settings: Settings interface to read from; the application's when omitted.

        Returns:
            The persisted settings, with defaults in place of whatever was missing or invalid.
        """
        settings = settings or carb.settings.get_settings()
        raw = settings.get(BRUSH_SETTINGS_PATH)
        if not raw or not isinstance(raw, str):
            return cls()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            carb.log_warn(f"{_LOG_PREFIX} Ignoring invalid persisted brush settings: {exc}")
            return cls()
        if not isinstance(data, dict):
            carb.log_warn(f"{_LOG_PREFIX} Ignoring persisted brush settings that are not a JSON object")
            return cls()
        try:
            return cls.from_json_dict(data)
        except ValidationError as exc:
            dropped = _failing_fields(exc, data)
        try:
            recovered = cls.from_json_dict({name: value for name, value in data.items() if name not in dropped})
        except ValidationError as exc:
            carb.log_warn(f"{_LOG_PREFIX} Ignoring invalid persisted brush settings: {exc}")
            return cls()
        carb.log_warn(
            f"{_LOG_PREFIX} Reset invalid persisted brush settings to their defaults: {', '.join(sorted(dropped))}"
        )
        return recovered


def prim_budget_status(rendered_prims: int, budget: int) -> PrimBudgetStatus:
    """Classify the prims a stroke would render against the brush prim budget.

    Args:
        rendered_prims: Placements authored so far multiplied by the instance count of the prototype each landed on.
        budget: ``prim_budget`` of the brush settings.

    Returns:
        ``WARNING`` from the budget upward, ``ALERT`` from ``ALERT_BUDGET_MULTIPLE`` times the budget upward, and
        ``OK`` below the budget or when the budget is not positive.
    """
    if budget <= 0:
        return PrimBudgetStatus.OK
    if rendered_prims >= budget * ALERT_BUDGET_MULTIPLE:
        return PrimBudgetStatus.ALERT
    if rendered_prims >= budget:
        return PrimBudgetStatus.WARNING
    return PrimBudgetStatus.OK


def save_assets_to_carb(assets: Sequence[ScatterAssetEntry], settings: carb.settings.ISettings | None = None) -> None:
    settings = settings or carb.settings.get_settings()
    settings.set(ASSETS_SETTING_PATH, json.dumps([asset.model_dump(mode="json") for asset in assets]))


def load_assets_from_carb(settings: carb.settings.ISettings | None = None) -> list[ScatterAssetEntry]:
    """Read the persisted brush palette, keeping every valid entry when some of them are not.

    The palette is written back whenever it changes, so dropping it whole for one bad entry would lose every other
    asset for good.

    Args:
        settings: Settings interface to read from; the application's when omitted.

    Returns:
        The valid entries in palette order; empty when nothing is persisted or the value is not a JSON list.
    """
    settings = settings or carb.settings.get_settings()
    raw = settings.get(ASSETS_SETTING_PATH)
    if not raw or not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        carb.log_warn(f"{_LOG_PREFIX} Ignoring invalid persisted brush assets: {exc}")
        return []
    if not isinstance(data, list):
        carb.log_warn(f"{_LOG_PREFIX} Ignoring persisted brush assets that are not a JSON list")
        return []
    assets: list[ScatterAssetEntry] = []
    dropped: list[str] = []
    for index, entry in enumerate(data):
        try:
            assets.append(ScatterAssetEntry.model_validate(entry))
        except ValidationError:
            path = entry.get("path") if isinstance(entry, dict) else None
            dropped.append(f"'{path}'" if path else f"entry {index}")
    if dropped:
        carb.log_warn(f"{_LOG_PREFIX} Dropped invalid persisted brush assets: {', '.join(dropped)}")
    return assets


def _failing_fields(exc: ValidationError, data: Mapping[str, Any]) -> set[str]:
    """Return the persisted settings fields to drop so that the ones left validate.

    A field outside its own bounds is named in the error's location. A low/high pair that is out of order is only
    reported once every field has passed on its own, and then by the model as a whole with no location, so the pairs
    are checked here over the values that are kept and the defaults of the ones that are not; both ends of a pair
    out of order are dropped.

    Args:
        exc: The error validating the persisted settings.
        data: The persisted settings.

    Returns:
        The names of the fields to drop.
    """
    failing = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
    kept = ScatterBrushSettings().model_dump()
    kept.update((name, value) for name, value in data.items() if name not in failing)
    for low_name, high_name in RANGE_PAIRS:
        if _out_of_order(kept[low_name], kept[high_name]):
            failing.update((low_name, high_name))
    return failing


def _out_of_order(low: Any, high: Any) -> bool:
    """Whether two persisted ends of a low/high pair are out of order, or cannot be compared at all."""
    try:
        return float(low) > float(high)
    except (TypeError, ValueError):
        return True
