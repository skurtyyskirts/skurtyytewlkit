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
    "ScatterAmount",
    "ScatterGroupAssetModel",
    "ScatterGroupCreateRequestModel",
    "ScatterGroupListResponseModel",
    "ScatterGroupModel",
    "ScatterGroupPathParamModel",
    "ScatterGroupResponseModel",
    "ScatterGroupTuningModel",
    "ScatterGroupUpdateRequestModel",
    "ScatterSurfaceListResponseModel",
    "ScatterSurfaceModel",
]

from enum import StrEnum

from omni.flux.service.shared import BaseServiceModel
from pydantic import ConfigDict, Field, model_validator

_GROUP_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$"


class ScatterAmount(StrEnum):
    """How much of its surfaces a scatter group covers; each amount sets a density and an instance cap."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class ScatterGroupAssetModel(BaseServiceModel):
    """One ingested asset a scatter group places, with how often it is picked."""

    path: str = Field(
        min_length=1,
        description=(
            "Absolute path of an ingested USD model, as `remix_get_available_ingested_assets` lists it. Files that "
            "have not been ingested are refused."
        ),
    )
    weight: float = Field(
        1.0,
        gt=0.0,
        le=100.0,
        description="Relative chance of this asset being picked: an asset with weight 2 appears twice as often as 1.",
    )


class ScatterGroupTuningModel(BaseServiceModel):
    """Optional adjustments that refine a scatter group beyond its amount.

    Unknown fields are refused, so a misspelt adjustment fails instead of being ignored.
    """

    model_config = ConfigDict(extra="forbid")

    density: float | None = Field(
        None,
        ge=0.1,
        le=500.0,
        description=(
            "Placements per 50-unit brush disk of surface; overrides the amount's density. Raise it to pack more in."
        ),
    )
    max_instances: int | None = Field(
        None,
        ge=1,
        le=100000,
        description="Cap on the group's placements across all its surfaces; overrides the amount's cap.",
    )
    padding: float | None = Field(
        None,
        ge=0.0,
        le=5000.0,
        description="Smallest distance between two placements, in stage units. Lower it to space them closer.",
    )
    size_min: float | None = Field(
        None,
        ge=0.001,
        le=100000.0,
        description="Smallest height a placement is scaled to, in stage units. Raise both sizes to make them longer.",
    )
    size_max: float | None = Field(
        None,
        ge=0.001,
        le=100000.0,
        description="Largest height a placement is scaled to, in stage units. Must not be below size_min.",
    )
    max_surface_angle: float | None = Field(
        None,
        ge=0.0,
        le=180.0,
        description="Steepest slope, in degrees from the stage up axis, that still receives placements.",
    )
    conform_to_surface: bool | None = Field(
        None, description="Tilt placements with the surface normal instead of standing them straight up."
    )
    vertical_offset: float | None = Field(
        None,
        ge=-10000.0,
        le=10000.0,
        description="Distance placements are moved along their up axis, in stage units; negative sinks them in.",
    )
    rotation_x_min: float | None = Field(None, ge=-180.0, le=180.0, description="Smallest random tilt about X.")
    rotation_x_max: float | None = Field(None, ge=-180.0, le=180.0, description="Largest random tilt about X.")
    rotation_y_min: float | None = Field(None, ge=-180.0, le=180.0, description="Smallest random tilt about Y.")
    rotation_y_max: float | None = Field(None, ge=-180.0, le=180.0, description="Largest random tilt about Y.")
    rotation_z_min: float | None = Field(None, ge=-360.0, le=360.0, description="Smallest random turn about Z.")
    rotation_z_max: float | None = Field(None, ge=-360.0, le=360.0, description="Largest random turn about Z.")


class ScatterGroupModel(BaseServiceModel):
    """A scatter group as it is recorded in the layer it authors."""

    name: str = Field(description="Name of the group.")
    amount: ScatterAmount = Field(description="How much of its surfaces the group covers.")
    tuning: ScatterGroupTuningModel = Field(description="Adjustments applied on top of the amount.")
    assets: list[ScatterGroupAssetModel] = Field(description="Assets the group places.")
    material_paths: list[str] = Field(description="Materials whose captured meshes the group fills.")
    prim_paths: list[str] = Field(description="Captured prims the group fills as well.")
    target_prim_paths: list[str] = Field(description="Captured mesh prototypes the group filled.")
    seed: int = Field(description="Seed of the group's random draws; the same seed reproduces the same layout.")
    placement_count: int = Field(description="Placements the group authored when it was last scattered.")
    max_instances: int = Field(description="Cap the group's placements were held to.")
    prim_limited: bool = Field(
        False,
        description=(
            "Whether the group stopped at the limit on what one scatter may add to the rendered stage. That limit "
            "counts every placement once per instance of the mesh it is on, so it is reached soonest on surfaces "
            "that repeat many times. A higher amount or max_instances adds nothing then."
        ),
    )


class ScatterGroupResponseModel(ScatterGroupModel):
    """A scatter group together with where it is authored."""

    layer_id: str = Field(description="Identifier of the layer the group's placements and record are authored in.")
    capped: bool = Field(
        description="Whether the group reached its cap, so a higher amount or max_instances would add more."
    )


class ScatterGroupListResponseModel(BaseServiceModel):
    """Every scatter group recorded in the layers of the stage."""

    groups: list[ScatterGroupResponseModel] = Field(description="The groups, by layer and then by name.")


class ScatterGroupCreateRequestModel(BaseServiceModel):
    """Request body that creates a scatter group and fills its surfaces."""

    name: str = Field(
        pattern=_GROUP_NAME_PATTERN,
        description="Name of the new group, such as 'grass'. Letters, digits, spaces, '_' and '-'; 64 at most.",
    )
    material_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Material paths from `remix_get_scatter_surfaces`. Every captured mesh bound to one of them is filled."
        ),
    )
    prim_paths: list[str] = Field(
        default_factory=list, description="Captured mesh or instance prim paths to fill in addition to the materials."
    )
    assets: list[ScatterGroupAssetModel] = Field(min_length=1, description="Ingested assets to place.")
    amount: ScatterAmount = Field(
        ScatterAmount.MEDIUM,
        description="'light', 'medium' or 'heavy': how much of the surfaces is covered.",
    )
    tuning: ScatterGroupTuningModel = Field(
        default_factory=ScatterGroupTuningModel, description="Optional adjustments on top of the amount."
    )
    seed: int | None = Field(
        None, ge=0, le=2**31 - 1, description="Seed of the random layout; a random one is drawn when omitted."
    )

    @model_validator(mode="after")
    def _check_targets(self) -> ScatterGroupCreateRequestModel:
        """Refuse a group with nothing to scatter on."""
        if not self.material_paths and not self.prim_paths:
            raise ValueError("Give at least one material path or prim path to scatter on")
        return self


class ScatterGroupUpdateRequestModel(BaseServiceModel):
    """Request body that changes a scatter group and scatters it again in place."""

    amount: ScatterAmount | None = Field(
        None,
        description=(
            "New amount. It also drops any density or max_instances adjustment, since the amount sets both again."
        ),
    )
    tuning: ScatterGroupTuningModel | None = Field(
        None, description="Adjustments to merge over the group's current ones; fields left out keep their value."
    )
    add_assets: list[ScatterGroupAssetModel] = Field(
        default_factory=list,
        description="Ingested assets to add, such as more grass types. An asset already in the group gets the weight.",
    )
    remove_asset_paths: list[str] = Field(default_factory=list, description="Asset paths to take out of the group.")
    material_paths: list[str] | None = Field(None, description="Replacement list of materials to fill.")
    prim_paths: list[str] | None = Field(None, description="Replacement list of captured prims to fill.")
    reseed: bool = Field(False, description="Draw a new random layout instead of keeping the current one.")


class ScatterGroupPathParamModel(BaseServiceModel):
    """Path parameter naming a scatter group."""

    name: str = Field(description="Name of the scatter group.")


class ScatterSurfaceModel(BaseServiceModel):
    """One captured material and the meshes that render with it."""

    number: int = Field(description="Number of the surface; it labels the surface's tile on the contact sheets.")
    material_path: str = Field(description="Prim path of the material; pass it to a scatter group to fill it.")
    texture_path: str | None = Field(description="Albedo texture the material renders with, or null when it has none.")
    preview_path: str | None = Field(description="PNG preview of the albedo texture, or null when none was written.")
    mesh_prim_paths: list[str] = Field(description="Captured mesh prototypes bound to the material.")
    instance_count: int = Field(description="How many instances of those meshes the capture placed in the scene.")


class ScatterSurfaceListResponseModel(BaseServiceModel):
    """The captured surfaces a scatter group can fill."""

    surfaces: list[ScatterSurfaceModel] = Field(description="Surfaces by material path.")
    contact_sheet_paths: list[str] = Field(
        description="PNG images tiling every preview with its surface number, 36 per sheet; empty without previews."
    )
