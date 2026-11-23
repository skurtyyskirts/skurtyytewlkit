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

__all__ = ["ScatterService"]

from lightspeed.trex.scatter.core import (
    ScatterGroupCreateRequestModel,
    ScatterGroupListResponseModel,
    ScatterGroupPathParamModel,
    ScatterGroupResponseModel,
    ScatterGroupsCore,
    ScatterGroupUpdateRequestModel,
    ScatterSurfaceListResponseModel,
)
from omni.flux.service.factory import ServiceBase


class ScatterService(ServiceBase):
    def __init__(self, context_name: str = ""):
        """
        A service class that provides access to scatter groups in a RestAPI.

        Args:
            context_name: The USD context name
        """

        self.__groups_core = ScatterGroupsCore(context_name=context_name)

        super().__init__()

    @classmethod
    @property
    def prefix(cls) -> str:
        return "/scatter"

    def register_endpoints(self):
        group_name_description = "Name of the scatter group"

        @self.router.get(
            path="/surfaces",
            operation_id="get_scatter_surfaces",
            description=(
                "List the captured surfaces a scatter group can fill: each material with the albedo texture it renders "
                "with and the captured meshes bound to it. Capture textures are named by hash, so look at the numbered "
                "contact sheets to recognize surfaces such as grass, then pass their material paths to "
                "`remix_create_scatter_group`."
            ),
            response_model=ScatterSurfaceListResponseModel,
        )
        async def get_scatter_surfaces(
            previews: bool = ServiceBase.describe_query_param(
                True, "Write a PNG preview of each texture and numbered contact sheets of them all"
            ),
        ) -> ScatterSurfaceListResponseModel:
            try:
                return await self.__groups_core.get_surfaces(previews=previews)
            except ValueError as e:
                ServiceBase.raise_error(400, e)

        @self.router.get(
            path="/groups",
            operation_id="get_scatter_groups",
            description=(
                "List the scatter groups of the project with their assets, surfaces, amount, tuning and how many "
                "placements each authored."
            ),
            response_model=ScatterGroupListResponseModel,
        )
        async def get_scatter_groups() -> ScatterGroupListResponseModel:
            try:
                return self.__groups_core.get_groups()
            except ValueError as e:
                ServiceBase.raise_error(400, e)

        @self.router.post(
            path="/groups",
            operation_id="create_scatter_group",
            description=(
                "Create a named scatter group that fills every captured mesh bound to the given materials, and any "
                "given captured prims, with the given ingested assets, in the current edit target layer and in one "
                "undo step. The amount sets coverage: 'light', 'medium' or 'heavy'. Tuning refines it: size_min and "
                "size_max are the height range in stage units, padding the smallest spacing, density the packing. "
                "The response says how many placements were made and whether the amount's cap was reached. Save the "
                "layer afterwards to keep the result."
            ),
            response_model=ScatterGroupResponseModel,
        )
        async def create_scatter_group(body: ScatterGroupCreateRequestModel) -> ScatterGroupResponseModel:
            try:
                return self.__groups_core.create_group(body)
            except ValueError as e:
                ServiceBase.raise_error(400, e)

        @self.router.put(
            path="/groups/{name}",
            operation_id="update_scatter_group",
            description=(
                "Change a scatter group and scatter it again in place, in one undo step: a new amount, merged tuning "
                "such as longer (raise size_min and size_max) or closer together (lower padding, raise density), more "
                "or fewer asset types, other surfaces, or a new random layout. Fields left out keep their value and "
                "the layout stays the same unless reseed is set. Save the layer afterwards to keep the result."
            ),
            response_model=ScatterGroupResponseModel,
        )
        async def update_scatter_group(
            body: ScatterGroupUpdateRequestModel,
            name: ScatterGroupPathParamModel = ServiceBase.validate_path_param(
                ScatterGroupPathParamModel, description=group_name_description
            ),
        ) -> ScatterGroupResponseModel:
            try:
                return self.__groups_core.update_group(name.name, body)
            except LookupError as e:
                ServiceBase.raise_error(404, e)
            except ValueError as e:
                ServiceBase.raise_error(400, e)

        @self.router.delete(
            path="/groups/{name}",
            operation_id="remove_scatter_group",
            description=(
                "Remove a scatter group: every placement it authored and its record, in one undo step. Only the "
                "group's own scatter is removed; captured meshes and scatter painted by hand are left alone. Save the "
                "layer afterwards to keep the result."
            ),
            response_model=ScatterGroupResponseModel,
        )
        async def remove_scatter_group(
            name: ScatterGroupPathParamModel = ServiceBase.validate_path_param(
                ScatterGroupPathParamModel, description=group_name_description
            ),
        ) -> ScatterGroupResponseModel:
            try:
                return self.__groups_core.remove_group(name.name)
            except LookupError as e:
                ServiceBase.raise_error(404, e)
            except ValueError as e:
                ServiceBase.raise_error(400, e)
