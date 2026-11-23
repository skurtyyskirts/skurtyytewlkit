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

import json
from pathlib import Path

import omni.kit.undo
import omni.usd
from omni.flux.utils.common.api import send_request
from omni.flux.utils.tests.context_managers import open_test_project
from omni.kit.test import AsyncTestCase
from omni.services.core import main
from pxr import Sdf, Usd

from lightspeed.trex.service.core.routes.stagecraft.scatter import ScatterService

_TEST_PROJECT = "usd/project_example/combined.usda"
_MATERIAL = "/RootNode/Looks/mat_BC868CE5A075ABB1"
# The capture binds these two prototypes to _MATERIAL; the replacement layer swaps the third one's capture mesh for a
# cube, which renders with the cube's own material and so is a surface of its own.
_PROTOTYPES = [
    "/RootNode/meshes/mesh_0AB745B8BEE1F16B",
    "/RootNode/meshes/mesh_CED45075A077A49A",
]
_REPLACED_PROTOTYPE = "/RootNode/meshes/mesh_BAC90CAA733B0859"
_GROUP_SPECS = ("instanced_group_grass", "scatter_group_grass")
_GROUPS_KEY = "remixScatterGroups"


class TestScatterService(AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self.service = ScatterService()
        main.register_router(router=self.service.router, prefix=self.service.prefix)
        omni.kit.undo.clear_stack()

    # After running each test
    async def tearDown(self):
        main.deregister_router(router=self.service.router, prefix=self.service.prefix)
        self.service = None
        omni.kit.undo.clear_stack()

    async def test_get_scatter_surfaces_lists_rendered_materials_with_meshes_and_writes_previews(self):
        async with open_test_project(_TEST_PROJECT):
            # Act
            response = await send_request("GET", f"{self.service.prefix}/surfaces", raw_response=True)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            body = response.json()
            self.assertEqual([surface["number"] for surface in body["surfaces"]], [1, 2])
            capture, replaced = body["surfaces"]
            self.assertEqual(capture["material_path"], _MATERIAL)
            self.assertEqual(Path(capture["texture_path"]).name, "BC868CE5A075ABB1.dds")
            self.assertEqual(sorted(capture["mesh_prim_paths"]), _PROTOTYPES)
            self.assertGreater(capture["instance_count"], 0)
            self.assertTrue(Path(capture["preview_path"]).is_file())
            self.assertEqual(replaced["mesh_prim_paths"], [_REPLACED_PROTOTYPE])
            self.assertEqual(len(body["contact_sheet_paths"]), 1)
            self.assertTrue(Path(body["contact_sheet_paths"][0]).is_file())

    async def test_get_scatter_surfaces_without_previews_returns_no_image_paths(self):
        async with open_test_project(_TEST_PROJECT):
            # Act
            response = await send_request("GET", f"{self.service.prefix}/surfaces?previews=false", raw_response=True)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            body = response.json()
            self.assertEqual([surface["preview_path"] for surface in body["surfaces"]], [None, None])
            self.assertEqual(body["contact_sheet_paths"], [])

    async def test_create_scatter_group_floods_material_meshes_and_records_group_in_edit_layer(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)

            # Act
            response = await self._create_group(project_url)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            body = response.json()
            self.assertEqual(body["layer_id"], layer.identifier)
            self.assertEqual(sorted(body["target_prim_paths"]), _PROTOTYPES)
            self.assertGreater(body["placement_count"], 0)
            self.assertEqual(self._authored_placements(layer), body["placement_count"])
            self.assertFalse(body["prim_limited"])
            record = json.loads(layer.customLayerData[_GROUPS_KEY]["grass"])
            self.assertEqual(record["material_paths"], [_MATERIAL])
            self.assertFalse(Path(record["assets"][0]["path"]).is_absolute())
            self.assertEqual(layer.customLayerData["lightspeed_layer_type"], "replacement")

    async def test_get_scatter_groups_lists_created_group_with_absolute_asset_paths(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            self._edit_replacements_layer(project_url)
            created = (await self._create_group(project_url)).json()

            # Act
            response = await send_request("GET", f"{self.service.prefix}/groups", raw_response=True)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            groups = response.json()["groups"]
            self.assertEqual([group["name"] for group in groups], ["grass"])
            self.assertEqual(groups[0]["placement_count"], created["placement_count"])
            self.assertEqual(
                Path(groups[0]["assets"][0]["path"]), Path(project_url.path).parent / "assets/ingested/cube.usda"
            )

    async def test_create_scatter_group_is_undone_in_one_step(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            await self._create_group(project_url)

            # Act
            omni.kit.undo.undo()

            # Assert
            self.assertEqual(self._authored_placements(layer), 0)
            self.assertNotIn(_GROUPS_KEY, layer.customLayerData)

    async def test_create_scatter_group_with_taken_name_returns_400_and_keeps_first_group(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            first = (await self._create_group(project_url)).json()

            # Act
            response = await self._create_group(project_url)

            # Assert
            self.assertEqual(response.status_code, 400, msg=response.json())
            self.assertEqual(self._authored_placements(layer), first["placement_count"])

    async def test_create_scatter_group_with_unknown_material_returns_400_without_editing(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)

            # Act
            response = await self._create_group(project_url, material_paths=["/RootNode/Looks/mat_0000000000000000"])

            # Assert
            self.assertEqual(response.status_code, 400, msg=response.json())
            self.assertIn("mat_0000000000000000", response.json()["detail"])
            self.assertNotIn(_GROUPS_KEY, layer.customLayerData)

    async def test_create_scatter_group_with_asset_not_ingested_returns_400_without_editing(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)

            # Act
            response = await self._create_group(project_url, asset="assets/models/cube.usda")

            # Assert
            self.assertEqual(response.status_code, 400, msg=response.json())
            self.assertIn("not an ingested asset", response.json()["detail"])
            self.assertEqual(self._authored_placements(layer), 0)

    async def test_create_scatter_group_without_surfaces_or_with_unknown_tuning_returns_422(self):
        for name, payload in (
            ("no_surfaces", {"material_paths": []}),
            ("misspelt_tuning", {"tuning": {"lenght": 30.0}}),
        ):
            with self.subTest(name=name):
                async with open_test_project(_TEST_PROJECT) as project_url:
                    # Arrange
                    self._edit_replacements_layer(project_url)

                    # Act
                    response = await self._create_group(project_url, **payload)

                    # Assert
                    self.assertEqual(response.status_code, 422, msg=response.json())

    async def test_update_scatter_group_with_heavier_amount_replaces_placements_with_more(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            light = (await self._create_group(project_url, amount="light")).json()

            # Act
            response = await send_request(
                "PUT", f"{self.service.prefix}/groups/grass", json={"amount": "heavy"}, raw_response=True
            )

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            heavy = response.json()
            self.assertEqual(heavy["amount"], "heavy")
            self.assertGreater(heavy["placement_count"], light["placement_count"])
            self.assertEqual(self._authored_placements(layer), heavy["placement_count"])

    async def test_update_scatter_group_with_taller_size_keeps_layout_and_scales_placements_up(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            before = (await self._create_group(project_url)).json()
            tallest_before = max(self._authored_scales(layer))

            # Act
            response = await send_request(
                "PUT",
                f"{self.service.prefix}/groups/grass",
                json={"tuning": {"size_min": 60.0, "size_max": 75.0}},
                raw_response=True,
            )

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            after = response.json()
            self.assertEqual(after["tuning"]["size_min"], 60.0)
            self.assertEqual(after["seed"], before["seed"])
            self.assertEqual(after["placement_count"], before["placement_count"])
            self.assertGreater(min(self._authored_scales(layer)), tallest_before)

    async def test_update_scatter_group_adds_asset_types(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            self._edit_replacements_layer(project_url)
            await self._create_group(project_url)
            textured = (Path(project_url.path).parent / "assets/ingested/cube_textured.usda").as_posix()

            # Act
            response = await send_request(
                "PUT",
                f"{self.service.prefix}/groups/grass",
                json={"add_assets": [{"path": textured, "weight": 2.0}]},
                raw_response=True,
            )

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            assets = response.json()["assets"]
            self.assertEqual([Path(asset["path"]).name for asset in assets], ["cube.usda", "cube_textured.usda"])
            self.assertEqual(assets[1]["weight"], 2.0)

    async def test_update_or_remove_unknown_scatter_group_returns_404(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            self._edit_replacements_layer(project_url)

            for method, kwargs in (("PUT", {"json": {"amount": "heavy"}}), ("DELETE", {})):
                with self.subTest(method=method):
                    # Act
                    response = await send_request(
                        method, f"{self.service.prefix}/groups/missing", raw_response=True, **kwargs
                    )

                    # Assert
                    self.assertEqual(response.status_code, 404, msg=response.json())

    async def test_remove_scatter_group_deletes_its_placements_and_record(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            await self._create_group(project_url)

            # Act
            response = await send_request("DELETE", f"{self.service.prefix}/groups/grass", raw_response=True)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            self.assertEqual(self._authored_placements(layer), 0)
            self.assertNotIn(_GROUPS_KEY, layer.customLayerData)

    async def test_remove_scatter_group_after_update_deletes_every_instancer_the_group_merged_into(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            await self._create_group(project_url)
            await send_request(
                "PUT", f"{self.service.prefix}/groups/grass", json={"amount": "heavy"}, raw_response=True
            )

            # Act
            response = await send_request("DELETE", f"{self.service.prefix}/groups/grass", raw_response=True)

            # Assert
            self.assertEqual(response.status_code, 200, msg=response.json())
            self.assertEqual(self._group_instancers(layer), [])
            self.assertEqual(self._authored_placements(layer), 0)

    async def test_remove_scatter_group_is_undone_in_one_step(self):
        async with open_test_project(_TEST_PROJECT) as project_url:
            # Arrange
            layer = self._edit_replacements_layer(project_url)
            created = (await self._create_group(project_url)).json()
            await send_request("DELETE", f"{self.service.prefix}/groups/grass", raw_response=True)

            # Act
            omni.kit.undo.undo()

            # Assert
            self.assertEqual(self._authored_placements(layer), created["placement_count"])
            self.assertIn("grass", layer.customLayerData[_GROUPS_KEY])

    def _edit_replacements_layer(self, project_url) -> Sdf.Layer:
        """Make the project's replacement layer the edit target, as the Toolkit does for a mod."""
        stage = omni.usd.get_context().get_stage()
        layer = Sdf.Layer.FindOrOpen((Path(project_url.path).parent / "replacements.usda").as_posix())
        stage.SetEditTarget(Usd.EditTarget(layer))
        return layer

    async def _create_group(
        self,
        project_url,
        asset: str = "assets/ingested/cube.usda",
        material_paths: list[str] | None = None,
        **payload,
    ):
        """Send a create request for a group named 'grass' on the fixture's material."""
        body = {
            "name": "grass",
            "material_paths": [_MATERIAL] if material_paths is None else material_paths,
            "assets": [{"path": (Path(project_url.path).parent / asset).as_posix()}],
            "seed": 7,
            **payload,
        }
        return await send_request("POST", f"{self.service.prefix}/groups", json=body, raw_response=True)

    def _authored_placements(self, layer: Sdf.Layer) -> int:
        """Count the group's instanced entries and loose placements on the fixture prototypes."""
        count = 0
        for instancer in self._group_instancers(layer):
            count += len(layer.GetAttributeAtPath(f"{instancer.path}.protoIndices").default)
        for prototype in _PROTOTYPES:
            loose = layer.GetPrimAtPath(f"{prototype}/{_GROUP_SPECS[1]}")
            count += len(loose.nameChildren) if loose is not None else 0
        return count

    def _authored_scales(self, layer: Sdf.Layer) -> list[float]:
        """Return the uniform scale of every instanced entry of the group on the fixture prototypes."""
        return [
            scale[0]
            for instancer in self._group_instancers(layer)
            for scale in layer.GetAttributeAtPath(f"{instancer.path}.scales").default
        ]

    def _group_instancers(self, layer: Sdf.Layer) -> list[Sdf.PrimSpec]:
        """Return the group's instancers on the fixture prototypes.

        A merge names the instancer ``instanced_group_grass``, or ``instanced_group_grass_<n>`` while the viewport's
        stand-in of a just-removed one still holds that name on the stage.
        """
        instancers = []
        for prototype in _PROTOTYPES:
            spec = layer.GetPrimAtPath(prototype)
            if spec is None:
                continue
            instancers.extend(child for child in spec.nameChildren.values() if child.name.startswith(_GROUP_SPECS[0]))
        return instancers
