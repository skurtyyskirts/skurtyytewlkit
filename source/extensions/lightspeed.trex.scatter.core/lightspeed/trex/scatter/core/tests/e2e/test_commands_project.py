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

import os
import pathlib

import omni.kit.commands
import omni.kit.test
import omni.kit.undo
import omni.usd
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from lightspeed.trex.scatter.core import commands as _commands
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX, IS_REMIX_SCATTER_ATTR
from lightspeed.trex.scatter.core.placement import PlacementRecord, make_relative_asset_path
from omni.flux.utils.tests.context_managers import open_test_project
from pxr import Gf, Sdf, Usd, UsdGeom

_PROJECT_STAGE = "usd/project_example/combined.usda"
_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_0AB745B8BEE1F16B"
_INSTANCE_ROOT = "/RootNode/instances/inst_0AB745B8BEE1F16B_0"
_CONTAINER_NAME = f"{CONTAINER_PREFIX}default"
_PLACEMENT_NAME = "s_e2e000000001"
_XFORM_OP_ORDER = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]


class TestScatterCommandsOnProjectExample(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # The extension registers these commands on startup; register them here too so the test is self-contained,
        # and restore that startup registration in tearDown so later tests in the same process still find them.
        self._commands_were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        self._registered = omni.kit.commands.register_all_commands_in_module(_commands)
        omni.kit.undo.clear_stack()

    async def tearDown(self):
        omni.kit.commands.unregister_module_commands(self._registered)
        if self._commands_were_registered:
            omni.kit.commands.register_all_commands_in_module(_commands)

    async def test_flood_command_authors_placement_under_prototype_and_undo_redo_round_trip(self):
        """A flood placement lands in the replacement layer, composes under the instance, and survives undo/redo."""
        async with open_test_project(_PROJECT_STAGE) as project_url:
            # The user works in the project's replacement layer, exactly like the Toolkit's authoring layer setup.
            stage = omni.usd.get_context().get_stage()
            project_dir = pathlib.Path(project_url.path).parent
            replacements_layer = Sdf.Layer.FindOrOpen((project_dir / "replacements.usda").as_posix())
            stage.SetEditTarget(Usd.EditTarget(replacements_layer))
            edit_layer = stage.GetEditTarget().GetLayer()
            self.assertEqual(edit_layer.identifier, replacements_layer.identifier)

            # One flood record targets the captured prototype and references the ingested cube relative to the
            # replacement layer, which is how the brush keeps mods portable.
            asset_abs_path = (project_dir / "assets" / "ingested" / "cube.usda").as_posix()
            asset_rel_path = make_relative_asset_path(edit_layer, asset_abs_path)
            self.assertFalse(os.path.isabs(asset_rel_path))
            self.assertTrue(asset_rel_path.endswith("assets/ingested/cube.usda"))
            record = PlacementRecord(
                container_path=f"{_PROTOTYPE_ROOT}/{_CONTAINER_NAME}",
                prim_name=_PLACEMENT_NAME,
                asset_rel_path=asset_rel_path,
                asset_abs_path=asset_abs_path,
                translate=(10.0, 20.0, 30.0),
                rotate_xyz=(0.0, 0.0, 45.0),
                scale=(1.0, 1.0, 1.0),
                brush_id="Default",
            )
            instance_placement_path = f"{_INSTANCE_ROOT}/{_CONTAINER_NAME}/{_PLACEMENT_NAME}"

            # The controller commits a flood through the command system so it lands on the undo stack.
            success, created = omni.kit.commands.execute(
                "ScatterFloodCommand",
                context_name="",
                layer_identifier=edit_layer.identifier,
                records=[record.to_dict()],
            )
            self.assertTrue(success)
            self.assertEqual(created, [record.prim_path])

            # The placement is authored under the prototype in the replacement layer only.
            self.assertIsNotNone(edit_layer.GetPrimAtPath(record.prim_path))
            self.assertIsNone(stage.GetRootLayer().GetPrimAtPath(record.prim_path))
            self.assertTrue(stage.GetPrimAtPath(record.prim_path).IsValid())

            # Because the capture instance references the prototype, the placement composes under the instance too,
            # which is what the viewport renders and what the Selection Panel lists as a reference.
            instance_placement = stage.GetPrimAtPath(instance_placement_path)
            self.assertTrue(instance_placement.IsValid())
            self.assertEqual(instance_placement.GetAttribute(IS_REMIX_REF_ATTR).Get(), True)
            self.assertEqual(instance_placement.GetAttribute(IS_REMIX_SCATTER_ATTR).Get(), True)
            self.assertEqual(list(instance_placement.GetAttribute("xformOpOrder").Get()), _XFORM_OP_ORDER)
            self.assertEqual(instance_placement.GetAttribute("xformOp:translate").Get(), Gf.Vec3d(10.0, 20.0, 30.0))
            self.assertEqual(instance_placement.GetAttribute("xformOp:rotateXYZ").Get(), Gf.Vec3f(0.0, 0.0, 45.0))

            # The relative reference resolves against the replacement layer, so the cube's meshes appear under it.
            self.assertTrue(instance_placement.GetChild("Toto").IsA(UsdGeom.Mesh))

            # Undo removes the placement together with the now-empty container from the replacement layer.
            omni.kit.undo.undo()
            self.assertFalse(stage.GetPrimAtPath(record.prim_path).IsValid())
            self.assertFalse(stage.GetPrimAtPath(instance_placement_path).IsValid())
            self.assertIsNone(edit_layer.GetPrimAtPath(record.prim_path))
            self.assertIsNone(edit_layer.GetPrimAtPath(record.container_path))

            # Redo brings the placement back with the same transform.
            omni.kit.undo.redo()
            restored = stage.GetPrimAtPath(instance_placement_path)
            self.assertTrue(restored.IsValid())
            self.assertEqual(restored.GetAttribute("xformOp:translate").Get(), Gf.Vec3d(10.0, 20.0, 30.0))
            self.assertTrue(restored.GetChild("Toto").IsA(UsdGeom.Mesh))
