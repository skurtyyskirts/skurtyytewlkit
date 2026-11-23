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

import pathlib
import tempfile

import numpy as np
import omni.kit.commands
import omni.kit.test
import omni.kit.undo
from lightspeed.trex.scatter.core import commands as _commands
from lightspeed.trex.scatter.core.constants import (
    CONTAINER_PREFIX,
    INSTANCED_ROTATIONS_ATTR,
    INSTANCED_TRANSLATES_ATTR,
    INSTANCER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    PROTOTYPES_SCOPE_NAME,
    SCATTER_KEY_ATTR,
    SCATTER_TRANSFORM_KEY_ATTR,
    SCATTER_TURN_ATTR,
    SCATTER_YAW_ATTR,
)
from lightspeed.trex.scatter.core.instancing import (
    InstancedEraser,
    euler_xyz_to_quaternions,
    instanced_entry_count,
    instanced_points,
    instanced_pool,
    instancers_of_selection,
    is_scatter_instancer,
    merge_loose_placements,
    quaternions_to_euler_xyz,
    read_block,
    split_instancers,
)
from lightspeed.trex.scatter.core.placement import PlacementRecord, author_placements, scene_scatter_cost
from lightspeed.trex.scatter.core.reapply import is_inside_instancer
from lightspeed.trex.scatter.core.settings import EraseScope
from lightspeed.trex.scatter.core.targets import ScatterTarget
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_0123456789ABCDEF"
_INSTANCE_ROOT = "/RootNode/instances/inst_0123456789ABCDEF_0"
_CONTAINER_PATH = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_INSTANCER_PATH = f"{_PROTOTYPE_ROOT}/{INSTANCER_PREFIX}default"
_OTHER_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_FEDCBA9876543210"
_OTHER_CONTAINER_PATH = f"{_OTHER_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_SHADER_REL_PATH = "XForms/Looks/brick/Shader"
_PARENT_OFFSET = Gf.Vec3d(100.0, 0.0, 0.0)
_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{
    double3 xformOp:translate = (5, 6, 7)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "Cube"
    {
    }
}
"""


def _write_asset(directory: tempfile.TemporaryDirectory, file_name: str) -> str:
    path = pathlib.Path(directory.name) / file_name
    path.write_text(_ASSET_USDA, encoding="utf-8")
    return str(path).replace("\\", "/")


def _material(texture: str = "./textures/brick.a.rtex.dds") -> list[dict]:
    """One shader override payload of the shape ``build_materials_payload`` produces."""
    return [{"shader_rel_path": _SHADER_REL_PATH, "inputs": {"inputs:diffuse_texture": texture}, "encoding": 0}]


def _record(name: str, asset: str, index: int, container_path: str = _CONTAINER_PATH) -> PlacementRecord:
    return PlacementRecord(
        container_path=container_path,
        prim_name=name,
        asset_rel_path=asset,
        asset_abs_path=asset,
        translate=(10.0 * index + 0.125, -3.5 * index, 1.0 + index),
        rotate_xyz=(5.0 * index, -30.0 + index, 17.0 * index),
        scale=(0.5 + 0.1 * index,) * 3,
        brush_id="Default",
        origin=(
            (float(index), 0.0, 0.0, 0.0, 0.0, 1.0)
            if index % 2
            else (float(index), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        ),
        key=1000 + index,
    )


def _placement_state(layer: Sdf.Layer, path: str) -> dict:
    """Everything a placement authors that a merge and a split must give back, as plain values."""
    spec = layer.GetPrimAtPath(path)
    state = {
        "specifier": spec.specifier,
        "typeName": spec.typeName,
        "references": [ref.assetPath for ref in spec.referenceList.GetAddedOrExplicitItems()],
    }
    for attribute in spec.attributes:
        value = attribute.default
        if isinstance(value, (Gf.Vec3d, Gf.Vec3f)):
            value = tuple(round(float(component), 6) for component in value)
        elif isinstance(value, (Vt.DoubleArray, Vt.TokenArray)):
            value = tuple(value)
        state[attribute.name] = value
    shader = layer.GetPrimAtPath(Sdf.Path(path).AppendPath(_SHADER_REL_PATH))
    state["overrides"] = None if shader is None else {a.name: str(a.default) for a in shader.attributes}
    return state


class TestInstancingConversions(omni.kit.test.AsyncTestCase):
    async def test_euler_xyz_to_quaternions_matches_the_rotation_usd_composes_for_rotate_xyz(self):
        # Arrange
        rotations = np.random.default_rng(7).uniform(-180.0, 180.0, size=(64, 3))

        # Act
        quaternions = euler_xyz_to_quaternions(rotations)

        # Assert
        for (x, y, z), quaternion in zip(rotations, quaternions):
            expected = (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), x)
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), y)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), z)
            ).GetQuat()
            expected_values = np.array([*expected.GetImaginary(), expected.GetReal()])
            difference = min(np.abs(expected_values - quaternion).max(), np.abs(expected_values + quaternion).max())
            self.assertLess(difference, 1e-12)

    async def test_quaternions_to_euler_xyz_gives_back_the_same_rotation(self):
        # Arrange
        rotations = np.random.default_rng(11).uniform(-180.0, 180.0, size=(64, 3))
        quaternions = euler_xyz_to_quaternions(rotations)

        # Act
        recovered = euler_xyz_to_quaternions(quaternions_to_euler_xyz(quaternions))

        # Assert
        difference = np.minimum(
            np.abs(recovered - quaternions).max(axis=1), np.abs(recovered + quaternions).max(axis=1)
        )
        self.assertLess(float(difference.max()), 1e-12)


class TestMergeAndSplit(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._asset = _write_asset(self._temp_dir, "cube.usda")
        self._other_asset = _write_asset(self._temp_dir, "other.usda")
        self._stage = Usd.Stage.CreateInMemory()
        instance = UsdGeom.Xform.Define(self._stage, _INSTANCE_ROOT)
        instance.AddTranslateOp().Set(_PARENT_OFFSET)
        self._stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
        self._stage.DefinePrim(_OTHER_PROTOTYPE_ROOT, "Xform")
        self._layer = self._stage.GetRootLayer()

    async def tearDown(self):
        self._stage = None
        self._layer = None
        self._temp_dir.cleanup()

    def _paint(self, count: int, start: int = 0, container_path: str = _CONTAINER_PATH, asset: str | None = None):
        records = [
            _record(f"s_{index:012x}", asset or self._asset, index, container_path)
            for index in range(start, start + count)
        ]
        author_placements(self._layer, records, {(asset or self._asset): _material()})
        return [record.prim_path for record in records]

    def _target(self) -> ScatterTarget:
        return ScatterTarget(
            prototype_root=Sdf.Path(_PROTOTYPE_ROOT),
            parent_instance_root=Sdf.Path(_INSTANCE_ROOT),
            mesh_path=Sdf.Path(f"{_INSTANCE_ROOT}/mesh"),
            parent_world=Gf.Matrix4d().SetTranslate(_PARENT_OFFSET),
            instance_count=1,
        )

    async def test_merge_moves_every_loose_placement_into_one_instancer_beside_its_container(self):
        # Arrange
        paths = self._paint(5)

        # Act
        edits = merge_loose_placements(self._layer, self._stage)

        # Assert
        self.assertEqual(len(edits), 1)
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
        instancer = self._stage.GetPrimAtPath(_INSTANCER_PATH)
        self.assertTrue(is_scatter_instancer(instancer))
        point_instancer = UsdGeom.PointInstancer(instancer)
        self.assertEqual(len(point_instancer.GetPositionsAttr().Get()), 5)
        self.assertEqual(list(point_instancer.GetProtoIndicesAttr().Get()), [0] * 5)
        self.assertEqual(sorted(point_instancer.GetIdsAttr().Get()), list(range(5)))
        self.assertEqual(len(point_instancer.GetPrototypesRel().GetTargets()), 1)
        self.assertEqual(edits[0].loose_paths, tuple(paths))

    async def test_merged_entries_render_where_the_placements_did(self):
        # Arrange
        paths = self._paint(6)
        expected = {
            path: UsdGeom.Xformable(self._stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            for path in paths
        }

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        transforms = instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
        ids = list(instancer.GetIdsAttr().Get())
        for path in paths:
            index = ids.index(int(Sdf.Path(path).name.removeprefix("s_"), 16))
            # Half-precision orientations: the rendered rotation is within a thousandth of the authored one.
            self.assertTrue(Gf.IsClose(transforms[index], expected[path], 2e-3), f"{path} moved when instanced")

    async def test_prototype_hides_the_asset_root_transform_the_placement_op_order_hid(self):
        # Arrange
        self._paint(1)

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        prototype = self._stage.GetPrimAtPath(instancer.GetPrototypesRel().GetTargets()[0])
        self.assertEqual(UsdGeom.Xformable(prototype).GetLocalTransformation(), Gf.Matrix4d(1.0))
        self.assertEqual(prototype.GetParent().GetName(), PROTOTYPES_SCOPE_NAME)
        shader = prototype.GetPrimAtPath(_SHADER_REL_PATH)
        self.assertEqual(shader.GetAttribute("inputs:diffuse_texture").Get().path, "./textures/brick.a.rtex.dds")

    async def test_merge_leaves_excluded_and_customised_placements_loose(self):
        # Arrange
        paths = self._paint(4)
        hidden = self._layer.GetPrimAtPath(paths[1])
        Sdf.AttributeSpec(hidden, "visibility", Sdf.ValueTypeNames.Token).default = "invisible"

        # Act
        edits = merge_loose_placements(self._layer, self._stage, exclude=[paths[0]])

        # Assert
        self.assertEqual(len(edits[0].block), 2)
        self.assertIsNotNone(self._layer.GetPrimAtPath(paths[0]))
        self.assertIsNotNone(self._layer.GetPrimAtPath(paths[1]))
        self.assertIsNone(self._layer.GetPrimAtPath(paths[2]))
        self.assertIsNone(self._layer.GetPrimAtPath(paths[3]))

    async def test_placements_with_different_textures_get_prototypes_of_their_own(self):
        # Arrange
        self._paint(2)
        records = [_record("s_0000000000aa", self._asset, 3)]
        author_placements(self._layer, records, {self._asset: _material("./textures/moss.a.rtex.dds")})
        self._paint(1, start=4, asset=self._other_asset)

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        self.assertEqual(len(instancer.GetPrototypesRel().GetTargets()), 3)
        self.assertEqual(sorted(set(instancer.GetProtoIndicesAttr().Get())), [0, 1, 2])

    async def test_uninstance_restores_the_layer_exactly_as_it_was(self):
        # Arrange
        self._paint(5)
        self._paint(2, start=10, container_path=_OTHER_CONTAINER_PATH)
        before = self._layer.ExportToString()
        edits = merge_loose_placements(self._layer, self._stage)

        # Act
        for edit in reversed(edits):
            edit.uninstance(self._layer)

        # Assert
        self.assertEqual(self._layer.ExportToString(), before)

    async def test_instance_again_after_uninstance_gives_the_same_instancer(self):
        # Arrange
        self._paint(5)
        edits = merge_loose_placements(self._layer, self._stage)
        merged = self._layer.ExportToString()
        for edit in reversed(edits):
            edit.uninstance(self._layer)

        # Act
        for edit in edits:
            edit.instance(self._layer)

        # Assert
        self.assertEqual(self._layer.ExportToString(), merged)

    async def test_a_second_merge_appends_to_the_instancer_the_layer_already_has(self):
        # Arrange
        self._paint(3)
        merge_loose_placements(self._layer, self._stage)
        self._paint(2, start=3)

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        self.assertEqual(sorted(instancer.GetIdsAttr().Get()), list(range(5)))

    async def test_merge_does_not_take_an_instancer_name_another_layer_defines(self):
        # Arrange
        stronger = Sdf.Layer.CreateAnonymous("stronger.usda")
        self._layer.subLayerPaths.insert(0, stronger.identifier)
        stronger_spec = Sdf.CreatePrimInLayer(stronger, _INSTANCER_PATH)
        stronger_spec.specifier = Sdf.SpecifierDef
        stronger_spec.typeName = "PointInstancer"
        self._paint(2)

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(_INSTANCER_PATH))
        self.assertIsNotNone(self._layer.GetPrimAtPath(f"{_INSTANCER_PATH}_1"))

    async def test_split_gives_every_placement_back_with_what_it_recorded(self):
        # Arrange
        paths = self._paint(5)
        spec = self._layer.GetPrimAtPath(paths[2])
        Sdf.AttributeSpec(spec, SCATTER_YAW_ATTR, Sdf.ValueTypeNames.Double).default = 12.5
        Sdf.AttributeSpec(spec, SCATTER_TURN_ATTR, Sdf.ValueTypeNames.DoubleArray).default = Vt.DoubleArray([1, 2, 3])
        # A placement moved by hand keeps a checksum that no longer matches its transform.
        spec.attributes[SCATTER_TRANSFORM_KEY_ATTR].default = 42
        keyless = self._layer.GetPrimAtPath(paths[3])
        keyless.RemoveProperty(keyless.attributes[SCATTER_KEY_ATTR])
        expected = {path: _placement_state(self._layer, path) for path in paths}
        merge_loose_placements(self._layer, self._stage)

        # Act
        edits = split_instancers(self._layer, [_INSTANCER_PATH])

        # Assert
        self.assertEqual(sorted(edits[0].loose_paths), sorted(paths))
        self.assertIsNone(self._layer.GetPrimAtPath(_INSTANCER_PATH))
        for path in paths:
            self.assertEqual(_placement_state(self._layer, path), expected[path], path)

    async def test_undoing_a_split_merges_the_same_entries_back(self):
        # Arrange
        self._paint(4)
        merge_loose_placements(self._layer, self._stage)
        merged = self._layer.ExportToString()
        edits = split_instancers(self._layer, [_INSTANCER_PATH])

        # Act
        for edit in reversed(edits):
            edit.instance(self._layer)

        # Assert
        self.assertEqual(self._layer.ExportToString(), merged)

    async def test_read_block_recovers_rotations_from_orientations_when_the_recorded_ones_are_gone(self):
        # Arrange
        self._paint(3)
        merge_loose_placements(self._layer, self._stage)
        spec = self._layer.GetPrimAtPath(_INSTANCER_PATH)
        recorded = np.asarray(spec.attributes[INSTANCED_ROTATIONS_ATTR].default, dtype=np.float64)
        spec.RemoveProperty(spec.attributes[INSTANCED_ROTATIONS_ATTR])
        spec.RemoveProperty(spec.attributes[INSTANCED_TRANSLATES_ATTR])

        # Act
        block = read_block(spec)

        # Assert
        recovered = euler_xyz_to_quaternions(block.rotations)
        expected = euler_xyz_to_quaternions(recorded)
        difference = np.minimum(np.abs(recovered - expected).max(axis=1), np.abs(recovered + expected).max(axis=1))
        self.assertLess(float(difference.max()), 2e-3)
        self.assertEqual(len(block), 3)

    async def test_eraser_removes_entries_and_its_edit_puts_them_back(self):
        # Arrange
        self._paint(4)
        merge_loose_placements(self._layer, self._stage)
        merged = self._layer.ExportToString()
        eraser = InstancedEraser(self._layer)

        # Act
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([1, 3]))
        erased = eraser.flush()
        remaining = sorted(UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH)).GetIdsAttr().Get())
        for edit in eraser.edits:
            edit.instance(self._layer)

        # Assert
        self.assertEqual(erased, 2)
        self.assertEqual(remaining, [0, 2])
        self.assertEqual(
            sorted(UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH)).GetIdsAttr().Get()), [0, 1, 2, 3]
        )
        self.assertEqual(len(merged), len(self._layer.ExportToString()))

    async def test_erasing_every_entry_deletes_the_instancer_and_undo_recreates_it(self):
        # Arrange
        self._paint(2)
        merge_loose_placements(self._layer, self._stage)
        eraser = InstancedEraser(self._layer)

        # Act
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([0, 1]))
        eraser.flush()
        deleted = self._layer.GetPrimAtPath(_INSTANCER_PATH) is None
        for edit in eraser.edits:
            edit.instance(self._layer)

        # Assert
        self.assertTrue(deleted)
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        self.assertEqual(sorted(instancer.GetIdsAttr().Get()), [0, 1])
        self.assertTrue(self._stage.GetPrimAtPath(instancer.GetPrototypesRel().GetTargets()[0]))

    async def test_eraser_leaves_the_instancer_alone_until_it_is_flushed(self):
        # Arrange - a stroke notes the entries of every stamp of a frame and rewrites the instancer once for them all
        self._paint(4)
        merge_loose_placements(self._layer, self._stage)
        eraser = InstancedEraser(self._layer)
        instancer = UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH))

        # Act
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([1]))
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([3]))
        before_flush = sorted(instancer.GetIdsAttr().Get())
        erased = eraser.flush()

        # Assert
        self.assertEqual(before_flush, [0, 1, 2, 3])
        self.assertEqual(erased, 2)
        self.assertEqual(sorted(instancer.GetIdsAttr().Get()), [0, 2])

    async def test_eraser_edits_cover_entries_that_were_noted_but_not_flushed(self):
        # Arrange
        self._paint(3)
        merge_loose_placements(self._layer, self._stage)
        eraser = InstancedEraser(self._layer)
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([2]))

        # Act
        edits = eraser.edits

        # Assert
        self.assertEqual([edit.block.ids.tolist() for edit in edits], [[2]])
        self.assertEqual(eraser.erased_count, 1)

    async def test_erasing_the_last_entry_under_an_untouched_mesh_leaves_no_empty_over_behind(self):
        # Arrange - the prototype is an over the brush created only to hold its scatter
        layer = Sdf.Layer.CreateAnonymous("untouched.usda")
        author_placements(layer, [_record("s_000000000000", self._asset, 0)])
        merge_loose_placements(layer)
        eraser = InstancedEraser(layer)

        # Act
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([0]))
        eraser.flush()

        # Assert
        self.assertEqual(list(layer.pseudoRoot.nameChildren.keys()), [])

    async def test_instanced_points_and_pool_place_entries_in_world_space(self):
        # Arrange
        self._paint(3)
        self._paint(1, start=3, asset=self._other_asset)
        merge_loose_placements(self._layer, self._stage)
        target = self._target()

        # Act
        points = instanced_points(self._stage, target)
        pool = instanced_pool(self._stage, self._layer, target, EraseScope.BRUSH_ASSETS, [self._other_asset])

        # Assert - the points are the surface each entry recorded, which is what padding is measured between
        self.assertEqual(points.shape, (4, 3))
        self.assertTrue(np.allclose(sorted(points[:, 0]), [100.0, 101.0, 102.0, 103.0]))
        self.assertEqual(len(pool), 1)
        taken = pool.take_within(Gf.Vec3d(130.125, -10.5, 4.0), 1.0)
        self.assertEqual({str(path): list(ids) for path, ids in taken.items()}, {_INSTANCER_PATH: [3]})
        self.assertEqual(len(pool), 0)

    async def test_pool_reaches_an_entry_through_the_surface_point_it_stands_on(self):
        # Arrange - entry 2 stands at x = 120.125 and recorded the surface at x = 102, out of reach of each other
        self._paint(3)
        merge_loose_placements(self._layer, self._stage)
        pool = instanced_pool(self._stage, self._layer, self._target())

        # Act
        taken = pool.take_within(Gf.Vec3d(102.0, 0.0, 0.0), 0.5)

        # Assert
        self.assertEqual({str(path): list(ids) for path, ids in taken.items()}, {_INSTANCER_PATH: [2]})

    async def test_pool_leaves_out_instancers_another_layer_authors(self):
        # Arrange
        self._paint(2)
        merge_loose_placements(self._layer, self._stage)

        # Act
        pool = instanced_pool(self._stage, Sdf.Layer.CreateAnonymous("other.usda"), self._target())

        # Assert
        self.assertEqual(len(pool), 0)

    async def test_selection_names_the_instancer_by_itself_its_container_or_its_mesh(self):
        # Arrange
        self._paint(2)
        self._paint(1, start=5, container_path=f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}moss")
        merge_loose_placements(self._layer, self._stage)
        moss = f"{_PROTOTYPE_ROOT}/{INSTANCER_PREFIX}moss"
        prototype = (
            UsdGeom.PointInstancer(self._stage.GetPrimAtPath(_INSTANCER_PATH)).GetPrototypesRel().GetTargets()[0]
        )

        # Act
        by_prototype_mesh = instancers_of_selection(self._stage, [f"{prototype}/Cube"])
        by_container = instancers_of_selection(self._stage, [_CONTAINER_PATH])
        by_mesh = instancers_of_selection(self._stage, [_PROTOTYPE_ROOT])
        by_instance = instancers_of_selection(self._stage, [_INSTANCE_ROOT])

        # Assert
        self.assertEqual(by_prototype_mesh, [Sdf.Path(_INSTANCER_PATH)])
        self.assertEqual(by_container, [Sdf.Path(_INSTANCER_PATH)])
        self.assertEqual(sorted(map(str, by_mesh)), sorted([_INSTANCER_PATH, moss]))
        self.assertEqual(sorted(map(str, by_instance)), sorted([_INSTANCER_PATH, moss]))
        self.assertTrue(is_inside_instancer(self._stage.GetPrimAtPath(f"{prototype}/Cube")))

    async def test_scene_cost_counts_instanced_and_loose_placements(self):
        # Arrange
        self._paint(3)
        merge_loose_placements(self._layer, self._stage)
        self._paint(2, start=3)

        # Act
        cost = scene_scatter_cost(self._stage)

        # Assert
        self.assertEqual(cost.placements, 5)
        self.assertEqual(cost.instanced, 3)
        self.assertEqual(instanced_entry_count(self._stage, _PROTOTYPE_ROOT), 3)

    async def test_instancer_carries_the_scatter_marker_so_pickers_skip_it(self):
        # Arrange
        self._paint(1)

        # Act
        merge_loose_placements(self._layer, self._stage)

        # Assert
        self.assertTrue(self._stage.GetPrimAtPath(_INSTANCER_PATH).HasAttribute(IS_REMIX_SCATTER_ATTR))


class TestInstancingCommands(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._commands_were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        self._registered = omni.kit.commands.register_all_commands_in_module(_commands)
        omni.kit.undo.clear_stack()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._asset = _write_asset(self._temp_dir, "cube.usda")
        self._stage = Usd.Stage.CreateInMemory()
        self._stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
        self._layer = self._stage.GetRootLayer()

    async def tearDown(self):
        omni.kit.undo.clear_stack()
        omni.kit.commands.unregister_module_commands(self._registered)
        if self._commands_were_registered:
            omni.kit.commands.register_all_commands_in_module(_commands)
        self._stage = None
        self._layer = None
        self._temp_dir.cleanup()

    def _records(self, count: int, start: int = 0) -> list[PlacementRecord]:
        return [_record(f"s_{index:012x}", self._asset, index) for index in range(start, start + count)]

    def _entry_ids(self) -> list[int]:
        prim = self._stage.GetPrimAtPath(_INSTANCER_PATH)
        return sorted(UsdGeom.PointInstancer(prim).GetIdsAttr().Get()) if prim else []

    def _place(self, records: list[PlacementRecord], instance_others: bool = True):
        author_placements(self._layer, records)
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=self._layer.identifier,
            kind="PLACE",
            records=records,
            already_applied=True,
            instance_others=instance_others,
        )

    async def test_a_stroke_merges_the_stroke_before_it_and_keeps_itself_loose(self):
        # Arrange
        first = self._records(3)
        second = self._records(2, start=3)
        self._place(first)

        # Act
        self._place(second)

        # Assert
        self.assertEqual(self._entry_ids(), [0, 1, 2])
        for record in second:
            self.assertIsNotNone(self._layer.GetPrimAtPath(record.prim_path))

    async def test_undoing_a_stroke_gives_the_stroke_before_it_back_as_prims(self):
        # Arrange
        first = self._records(3)
        self._place(first)
        before = self._layer.ExportToString()
        self._place(self._records(2, start=3))

        # Act
        omni.kit.undo.undo()

        # Assert
        self.assertEqual(self._layer.ExportToString(), before)

    async def test_redoing_a_stroke_merges_again(self):
        # Arrange
        self._place(self._records(3))
        self._place(self._records(2, start=3))
        merged = self._layer.ExportToString()
        omni.kit.undo.undo()

        # Act
        omni.kit.undo.redo()

        # Assert
        self.assertEqual(self._layer.ExportToString(), merged)

    async def test_a_stroke_without_instancing_merges_nothing(self):
        # Arrange
        self._place(self._records(3), instance_others=False)

        # Act
        self._place(self._records(2, start=3), instance_others=False)

        # Assert
        self.assertEqual(self._entry_ids(), [])

    async def test_flood_with_instance_merges_itself_and_undo_removes_it(self):
        # Arrange
        self._place(self._records(2))
        before = self._layer.ExportToString()

        # Act
        omni.kit.commands.execute(
            "ScatterFloodCommand",
            context_name="",
            layer_identifier=self._layer.identifier,
            records=self._records(3, start=2),
            instance=True,
        )
        merged_ids = self._entry_ids()
        omni.kit.undo.undo()

        # Assert
        self.assertEqual(merged_ids, [0, 1, 2, 3, 4])
        self.assertEqual(self._layer.ExportToString(), before)

    async def test_instance_and_split_commands_undo_and_redo(self):
        # Arrange
        self._place(self._records(3), instance_others=False)
        loose = self._layer.ExportToString()

        # Act
        omni.kit.commands.execute("ScatterInstanceCommand", context_name="", layer_identifier=self._layer.identifier)
        merged = self._layer.ExportToString()
        omni.kit.commands.execute(
            "ScatterSplitCommand",
            context_name="",
            layer_identifier=self._layer.identifier,
            instancer_paths=[_INSTANCER_PATH],
        )
        split_ids = self._entry_ids()
        omni.kit.undo.undo()
        after_split_undo = self._layer.ExportToString()
        omni.kit.undo.undo()
        after_instance_undo = self._layer.ExportToString()
        omni.kit.undo.redo()

        # Assert
        self.assertEqual(split_ids, [])
        self.assertEqual(after_split_undo, merged)
        self.assertEqual(after_instance_undo, loose)
        self.assertEqual(self._layer.ExportToString(), merged)

    async def test_an_erase_stroke_undoes_and_repeats_the_entries_it_erased(self):
        # Arrange
        self._place(self._records(3), instance_others=False)
        omni.kit.commands.execute("ScatterInstanceCommand", context_name="", layer_identifier=self._layer.identifier)
        eraser = InstancedEraser(self._layer)
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([1]))
        omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=self._layer.identifier,
            kind="ERASE",
            prim_paths=[],
            already_applied=True,
            erased_entries=eraser.edits,
        )
        erased = self._entry_ids()

        # Act
        omni.kit.undo.undo()
        restored = self._entry_ids()
        omni.kit.undo.redo()

        # Assert
        self.assertEqual(erased, [0, 2])
        self.assertEqual(restored, [0, 1, 2])
        self.assertEqual(self._entry_ids(), [0, 2])


__all__ = ["TestInstancingCommands", "TestInstancingConversions", "TestMergeAndSplit"]
