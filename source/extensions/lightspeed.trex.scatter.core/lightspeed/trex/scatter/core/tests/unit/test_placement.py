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
import os
import pathlib
import tempfile

import numpy as np
import omni.kit.test
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from lightspeed.trex.scatter.core.constants import (
    CONTAINER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    PLACEMENT_PREFIX,
    SCATTER_ASSET_ATTR,
    SCATTER_BRUSH_ID_ATTR,
)
from lightspeed.trex.scatter.core.geometry import MeshSurfaceCache, SurfaceSample
from lightspeed.trex.scatter.core.placement import (
    PlacementRecord,
    author_placements,
    container_path_for,
    erase_candidates,
    existing_placement_points,
    generate_flood,
    generate_stamp,
    make_relative_asset_path,
    new_placement_name,
    remove_placements,
    restore_prims,
    snapshot_prims,
)
from lightspeed.trex.scatter.core.sampling import PaddingIndex
from lightspeed.trex.scatter.core.settings import EraseScope, Falloff, ScatterAssetEntry, ScatterBrushSettings, UpAxis
from lightspeed.trex.scatter.core.targets import ScatterTarget
from pxr import Gf, Sdf, Usd, UsdGeom

_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_0123456789ABCDEF"
_INSTANCE_ROOT = "/RootNode/instances/inst_0123456789ABCDEF_0"
_MESH_PATH = f"{_INSTANCE_ROOT}/mesh"
_CONTAINER_PATH = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_OTHER_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_FEDCBA9876543210"
_OTHER_CONTAINER_PATH = f"{_OTHER_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_PARENT_OFFSET = Gf.Vec3d(100.0, 0.0, 0.0)
_CENTER_WORLD = np.array([100.0, 0.0, 0.0])
_HALF_SIZE = 500.0
_XFORM_OP_ORDER = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{
    def Cube "Cube"
    {
    }
}
"""


def _build_flat_quad_scene(stage: Usd.Stage) -> None:
    """Author an instance translated by ``_PARENT_OFFSET`` holding a flat quad in the XY plane, plus the prototype."""
    instance = UsdGeom.Xform.Define(stage, _INSTANCE_ROOT)
    instance.AddTranslateOp().Set(_PARENT_OFFSET)
    mesh = UsdGeom.Mesh.Define(stage, _MESH_PATH)
    mesh.GetPointsAttr().Set(
        [
            (-_HALF_SIZE, -_HALF_SIZE, 0.0),
            (_HALF_SIZE, -_HALF_SIZE, 0.0),
            (_HALF_SIZE, _HALF_SIZE, 0.0),
            (-_HALF_SIZE, _HALF_SIZE, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")


def _make_record(
    name: str,
    asset_path: str,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    container_path: str = _CONTAINER_PATH,
) -> PlacementRecord:
    return PlacementRecord(
        container_path=container_path,
        prim_name=name,
        asset_rel_path=asset_path,
        asset_abs_path=asset_path,
        translate=translate,
        rotate_xyz=(0.0, 0.0, 90.0),
        scale=(2.0, 2.0, 2.0),
        brush_id="Default",
    )


def _center_sample() -> SurfaceSample:
    return SurfaceSample(
        position=_CENTER_WORLD.copy(), normal=np.array([0.0, 0.0, 1.0]), triangle_index=0, distance=0.0
    )


class TestPlacementRecord(omni.kit.test.AsyncTestCase):
    async def test_from_dict_of_to_dict_round_trips_every_field(self):
        # Arrange
        record = _make_record("s_0123456789ab", "C:/assets/cube.usda", translate=(1.5, -2.0, 3.25))

        # Act
        restored = PlacementRecord.from_dict(record.to_dict())

        # Assert
        self.assertEqual(restored, record)
        self.assertIsInstance(restored.translate, tuple)
        self.assertIsInstance(restored.rotate_xyz, tuple)
        self.assertIsInstance(restored.scale, tuple)

    async def test_to_dict_is_json_serializable_with_list_vectors(self):
        # Arrange
        record = _make_record("s_0123456789ab", "C:/assets/cube.usda", translate=(1.0, 2.0, 3.0))

        # Act
        payload = json.loads(json.dumps(record.to_dict()))

        # Assert
        self.assertEqual(payload["container_path"], _CONTAINER_PATH)
        self.assertEqual(payload["prim_name"], "s_0123456789ab")
        self.assertEqual(payload["asset_rel_path"], "C:/assets/cube.usda")
        self.assertEqual(payload["asset_abs_path"], "C:/assets/cube.usda")
        self.assertEqual(payload["translate"], [1.0, 2.0, 3.0])
        self.assertEqual(payload["rotate_xyz"], [0.0, 0.0, 90.0])
        self.assertEqual(payload["scale"], [2.0, 2.0, 2.0])
        self.assertEqual(payload["brush_id"], "Default")

    async def test_prim_path_joins_container_path_and_prim_name(self):
        # Arrange
        record = _make_record("s_0123456789ab", "C:/assets/cube.usda")

        # Act
        prim_path = record.prim_path

        # Assert
        self.assertEqual(prim_path, f"{_CONTAINER_PATH}/s_0123456789ab")


class TestPlacementFunctions(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._asset_a = self._write_asset("cube_a.usda")
        self._asset_b = self._write_asset("cube_b.usda")
        self._stage = Usd.Stage.CreateInMemory()
        self._layer = self._stage.GetRootLayer()
        _build_flat_quad_scene(self._stage)
        self._cache = MeshSurfaceCache(lambda: self._stage)
        instance_prim = self._stage.GetPrimAtPath(_INSTANCE_ROOT)
        self._target = ScatterTarget(
            prototype_root=Sdf.Path(_PROTOTYPE_ROOT),
            parent_instance_root=Sdf.Path(_INSTANCE_ROOT),
            mesh_path=Sdf.Path(_MESH_PATH),
            parent_world=UsdGeom.Xformable(instance_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            instance_count=1,
        )
        self._settings = ScatterBrushSettings(
            radius=50.0, density=8.0, strength=1.0, padding=0.0, falloff=Falloff.CONSTANT, vertical_offset=0.0
        )
        self._assets = [ScatterAssetEntry(path=self._asset_a, up_axis=UpAxis.Z)]

    async def tearDown(self):
        self._cache.destroy()
        self._cache = None
        self._stage = None
        self._layer = None
        self._temp_dir.cleanup()

    def _write_asset(self, file_name: str) -> str:
        path = pathlib.Path(self._temp_dir.name) / "assets" / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_ASSET_USDA, encoding="utf-8")
        return path.as_posix()

    def _author(self, *records: PlacementRecord) -> list[str]:
        return author_placements(self._layer, list(records))

    def _stamp(
        self,
        settings: ScatterBrushSettings | None = None,
        assets: list[ScatterAssetEntry] | None = None,
        padding_index: PaddingIndex | None = None,
        seed: int = 1234,
    ) -> list[PlacementRecord]:
        return generate_stamp(
            self._cache,
            self._target,
            _center_sample(),
            self._settings if settings is None else settings,
            self._assets if assets is None else assets,
            np.random.default_rng(seed),
            PaddingIndex(1.0) if padding_index is None else padding_index,
            None,
            UpAxis.Z,
            self._layer,
        )

    def _flood(self, max_count: int, settings: ScatterBrushSettings | None = None) -> list[PlacementRecord]:
        return generate_flood(
            self._cache,
            self._target,
            self._settings if settings is None else settings,
            self._assets,
            np.random.default_rng(42),
            PaddingIndex(1.0),
            UpAxis.Z,
            self._layer,
            max_count,
        )

    def _world_translate(self, record: PlacementRecord) -> Gf.Vec3d:
        return self._target.parent_world.Transform(Gf.Vec3d(*record.translate))

    async def test_container_path_for_appends_prefixed_slug_to_prototype(self):
        # Arrange
        prototype_root = Sdf.Path(_PROTOTYPE_ROOT)

        # Act
        container_path = container_path_for(prototype_root, "mossy_rocks")

        # Assert
        self.assertEqual(container_path, Sdf.Path(f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}mossy_rocks"))

    async def test_container_path_for_accepts_string_prototype(self):
        # Arrange
        prototype_root = _PROTOTYPE_ROOT

        # Act
        container_path = container_path_for(prototype_root, "default")

        # Assert
        self.assertEqual(container_path, Sdf.Path(_CONTAINER_PATH))

    async def test_make_relative_asset_path_on_disk_layer_returns_relative_forward_slash_path(self):
        # Arrange
        project_dir = pathlib.Path(self._temp_dir.name) / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        layer = Sdf.Layer.CreateNew((project_dir / "mod.usda").as_posix())
        asset_abs_path = (project_dir / "assets" / "cube.usda").as_posix()

        # Act
        relative_path = make_relative_asset_path(layer, asset_abs_path)

        # Assert
        self.assertIn(relative_path, ("./assets/cube.usda", "assets/cube.usda"))
        self.assertNotIn("\\", relative_path)
        self.assertFalse(os.path.isabs(relative_path))
        self.assertEqual(
            os.path.normcase(os.path.normpath(layer.ComputeAbsolutePath(relative_path))),
            os.path.normcase(os.path.normpath(asset_abs_path)),
        )

    async def test_make_relative_asset_path_anonymous_layer_returns_absolute_forward_slash_path(self):
        # Arrange
        layer = Sdf.Layer.CreateAnonymous()
        asset_abs_path = "C:\\projects\\assets\\cube.usda"

        # Act
        relative_path = make_relative_asset_path(layer, asset_abs_path)

        # Assert
        self.assertEqual(relative_path, "C:/projects/assets/cube.usda")

    async def test_new_placement_name_has_prefix_and_twelve_hex_characters(self):
        # Arrange
        expected_length = len(PLACEMENT_PREFIX) + 12

        # Act
        name = new_placement_name()

        # Assert
        self.assertTrue(name.startswith(PLACEMENT_PREFIX))
        self.assertEqual(len(name), expected_length)
        self.assertTrue(all(character in "0123456789abcdef" for character in name[len(PLACEMENT_PREFIX) :]))
        self.assertTrue(Sdf.Path.IsValidIdentifier(name))

    async def test_existing_placement_points_composes_local_translate_with_parent_world(self):
        # Arrange
        self._author(
            _make_record("s_000000000001", self._asset_a, translate=(0.0, 0.0, 0.0)),
            _make_record("s_000000000002", self._asset_a, translate=(10.0, 20.0, 30.0)),
        )
        # A plain child without the scatter markers must not count as a placement.
        UsdGeom.Xform.Define(self._stage, f"{_CONTAINER_PATH}/helper")

        # Act
        points = existing_placement_points(self._stage, self._target)

        # Assert
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points[np.argsort(points[:, 0])], [[100.0, 0.0, 0.0], [110.0, 20.0, 30.0]])

    async def test_existing_placement_points_without_container_returns_empty_array(self):
        # Arrange
        target = self._target

        # Act
        points = existing_placement_points(self._stage, target)

        # Assert
        self.assertEqual(points.shape, (0, 3))

    async def test_generate_stamp_density_eight_yields_eight_records(self):
        # Arrange
        padding_index = PaddingIndex(1.0)

        # Act
        records = self._stamp(padding_index=padding_index)

        # Assert
        self.assertEqual(len(records), 8)
        self.assertEqual(len(padding_index), 8)

    async def test_generate_stamp_strength_zero_yields_no_records(self):
        # Arrange
        settings = self._settings.model_copy(update={"strength": 0.0})

        # Act
        records = self._stamp(settings=settings)

        # Assert
        self.assertEqual(records, [])

    async def test_generate_stamp_padding_skips_points_near_existing_placements(self):
        # Arrange
        settings = self._settings.model_copy(update={"padding": 200.0})
        padding_index = PaddingIndex(200.0)
        padding_index.add(_CENTER_WORLD)

        # Act
        records = self._stamp(settings=settings, padding_index=padding_index)

        # Assert
        self.assertEqual(records, [])
        self.assertEqual(len(padding_index), 1)

    async def test_generate_stamp_records_target_container_and_relative_asset_path(self):
        # Arrange
        expected_container = str(container_path_for(self._target.prototype_root, self._settings.slug()))
        expected_asset = make_relative_asset_path(self._layer, self._asset_a)

        # Act
        records = self._stamp(seed=7)

        # Assert
        self.assertEqual(len(records), 8)
        self.assertEqual({record.container_path for record in records}, {expected_container})
        self.assertEqual({record.asset_rel_path for record in records}, {expected_asset})
        self.assertEqual({record.asset_abs_path for record in records}, {self._asset_a})
        self.assertEqual({record.brush_id for record in records}, {self._settings.preset_name})
        self.assertEqual(len({record.prim_name for record in records}), 8)
        self.assertTrue(all(record.prim_name.startswith(PLACEMENT_PREFIX) for record in records))

    async def test_generate_stamp_transforms_are_in_parent_space(self):
        # Arrange
        center = Gf.Vec3d(*_CENTER_WORLD)

        # Act
        records = self._stamp(seed=99)

        # Assert
        self.assertEqual(len(records), 8)
        for record in records:
            world = self._world_translate(record)
            self.assertLessEqual((world - center).GetLength(), self._settings.radius + 1e-3)
            self.assertAlmostEqual(world[2], 0.0, places=3)
            self.assertAlmostEqual(record.translate[0] + _PARENT_OFFSET[0], world[0], places=6)

    async def test_generate_stamp_without_enabled_assets_returns_no_records(self):
        # Arrange
        assets = [ScatterAssetEntry(path=self._asset_a, enabled=False)]
        padding_index = PaddingIndex(1.0)

        # Act
        records = self._stamp(assets=assets, padding_index=padding_index)

        # Assert
        self.assertEqual(records, [])
        self.assertEqual(len(padding_index), 0)

    async def test_generate_flood_caps_at_max_count_and_stays_within_mesh_bbox(self):
        # Arrange
        geometry = self._cache.get(self._target.mesh_path)

        # Act
        records = self._flood(20)

        # Assert
        self.assertEqual(len(records), 20)
        for record in records:
            world = np.array(self._world_translate(record))
            self.assertTrue(np.all(world >= geometry.bbox_min - 1e-3), msg=f"{world} below {geometry.bbox_min}")
            self.assertTrue(np.all(world <= geometry.bbox_max + 1e-3), msg=f"{world} above {geometry.bbox_max}")

    async def test_generate_flood_large_radius_yields_single_record(self):
        # Arrange
        settings = self._settings.model_copy(update={"radius": 10000.0})

        # Act
        records = self._flood(300, settings=settings)

        # Assert
        self.assertEqual(len(records), 1)

    async def test_generate_flood_zero_max_count_yields_no_records(self):
        # Arrange
        max_count = 0

        # Act
        records = self._flood(max_count)

        # Assert
        self.assertEqual(records, [])

    async def test_erase_candidates_returns_only_marker_prims_within_radius(self):
        # Arrange
        near = _make_record("s_00000000near", self._asset_a, translate=(0.0, 0.0, 0.0))
        far = _make_record("s_000000000far", self._asset_a, translate=(300.0, 0.0, 0.0))
        self._author(near, far)
        UsdGeom.Xform.Define(self._stage, f"{_CONTAINER_PATH}/helper")

        # Act
        candidates = erase_candidates(
            self._stage, self._target, Gf.Vec3d(*_CENTER_WORLD), 50.0, EraseScope.ALL_SCATTERED, []
        )

        # Assert
        self.assertEqual(candidates, [Sdf.Path(near.prim_path)])

    async def test_erase_candidates_brush_assets_scope_filters_by_asset(self):
        # Arrange
        with_a = _make_record("s_0000000000aa", self._asset_a)
        with_b = _make_record("s_0000000000bb", self._asset_b)
        self._author(with_a, with_b)

        # Act
        candidates = erase_candidates(
            self._stage, self._target, Gf.Vec3d(*_CENTER_WORLD), 50.0, EraseScope.BRUSH_ASSETS, [self._asset_a]
        )

        # Assert
        self.assertEqual(candidates, [Sdf.Path(with_a.prim_path)])

    async def test_author_placements_creates_container_and_placement_specs(self):
        # Arrange
        record = _make_record("s_000000000001", self._asset_a, translate=(1.0, 2.0, 3.0))

        # Act
        created = author_placements(self._layer, [record])

        # Assert
        self.assertEqual(created, [record.prim_path])
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(container.specifier, Sdf.SpecifierDef)
        self.assertEqual(container.typeName, "Xform")
        self.assertTrue(container.attributes[IS_REMIX_SCATTER_ATTR].custom)
        self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertTrue(container.attributes[SCATTER_BRUSH_ID_ATTR].custom)
        self.assertEqual(container.attributes[SCATTER_BRUSH_ID_ATTR].default, "Default")
        self.assertFalse(container.hasReferences)
        spec = self._layer.GetPrimAtPath(record.prim_path)
        self.assertEqual(spec.specifier, Sdf.SpecifierDef)
        self.assertEqual(spec.typeName, "Xform")
        self.assertEqual(list(spec.referenceList.prependedItems), [Sdf.Reference(assetPath=self._asset_a)])
        self.assertEqual(spec.attributes[IS_REMIX_REF_ATTR].default, True)
        self.assertTrue(spec.attributes[IS_REMIX_REF_ATTR].custom)
        self.assertEqual(spec.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertTrue(spec.attributes[IS_REMIX_SCATTER_ATTR].custom)
        self.assertEqual(spec.attributes[SCATTER_ASSET_ATTR].default, self._asset_a)
        self.assertTrue(spec.attributes[SCATTER_ASSET_ATTR].custom)
        self.assertEqual(spec.attributes["xformOp:translate"].typeName, Sdf.ValueTypeNames.Double3)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(spec.attributes["xformOp:rotateXYZ"].typeName, Sdf.ValueTypeNames.Float3)
        self.assertEqual(spec.attributes["xformOp:rotateXYZ"].default, Gf.Vec3f(0.0, 0.0, 90.0))
        self.assertEqual(spec.attributes["xformOp:scale"].typeName, Sdf.ValueTypeNames.Float3)
        self.assertEqual(spec.attributes["xformOp:scale"].default, Gf.Vec3f(2.0, 2.0, 2.0))
        self.assertEqual(spec.attributes["xformOpOrder"].typeName, Sdf.ValueTypeNames.TokenArray)
        self.assertEqual(spec.attributes["xformOpOrder"].variability, Sdf.VariabilityUniform)
        self.assertEqual(list(spec.attributes["xformOpOrder"].default), _XFORM_OP_ORDER)
        prim = self._stage.GetPrimAtPath(record.prim_path)
        self.assertTrue(prim.IsValid())
        self.assertTrue(prim.GetChild("Cube").IsValid())

    async def test_author_placements_skips_existing_names(self):
        # Arrange
        original = _make_record("s_000000000001", self._asset_a, translate=(1.0, 2.0, 3.0))
        self._author(original)
        duplicate = _make_record("s_000000000001", self._asset_a, translate=(9.0, 9.0, 9.0))

        # Act
        created = author_placements(self._layer, [duplicate])

        # Assert
        self.assertEqual(created, [])
        spec = self._layer.GetPrimAtPath(original.prim_path)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(len(self._layer.GetPrimAtPath(_CONTAINER_PATH).nameChildren), 1)

    async def test_author_placements_with_invalid_prim_name_creates_nothing(self):
        # Arrange
        record = _make_record("s_bad name", self._asset_a)

        # Act
        created = author_placements(self._layer, [record])

        # Assert
        self.assertEqual(created, [])
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_remove_placements_deletes_specs_and_empty_container(self):
        # Arrange
        first = _make_record("s_000000000001", self._asset_a)
        second = _make_record("s_000000000002", self._asset_a)
        self._author(first, second)

        # Act
        remove_placements(self._layer, [first.prim_path, second.prim_path])

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(first.prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(second.prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
        self.assertIsNotNone(self._layer.GetPrimAtPath(_PROTOTYPE_ROOT))

    async def test_remove_placements_keeps_container_with_remaining_children(self):
        # Arrange
        first = _make_record("s_000000000001", self._asset_a)
        second = _make_record("s_000000000002", self._asset_a)
        self._author(first, second)

        # Act
        remove_placements(self._layer, [first.prim_path])

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(first.prim_path))
        self.assertIsNotNone(self._layer.GetPrimAtPath(second.prim_path))
        self.assertEqual(list(self._layer.GetPrimAtPath(_CONTAINER_PATH).nameChildren.keys()), ["s_000000000002"])

    async def test_remove_placements_without_container_cleanup_leaves_empty_container(self):
        # Arrange
        record = _make_record("s_000000000001", self._asset_a)
        self._author(record)

        # Act
        remove_placements(self._layer, [record.prim_path], remove_empty_containers=False)

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(record.prim_path))
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(container.specifier, Sdf.SpecifierDef)
        self.assertEqual(len(container.nameChildren), 0)

    async def test_remove_placements_ignores_missing_paths(self):
        # Arrange
        record = _make_record("s_000000000001", self._asset_a)
        self._author(record)

        # Act
        remove_placements(self._layer, [f"{_CONTAINER_PATH}/s_00000missing"])

        # Assert
        self.assertIsNotNone(self._layer.GetPrimAtPath(record.prim_path))
        self.assertIsNotNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_restore_prims_after_snapshot_and_removal_round_trips_attributes(self):
        # Arrange
        record = _make_record("s_000000000001", self._asset_a, translate=(4.0, 5.0, 6.0))
        self._author(record)
        snapshot = snapshot_prims(self._layer, [record.prim_path])
        remove_placements(self._layer, [record.prim_path])

        # Act
        restore_prims(self._layer, snapshot, [record.prim_path])

        # Assert
        spec = self._layer.GetPrimAtPath(record.prim_path)
        self.assertEqual(spec.specifier, Sdf.SpecifierDef)
        self.assertEqual(spec.typeName, "Xform")
        self.assertEqual(list(spec.referenceList.prependedItems), [Sdf.Reference(assetPath=self._asset_a)])
        self.assertEqual(spec.attributes[IS_REMIX_REF_ATTR].default, True)
        self.assertEqual(spec.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertEqual(spec.attributes[SCATTER_ASSET_ATTR].default, self._asset_a)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(4.0, 5.0, 6.0))
        self.assertEqual(spec.attributes["xformOp:rotateXYZ"].default, Gf.Vec3f(0.0, 0.0, 90.0))
        self.assertEqual(spec.attributes["xformOp:scale"].default, Gf.Vec3f(2.0, 2.0, 2.0))
        self.assertEqual(list(spec.attributes["xformOpOrder"].default), _XFORM_OP_ORDER)
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(container.specifier, Sdf.SpecifierDef)
        self.assertEqual(container.typeName, "Xform")
        self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertEqual(container.attributes[SCATTER_BRUSH_ID_ATTR].default, "Default")
        self.assertTrue(self._stage.GetPrimAtPath(record.prim_path).IsValid())

    async def test_snapshot_prims_copies_only_existing_specs(self):
        # Arrange
        record = _make_record("s_000000000001", self._asset_a)
        self._author(record)
        missing_path = f"{_CONTAINER_PATH}/s_00000missing"

        # Act
        snapshot = snapshot_prims(self._layer, [record.prim_path, missing_path])

        # Assert
        self.assertTrue(snapshot.anonymous)
        self.assertIsNotNone(snapshot.GetPrimAtPath(record.prim_path))
        self.assertIsNone(snapshot.GetPrimAtPath(missing_path))
        self.assertEqual(snapshot.GetPrimAtPath(record.prim_path).attributes[SCATTER_ASSET_ATTR].default, self._asset_a)
        self.assertEqual(list(snapshot.GetPrimAtPath(_CONTAINER_PATH).nameChildren.keys()), ["s_000000000001"])

    async def test_snapshot_prims_into_existing_snapshot_accumulates_specs_in_that_layer(self):
        # Arrange
        first = _make_record("s_000000000001", self._asset_a)
        second = _make_record("s_000000000002", self._asset_a)
        self._author(first, second)
        snapshot = snapshot_prims(self._layer, [first.prim_path])

        # Act
        result = snapshot_prims(self._layer, [second.prim_path], into=snapshot)

        # Assert
        self.assertIs(result, snapshot)
        self.assertEqual(
            list(snapshot.GetPrimAtPath(_CONTAINER_PATH).nameChildren.keys()), ["s_000000000001", "s_000000000002"]
        )
        self.assertEqual(snapshot.GetPrimAtPath(second.prim_path).attributes[SCATTER_ASSET_ATTR].default, self._asset_a)

    async def test_snapshot_prims_into_existing_snapshot_copies_definition_of_new_container(self):
        # Arrange
        first = _make_record("s_000000000001", self._asset_a)
        second = _make_record("s_000000000002", self._asset_a, container_path=_OTHER_CONTAINER_PATH)
        self._author(first, second)
        snapshot = snapshot_prims(self._layer, [first.prim_path])

        # Act
        snapshot_prims(self._layer, [second.prim_path], into=snapshot)

        # Assert
        container = snapshot.GetPrimAtPath(_OTHER_CONTAINER_PATH)
        self.assertEqual(container.specifier, Sdf.SpecifierDef)
        self.assertEqual(container.typeName, "Xform")
        self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertEqual(container.attributes[SCATTER_BRUSH_ID_ATTR].default, "Default")

    async def test_restore_prims_from_accumulated_snapshot_redefines_every_removed_container(self):
        # Arrange
        self._stage.DefinePrim(_OTHER_PROTOTYPE_ROOT, "Xform")
        first = _make_record("s_000000000001", self._asset_a)
        second = _make_record("s_000000000002", self._asset_a, container_path=_OTHER_CONTAINER_PATH)
        self._author(first, second)
        snapshot = snapshot_prims(self._layer, [first.prim_path])
        remove_placements(self._layer, [first.prim_path])
        snapshot_prims(self._layer, [second.prim_path], into=snapshot)
        remove_placements(self._layer, [second.prim_path])
        self.assertIsNone(self._layer.GetPrimAtPath(_OTHER_CONTAINER_PATH))

        # Act
        restore_prims(self._layer, snapshot, [first.prim_path, second.prim_path])

        # Assert
        for container_path in (_CONTAINER_PATH, _OTHER_CONTAINER_PATH):
            self.assertEqual(self._layer.GetPrimAtPath(container_path).specifier, Sdf.SpecifierDef)
            self.assertTrue(self._stage.GetPrimAtPath(container_path).IsDefined())
        self.assertTrue(self._stage.GetPrimAtPath(first.prim_path).IsDefined())
        self.assertTrue(self._stage.GetPrimAtPath(second.prim_path).IsDefined())

    async def test_restore_prims_ignores_paths_missing_from_snapshot(self):
        # Arrange
        snapshot = Sdf.Layer.CreateAnonymous()
        missing_path = f"{_CONTAINER_PATH}/s_00000missing"

        # Act
        restore_prims(self._layer, snapshot, [missing_path])

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(missing_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
