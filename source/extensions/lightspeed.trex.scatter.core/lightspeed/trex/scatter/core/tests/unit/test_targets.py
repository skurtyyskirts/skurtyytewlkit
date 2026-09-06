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

__all__ = ["TestTargets"]

from collections.abc import Sequence
from unittest import mock

import omni.kit.test
from lightspeed.common.constants import INSTANCE_PATH, IS_REMIX_REF_ATTR, ROOTNODE_INSTANCES, ROOTNODE_MESHES
from lightspeed.trex.scatter.core import targets
from lightspeed.trex.scatter.core.constants import IS_REMIX_SCATTER_ATTR
from lightspeed.trex.scatter.core.picking import SurfaceHit
from lightspeed.trex.scatter.core.settings import TargetMode
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_HASH = "0AB745B8BEE1F16B"
_OTHER_HASH = "CED45075A077A49A"
_PROTOTYPE = Sdf.Path(f"{ROOTNODE_MESHES}/mesh_{_HASH}")
_OTHER_PROTOTYPE = Sdf.Path(f"{ROOTNODE_MESHES}/mesh_{_OTHER_HASH}")
_INSTANCE_0 = Sdf.Path(f"{INSTANCE_PATH}{_HASH}_0")
_INSTANCE_1 = Sdf.Path(f"{INSTANCE_PATH}{_HASH}_1")
# Every instance gets a distinct, non-identity translation so parent transforms are distinguishable in assertions.
_INSTANCE_TRANSLATES = {
    0: Gf.Vec3d(10.0, 20.0, 30.0),
    1: Gf.Vec3d(100.0, 0.0, 0.0),
    2: Gf.Vec3d(200.0, 0.0, 0.0),
    3: Gf.Vec3d(300.0, 0.0, 0.0),
}
_CONTAINER_NAME = "scatter_default"
_PLACEMENT_NAME = "s_0123456789ab"
_SCATTERED_MESH_NAME = "asset_mesh"
_QUAD_HALF_SIZE = 10.0


def _author_quad(stage: Usd.Stage, path: Sdf.Path, offset: Gf.Vec3f | None = None) -> UsdGeom.Mesh:
    """Author a flat square mesh in the XY plane centred on ``offset`` with an explicit extent."""
    offset = offset if offset is not None else Gf.Vec3f(0.0)
    mesh = UsdGeom.Mesh.Define(stage, path)
    half = _QUAD_HALF_SIZE
    points = [
        Gf.Vec3f(-half, -half, 0.0) + offset,
        Gf.Vec3f(half, -half, 0.0) + offset,
        Gf.Vec3f(half, half, 0.0) + offset,
        Gf.Vec3f(-half, half, 0.0) + offset,
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateExtentAttr(Vt.Vec3fArray([points[0], points[2]]))
    return mesh


def _mark(prim: Usd.Prim, attribute_name: str) -> None:
    prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool, custom=True).Set(True)


def _author_scatter_placement(stage: Usd.Stage, prototype_path: Sdf.Path) -> Sdf.Path:
    """Author ``<prototype>/scatter_default/s_<id>/asset_mesh`` and return the scattered mesh path."""
    container = UsdGeom.Xform.Define(stage, prototype_path.AppendChild(_CONTAINER_NAME))
    _mark(container.GetPrim(), IS_REMIX_SCATTER_ATTR)
    placement = UsdGeom.Xform.Define(stage, container.GetPath().AppendChild(_PLACEMENT_NAME))
    _mark(placement.GetPrim(), IS_REMIX_SCATTER_ATTR)
    _mark(placement.GetPrim(), IS_REMIX_REF_ATTR)
    return _author_quad(stage, placement.GetPath().AppendChild(_SCATTERED_MESH_NAME)).GetPath()


def _build_capture_stage(instance_indices: Sequence[int] = (0, 1), with_scatter: bool = False) -> Usd.Stage:
    """Mirror the capture topology: an invisible prototype holding ``mesh`` and instances referencing it.

    When ``with_scatter`` is set, a scatter container with one placement is authored under the prototype BEFORE the
    capture mesh so that traversal order alone would pick the scattered mesh first.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/RootNode")
    stage.DefinePrim(ROOTNODE_MESHES)
    stage.DefinePrim(ROOTNODE_INSTANCES)
    prototype = UsdGeom.Xform.Define(stage, _PROTOTYPE)
    prototype.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if with_scatter:
        _author_scatter_placement(stage, _PROTOTYPE)
    _author_quad(stage, _PROTOTYPE.AppendChild("mesh"))
    for index in instance_indices:
        instance = UsdGeom.Xform.Define(stage, Sdf.Path(f"{INSTANCE_PATH}{_HASH}_{index}"))
        instance.GetPrim().GetReferences().AddInternalReference(_PROTOTYPE)
        instance.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        instance.AddTranslateOp().Set(_INSTANCE_TRANSLATES[index])
    return stage


def _translation_matrix(translate: Gf.Vec3d) -> Gf.Matrix4d:
    return Gf.Matrix4d(1.0).SetTranslate(translate)


class TestTargets(omni.kit.test.AsyncTestCase):
    async def test_get_prototype_root_with_instance_child_path_returns_prototype(self):
        # Arrange
        path = f"{_INSTANCE_1}/mesh"

        # Act
        result = targets.get_prototype_root(path)

        # Assert
        self.assertEqual(result, _PROTOTYPE)

    async def test_get_prototype_root_with_prototype_child_path_returns_prototype(self):
        # Arrange
        path = _PROTOTYPE.AppendChild("mesh")

        # Act
        result = targets.get_prototype_root(path)

        # Assert
        self.assertEqual(result, _PROTOTYPE)

    async def test_get_prototype_root_with_prototype_itself_returns_same_path(self):
        # Arrange
        path = _PROTOTYPE

        # Act
        result = targets.get_prototype_root(path)

        # Assert
        self.assertEqual(result, _PROTOTYPE)

    async def test_get_prototype_root_with_instance_root_returns_prototype(self):
        # Arrange
        path = _INSTANCE_0

        # Act
        result = targets.get_prototype_root(path)

        # Assert
        self.assertEqual(result, _PROTOTYPE)

    async def test_get_prototype_root_with_unrelated_path_returns_none(self):
        cases = [
            "/World/Cube",
            f"/RootNode/lights/light_{_HASH}_0",
            ROOTNODE_MESHES,
            f"{ROOTNODE_MESHES}/mesh_ABC",
            f"{ROOTNODE_MESHES}/mesh_{_HASH}x",
            "",
        ]
        for path in cases:
            with self.subTest(title=f"path={path!r}"):
                # Arrange
                candidate = path

                # Act
                result = targets.get_prototype_root(candidate)

                # Assert
                self.assertIsNone(result)

    async def test_get_instance_root_with_instance_child_path_returns_instance(self):
        # Arrange
        path = _INSTANCE_1.AppendPath("scatter_default/s_0123456789ab/asset_mesh")

        # Act
        result = targets.get_instance_root(path)

        # Assert
        self.assertEqual(result, _INSTANCE_1)

    async def test_get_instance_root_with_instance_itself_returns_same_path(self):
        # Arrange
        path = str(_INSTANCE_0)

        # Act
        result = targets.get_instance_root(path)

        # Assert
        self.assertEqual(result, _INSTANCE_0)

    async def test_get_instance_root_with_prototype_or_unrelated_path_returns_none(self):
        cases = [str(_PROTOTYPE.AppendChild("mesh")), str(_PROTOTYPE), "/World/Cube", ROOTNODE_INSTANCES]
        for path in cases:
            with self.subTest(title=f"path={path}"):
                # Arrange
                candidate = path

                # Act
                result = targets.get_instance_root(candidate)

                # Assert
                self.assertIsNone(result)

    async def test_is_scatter_prim_with_marker_attribute_returns_true(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)
        container = stage.GetPrimAtPath(_PROTOTYPE.AppendChild(_CONTAINER_NAME))

        # Act
        result = targets.is_scatter_prim(container)

        # Assert
        self.assertTrue(result)

    async def test_is_scatter_prim_without_marker_or_invalid_prim_returns_false(self):
        stage = _build_capture_stage()
        cases = [
            ("capture mesh", stage.GetPrimAtPath(_PROTOTYPE.AppendChild("mesh"))),
            ("invalid prim", stage.GetPrimAtPath("/Does/Not/Exist")),
        ]
        for title, prim in cases:
            with self.subTest(title=title):
                # Arrange
                candidate = prim

                # Act
                result = targets.is_scatter_prim(candidate)

                # Assert
                self.assertFalse(result)

    async def test_has_scatter_ancestor_with_scattered_mesh_returns_true(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)
        scattered_mesh = stage.GetPrimAtPath(
            _INSTANCE_0.AppendPath(f"{_CONTAINER_NAME}/{_PLACEMENT_NAME}/{_SCATTERED_MESH_NAME}")
        )

        # Act
        result = targets.has_scatter_ancestor(scattered_mesh)

        # Assert
        self.assertTrue(result)

    async def test_has_scatter_ancestor_with_capture_mesh_returns_false(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)
        capture_mesh = stage.GetPrimAtPath(_INSTANCE_0.AppendChild("mesh"))

        # Act
        result = targets.has_scatter_ancestor(capture_mesh)

        # Assert
        self.assertFalse(result)

    async def test_has_scatter_ancestor_with_stop_at_below_marker_returns_false(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)
        placement_path = _PROTOTYPE.AppendPath(f"{_CONTAINER_NAME}/{_PLACEMENT_NAME}")
        scattered_mesh = stage.GetPrimAtPath(placement_path.AppendChild(_SCATTERED_MESH_NAME))

        # Act
        result = targets.has_scatter_ancestor(scattered_mesh, stop_at=placement_path)

        # Assert
        self.assertFalse(result)

    async def test_find_capture_mesh_under_instance_returns_first_mesh(self):
        # Arrange
        stage = _build_capture_stage()

        # Act
        result = targets.find_capture_mesh(stage, _INSTANCE_1)

        # Assert
        self.assertEqual(result, _INSTANCE_1.AppendChild("mesh"))

    async def test_find_capture_mesh_skips_meshes_under_scatter_container(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)

        # Act
        result = targets.find_capture_mesh(stage, _INSTANCE_0)

        # Assert
        self.assertEqual(result, _INSTANCE_0.AppendChild("mesh"))

    async def test_find_capture_mesh_with_hint_prefers_mesh_whose_bbox_contains_it(self):
        # Arrange
        stage = _build_capture_stage()
        _author_quad(stage, _PROTOTYPE.AppendChild("mesh_far"), offset=Gf.Vec3f(1000.0, 0.0, 0.0))
        hint = _INSTANCE_TRANSLATES[0] + Gf.Vec3d(1000.0, 0.0, 0.0)

        # Act
        result = targets.find_capture_mesh(stage, _INSTANCE_0, hint_point=hint)

        # Assert
        self.assertEqual(result, _INSTANCE_0.AppendChild("mesh_far"))

    async def test_find_capture_mesh_with_hint_outside_every_bbox_returns_first_mesh(self):
        # Arrange
        stage = _build_capture_stage()
        _author_quad(stage, _PROTOTYPE.AppendChild("mesh_far"), offset=Gf.Vec3f(1000.0, 0.0, 0.0))
        hint = Gf.Vec3d(-5000.0, -5000.0, -5000.0)

        # Act
        result = targets.find_capture_mesh(stage, _INSTANCE_0, hint_point=hint)

        # Assert
        self.assertEqual(result, _INSTANCE_0.AppendChild("mesh"))

    async def test_find_capture_mesh_without_mesh_or_missing_root_returns_none(self):
        stage = _build_capture_stage()
        UsdGeom.Xform.Define(stage, _OTHER_PROTOTYPE)
        cases = [("prototype without mesh", _OTHER_PROTOTYPE), ("missing root", Sdf.Path("/Does/Not/Exist"))]
        for title, root_path in cases:
            with self.subTest(title=title):
                # Arrange
                candidate = root_path

                # Act
                result = targets.find_capture_mesh(stage, candidate)

                # Assert
                self.assertIsNone(result)

    async def test_canonical_instance_returns_lowest_index_regardless_of_authoring_order(self):
        # Arrange
        stage = _build_capture_stage(instance_indices=(3, 1, 2))

        # Act
        result = targets.canonical_instance(stage, _PROTOTYPE)

        # Assert
        self.assertEqual(result, _INSTANCE_1)

    async def test_canonical_instance_without_instances_returns_none(self):
        stage = _build_capture_stage()
        UsdGeom.Xform.Define(stage, _OTHER_PROTOTYPE)
        cases = [("prototype without instances", _OTHER_PROTOTYPE), ("not a prototype", Sdf.Path("/World/Cube"))]
        for title, prototype in cases:
            with self.subTest(title=title):
                # Arrange
                candidate = prototype

                # Act
                result = targets.canonical_instance(stage, candidate)

                # Assert
                self.assertIsNone(result)

    async def test_instance_count_counts_both_instances(self):
        # Arrange
        stage = _build_capture_stage()

        # Act
        result = targets.instance_count(stage, _PROTOTYPE)

        # Assert
        self.assertEqual(result, 2)

    async def test_instance_count_with_unknown_hash_returns_zero(self):
        # Arrange
        stage = _build_capture_stage()

        # Act
        result = targets.instance_count(stage, _OTHER_PROTOTYPE)

        # Assert
        self.assertEqual(result, 0)

    async def test_resolve_target_hit_surface_under_instance_uses_that_instance(self):
        # Arrange
        stage = _build_capture_stage()
        hit = SurfaceHit(path=_INSTANCE_1.AppendChild("mesh"), world_position=_INSTANCE_TRANSLATES[1])

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertEqual(target.prototype_root, _PROTOTYPE)
        self.assertEqual(target.parent_instance_root, _INSTANCE_1)
        self.assertEqual(target.mesh_path, _INSTANCE_1.AppendChild("mesh"))
        self.assertEqual(target.parent_world, _translation_matrix(_INSTANCE_TRANSLATES[1]))
        self.assertEqual(target.instance_count, 2)

    async def test_resolve_target_hit_surface_under_prototype_uses_canonical_instance(self):
        # Arrange
        stage = _build_capture_stage()
        hit = SurfaceHit(path=_PROTOTYPE.AppendChild("mesh"), world_position=_INSTANCE_TRANSLATES[0])

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertEqual(target.prototype_root, _PROTOTYPE)
        self.assertEqual(target.parent_instance_root, _INSTANCE_0)
        self.assertEqual(target.mesh_path, _INSTANCE_0.AppendChild("mesh"))
        self.assertEqual(target.parent_world, _translation_matrix(_INSTANCE_TRANSLATES[0]))
        self.assertEqual(target.instance_count, 2)

    async def test_resolve_target_hit_surface_on_scattered_child_resolves_to_capture_mesh(self):
        # Arrange
        stage = _build_capture_stage(with_scatter=True)
        scattered_mesh = _INSTANCE_0.AppendPath(f"{_CONTAINER_NAME}/{_PLACEMENT_NAME}/{_SCATTERED_MESH_NAME}")
        hit = SurfaceHit(path=scattered_mesh, world_position=_INSTANCE_TRANSLATES[0])

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertEqual(target.prototype_root, _PROTOTYPE)
        self.assertEqual(target.parent_instance_root, _INSTANCE_0)
        self.assertEqual(target.mesh_path, _INSTANCE_0.AppendChild("mesh"))

    async def test_resolve_target_hit_surface_outside_topology_returns_none(self):
        # Arrange
        stage = _build_capture_stage()
        hit = SurfaceHit(path=Sdf.Path("/World/Cube"), world_position=Gf.Vec3d(0.0))

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertIsNone(target)

    async def test_resolve_target_hit_surface_without_capture_mesh_returns_none(self):
        # Arrange
        stage = _build_capture_stage()
        UsdGeom.Xform.Define(stage, _OTHER_PROTOTYPE)
        hit = SurfaceHit(path=_OTHER_PROTOTYPE.AppendChild("mesh"), world_position=Gf.Vec3d(0.0))

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertIsNone(target)

    async def test_resolve_target_hit_surface_without_instances_parents_to_identity(self):
        # Arrange
        stage = _build_capture_stage(instance_indices=())
        hit = SurfaceHit(path=_PROTOTYPE.AppendChild("mesh"), world_position=Gf.Vec3d(0.0))

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.HIT_SURFACE, None)

        # Assert
        self.assertEqual(target.prototype_root, _PROTOTYPE)
        self.assertIsNone(target.parent_instance_root)
        self.assertEqual(target.mesh_path, _PROTOTYPE.AppendChild("mesh"))
        self.assertEqual(target.parent_world, Gf.Matrix4d(1.0))
        self.assertEqual(target.instance_count, 0)

    async def test_resolve_target_anchor_uses_anchor_canonical_instance_world(self):
        # Arrange
        stage = _build_capture_stage()
        hit = SurfaceHit(path=_INSTANCE_1.AppendChild("mesh"), world_position=_INSTANCE_TRANSLATES[1])

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.ANCHOR, str(_PROTOTYPE))

        # Assert
        self.assertEqual(target.prototype_root, _PROTOTYPE)
        self.assertEqual(target.parent_instance_root, _INSTANCE_0)
        self.assertNotEqual(target.parent_instance_root, targets.get_instance_root(hit.path))
        self.assertEqual(target.mesh_path, _INSTANCE_1.AppendChild("mesh"))
        self.assertEqual(target.parent_world, _translation_matrix(_INSTANCE_TRANSLATES[0]))
        self.assertEqual(target.instance_count, 2)

    async def test_resolve_target_anchor_with_invalid_anchor_returns_none(self):
        stage = _build_capture_stage()
        hit = SurfaceHit(path=_INSTANCE_1.AppendChild("mesh"), world_position=_INSTANCE_TRANSLATES[1])
        cases = [
            ("none", None),
            ("empty", ""),
            ("not a prototype", "/World/Cube"),
            ("instance instead of prototype", str(_INSTANCE_0)),
            ("prototype missing from stage", str(_OTHER_PROTOTYPE)),
        ]
        for title, anchor in cases:
            with self.subTest(title=title):
                # Arrange
                candidate = anchor

                # Act
                target = targets.resolve_target(stage, hit, TargetMode.ANCHOR, candidate)

                # Assert
                self.assertIsNone(target)

    async def test_resolve_target_anchor_without_instances_returns_none(self):
        # Arrange
        stage = _build_capture_stage()
        UsdGeom.Xform.Define(stage, _OTHER_PROTOTYPE)
        hit = SurfaceHit(path=_INSTANCE_1.AppendChild("mesh"), world_position=_INSTANCE_TRANSLATES[1])

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.ANCHOR, _OTHER_PROTOTYPE)

        # Assert
        self.assertIsNone(target)

    async def test_resolve_target_anchor_with_hit_outside_topology_returns_none(self):
        # Arrange
        stage = _build_capture_stage()
        hit = SurfaceHit(path=Sdf.Path("/World/Cube"), world_position=Gf.Vec3d(0.0))

        # Act
        target = targets.resolve_target(stage, hit, TargetMode.ANCHOR, _PROTOTYPE)

        # Assert
        self.assertIsNone(target)

    async def test_validated_anchor_prototype_with_existing_prototype_returns_its_path(self):
        # Arrange
        stage = _build_capture_stage()

        # Act
        result = targets.validated_anchor_prototype(stage, str(_PROTOTYPE))

        # Assert
        self.assertEqual(result, _PROTOTYPE)

    async def test_validated_anchor_prototype_with_malformed_path_returns_none(self):
        # Arrange
        stage = _build_capture_stage()

        # Act
        result = targets.validated_anchor_prototype(stage, "my anchor")

        # Assert
        self.assertIsNone(result)

    async def test_selected_prototypes_maps_selection_to_prototype_paths(self):
        # Arrange
        usd_context = mock.MagicMock()
        usd_context.get_selection.return_value.get_selected_prim_paths.return_value = [
            f"{_INSTANCE_1}/mesh",
            str(_PROTOTYPE),
            f"{_INSTANCE_0}",
            f"{_OTHER_PROTOTYPE}/mesh",
            "/World/Other",
        ]

        # Act
        result = targets.selected_prototypes(usd_context)

        # Assert
        self.assertEqual(result, {_PROTOTYPE, _OTHER_PROTOTYPE})

    async def test_selected_prototypes_with_empty_selection_returns_empty_set(self):
        # Arrange
        usd_context = mock.MagicMock()
        usd_context.get_selection.return_value.get_selected_prim_paths.return_value = []

        # Act
        result = targets.selected_prototypes(usd_context)

        # Assert
        self.assertEqual(result, set())
