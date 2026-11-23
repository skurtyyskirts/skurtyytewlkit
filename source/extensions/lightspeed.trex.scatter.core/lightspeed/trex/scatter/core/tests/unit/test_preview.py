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
import omni.kit.app
import omni.kit.test
import omni.usd
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX, INSTANCER_PREFIX, IS_REMIX_SCATTER_ATTR
from lightspeed.trex.scatter.core.instancing import (
    InstancedEraser,
    instancers_of_selection,
    merge_loose_placements,
    split_instancers,
)
from lightspeed.trex.scatter.core.placement import PlacementRecord, author_placements, scene_scatter_cost
from lightspeed.trex.scatter.core.preview import (
    InstancerViewportPreview,
    clear_instancer_preview,
    preview_path_for,
    sync_instancer_preview,
)
from lightspeed.trex.scatter.core.reapply import is_inside_instancer
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_0123456789ABCDEF"
_CONTAINER_PATH = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_INSTANCER_PATH = f"{_PROTOTYPE_ROOT}/{INSTANCER_PREFIX}default"
_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Z"
)

def Xform "Root"
{
    def Mesh "Blade"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 0, 10)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
}
"""


def _records(asset: str, count: int, start: int = 0) -> list[PlacementRecord]:
    return [
        PlacementRecord(
            container_path=_CONTAINER_PATH,
            prim_name=f"s_{index:012x}",
            asset_rel_path=asset,
            asset_abs_path=asset,
            translate=(25.0 * index, 3.0 - index, 0.5 * index),
            rotate_xyz=(3.0 * index, -7.0, 40.0 * index),
            scale=(0.25 + 0.05 * index,) * 3,
            brush_id="Default",
            key=index + 1,
        )
        for index in range(start, start + count)
    ]


class TestSyncInstancerPreview(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        asset = pathlib.Path(self._temp_dir.name) / "blade.usda"
        asset.write_text(_ASSET_USDA, encoding="utf-8")
        self._asset = str(asset).replace("\\", "/")
        self._stage = Usd.Stage.CreateInMemory()
        self._stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
        self._layer = self._stage.GetRootLayer()
        self._session = self._stage.GetSessionLayer()

    async def tearDown(self):
        self._stage = None
        self._temp_dir.cleanup()

    def _blade_transforms(self, root: str) -> dict[str, Gf.Matrix4d]:
        """World transform of every blade mesh under ``root``, keyed by the name of the prim directly under it."""
        cache = UsdGeom.XformCache()
        found = {}
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Mesh):
                key = prim.GetPath().GetPrefixes()[Sdf.Path(root).pathElementCount].name
                found[key] = cache.GetLocalToWorldTransform(prim)
        return found

    async def test_stand_in_draws_every_blade_where_the_loose_placement_drew_it(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 6))
        loose = self._blade_transforms(_CONTAINER_PATH)
        merge_loose_placements(self._layer, self._stage)

        # Act
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)

        # Assert
        shown = self._blade_transforms(str(preview_path_for(_INSTANCER_PATH)))
        self.assertEqual(len(state.ids), 6)
        self.assertEqual(len(shown), 6)
        for name, matrix in loose.items():
            stand_in = shown[f"e_{name[2:]}"]
            self.assertTrue(Gf.IsClose(stand_in, matrix, 1e-6), name)

    async def test_stand_in_hides_the_instancer_and_lives_only_in_the_given_layer(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 2))
        merge_loose_placements(self._layer, self._stage)

        # Act
        sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)

        # Assert
        instancer = UsdGeom.Imageable(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        self.assertEqual(instancer.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertIsNone(self._layer.GetPrimAtPath(preview_path_for(_INSTANCER_PATH)))
        container = self._stage.GetPrimAtPath(preview_path_for(_INSTANCER_PATH))
        self.assertTrue(container.HasAttribute(IS_REMIX_SCATTER_ATTR))
        self.assertEqual(scene_scatter_cost(self._stage).placements, 2)

    async def test_a_sync_after_an_erase_removes_only_the_erased_blades(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 5))
        merge_loose_placements(self._layer, self._stage)
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)
        container_path = preview_path_for(_INSTANCER_PATH)
        # A blade the sync leaves alone keeps whatever it holds, so a marker value shows it was not written again.
        kept = self._session.GetPrimAtPath(container_path.AppendChild("e_000000000000"))
        kept.attributes["xformOp:translate"].default = Gf.Vec3d(-1.0, -1.0, -1.0)
        eraser = InstancedEraser(self._layer)
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([1, 3]))
        eraser.flush()

        # Act
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH, state)

        # Assert - the stage is what the viewport draws; an erased blade is switched off there rather than deleted
        names = sorted(prim.GetName() for prim in self._stage.GetPrimAtPath(container_path).GetChildren())
        self.assertEqual(names, ["e_000000000000", "e_000000000002", "e_000000000004"])
        self.assertEqual(sorted(state.ids.tolist()), [0, 2, 4])
        self.assertEqual(kept.attributes["xformOp:translate"].default, Gf.Vec3d(-1.0, -1.0, -1.0))

    async def test_a_sync_after_an_erase_is_undone_shows_the_erased_blades_again(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 4))
        merge_loose_placements(self._layer, self._stage)
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)
        eraser = InstancedEraser(self._layer)
        eraser.erase(Sdf.Path(_INSTANCER_PATH), np.array([1, 2]))
        eraser.flush()
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH, state)
        for edit in eraser.edits:
            edit.instance(self._layer)

        # Act
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH, state)

        # Assert
        container = self._stage.GetPrimAtPath(preview_path_for(_INSTANCER_PATH))
        self.assertEqual(len(container.GetChildren()), 4)
        self.assertEqual(sorted(state.ids.tolist()), [0, 1, 2, 3])

    async def test_a_sync_after_one_blade_changed_authors_only_that_blade(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 3))
        merge_loose_placements(self._layer, self._stage)
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)
        container_path = preview_path_for(_INSTANCER_PATH)
        # A blade the sync leaves alone keeps a marker written over what it shows; one it authors again loses it.
        for name in ("e_000000000000", "e_000000000001", "e_000000000002"):
            spec = self._session.GetPrimAtPath(container_path.AppendChild(name))
            spec.attributes["xformOp:scale"].default = Gf.Vec3f(9.0, 9.0, 9.0)
        instancer = self._layer.GetPrimAtPath(_INSTANCER_PATH)
        scales = np.array(instancer.attributes["scales"].default)
        scales[1] = (2.0, 2.0, 2.0)
        instancer.attributes["scales"].default = Vt.Vec3fArray.FromNumpy(scales)

        # Act
        sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH, state)

        # Assert
        shown = {
            name: tuple(
                self._session.GetPrimAtPath(container_path.AppendChild(name)).attributes["xformOp:scale"].default
            )
            for name in ("e_000000000000", "e_000000000001", "e_000000000002")
        }
        self.assertEqual(
            shown,
            {"e_000000000000": (9.0, 9.0, 9.0), "e_000000000001": (2.0, 2.0, 2.0), "e_000000000002": (9.0, 9.0, 9.0)},
        )

    async def test_a_sync_after_the_instancer_is_gone_removes_the_stand_in_and_the_hiding(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 3))
        merge_loose_placements(self._layer, self._stage)
        state = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)
        split_instancers(self._layer, [_INSTANCER_PATH])

        # Act
        result = sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH, state)

        # Assert
        self.assertIsNone(result)
        self.assertIsNone(self._session.GetPrimAtPath(preview_path_for(_INSTANCER_PATH)))
        self.assertIsNone(self._session.GetPrimAtPath(_INSTANCER_PATH))

    async def test_clear_leaves_an_instancer_visible_again(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 2))
        merge_loose_placements(self._layer, self._stage)
        sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)

        # Act
        clear_instancer_preview(self._session, _INSTANCER_PATH)

        # Assert
        instancer = UsdGeom.Imageable(self._stage.GetPrimAtPath(_INSTANCER_PATH))
        self.assertEqual(instancer.ComputeVisibility(), UsdGeom.Tokens.inherited)

    async def test_a_blade_of_the_stand_in_counts_as_instanced_scatter_and_names_its_instancer(self):
        # Arrange
        author_placements(self._layer, _records(self._asset, 2))
        merge_loose_placements(self._layer, self._stage)
        sync_instancer_preview(self._stage, self._session, _INSTANCER_PATH)
        blade = preview_path_for(_INSTANCER_PATH).AppendPath("e_000000000001/Blade")

        # Act
        inside = is_inside_instancer(self._stage.GetPrimAtPath(blade))
        named = instancers_of_selection(self._stage, [blade])

        # Assert
        self.assertTrue(inside)
        self.assertEqual(named, [Sdf.Path(_INSTANCER_PATH)])


class TestInstancerViewportPreview(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        asset = pathlib.Path(self._temp_dir.name) / "blade.usda"
        asset.write_text(_ASSET_USDA, encoding="utf-8")
        self._asset = str(asset).replace("\\", "/")
        self._context = omni.usd.get_context()
        await self._context.new_stage_async()
        self._stage = self._context.get_stage()
        self._stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
        self._preview = InstancerViewportPreview("")

    async def tearDown(self):
        self._preview.destroy()
        self._preview = None
        await self._context.close_stage_async()
        self._stage = None
        self._temp_dir.cleanup()

    async def _updates(self, count: int = 3) -> None:
        for _ in range(count):
            await omni.kit.app.get_app().next_update_async()

    async def test_a_merge_shows_up_as_a_stand_in_on_the_next_update(self):
        # Arrange
        layer = self._stage.GetRootLayer()
        author_placements(layer, _records(self._asset, 4))

        # Act
        merge_loose_placements(layer, self._stage)
        await self._updates()

        # Assert
        container = self._stage.GetSessionLayer().GetPrimAtPath(preview_path_for(_INSTANCER_PATH))
        self.assertIsNotNone(container)
        self.assertEqual(len(container.nameChildren), 4)
        self.assertIn(Sdf.Path(_INSTANCER_PATH), self._preview.shown_instancers)

    async def test_destroy_takes_the_stand_in_back_out_of_the_session_layer(self):
        # Arrange
        layer = self._stage.GetRootLayer()
        author_placements(layer, _records(self._asset, 2))
        merge_loose_placements(layer, self._stage)
        await self._updates()

        # Act
        self._preview.destroy()

        # Assert
        session = self._stage.GetSessionLayer()
        self.assertIsNone(session.GetPrimAtPath(preview_path_for(_INSTANCER_PATH)))
        self.assertIsNone(session.GetPrimAtPath(_INSTANCER_PATH))
        self._preview = InstancerViewportPreview("")

    async def test_opening_a_stage_with_instancers_builds_their_stand_ins(self):
        # Arrange
        layer = self._stage.GetRootLayer()
        author_placements(layer, _records(self._asset, 3))
        merge_loose_placements(layer, self._stage)
        saved = pathlib.Path(self._temp_dir.name) / "merged.usda"
        layer.Export(str(saved))

        # Act
        await self._context.open_stage_async(str(saved))
        await self._updates(5)

        # Assert
        stage = self._context.get_stage()
        container = stage.GetSessionLayer().GetPrimAtPath(preview_path_for(_INSTANCER_PATH))
        self.assertIsNotNone(container)
        self.assertEqual(len(container.nameChildren), 3)
        self.assertIsNone(Sdf.Layer.FindOrOpen(str(saved)).GetPrimAtPath(preview_path_for(_INSTANCER_PATH)))


__all__ = ["TestInstancerViewportPreview", "TestSyncInstancerPreview"]
