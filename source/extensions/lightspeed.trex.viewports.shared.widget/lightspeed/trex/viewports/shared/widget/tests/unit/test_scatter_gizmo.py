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

__all__ = ["TestScatterGizmoTool"]

import contextlib
import pathlib
import tempfile
from unittest import mock

import carb.settings
import numpy as np
import omni.kit.test
import omni.usd
from lightspeed.common.constants import INSTANCE_PATH, ROOTNODE_INSTANCES, ROOTNODE_MESHES
from lightspeed.trex.scatter.core import (
    MeshSurfaceCache,
    PaddingIndex,
    ScatterAssetEntry,
    ScatterBrushSettings,
    ScatterMode,
    ScatterTarget,
    UpAxis,
    author_placements,
    generate_stamp,
    get_scatter_brush_controller,
)
from lightspeed.trex.scatter.core import controller as controller_module
from lightspeed.trex.scatter.core.constants import CONTAINER_PREFIX
from lightspeed.trex.viewports.shared.widget.tools import scatter_gizmo as scatter_gizmo_module
from lightspeed.trex.viewports.shared.widget.tools.scatter_gizmo import ScatterGizmoTool, scatter_gizmo_factory
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

_CONTEXT_NAME = "scatter_gizmo_unit_test"
_HASH = "0AB745B8BEE1F16B"
_PROTOTYPE = Sdf.Path(f"{ROOTNODE_MESHES}/mesh_{_HASH}")
_INSTANCE_0 = Sdf.Path(f"{INSTANCE_PATH}{_HASH}_0")
_MESH_PATH = _INSTANCE_0.AppendChild("mesh")
_CONTAINER = _PROTOTYPE.AppendChild(f"{CONTAINER_PREFIX}grass")
_HALF_SIZE = 500.0
_TRANSFORM_OP_SETTING = "/app/transform/operation"
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


def _build_capture_stage(stage: Usd.Stage) -> None:
    """Mirror the capture topology: a prototype holding a flat quad in the XY plane and one instance referencing it."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/RootNode")
    stage.DefinePrim(ROOTNODE_MESHES)
    stage.DefinePrim(ROOTNODE_INSTANCES)
    UsdGeom.Xform.Define(stage, _PROTOTYPE)
    mesh = UsdGeom.Mesh.Define(stage, _PROTOTYPE.AppendChild("mesh"))
    points = [
        Gf.Vec3f(-_HALF_SIZE, -_HALF_SIZE, 0.0),
        Gf.Vec3f(_HALF_SIZE, -_HALF_SIZE, 0.0),
        Gf.Vec3f(_HALF_SIZE, _HALF_SIZE, 0.0),
        Gf.Vec3f(-_HALF_SIZE, _HALF_SIZE, 0.0),
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    instance = UsdGeom.Xform.Define(stage, _INSTANCE_0)
    instance.GetPrim().GetReferences().AddInternalReference(_PROTOTYPE)


class TestScatterGizmoTool(omni.kit.test.AsyncTestCase):
    """Drives the gizmo's scope, visibility and drag lifecycle with the manipulator itself mocked away."""

    async def setUp(self):
        self._stack = contextlib.ExitStack()
        enter = self._stack.enter_context
        self._previous_controller = controller_module._CONTROLLER_INSTANCE
        controller_module._CONTROLLER_INSTANCE = None
        enter(mock.patch.object(ScatterBrushSettings, "save_to_carb"))
        enter(mock.patch.object(ScatterBrushSettings, "load_from_carb", return_value=ScatterBrushSettings()))
        enter(mock.patch.object(controller_module, "save_assets_to_carb"))
        enter(mock.patch.object(controller_module, "load_assets_from_carb", return_value=[]))

        # The gizmo widget itself is omni.ui.scene; the rules under test are which placements it offers and what a
        # drag of it writes, so it is replaced rather than drawn.
        self._manipulator = mock.MagicMock(name="manipulator")
        self._model = mock.MagicMock(name="model")
        enter(mock.patch.object(scatter_gizmo_module, "TransformManipulator", return_value=self._manipulator))
        enter(mock.patch.object(scatter_gizmo_module, "SimpleTransformModel", return_value=self._model))
        self._execute = enter(mock.patch.object(scatter_gizmo_module.omni.kit.commands, "execute"))

        self._temp_dir = tempfile.TemporaryDirectory()
        asset_path = pathlib.Path(self._temp_dir.name) / "cube.usda"
        asset_path.write_text(_ASSET_USDA, encoding="utf-8")
        self._asset_path = asset_path.as_posix()

        self._usd_context = omni.usd.get_context(_CONTEXT_NAME) or omni.usd.create_context(_CONTEXT_NAME)
        await self._usd_context.new_stage_async()
        self._stage = self._usd_context.get_stage()
        _build_capture_stage(self._stage)
        self._layer = self._stage.GetRootLayer()
        self._placements = self._paint()

        self._settings = carb.settings.get_settings()
        self._previous_operation = self._settings.get(_TRANSFORM_OP_SETTING)
        self._settings.set(_TRANSFORM_OP_SETTING, "move")

        self._controller = get_scatter_brush_controller()
        self._controller.replace_settings(ScatterBrushSettings(preset_name="Grass"))
        self._controller.set_mode(ScatterMode.OFF)

        viewport_api = mock.MagicMock(name="viewport_api")
        viewport_api.usd_context_name = _CONTEXT_NAME
        self._tool = ScatterGizmoTool(viewport_api, lambda: True, _CONTEXT_NAME)

    async def tearDown(self):
        self._tool.destroy()
        self._tool = None
        self._settings.set(_TRANSFORM_OP_SETTING, self._previous_operation or "select")
        self._usd_context.get_selection().set_selected_prim_paths([], True)
        controller_module.destroy_scatter_brush_controller()
        controller_module._CONTROLLER_INSTANCE = self._previous_controller
        self._temp_dir.cleanup()
        self._stack.close()

    def _paint(self) -> list[str]:
        """Author one stamp of grass onto the quad, the way a brush stroke does."""
        cache = MeshSurfaceCache(lambda: self._stage)
        target = ScatterTarget(
            prototype_root=_PROTOTYPE,
            parent_instance_root=_INSTANCE_0,
            mesh_path=_MESH_PATH,
            parent_world=Gf.Matrix4d(1.0),
            instance_count=1,
        )
        sample = cache.closest_point(_MESH_PATH, Gf.Vec3d(0.0, 0.0, 0.0), 10.0)
        records = generate_stamp(
            cache,
            target,
            sample,
            ScatterBrushSettings(preset_name="Grass", seed=5, randomize_seed=False, padding=0.0),
            [ScatterAssetEntry(path=self._asset_path, up_axis=UpAxis.Z, natural_extent=(1.0, 1.0, 2.0))],
            np.random.default_rng(1),
            PaddingIndex(1.0),
            None,
            UpAxis.Z,
            1.0,
            self._layer,
        )
        created = author_placements(self._layer, records)
        cache.destroy()
        self.assertTrue(created, "the test stamp authored no placement")
        return created

    def _select(self, paths: list[str]) -> None:
        """Set the USD selection and let the gizmo resolve it again."""
        self._usd_context.get_selection().set_selected_prim_paths(paths, True)
        self._tool.refresh()

    def _positions(self) -> list[tuple]:
        """Return the authored translation of every painted placement."""
        return [
            tuple(self._layer.GetPrimAtPath(Sdf.Path(path)).attributes["xformOp:translate"].default)
            for path in self._placements
        ]

    async def test_selecting_a_scatter_container_shows_the_gizmo_on_it(self):
        # Arrange
        self._manipulator.reset_mock()

        # Act
        self._select([str(_CONTAINER)])

        # Assert
        self.assertEqual(self._tool.scope.placement_count, len(self._placements))
        self.assertTrue(self._manipulator.enabled)

    async def test_selecting_nothing_hides_the_gizmo(self):
        # Arrange
        self._select([str(_CONTAINER)])

        # Act
        self._select([])

        # Assert
        self.assertTrue(self._tool.scope.is_empty)
        self.assertFalse(self._manipulator.enabled)

    async def test_arming_the_brush_hides_the_gizmo_so_it_cannot_take_the_paint_drag(self):
        # Arrange
        self._select([str(_CONTAINER)])

        # Act
        self._controller.set_mode(ScatterMode.PAINT)

        # Assert
        self.assertFalse(self._manipulator.enabled)

    async def test_switching_the_viewport_out_of_move_and_rotate_hides_the_gizmo(self):
        # Arrange
        self._select([str(_CONTAINER)])

        # Act
        self._settings.set(_TRANSFORM_OP_SETTING, "select")
        self._tool.refresh()

        # Assert
        self.assertFalse(self._manipulator.enabled)

    async def test_the_gizmo_stays_hidden_in_a_viewport_that_is_not_the_active_one(self):
        # Arrange
        viewport_api = mock.MagicMock(name="viewport_api")
        viewport_api.usd_context_name = _CONTEXT_NAME
        inactive = ScatterGizmoTool(viewport_api, lambda: False, _CONTEXT_NAME)
        self._usd_context.get_selection().set_selected_prim_paths([str(_CONTAINER)], True)

        # Act
        inactive.refresh()

        # Assert
        self.assertFalse(self._manipulator.enabled)
        inactive.destroy()

    async def test_dragging_moves_every_placement_of_the_selected_patch(self):
        # Arrange
        self._select([str(_CONTAINER)])
        before = self._positions()

        # Act
        self._tool.begin_drag()
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(40.0, 0.0, 0.0)))
        self._tool.end_drag()

        # Assert
        moved = [new[0] - old[0] for new, old in zip(self._positions(), before)]
        self.assertTrue(all(abs(value - 40.0) < 1e-6 for value in moved))

    async def test_a_drag_commits_one_undoable_command(self):
        # Arrange
        self._select([str(_CONTAINER)])
        self._tool.begin_drag()
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, 0.0, 0.0)))
        self._execute.reset_mock()

        # Act
        self._tool.end_drag()

        # Assert
        self._execute.assert_called_once()
        self.assertEqual(self._execute.call_args.args[0], "ScatterReapplyCommand")

    async def test_every_step_of_a_drag_is_measured_from_where_the_drag_started(self):
        # Arrange
        self._select([str(_CONTAINER)])
        before = self._positions()
        self._tool.begin_drag()

        # Act - the gizmo reports the total moved so far, so two steps are not two moves
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(10.0, 0.0, 0.0)))
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(25.0, 0.0, 0.0)))
        self._tool.end_drag()

        # Assert
        moved = [new[0] - old[0] for new, old in zip(self._positions(), before)]
        self.assertTrue(all(abs(value - 25.0) < 1e-6 for value in moved))

    async def test_cancelling_a_drag_puts_the_patch_back_and_commits_nothing(self):
        # Arrange
        self._select([str(_CONTAINER)])
        before = self._positions()
        self._tool.begin_drag()
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(60.0, 0.0, 0.0)))
        self._execute.reset_mock()

        # Act
        self._tool.cancel_drag()

        # Assert
        self.assertEqual(self._positions(), before)
        self._execute.assert_not_called()

    async def test_dragging_with_nothing_selected_writes_nothing(self):
        # Arrange
        self._select([])
        before = self._positions()

        # Act
        self._tool.begin_drag()
        self._tool.update_drag(Gf.Matrix4d().SetTranslate(Gf.Vec3d(40.0, 0.0, 0.0)))
        self._tool.end_drag()

        # Assert
        self.assertEqual(self._positions(), before)
        self._execute.assert_not_called()

    async def test_the_factory_builds_a_gizmo_for_the_viewport_it_is_given(self):
        # Arrange
        viewport_api = mock.MagicMock(name="viewport_api")
        viewport_api.usd_context_name = _CONTEXT_NAME

        # Act
        tool = scatter_gizmo_factory(
            {"viewport_api": viewport_api, "is_active_fn": lambda: True, "usd_context_name": _CONTEXT_NAME}
        )

        # Assert
        self.assertIsInstance(tool, ScatterGizmoTool)
        tool.destroy()
