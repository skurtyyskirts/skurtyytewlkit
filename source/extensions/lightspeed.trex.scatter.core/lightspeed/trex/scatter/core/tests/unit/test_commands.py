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
from unittest import mock

import omni.kit.commands
import omni.kit.test
import omni.kit.undo
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from lightspeed.trex.scatter.core import commands as _commands
from lightspeed.trex.scatter.core.commands import ScatterStrokeCommand, ScatterStrokeKind
from lightspeed.trex.scatter.core.constants import (
    CONTAINER_PREFIX,
    IS_REMIX_SCATTER_ATTR,
    SCATTER_ASSET_ATTR,
    SCATTER_BRUSH_ID_ATTR,
)
from lightspeed.trex.scatter.core.placement import PlacementRecord, author_placements, remove_placements, snapshot_prims
from pxr import Gf, Sdf, Usd

_PROTOTYPE_ROOT = "/RootNode/meshes/mesh_0123456789ABCDEF"
_CONTAINER_PATH = f"{_PROTOTYPE_ROOT}/{CONTAINER_PREFIX}default"
_MISSING_LAYER_IDENTIFIER = "anon:0x1:scatter_missing_layer.usda"
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


def _write_asset(temp_dir: tempfile.TemporaryDirectory, file_name: str) -> str:
    path = pathlib.Path(temp_dir.name) / "assets" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ASSET_USDA, encoding="utf-8")
    return path.as_posix()


def _make_record(
    name: str, asset_path: str, translate: tuple[float, float, float] = (1.0, 2.0, 3.0)
) -> PlacementRecord:
    return PlacementRecord(
        container_path=_CONTAINER_PATH,
        prim_name=name,
        asset_rel_path=asset_path,
        asset_abs_path=asset_path,
        translate=translate,
        rotate_xyz=(0.0, 0.0, 45.0),
        scale=(1.5, 1.5, 1.5),
        brush_id="Default",
    )


def _new_stage_with_prototype() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(_PROTOTYPE_ROOT, "Xform")
    return stage


class TestScatterStrokeCommand(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # The extension registers these commands on startup; register them here too so the test is self-contained,
        # and restore that startup registration in tearDown so later tests in the same process still find them.
        self._commands_were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        self._registered = omni.kit.commands.register_all_commands_in_module(_commands)
        omni.kit.undo.clear_stack()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._asset = _write_asset(self._temp_dir, "cube.usda")
        self._stage = _new_stage_with_prototype()
        self._layer = self._stage.GetRootLayer()
        self._records = [
            _make_record("s_000000000001", self._asset, translate=(1.0, 2.0, 3.0)),
            _make_record("s_000000000002", self._asset, translate=(4.0, 5.0, 6.0)),
        ]

    async def tearDown(self):
        omni.kit.commands.unregister_module_commands(self._registered)
        if self._commands_were_registered:
            omni.kit.commands.register_all_commands_in_module(_commands)
        self._stage = None
        self._layer = None
        self._temp_dir.cleanup()

    def _execute_place(
        self, records: list[PlacementRecord], already_applied: bool = True, layer_identifier: str | None = None
    ) -> tuple[bool, object]:
        return omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=self._layer.identifier if layer_identifier is None else layer_identifier,
            kind=ScatterStrokeKind.PLACE.value,
            records=[record.to_dict() for record in records],
            already_applied=already_applied,
        )

    def _execute_erase(
        self, prim_paths: list[str], snapshot_layer: Sdf.Layer | None, already_applied: bool = True
    ) -> tuple[bool, object]:
        return omni.kit.commands.execute(
            "ScatterStrokeCommand",
            context_name="",
            layer_identifier=self._layer.identifier,
            kind=ScatterStrokeKind.ERASE.value,
            prim_paths=prim_paths,
            snapshot_layer=snapshot_layer,
            already_applied=already_applied,
        )

    def _erase_live(self, record: PlacementRecord) -> Sdf.Layer:
        """Mimic a live erase stroke: author, snapshot, then remove the placement before the command is committed."""
        author_placements(self._layer, [record])
        snapshot = snapshot_prims(self._layer, [record.prim_path])
        remove_placements(self._layer, [record.prim_path])
        return snapshot

    def _container_child_names(self) -> list[str]:
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        return [] if container is None else list(container.nameChildren.keys())

    async def test_place_already_applied_first_execute_keeps_authored_prims_without_duplicates(self):
        # Arrange
        author_placements(self._layer, self._records)

        # Act
        success, _ = self._execute_place(self._records, already_applied=True)

        # Assert
        self.assertTrue(success)
        self.assertEqual(self._container_child_names(), ["s_000000000001", "s_000000000002"])
        spec = self._layer.GetPrimAtPath(self._records[0].prim_path)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))

    async def test_place_already_applied_undo_removes_created_specs_and_empty_container(self):
        # Arrange
        author_placements(self._layer, self._records)
        self._execute_place(self._records, already_applied=True)

        # Act
        omni.kit.undo.undo()

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(self._records[0].prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(self._records[1].prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
        self.assertIsNotNone(self._layer.GetPrimAtPath(_PROTOTYPE_ROOT))

    async def test_place_already_applied_undo_keeps_unrelated_sibling_placement(self):
        # Arrange
        sibling = _make_record("s_00000sibling", self._asset)
        author_placements(self._layer, [sibling])
        author_placements(self._layer, self._records)
        self._execute_place(self._records, already_applied=True)

        # Act
        omni.kit.undo.undo()

        # Assert
        self.assertEqual(self._container_child_names(), ["s_00000sibling"])
        self.assertEqual(self._layer.GetPrimAtPath(_CONTAINER_PATH).specifier, Sdf.SpecifierDef)

    async def test_place_already_applied_redo_recreates_specs(self):
        # Arrange
        author_placements(self._layer, self._records)
        self._execute_place(self._records, already_applied=True)
        omni.kit.undo.undo()

        # Act
        omni.kit.undo.redo()

        # Assert
        self.assertEqual(self._container_child_names(), ["s_000000000001", "s_000000000002"])
        spec = self._layer.GetPrimAtPath(self._records[1].prim_path)
        self.assertEqual(list(spec.referenceList.prependedItems), [Sdf.Reference(assetPath=self._asset)])
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(4.0, 5.0, 6.0))
        self.assertEqual(spec.attributes[IS_REMIX_REF_ATTR].default, True)
        self.assertTrue(self._stage.GetPrimAtPath(self._records[1].prim_path).IsValid())

    async def test_place_not_applied_do_authors_specs(self):
        # Arrange
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

        # Act
        success, created = self._execute_place(self._records, already_applied=False)

        # Assert
        self.assertTrue(success)
        self.assertEqual(created, [record.prim_path for record in self._records])
        self.assertEqual(self._container_child_names(), ["s_000000000001", "s_000000000002"])
        self.assertEqual(
            self._layer.GetPrimAtPath(_CONTAINER_PATH).attributes[SCATTER_BRUSH_ID_ATTR].default, "Default"
        )

    async def test_place_not_applied_on_disk_layer_authors_specs_in_that_layer(self):
        # Arrange
        layer = Sdf.Layer.CreateNew((pathlib.Path(self._temp_dir.name) / "mod.usda").as_posix())
        record = _make_record("s_000000000001", "./assets/cube.usda")

        # Act
        success, _ = self._execute_place([record], already_applied=False, layer_identifier=layer.identifier)

        # Assert
        self.assertTrue(success)
        spec = layer.GetPrimAtPath(record.prim_path)
        self.assertIsNotNone(spec)
        self.assertEqual(list(spec.referenceList.prependedItems), [Sdf.Reference(assetPath="./assets/cube.usda")])
        self.assertIsNone(self._layer.GetPrimAtPath(record.prim_path))

    async def test_erase_already_applied_undo_restores_prim_with_all_attributes(self):
        # Arrange
        record = self._records[0]
        snapshot = self._erase_live(record)
        self._execute_erase([record.prim_path], snapshot, already_applied=True)

        # Act
        omni.kit.undo.undo()

        # Assert
        spec = self._layer.GetPrimAtPath(record.prim_path)
        self.assertEqual(spec.specifier, Sdf.SpecifierDef)
        self.assertEqual(spec.typeName, "Xform")
        self.assertEqual(list(spec.referenceList.prependedItems), [Sdf.Reference(assetPath=self._asset)])
        self.assertEqual(spec.attributes[IS_REMIX_REF_ATTR].default, True)
        self.assertEqual(spec.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertEqual(spec.attributes[SCATTER_ASSET_ATTR].default, self._asset)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(spec.attributes["xformOp:rotateXYZ"].default, Gf.Vec3f(0.0, 0.0, 45.0))
        self.assertEqual(spec.attributes["xformOp:scale"].default, Gf.Vec3f(1.5, 1.5, 1.5))
        self.assertEqual(list(spec.attributes["xformOpOrder"].default), _XFORM_OP_ORDER)
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(container.specifier, Sdf.SpecifierDef)
        self.assertEqual(container.typeName, "Xform")
        self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        self.assertTrue(self._stage.GetPrimAtPath(record.prim_path).IsValid())

    async def test_erase_already_applied_first_execute_is_a_no_op(self):
        # Arrange
        record = self._records[0]
        snapshot = self._erase_live(record)

        # Act
        success, _ = self._execute_erase([record.prim_path], snapshot, already_applied=True)

        # Assert
        self.assertTrue(success)
        self.assertIsNone(self._layer.GetPrimAtPath(record.prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_erase_already_applied_redo_removes_prim_again(self):
        # Arrange
        record = self._records[0]
        snapshot = self._erase_live(record)
        self._execute_erase([record.prim_path], snapshot, already_applied=True)
        omni.kit.undo.undo()

        # Act
        omni.kit.undo.redo()

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(record.prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
        self.assertFalse(self._stage.GetPrimAtPath(record.prim_path).IsValid())

    async def test_erase_not_applied_do_removes_prims(self):
        # Arrange
        record = self._records[0]
        author_placements(self._layer, [record])

        # Act
        success, _ = self._execute_erase([record.prim_path], None, already_applied=False)

        # Assert
        self.assertTrue(success)
        self.assertIsNone(self._layer.GetPrimAtPath(record.prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_erase_not_applied_undo_restores_prims_from_internal_snapshot(self):
        # Arrange
        record = self._records[0]
        author_placements(self._layer, [record])
        self._execute_erase([record.prim_path], None, already_applied=False)

        # Act
        omni.kit.undo.undo()

        # Assert
        spec = self._layer.GetPrimAtPath(record.prim_path)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertEqual(self._layer.GetPrimAtPath(_CONTAINER_PATH).typeName, "Xform")

    async def test_missing_layer_on_do_logs_warning_and_does_not_raise(self):
        # Arrange
        with mock.patch.object(_commands.carb, "log_warn") as log_warn_mock:
            # Act
            success, _ = self._execute_place(
                self._records, already_applied=False, layer_identifier=_MISSING_LAYER_IDENTIFIER
            )

        # Assert
        self.assertTrue(success)
        log_warn_mock.assert_called_once()
        self.assertIn(_MISSING_LAYER_IDENTIFIER, log_warn_mock.call_args.args[0])
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_missing_layer_on_undo_logs_warning_and_does_not_raise(self):
        # Arrange
        with mock.patch.object(_commands.carb, "log_warn") as log_warn_mock:
            self._execute_place(self._records, already_applied=True, layer_identifier=_MISSING_LAYER_IDENTIFIER)
            log_warn_mock.reset_mock()

            # Act
            omni.kit.undo.undo()

        # Assert
        log_warn_mock.assert_called_once()
        self.assertIn(_MISSING_LAYER_IDENTIFIER, log_warn_mock.call_args.args[0])

    async def test_invalid_kind_raises_value_error(self):
        # Arrange
        kind = "SPRINKLE"

        # Act
        with self.assertRaises(ValueError):
            ScatterStrokeCommand(context_name="", layer_identifier=self._layer.identifier, kind=kind)

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))


class TestScatterFloodCommand(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # The extension registers these commands on startup; register them here too so the test is self-contained,
        # and restore that startup registration in tearDown so later tests in the same process still find them.
        self._commands_were_registered = omni.kit.commands.get_command_class("ScatterStroke") is not None
        self._registered = omni.kit.commands.register_all_commands_in_module(_commands)
        omni.kit.undo.clear_stack()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._asset = _write_asset(self._temp_dir, "cube.usda")
        self._stage = _new_stage_with_prototype()
        self._layer = self._stage.GetRootLayer()
        self._records = [
            _make_record("s_000000000001", self._asset, translate=(1.0, 2.0, 3.0)),
            _make_record("s_000000000002", self._asset, translate=(4.0, 5.0, 6.0)),
        ]

    async def tearDown(self):
        omni.kit.commands.unregister_module_commands(self._registered)
        if self._commands_were_registered:
            omni.kit.commands.register_all_commands_in_module(_commands)
        self._stage = None
        self._layer = None
        self._temp_dir.cleanup()

    def _execute_flood(self, layer_identifier: str | None = None) -> tuple[bool, object]:
        return omni.kit.commands.execute(
            "ScatterFloodCommand",
            context_name="",
            layer_identifier=self._layer.identifier if layer_identifier is None else layer_identifier,
            records=[record.to_dict() for record in self._records],
        )

    async def test_do_authors_placements(self):
        # Arrange
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

        # Act
        success, created = self._execute_flood()

        # Assert
        self.assertTrue(success)
        self.assertEqual(created, [record.prim_path for record in self._records])
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(list(container.nameChildren.keys()), ["s_000000000001", "s_000000000002"])
        self.assertEqual(container.attributes[IS_REMIX_SCATTER_ATTR].default, True)
        spec = self._layer.GetPrimAtPath(self._records[0].prim_path)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(1.0, 2.0, 3.0))
        self.assertTrue(self._stage.GetPrimAtPath(self._records[0].prim_path).GetChild("Cube").IsValid())

    async def test_undo_removes_placements_and_empty_container(self):
        # Arrange
        self._execute_flood()

        # Act
        omni.kit.undo.undo()

        # Assert
        self.assertIsNone(self._layer.GetPrimAtPath(self._records[0].prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(self._records[1].prim_path))
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))

    async def test_redo_recreates_placements(self):
        # Arrange
        self._execute_flood()
        omni.kit.undo.undo()

        # Act
        omni.kit.undo.redo()

        # Assert
        container = self._layer.GetPrimAtPath(_CONTAINER_PATH)
        self.assertEqual(list(container.nameChildren.keys()), ["s_000000000001", "s_000000000002"])
        spec = self._layer.GetPrimAtPath(self._records[1].prim_path)
        self.assertEqual(spec.attributes["xformOp:translate"].default, Gf.Vec3d(4.0, 5.0, 6.0))

    async def test_missing_layer_logs_warning_and_does_not_raise(self):
        # Arrange
        with mock.patch.object(_commands.carb, "log_warn") as log_warn_mock:
            # Act
            success, created = self._execute_flood(layer_identifier=_MISSING_LAYER_IDENTIFIER)

        # Assert
        self.assertTrue(success)
        self.assertEqual(created, [])
        log_warn_mock.assert_called_once()
        self.assertIsNone(self._layer.GetPrimAtPath(_CONTAINER_PATH))
