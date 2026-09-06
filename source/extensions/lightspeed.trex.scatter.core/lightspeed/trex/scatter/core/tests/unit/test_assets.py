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

__all__ = ["TestAssets"]

import tempfile
from pathlib import Path
from unittest import mock

import omni.kit.test
import omni.usd
from lightspeed.common.constants import IS_REMIX_REF_ATTR
from lightspeed.trex.asset_replacements.core.shared.data_models import AssetType, DefaultAssetDirectory
from lightspeed.trex.scatter.core import assets
from lightspeed.trex.scatter.core.settings import UpAxis
from pxr import Sdf, Usd, UsdGeom

_REF_PRIM_PATH = "/RootNode/meshes/mesh_0AB745B8BEE1F16B/ref_0123456789abcdef"


def _write_asset_layer(
    directory: Path, name: str, up_axis: str | None = None, default_prim: str | None = "Asset"
) -> str:
    """Write a small asset layer to disk and return its forward-slash path."""
    path = (directory / name).as_posix()
    stage = Usd.Stage.CreateNew(path)
    if default_prim:
        root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
        UsdGeom.Mesh.Define(stage, f"/{default_prim}/geo")
        stage.SetDefaultPrim(root.GetPrim())
    if up_axis:
        UsdGeom.SetStageUpAxis(stage, up_axis)
    stage.GetRootLayer().Save()
    return path


def _context_with_selection(stage: Usd.Stage | None, selected_paths: list[str]) -> mock.MagicMock:
    usd_context = mock.MagicMock()
    usd_context.get_stage.return_value = stage
    usd_context.get_selection.return_value.get_selected_prim_paths.return_value = selected_paths
    return usd_context


def _define_remix_ref_prim(stage: Usd.Stage) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, _REF_PRIM_PATH).GetPrim()
    prim.CreateAttribute(IS_REMIX_REF_ATTR, Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    return prim


class TestAssets(omni.kit.test.AsyncTestCase):
    async def test_resolve_reference_asset_from_selection_returns_reference_of_nearest_remix_ref_prim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            directory = Path(temp_dir)
            (directory / "assets").mkdir()
            asset_path = _write_asset_layer(directory / "assets", "asset.usda")
            stage = Usd.Stage.CreateNew((directory / "scene.usda").as_posix())
            ref_prim = _define_remix_ref_prim(stage)
            ref_prim.GetReferences().AddReference("./assets/asset.usda")
            usd_context = _context_with_selection(stage, [f"{_REF_PRIM_PATH}/geo"])

            # Act
            result = assets.resolve_reference_asset_from_selection(usd_context)

            # Assert
            self.assertEqual(Path(result), Path(asset_path))

    async def test_resolve_reference_asset_from_selection_with_no_selection_or_stage_returns_none(self):
        stage = Usd.Stage.CreateInMemory()
        cases = [("no selection", stage, []), ("no stage", None, [_REF_PRIM_PATH])]
        for title, candidate_stage, selection in cases:
            with self.subTest(title=title):
                # Arrange
                usd_context = _context_with_selection(candidate_stage, selection)

                # Act
                result = assets.resolve_reference_asset_from_selection(usd_context)

                # Assert
                self.assertIsNone(result)

    async def test_resolve_reference_asset_from_selection_without_remix_ref_ancestor_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/Cube")
        usd_context = _context_with_selection(stage, ["/World/Cube"])

        # Act
        result = assets.resolve_reference_asset_from_selection(usd_context)

        # Assert
        self.assertIsNone(result)

    async def test_resolve_reference_asset_from_selection_with_only_internal_reference_returns_none(self):
        # Arrange
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/Source")
        ref_prim = _define_remix_ref_prim(stage)
        ref_prim.GetReferences().AddInternalReference(Sdf.Path("/World/Source"))
        usd_context = _context_with_selection(stage, [_REF_PRIM_PATH])

        # Act
        result = assets.resolve_reference_asset_from_selection(usd_context)

        # Assert
        self.assertIsNone(result)

    async def test_list_ingested_models_returns_only_usd_files_from_core(self):
        # Arrange
        response = mock.Mock(file_paths=["C:/p/a.usd", "C:/p/b.dds", "C:/p/c.USDA", "C:/p/d.png", "C:/p/e.usdc"])
        with mock.patch.object(assets, "AssetReplacementsCore") as core_cls:
            core_cls.return_value.get_available_assets_with_data_model.return_value = response

            # Act
            result = assets.list_ingested_models("test_context")

        # Assert
        self.assertEqual(result, ["C:/p/a.usd", "C:/p/c.USDA", "C:/p/e.usdc"])
        core_cls.assert_called_once_with("test_context")
        call = core_cls.return_value.get_available_assets_with_data_model.call_args
        self.assertEqual(call.args[0], DefaultAssetDirectory.INGESTED)
        self.assertEqual(call.args[1].asset_type, AssetType.MODELS)

    async def test_list_ingested_models_when_core_raises_returns_empty_list(self):
        # Arrange
        with mock.patch.object(assets, "AssetReplacementsCore") as core_cls:
            core_cls.return_value.get_available_assets_with_data_model.side_effect = ValueError("No stage")

            # Act
            result = assets.list_ingested_models("test_context")

        # Assert
        self.assertEqual(result, [])

    async def test_read_asset_up_axis_returns_authored_axis(self):
        cases = [("Y", UpAxis.Y), ("Z", UpAxis.Z)]
        with tempfile.TemporaryDirectory() as temp_dir:
            for authored, expected in cases:
                with self.subTest(title=f"upAxis={authored}"):
                    # Arrange
                    path = _write_asset_layer(Path(temp_dir), f"asset_{authored}.usda", up_axis=authored)

                    # Act
                    result = assets.read_asset_up_axis(path)

                    # Assert
                    self.assertEqual(result, expected)

    async def test_read_asset_up_axis_without_authored_axis_defaults_to_z(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            path = _write_asset_layer(Path(temp_dir), "asset_no_axis.usda")

            # Act
            result = assets.read_asset_up_axis(path)

            # Assert
            self.assertEqual(result, UpAxis.Z)

    async def test_read_asset_up_axis_with_missing_file_defaults_to_z(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            path = (Path(temp_dir) / "does_not_exist.usda").as_posix()

            # Act
            result = assets.read_asset_up_axis(path)

            # Assert
            self.assertEqual(result, UpAxis.Z)

    async def test_asset_crate_supported_delegates_to_omni_usd(self):
        # Arrange
        path = "C:/project/assets/ingested/model.usd"
        with mock.patch.object(omni.usd, "is_usd_crate_file_version_supported", return_value=False) as check:
            # Act
            result = assets.asset_crate_supported(path)

        # Assert
        self.assertFalse(result)
        check.assert_called_once_with(path)

    async def test_asset_crate_supported_with_empty_path_returns_false_without_delegating(self):
        # Arrange
        with mock.patch.object(omni.usd, "is_usd_crate_file_version_supported", return_value=True) as check:
            # Act
            result = assets.asset_crate_supported("")

        # Assert
        self.assertFalse(result)
        check.assert_not_called()

    async def test_asset_default_prim_name_returns_layer_default_prim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            path = _write_asset_layer(Path(temp_dir), "asset.usda", default_prim="Asset")

            # Act
            result = assets.asset_default_prim_name(path)

            # Assert
            self.assertEqual(result, "Asset")

    async def test_asset_default_prim_name_without_default_prim_or_file_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cases = [
                ("no default prim", _write_asset_layer(directory, "asset_no_default.usda", default_prim=None)),
                ("missing file", (directory / "does_not_exist.usda").as_posix()),
            ]
            for title, path in cases:
                with self.subTest(title=title):
                    # Arrange
                    candidate = path

                    # Act
                    result = assets.asset_default_prim_name(candidate)

                    # Assert
                    self.assertIsNone(result)
