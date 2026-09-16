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
from pxr import Gf, Sdf, Usd, UsdGeom

_REF_PRIM_PATH = "/RootNode/meshes/mesh_0AB745B8BEE1F16B/ref_0123456789abcdef"


def _write_asset_layer(
    directory: Path,
    name: str,
    up_axis: str | None = None,
    default_prim: str | None = "Asset",
    meters_per_unit: float | None = None,
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
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.GetRootLayer().Save()
    return path


def _write_ingested_shaped_asset(
    directory: Path,
    name: str,
    cube_size: float,
    wrapper_scale: float | None = None,
    default_prim_scale: float | None = None,
) -> str:
    """Write an asset shaped the way ingestion leaves one: a default prim wrapping an ``XForms`` prim and geometry.

    ``wrapper_scale`` stands in for the unit scale ingestion bakes onto the wrapper; ``default_prim_scale`` is the
    op a placement would replace, so it must not reach the measurement.
    """
    path = (directory / name).as_posix()
    stage = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(stage, "/ReferenceTarget")
    stage.SetDefaultPrim(root.GetPrim())
    if default_prim_scale is not None:
        root.AddScaleOp().Set(Gf.Vec3f(default_prim_scale))
    wrapper = UsdGeom.Xform.Define(stage, "/ReferenceTarget/XForms")
    if wrapper_scale is not None:
        wrapper.AddScaleOp().Set(Gf.Vec3f(wrapper_scale))
    cube = UsdGeom.Cube.Define(stage, "/ReferenceTarget/XForms/Cube")
    cube.CreateSizeAttr(cube_size)
    half = cube_size / 2.0
    cube.CreateExtentAttr([Gf.Vec3f(-half, -half, -half), Gf.Vec3f(half, half, half)])
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

    async def test_read_asset_up_axis_without_authored_axis_or_file_defaults_to_y(self):
        # Y is USD's own fallback. Answering Z would claim the asset already stands the way a Z-up stage does and
        # skip the up-axis correction, which lays it on its side with nothing to point at.
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cases = [
                ("no authored axis", _write_asset_layer(directory, "asset_no_axis.usda")),
                ("missing file", (directory / "does_not_exist.usda").as_posix()),
            ]
            for title, path in cases:
                with self.subTest(title=title):
                    # Act
                    result = assets.read_asset_up_axis(path)

                    # Assert
                    self.assertEqual(result, UpAxis.Y)

    async def test_read_authored_asset_up_axis_returns_the_authored_axis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for authored, expected in (("Y", UpAxis.Y), ("Z", UpAxis.Z)):
                with self.subTest(authored=authored):
                    # Arrange
                    path = _write_asset_layer(Path(temp_dir), f"authored_{authored}.usda", up_axis=authored)

                    # Act
                    result = assets.read_authored_asset_up_axis(path)

                    # Assert
                    self.assertEqual(result, expected)

    async def test_read_authored_asset_up_axis_without_authored_axis_or_file_returns_none(self):
        # None is what lets the palette tell "the file says nothing" apart from "the file says Y".
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cases = [
                ("no authored axis", _write_asset_layer(directory, "unauthored.usda")),
                ("missing file", (directory / "does_not_exist.usda").as_posix()),
                ("empty path", ""),
            ]
            for title, path in cases:
                with self.subTest(title=title):
                    # Act
                    result = assets.read_authored_asset_up_axis(path)

                    # Assert
                    self.assertIsNone(result)

    async def test_read_asset_meters_per_unit_returns_authored_unit_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for authored in (0.01, 1.0, 100.0):
                with self.subTest(authored=authored):
                    # Arrange
                    path = _write_asset_layer(Path(temp_dir), f"asset_{authored}.usda", meters_per_unit=authored)

                    # Act
                    result = assets.read_asset_meters_per_unit(path)

                    # Assert
                    self.assertAlmostEqual(result, authored)

    async def test_read_asset_meters_per_unit_without_authored_unit_or_file_defaults_to_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cases = [
                ("no authored unit", _write_asset_layer(directory, "asset_no_unit.usda")),
                ("missing file", (directory / "does_not_exist.usda").as_posix()),
            ]
            for title, path in cases:
                with self.subTest(title=title):
                    # Act
                    result = assets.read_asset_meters_per_unit(path)

                    # Assert
                    self.assertAlmostEqual(result, 1.0)

    async def test_read_asset_natural_extent_returns_the_default_prim_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            path = _write_ingested_shaped_asset(Path(temp_dir), "cube.usda", cube_size=10.0)

            # Act
            result = assets.read_asset_natural_extent(path)

            # Assert
            for axis, value in enumerate(result):
                self.assertAlmostEqual(value, 10.0, places=4, msg=f"axis {axis}")

    async def test_read_asset_natural_extent_includes_a_scale_op_below_the_default_prim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange - the shape ingestion produces: the source file's unit scale baked onto the XForms wrapper.
            path = _write_ingested_shaped_asset(Path(temp_dir), "scaled.usda", cube_size=10.0, wrapper_scale=100.0)

            # Act
            result = assets.read_asset_natural_extent(path)

            # Assert
            for axis, value in enumerate(result):
                self.assertAlmostEqual(value, 1000.0, places=2, msg=f"axis {axis}")

    async def test_read_asset_natural_extent_ignores_the_default_prims_own_transform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange - a placement authors its own xformOpOrder, so the default prim's ops never reach the stage.
            path = _write_ingested_shaped_asset(
                Path(temp_dir), "outer.usda", cube_size=10.0, wrapper_scale=100.0, default_prim_scale=5.0
            )

            # Act
            result = assets.read_asset_natural_extent(path)

            # Assert
            for axis, value in enumerate(result):
                self.assertAlmostEqual(value, 1000.0, places=2, msg=f"axis {axis}")

    async def test_read_asset_natural_extent_without_default_prim_or_file_returns_zeros(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            cases = [
                ("no default prim", _write_asset_layer(directory, "asset_no_default.usda", default_prim=None)),
                ("missing file", (directory / "does_not_exist.usda").as_posix()),
                ("empty path", ""),
            ]
            for title, path in cases:
                with self.subTest(title=title):
                    # Act
                    result = assets.read_asset_natural_extent(path)

                    # Assert
                    self.assertEqual(result, (0.0, 0.0, 0.0))

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
