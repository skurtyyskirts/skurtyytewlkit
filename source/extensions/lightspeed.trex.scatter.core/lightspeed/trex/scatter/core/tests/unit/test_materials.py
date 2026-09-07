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

import omni.kit.test
from lightspeed.trex.scatter.core.materials import (
    APERTURE_PBR_TEXTURE_INPUTS,
    build_materials_payload,
    resolve_asset_texture_set,
)
from lightspeed.trex.scatter.core.placement import make_relative_asset_path
from lightspeed.trex.scatter.core.settings import ScatterAssetEntry, ScatterMaterialOverride, ScatterTextureSet
from omni.flux.asset_importer.core.data_models import TEXTURE_TYPE_INPUT_MAP
from omni.flux.utils.common.path_utils import hash_file, write_metadata
from omni.flux.validator.factory import BASE_HASH_KEY, VALIDATION_PASSED
from pxr import Sdf

_DIFFUSE = "inputs:diffuse_texture"
_NORMAL = "inputs:normalmap_texture"
_ROUGHNESS = "inputs:reflectionroughness_texture"
_ENCODING = "inputs:encoding"

_MATERIAL_TEMPLATE = """    def Xform "{group}"
    {{
        def Scope "Looks"
        {{
            def Material "{material}"
            {{
                def Shader "Shader"
                {{
                    uniform token info:implementationSource = "sourceAsset"
                    uniform asset info:mdl:sourceAsset = @{mdl}.mdl@
                    uniform token info:mdl:sourceAsset:subIdentifier = "{mdl}"
{inputs}
                    token outputs:out
                }}
            }}
        }}
    }}
"""

_ASSET_TEMPLATE = """#usda 1.0
(
    defaultPrim = "ReferenceTarget"
    upAxis = "Z"
)

def Xform "ReferenceTarget"
{{
{materials}}}
"""

_NO_MATERIAL_USDA = """#usda 1.0
(
    defaultPrim = "ReferenceTarget"
)

def Xform "ReferenceTarget"
{
    def Mesh "Geo"
    {
    }
}
"""

_NO_DEFAULT_PRIM_USDA = """#usda 1.0
(
)

def Xform "Root"
{
}
"""


def known_inputs() -> set[str]:
    """Every texture input an ``AperturePBR_Opacity`` shader declares."""
    return set(APERTURE_PBR_TEXTURE_INPUTS["AperturePBR_Opacity"])


def _texture_set(*overrides) -> ScatterTextureSet:
    """Build a texture set from ``(shader_rel_path, inputs, encoding)`` tuples."""
    return ScatterTextureSet(
        overrides=tuple(
            ScatterMaterialOverride(shader_rel_path=path, inputs=inputs, encoding=encoding)
            for path, inputs, encoding in overrides
        )
    )


class _AssetFixture(omni.kit.test.AsyncTestCase):
    """Writes ingested-looking assets and textures into a temporary project folder."""

    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._root = pathlib.Path(self._temp_dir.name)
        self._textures_dir = self._root / "SubUSDs" / "textures"
        self._textures_dir.mkdir(parents=True, exist_ok=True)

    async def tearDown(self):
        self._temp_dir.cleanup()

    def _ingest(self, path: pathlib.Path) -> str:
        """Give a file the ingestion metadata ``is_asset_ingested`` looks for, so the resolver accepts it."""
        write_metadata(str(path), BASE_HASH_KEY, hash_file(str(path)))
        write_metadata(str(path), VALIDATION_PASSED, True)
        return path.as_posix()

    def _write_texture(self, name: str, ingested: bool = True) -> str:
        path = self._textures_dir / name
        path.write_bytes(b"fake dds payload")
        return self._ingest(path) if ingested else path.as_posix()

    def _write_asset(
        self,
        file_name: str = "asset.usda",
        materials: tuple[tuple[str, str], ...] = (("XForms", "brick"),),
        mdl: str = "AperturePBR_Opacity",
        inputs: dict[str, str] | None = None,
        encoding: int | None = None,
    ) -> str:
        """Write an asset shaped like ingestion output: a default prim wrapping one Looks scope per material."""
        lines = [f"                    asset {name} = @{value}@" for name, value in (inputs or {}).items()]
        if encoding is not None:
            lines.append(f"                    int {_ENCODING} = {encoding}")
        blocks = "".join(
            _MATERIAL_TEMPLATE.format(group=group, material=material, mdl=mdl, inputs="\n".join(lines))
            for group, material in materials
        )
        path = self._root / file_name
        path.write_text(_ASSET_TEMPLATE.format(materials=blocks), encoding="utf-8")
        return self._ingest(path)


class TestResolveAssetTextureSet(_AssetFixture):
    async def test_aperture_pbr_texture_inputs_are_all_known_texture_input_names(self):
        # Arrange
        known = set(TEXTURE_TYPE_INPUT_MAP.values())

        # Act / Assert
        for shader, inputs in APERTURE_PBR_TEXTURE_INPUTS.items():
            self.assertTrue(inputs.issubset(known), f"{shader} declares an input that is not a known texture input")

    async def test_asset_with_a_full_texture_set_authors_nothing_and_reports_them_assigned(self):
        # Arrange
        textures = {name: f"./SubUSDs/textures/brick_{index}.dds" for index, name in enumerate(sorted(known_inputs()))}
        asset = self._write_asset(inputs=textures)

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides, ())
        self.assertEqual(set(texture_set.assigned), set(textures))
        self.assertEqual(texture_set.unresolved, ())
        self.assertEqual(texture_set.reason, "")

    async def test_asset_missing_normal_and_roughness_returns_those_two_inputs_only(self):
        # Arrange
        asset = self._write_asset(inputs={_DIFFUSE: "./SubUSDs/textures/brick_albedo.a.rtex.dds"})
        self._write_texture("brick_albedo.a.rtex.dds")
        self._write_texture("brick_normal.n.rtex.dds")
        self._write_texture("brick_rough.r.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(len(texture_set.overrides), 1)
        self.assertEqual(set(texture_set.overrides[0].inputs), {_NORMAL, _ROUGHNESS})
        self.assertIn(_DIFFUSE, texture_set.assigned)

    async def test_shader_path_is_relative_to_the_default_prim(self):
        # A reference without a prim path maps the default prim's children onto the placement, so the default prim
        # name must not be part of the path the brush authors under it.
        # Arrange
        asset = self._write_asset()
        self._write_texture("brick_albedo.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides[0].shader_rel_path, "XForms/Looks/brick/Shader")

    async def test_octahedral_normal_sets_encoding_zero(self):
        # Arrange
        asset = self._write_asset()
        self._write_texture("brick_octahedral.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertIn(_NORMAL, texture_set.overrides[0].inputs)
        self.assertEqual(texture_set.overrides[0].encoding, 0)

    async def test_opengl_normal_sets_encoding_one(self):
        # Arrange
        asset = self._write_asset()
        self._write_texture("brick_normal.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides[0].encoding, 1)

    async def test_no_encoding_is_authored_without_a_normal_map(self):
        # Arrange
        asset = self._write_asset(encoding=2)
        self._write_texture("brick_albedo.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE})
        self.assertIsNone(texture_set.overrides[0].encoding)

    async def test_a_stale_encoding_is_replaced_when_the_normal_map_is_authored(self):
        # The asset declares DirectX encoding but carries no normal map, so the value describes nothing. Keeping it
        # would silently mis-decode the octahedral map the brush is about to assign.
        # Arrange
        asset = self._write_asset(encoding=2)
        self._write_texture("brick_octahedral.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertIn(_NORMAL, texture_set.overrides[0].inputs)
        self.assertEqual(texture_set.overrides[0].encoding, 0)

    async def test_a_texture_set_of_a_longer_named_asset_is_rejected_even_mid_name(self):
        # The run is allowed to sit anywhere in the file name, so this is the case that proves the type-word test is
        # what does the real work: `grass` appears in every one of these names and must still claim none of them.
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass"),))
        self._write_texture("Quaternius_grass_medium_02_diff_4k.a.rtex.dds")
        self._write_texture("grass_bermuda_01_rough_4k.r.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides, ())

    async def test_a_texture_set_that_only_shares_a_leading_word_is_rejected(self):
        # Ingestion folders are shared between assets: a material called `grass` must not claim the textures of
        # `grass_medium_02`, whose UVs belong to a different mesh. This is the most important test in this file.
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass"),))
        self._write_texture("grass_medium_02_diff_alpha_1k.a.rtex.dds")
        self._write_texture("grass_medium_02_rough_1k.r.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides, ())
        self.assertNotEqual(texture_set.reason, "")

    async def test_a_texture_set_named_after_the_material_is_accepted(self):
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass"),))
        self._write_texture("grass_albedo.a.rtex.dds")
        self._write_texture("grass_rough.r.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE, _ROUGHNESS})

    async def test_a_texture_set_named_after_the_asset_file_is_accepted(self):
        # Arrange
        asset = self._write_asset(file_name="pebble.usda", materials=(("XForms", "Material_001"),))
        self._write_texture("pebble_albedo.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE})

    async def test_a_two_word_descriptor_like_nor_gl_is_recognised(self):
        # PolyHaven writes an OpenGL normal map as `nor_gl`, which only names a type once both words are read.
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass_medium_02"),))
        self._write_texture("grass_medium_02_diff_alpha_1k.a.rtex.dds")
        self._write_texture("grass_medium_02_nor_gl_1k_OTH_Normal.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE, _NORMAL})
        self.assertEqual(texture_set.overrides[0].encoding, 0)

    async def test_a_texture_prefixed_with_an_asset_pack_name_is_claimed_by_its_material(self):
        # Asset packs stamp their own name in front of every texture, so the material name sits in the middle.
        # Arrange
        asset = self._write_asset(materials=(("XForms", "Leaves"),))
        self._write_texture("Quaternius_Leaves_diff.a.rtex.dds")
        self._write_texture("Quaternius_Flowers_diff.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE})
        self.assertTrue(
            texture_set.overrides[0].inputs[_DIFFUSE].endswith("Quaternius_Leaves_diff.a.rtex.dds"),
            texture_set.overrides[0].inputs[_DIFFUSE],
        )

    async def test_a_pack_prefixed_texture_of_another_material_is_not_claimed(self):
        # Arrange
        asset = self._write_asset(materials=(("XForms", "Leaves"),))
        self._write_texture("Quaternius_Flowers_diff.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides, ())

    async def test_a_picked_texture_resolves_the_set_auto_detection_rejects(self):
        # The Kenney-grass case: the only textures in the project belong to another asset, so the user points at one
        # and the rest of its set follows.
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass"),))
        picked = self._write_texture("grass_medium_02_diff_alpha_1k.a.rtex.dds")
        self._write_texture("grass_medium_02_nor_gl_1k_OTH_Normal.n.rtex.dds")
        self._write_texture("grass_medium_02_rough_1k.r.rtex.dds")

        # Act
        self.assertEqual(resolve_asset_texture_set(asset).overrides, ())

        # Assert
        texture_set = resolve_asset_texture_set(asset, texture_source=picked)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE, _NORMAL, _ROUGHNESS})
        self.assertEqual(texture_set.overrides[0].encoding, 0)

    async def test_a_picked_texture_leaves_out_another_asset_in_the_same_folder(self):
        # Arrange
        asset = self._write_asset(materials=(("XForms", "grass"),))
        picked = self._write_texture("grass_medium_02_diff_alpha_1k.a.rtex.dds")
        self._write_texture("grass_medium_02_rough_1k.r.rtex.dds")
        self._write_texture("stone_normal.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset, texture_source=picked)

        # Assert
        self.assertEqual(set(texture_set.overrides[0].inputs), {_DIFFUSE, _ROUGHNESS})

    async def test_an_asset_with_two_materials_returns_one_override_per_shader(self):
        # Arrange
        asset = self._write_asset(materials=(("XForms", "brick"), ("XForms2", "stone")))
        self._write_texture("brick_albedo.a.rtex.dds")
        self._write_texture("stone_albedo.a.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(
            {override.shader_rel_path: set(override.inputs) for override in texture_set.overrides},
            {"XForms/Looks/brick/Shader": {_DIFFUSE}, "XForms2/Looks/stone/Shader": {_DIFFUSE}},
        )

    async def test_a_translucent_asset_never_returns_diffuse_or_roughness_inputs(self):
        # Arrange
        asset = self._write_asset(materials=(("XForms", "glass"),), mdl="AperturePBR_Translucent")
        self._write_texture("glass_albedo.a.rtex.dds")
        self._write_texture("glass_rough.r.rtex.dds")
        self._write_texture("glass_normal.n.rtex.dds")

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        authored = {name for override in texture_set.overrides for name in override.inputs}
        self.assertEqual(authored, {_NORMAL})

    async def test_an_asset_without_a_material_returns_a_reason_and_no_overrides(self):
        # Arrange
        path = self._root / "no_material.usda"
        path.write_text(_NO_MATERIAL_USDA, encoding="utf-8")

        # Act
        texture_set = resolve_asset_texture_set(self._ingest(path))

        # Assert
        self.assertEqual(texture_set.overrides, ())
        self.assertIn("no AperturePBR material", texture_set.reason)

    async def test_an_asset_without_a_default_prim_returns_a_reason(self):
        # Arrange
        path = self._root / "no_default_prim.usda"
        path.write_text(_NO_DEFAULT_PRIM_USDA, encoding="utf-8")

        # Act
        texture_set = resolve_asset_texture_set(self._ingest(path))

        # Assert
        self.assertIn("no default prim", texture_set.reason)

    async def test_a_texture_that_was_not_ingested_is_skipped(self):
        # Arrange
        asset = self._write_asset()
        self._write_texture("brick_albedo.a.rtex.dds", ingested=False)

        # Act
        texture_set = resolve_asset_texture_set(asset)

        # Assert
        self.assertEqual(texture_set.overrides, ())

    async def test_an_empty_path_returns_none(self):
        # Act / Assert
        self.assertIsNone(resolve_asset_texture_set(""))

    async def test_the_ingestion_hash_is_recorded_so_a_re_ingest_is_noticed(self):
        # Arrange
        asset = self._write_asset()

        # Act / Assert
        self.assertNotEqual(resolve_asset_texture_set(asset).source_hash, "")


class TestBuildMaterialsPayload(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._root = pathlib.Path(self._temp_dir.name)
        self._layer = Sdf.Layer.CreateNew((self._root / "mod.usda").as_posix())

    async def tearDown(self):
        self._layer = None
        self._temp_dir.cleanup()

    def _entry(self, texture_set: ScatterTextureSet | None) -> ScatterAssetEntry:
        entry = ScatterAssetEntry(path=(self._root / "assets" / "brick.usda").as_posix())
        entry.textures = texture_set
        return entry

    async def test_payload_keys_match_the_record_asset_rel_paths(self):
        # Arrange
        entry = self._entry(_texture_set(("XForms/Looks/brick/Shader", {_DIFFUSE: "brick.a.rtex.dds"}, None)))

        # Act
        payload = build_materials_payload(self._layer, [entry])

        # Assert
        self.assertEqual(list(payload), [make_relative_asset_path(self._layer, entry.path)])

    async def test_payload_relativizes_texture_paths_to_the_layer(self):
        # Arrange
        texture = (self._root / "assets" / "SubUSDs" / "textures" / "brick.a.rtex.dds").as_posix()
        entry = self._entry(_texture_set(("XForms/Looks/brick/Shader", {_DIFFUSE: texture}, 0)))

        # Act
        payload = build_materials_payload(self._layer, [entry])

        # Assert
        authored = next(iter(payload.values()))[0]
        self.assertEqual(authored["inputs"][_DIFFUSE], "./assets/SubUSDs/textures/brick.a.rtex.dds")
        self.assertEqual(authored["encoding"], 0)

    async def test_payload_skips_entries_without_a_texture_set(self):
        # Act / Assert
        self.assertEqual(build_materials_payload(self._layer, [self._entry(None)]), {})

    async def test_payload_skips_entries_whose_texture_set_authors_nothing(self):
        # Act / Assert
        self.assertEqual(build_materials_payload(self._layer, [self._entry(_texture_set())]), {})
