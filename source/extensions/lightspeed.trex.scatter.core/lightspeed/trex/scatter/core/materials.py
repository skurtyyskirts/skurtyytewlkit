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

from __future__ import annotations

__all__ = [
    "APERTURE_PBR_TEXTURE_INPUTS",
    "DETECTABLE_TEXTURE_INPUTS",
    "NORMAL_TYPE_ENCODINGS",
    "asset_source_hash",
    "build_materials_payload",
    "resolve_asset_texture_set",
]

import re
from collections.abc import Mapping, Sequence
from typing import Any

import carb
from lightspeed.trex.utils.common.asset_utils import is_asset_ingested
from omni.flux.asset_importer.core import determine_ideal_types, parse_texture_paths
from omni.flux.asset_importer.core.data_models import TEXTURE_TYPE_INPUT_MAP, TEXTURE_TYPE_REGEX_MAP, TextureTypes
from omni.flux.utils.common.omni_url import OmniUrl
from omni.flux.utils.common.path_utils import read_metadata
from omni.flux.validator.factory import BASE_HASH_KEY
from pxr import Sdf, Usd, UsdShade

from .assets import asset_crate_supported
from .placement import make_relative_asset_path
from .settings import ScatterAssetEntry, ScatterMaterialOverride, ScatterTextureSet

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"

_TEXTURES_SUBDIRECTORY = "SubUSDs/textures"

NORMAL_TYPE_ENCODINGS = {
    TextureTypes.NORMAL_OTH: 0,
    TextureTypes.NORMAL_OGL: 1,
    TextureTypes.NORMAL_DX: 2,
}
"""Normal map texture type to the ``inputs:encoding`` value AperturePBR expects for it.

Mirrors ``lightspeed.common.constants.NormalMapEncodings`` and the ``normalmap_encoding`` enum of
``AperturePBR_Normal.mdl``. The ingested ``.n.rtex.dds`` tag cannot be used for this: it maps every normal type to
``NORMAL_OGL``, so it would author ``1`` for the octahedral maps Remix ingestion actually produces.
"""

DETECTABLE_TEXTURE_INPUTS = frozenset(
    TEXTURE_TYPE_INPUT_MAP[texture_type]
    for texture_type, pattern in TEXTURE_TYPE_REGEX_MAP.items()
    if pattern is not None and texture_type is not TextureTypes.OTHER
)
"""Texture inputs that can be recognised from a file name at all.

Remix has no filename pattern for anisotropy or the subsurface inputs, so no search could ever find one. Reporting
them as "not found" would mean listing them for every asset that ever existed.
"""

APERTURE_PBR_TEXTURE_INPUTS = {
    "AperturePBR_Opacity": frozenset(TEXTURE_TYPE_INPUT_MAP.values()),
    "AperturePBR_Translucent": frozenset(
        {
            "inputs:normalmap_texture",
            "inputs:emissive_mask_texture",
            "inputs:subsurface_transmittance_texture",
        }
    ),
}
"""Texture inputs each AperturePBR shader declares.

``ShaderInfoAPI`` would answer this from the MDL itself, but it lives behind ``omni.kit.window.preferences``, which has
no place in a headless engine. ``AperturePBR_Translucent`` has no diffuse, roughness or height input, so authoring
those would only create orphan attributes.
"""


def resolve_asset_texture_set(path: str, texture_source: str = "") -> ScatterTextureSet | None:
    """Inspect an ingested asset and work out which texture inputs the brush should author for it.

    Only inputs the asset leaves empty are resolved; anything it authors itself is reported in ``assigned`` and left
    alone.

    Args:
        path: Absolute path of the ingested USD asset.
        texture_source: A texture the user picked for this asset. When set, the set is taken from that file's siblings
            by name similarity instead of being auto-detected.

    Returns:
        The resolved set, or None when the path is empty or the file cannot be read at all.
    """
    # Checked before the crate probe: probing a path that is not there logs an error for a case that is simply "the
    # asset is gone", which the palette already reports elsewhere.
    if not path or not OmniUrl(path).exists or not asset_crate_supported(path):
        return None
    try:
        stage = Usd.Stage.Open(path)
    except Exception as exc:  # noqa: BLE001 - a malformed asset must not take the palette down
        carb.log_warn(f"{_LOG_PREFIX} Cannot open '{path}' to resolve its textures: {exc}")
        return None
    if stage is None:
        carb.log_warn(f"{_LOG_PREFIX} Cannot open '{path}' to resolve its textures")
        return None

    source_hash = asset_source_hash(path)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        # Placements reference the asset without a prim path, so a file without a default prim composes to nothing.
        return ScatterTextureSet(
            source_hash=source_hash, reason="the asset has no default prim and cannot be scattered"
        )

    shaders = _collect_shaders(default_prim, OmniUrl(path).stem)
    if not shaders:
        return ScatterTextureSet(
            source_hash=source_hash,
            reason="the asset has no AperturePBR material; re-ingest it so ingestion can create one",
        )

    assigned = tuple(sorted({name for shader in shaders for name in shader.assigned}))
    if not any(shader.missing for shader in shaders):
        return ScatterTextureSet(source_hash=source_hash, assigned=assigned)

    all_missing = tuple(sorted({name for shader in shaders for name in shader.missing} & DETECTABLE_TEXTURE_INPUTS))
    candidates = _candidate_textures(path, texture_source)
    if not candidates:
        return ScatterTextureSet(
            source_hash=source_hash,
            assigned=assigned,
            unresolved=all_missing,
            reason=(
                "no ingested textures were found for this asset; use Set textures... on its row, or ingest "
                "textures named after the material"
            ),
        )

    ideal_types = determine_ideal_types([_normalized(candidate) for candidate in candidates])
    overrides = []
    unresolved = set()
    for shader in shaders:
        inputs, encoding = _match_shader_textures(shader, candidates, ideal_types, texture_source)
        if inputs:
            overrides.append(ScatterMaterialOverride(shader_rel_path=shader.rel_path, inputs=inputs, encoding=encoding))
        unresolved.update((shader.missing - set(inputs)) & DETECTABLE_TEXTURE_INPUTS)

    return ScatterTextureSet(
        source_hash=source_hash,
        overrides=tuple(overrides),
        assigned=assigned,
        unresolved=tuple(sorted(unresolved)),
        reason="" if overrides else "no matching ingested texture set was found for this asset",
    )


def build_materials_payload(layer: Sdf.Layer, assets: Sequence[ScatterAssetEntry]) -> dict[str, list[dict[str, Any]]]:
    """Turn the palette's resolved texture sets into the JSON payload ``author_placements`` consumes.

    Keys come from the same :func:`make_relative_asset_path` call against the same layer that builds
    ``PlacementRecord.asset_rel_path``, so a record always finds its own entry.

    Args:
        layer: Layer the placements will be authored in; texture paths are relativized against it.
        assets: Palette entries whose ``textures`` have already been resolved.

    Returns:
        Relative asset path to the shader overrides to author, JSON-friendly so it survives the undo commands.
    """
    payload: dict[str, list[dict[str, Any]]] = {}
    for entry in assets:
        texture_set = entry.textures
        if texture_set is None or not texture_set.overrides:
            continue
        payload[make_relative_asset_path(layer, entry.path)] = [
            {
                "shader_rel_path": override.shader_rel_path,
                "inputs": {name: make_relative_asset_path(layer, texture) for name, texture in override.inputs.items()},
                "encoding": override.encoding,
            }
            for override in texture_set.overrides
        ]
    return payload


class _ShaderInfo:
    """One AperturePBR shader of an asset, with the texture inputs it does and does not author."""

    __slots__ = ("asset_stem", "assigned", "material_name", "missing", "rel_path")

    def __init__(self, rel_path: str, material_name: str, asset_stem: str, assigned: set[str], missing: set[str]):
        self.rel_path = rel_path
        self.material_name = material_name
        self.asset_stem = asset_stem
        self.assigned = assigned
        self.missing = missing


def _collect_shaders(default_prim: Usd.Prim, asset_stem: str) -> list[_ShaderInfo]:
    """Return every AperturePBR shader under ``default_prim`` with its authored and missing texture inputs."""
    shaders = []
    default_path = default_prim.GetPath()
    # The all-prims predicate is required: a `def` under an `over` ancestor is not IsDefined() and would be skipped.
    for prim in Usd.PrimRange(default_prim, Usd.PrimAllPrimsPredicate):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        declared = APERTURE_PBR_TEXTURE_INPUTS.get(shader.GetSourceAssetSubIdentifier("mdl"))
        if declared is None:
            continue
        assigned = set()
        for shader_input in shader.GetInputs():
            if shader_input.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = shader_input.Get()
            if value is not None and getattr(value, "path", ""):
                assigned.add(shader_input.GetFullName())
        shaders.append(
            _ShaderInfo(
                rel_path=str(prim.GetPath().MakeRelativePath(default_path)),
                material_name=_material_name(prim),
                asset_stem=asset_stem,
                assigned=assigned,
                missing=set(declared) - assigned,
            )
        )
    return shaders


def _material_name(shader_prim: Usd.Prim) -> str:
    """Return the name of the material owning ``shader_prim``, or the shader's own name when it has none."""
    prim = shader_prim.GetParent()
    while prim and not prim.IsPseudoRoot():
        if prim.IsA(UsdShade.Material):
            return prim.GetName()
        prim = prim.GetParent()
    return shader_prim.GetName()


def _candidate_textures(asset_path: str, texture_source: str) -> list[OmniUrl]:
    """List the ingested textures that could belong to ``asset_path``.

    With a user-picked ``texture_source`` only that file's own directory is searched. Otherwise the conventional
    ingestion output folder is searched first, then the asset's own directory.
    """
    if texture_source:
        directories = [OmniUrl(texture_source).parent_url]
    else:
        asset_directory = OmniUrl(asset_path).parent_url
        directories = [f"{asset_directory}/{_TEXTURES_SUBDIRECTORY}", asset_directory]

    seen_directories = set()
    seen_files = set()
    candidates = []
    for directory in directories:
        directory_key = str(directory).replace("\\", "/").rstrip("/").lower()
        if directory_key in seen_directories:
            continue
        seen_directories.add(directory_key)
        for url in OmniUrl(directory).iterdir():
            if url.suffix.lower() != ".dds":
                continue
            path = _normalized(url)
            if path.lower() in seen_files:
                continue
            if not is_asset_ingested(path, ignore_invalid_paths=False):
                continue
            seen_files.add(path.lower())
            candidates.append(url)
    return candidates


def _match_shader_textures(
    shader: _ShaderInfo,
    candidates: Sequence[OmniUrl],
    ideal_types: Mapping[str, TextureTypes],
    texture_source: str,
) -> tuple[dict[str, str], int | None]:
    """Pick the textures of ``shader``'s own set and map them onto the inputs it is missing.

    Returns the inputs to author and the ``inputs:encoding`` that goes with the normal map, if one is authored.
    """
    if texture_source:
        chosen = _sharing_the_longest_prefix(candidates, texture_source)
    else:
        chosen = _named_after(candidates, (shader.material_name, shader.asset_stem))

    normal_input = TEXTURE_TYPE_INPUT_MAP[TextureTypes.NORMAL_OGL]
    inputs: dict[str, str] = {}
    encoding: int | None = None
    for url in chosen:
        path = _normalized(url)
        texture_type = ideal_types.get(path)
        if texture_type is None or texture_type is TextureTypes.OTHER:
            # OTHER falls back to diffuse in TEXTURE_TYPE_INPUT_MAP, which would silently mis-assign the texture.
            continue
        input_name = TEXTURE_TYPE_INPUT_MAP[texture_type]
        if input_name not in shader.missing or input_name in inputs:
            continue
        inputs[input_name] = path
        if input_name == normal_input:
            encoding = NORMAL_TYPE_ENCODINGS.get(texture_type)
    if normal_input not in inputs:
        # Only a normal map the brush authors itself justifies replacing the encoding the asset declares. An asset
        # with no normal map often declares a stale encoding, which would silently mis-decode the map we add.
        encoding = None
    return inputs, encoding


def _named_after(candidates: Sequence[OmniUrl], names: Sequence[str]) -> list[OmniUrl]:
    """Return the candidates named after one of ``names``, most specific name first.

    A candidate qualifies when its file name begins with the name's words and the very next word names a texture type
    in Remix's own vocabulary: ``grass`` claims ``grass_albedo.a.rtex.dds`` but not ``grass_medium_02_diff_1k.a.rtex
    .dds``, whose next word is ``medium``. That word test is what keeps one asset from stealing another's textures out
    of the ingestion folder they share, and it is why a descriptor Remix does not recognise on its own (``nor``) makes
    the map show up as missing instead of being guessed at.
    """
    for name in names:
        words = [word.lower() for word in _name_words(name)]
        if not words:
            continue
        chosen = [candidate for candidate in candidates if _is_named_after(candidate, words)]
        if chosen:
            return chosen
    return []


def _is_named_after(candidate: OmniUrl, words: Sequence[str]) -> bool:
    """Return whether a texture file name contains ``words`` followed by the name of a texture type.

    The run does not have to open the file name. Asset packs routinely stamp their own name in front of every
    texture, so ``Quaternius_Leaves_diff`` is the texture of a material called ``Leaves``. What has to come directly
    after the run is the texture type, and that is what still stops a material called ``grass`` from claiming
    ``grass_medium_02_diff_4k``: the word after ``grass`` there is ``medium``, which names no type.
    """
    parts = [part.lower() for part in _name_parts(candidate.name)]
    wanted = list(words)
    for start in range(len(parts) - len(wanted) + 1):
        if parts[start : start + len(wanted)] != wanted:
            continue
        rest = parts[start + len(wanted) :]
        if rest and _begins_a_texture_type(rest):
            return True
    return False


def _begins_a_texture_type(rest: Sequence[str]) -> bool:
    """Return whether the words following a name begin a texture-type descriptor.

    A descriptor can take two words: an OpenGL normal map is conventionally written ``nor_gl``, which names a type
    only once both words are read together.
    """
    if _names_a_texture_type(rest[0]):
        return True
    return len(rest) > 1 and _names_a_texture_type(rest[0] + rest[1])


def _names_a_texture_type(word: str) -> bool:
    """Return whether a single word is one of the texture-type keywords Remix detects."""
    return any(
        pattern is not None and re.search(pattern, word, re.IGNORECASE) for pattern in TEXTURE_TYPE_REGEX_MAP.values()
    )


def _sharing_the_longest_prefix(candidates: Sequence[OmniUrl], texture_source: str) -> list[OmniUrl]:
    """Return the candidates that share the longest run of leading words with the picked texture.

    The user has already said these are the right textures, so the only job left is to keep the other assets in the
    same folder out. Taking the longest shared run does that without assuming the files are named consistently:
    ``grass_medium_02_rough_1k`` and ``grass_medium_02_nor_gl_1k_OTH_Normal`` both share three words with
    ``grass_medium_02_diff_alpha_1k``, while a ``stone_albedo`` beside them shares none.
    """
    picked_name = OmniUrl(texture_source).name
    picked = [part.lower() for part in _name_parts(picked_name)]
    if not picked:
        return []
    # The picked file is itself a candidate and shares every word with itself, so it is scored separately or it
    # would set a bar none of its siblings could clear.
    siblings = [candidate for candidate in candidates if candidate.name != picked_name]
    shared = {
        _normalized(candidate): _shared_leading_words(picked, [part.lower() for part in _name_parts(candidate.name)])
        for candidate in siblings
    }
    longest = max(shared.values(), default=0)
    chosen = [candidate for candidate in candidates if candidate.name == picked_name]
    if longest:
        chosen.extend(candidate for candidate in siblings if shared[_normalized(candidate)] == longest)
    return chosen


def _shared_leading_words(first: Sequence[str], second: Sequence[str]) -> int:
    """Return how many leading words two file names have in common."""
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def _name_parts(name: str) -> list[str]:
    """Split a texture file name the way Remix's texture-set matching does, dropping the ingested ``rtex`` tag."""
    parts = parse_texture_paths([name]).get(name, [])
    if parts and parts[-1] == "rtex":
        parts = parts[:-2]
    return parts


def _name_words(name: str) -> list[str]:
    """Split a plain name (a material or a file stem) with the same tokenizer Remix uses for texture file names.

    ``parse_texture_paths`` only looks at files, so the name is given the extension it insists on and gets it removed
    again as part of the split.
    """
    return parse_texture_paths([f"{name}.dds"]).get(f"{name}.dds", [])


def _normalized(url: OmniUrl | str) -> str:
    """Return a path string with forward slashes, the form every lookup in this module compares on."""
    return str(url).replace("\\", "/")


def asset_source_hash(path: str) -> str:
    """Return the asset's ingestion base hash, used to notice that it has been re-ingested.

    Empty when the asset has no ``.meta`` sidecar, which simply means the set is re-resolved every time.
    """
    return str(read_metadata(path, BASE_HASH_KEY) or "")
