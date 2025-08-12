"""
* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

__all__ = [
    "get_texture_type_input_name",
    "is_asset_ingested",
    "is_layer_from_capture",
    "is_mesh_from_capture",
    "is_texture_from_capture",
    "list_ingested_usd_assets",
]

from pathlib import Path

from lightspeed.common import constants
from omni.flux.asset_importer.core.data_models import (
    TEXTURE_TYPE_CONVERTED_SUFFIX_MAP,
    TEXTURE_TYPE_INPUT_MAP,
    TextureTypes,
)
from omni.flux.utils.common import path_utils
from omni.flux.validator.factory import BASE_HASH_KEY, VALIDATION_PASSED


def get_texture_type_input_name(texture_type: TextureTypes) -> str | None:
    return TEXTURE_TYPE_INPUT_MAP.get(texture_type, None)


def get_ingested_texture_type(texture_file_path: str | Path) -> TextureTypes | None:
    texture_path = Path(str(texture_file_path))
    suffixes = texture_path.suffixes

    if not suffixes:
        return None

    # Expect .a.rtex.dds -> ['.a', '.rtex', '.dds'] -> Only keep 'a'
    return TEXTURE_TYPE_CONVERTED_SUFFIX_MAP.get(suffixes[0][1:], None)


def is_asset_ingested(asset_path: str | Path, ignore_invalid_paths: bool = True) -> bool:
    path = str(asset_path)

    # Ignore invalid paths unless set not to ignore
    if not path_utils.is_file_path_valid(path, log_error=False):
        return ignore_invalid_paths

    # Ignore assets from captures
    if is_mesh_from_capture(path) or is_texture_from_capture(path):
        return True

    if not bool(
        path_utils.hash_match_metadata(path, key=BASE_HASH_KEY) and path_utils.read_metadata(path, VALIDATION_PASSED)
    ):
        return False

    return True


def is_layer_from_capture(layer_path: str) -> bool:
    path = Path(layer_path).resolve()
    return bool(constants.CAPTURE_FOLDER in path.parts or constants.REMIX_CAPTURE_FOLDER in path.parts)


def is_mesh_from_capture(asset_path: str) -> bool:
    path = Path(asset_path)
    return (
        bool(constants.CAPTURE_FOLDER in path.parts or constants.REMIX_CAPTURE_FOLDER in path.parts)
        and constants.MESHES_FOLDER in path.parts
    )


def is_texture_from_capture(texture_path: str) -> bool:
    path = Path(texture_path)
    return (
        bool(constants.CAPTURE_FOLDER in path.parts or constants.REMIX_CAPTURE_FOLDER in path.parts)
        and constants.TEXTURES_FOLDER in path.parts
    )


def list_ingested_usd_assets(context_name: str = "") -> list[tuple[str, str]]:
    """
    Return a list of (display_name, url) for all ingested USD assets in the current project.

    - Uses the active stage URL to determine the project root
    - Scans under `assets/ingested` recursively
    - Filters by USD extensions and validates that the asset is ingested via metadata
    """
    try:
        import omni.client  # type: ignore
        import omni.usd  # type: ignore
    except Exception:  # pragma: no cover - available in Kit
        return []

    ctx = omni.usd.get_context(context_name)
    stage_url = ctx.get_stage_url()
    if not stage_url:
        return []

    root_url = omni.client.normalize_url(str(omni.client.Uri(stage_url).get_dirname()))
    ingested_url = omni.client.combine_urls(root_url, constants.REMIX_INGESTED_ASSETS_FOLDER)

    results: list[tuple[str, str]] = []

    def walk(url: str):
        res, entries = omni.client.list(url)
        if res != omni.client.Result.OK:
            return
        for e in entries:
            child = omni.client.combine_urls(url, e.relative_path)
            if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                walk(child)
            elif e.flags & omni.client.ItemFlags.READABLE_FILE:
                lower = e.relative_path.lower()
                if any(lower.endswith(ext) for ext in constants.USD_EXTENSIONS):
                    if is_asset_ingested(child):
                        results.append((e.relative_path, child))

    try:
        walk(ingested_url)
    except Exception:  # pragma: no cover
        return []

    results.sort(key=lambda t: t[0].lower())
    return results
