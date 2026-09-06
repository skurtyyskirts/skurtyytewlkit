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
    "asset_crate_supported",
    "asset_default_prim_name",
    "list_ingested_models",
    "read_asset_up_axis",
    "resolve_reference_asset_from_selection",
]

from pathlib import PurePosixPath

import carb
import omni.usd
from lightspeed.common import constants as common_constants
from lightspeed.trex.asset_replacements.core.shared import Setup as AssetReplacementsCore
from lightspeed.trex.asset_replacements.core.shared.data_models import (
    AssetType,
    DefaultAssetDirectory,
    GetAvailableAssetsQueryModel,
)
from pxr import Sdf, Usd, UsdGeom

from .settings import UpAxis

_USD_SUFFIXES = frozenset(suffix.lower() for suffix in common_constants.USD_EXTENSIONS)


def resolve_reference_asset_from_selection(usd_context: omni.usd.UsdContext) -> str | None:
    """Return the asset file referenced by the selected replacement, so it can be added to the brush.

    The first selected prim is walked up to the nearest prim carrying ``IsRemixRef`` (itself included) and the
    first file reference authored on any of that prim's specs is resolved against the layer that authored it.

    Args:
        usd_context: Context whose selection and stage are read.

    Returns:
        Absolute asset path with forward slashes, or None without a selection, a stage, a reference prim or a file
        reference (internal references are skipped).
    """
    selected_paths = usd_context.get_selection().get_selected_prim_paths()
    if not selected_paths:
        return None
    stage = usd_context.get_stage()
    if stage is None:
        return None
    reference_prim = _nearest_remix_ref_prim(stage.GetPrimAtPath(selected_paths[0]))
    if reference_prim is None:
        return None
    for spec in reference_prim.GetPrimStack():
        for reference in spec.referenceList.GetAddedOrExplicitItems():
            asset_path = reference.assetPath
            if not asset_path:
                continue
            layer = spec.layer
            absolute_path = asset_path if layer.anonymous else layer.ComputeAbsolutePath(asset_path)
            return absolute_path.replace("\\", "/")
    return None


def list_ingested_models(context_name: str) -> list[str]:
    """List the USD files in the project's ingested assets directory.

    Args:
        context_name: USD context whose project is inspected.

    Returns:
        USD file paths, or an empty list when no project is loaded or the directory cannot be listed.
    """
    try:
        response = AssetReplacementsCore(context_name).get_available_assets_with_data_model(
            DefaultAssetDirectory.INGESTED, GetAvailableAssetsQueryModel(asset_type=AssetType.MODELS)
        )
    except (ValueError, RuntimeError, OSError) as exc:
        carb.log_warn(f"[lightspeed.trex.scatter.core] Unable to list ingested models: {exc}")
        return []
    return [path for path in response.file_paths if PurePosixPath(path).suffix.lower() in _USD_SUFFIXES]


def read_asset_up_axis(path: str) -> UpAxis:
    """Read the ``upAxis`` metadata of a USD file without composing it.

    Args:
        path: USD file path.

    Returns:
        The authored up axis; Z when the file cannot be opened or authors no up axis.
    """
    layer = Sdf.Layer.FindOrOpen(path) if path else None
    if layer is None:
        return UpAxis.Z
    pseudo_root = layer.pseudoRoot
    if not pseudo_root.HasInfo(UsdGeom.Tokens.upAxis):
        return UpAxis.Z
    return UpAxis.Y if str(pseudo_root.GetInfo(UsdGeom.Tokens.upAxis)) == UsdGeom.Tokens.y else UpAxis.Z


def asset_crate_supported(path: str) -> bool:
    """Return whether the USD crate version of ``path`` can be read by this Kit (true for non-crate files)."""
    if not path:
        return False
    return omni.usd.is_usd_crate_file_version_supported(path)


def asset_default_prim_name(path: str) -> str | None:
    """Return the default prim name of a USD file, or None when unset or the file cannot be opened."""
    layer = Sdf.Layer.FindOrOpen(path) if path else None
    if layer is None or not layer.HasDefaultPrim():
        return None
    return str(layer.defaultPrim) or None


def _nearest_remix_ref_prim(prim: Usd.Prim) -> Usd.Prim | None:
    """Return ``prim`` or its nearest ancestor carrying the Remix reference marker, or None when there is none."""
    while prim and not prim.IsPseudoRoot():
        if prim.HasAttribute(common_constants.IS_REMIX_REF_ATTR):
            return prim
        prim = prim.GetParent()
    return None
