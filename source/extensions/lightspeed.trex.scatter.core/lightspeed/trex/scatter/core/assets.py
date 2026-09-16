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
    "asset_comparison_key",
    "asset_crate_supported",
    "asset_default_prim_name",
    "list_ingested_models",
    "normalize_asset_path",
    "read_asset_meters_per_unit",
    "read_asset_natural_extent",
    "read_asset_up_axis",
    "read_authored_asset_up_axis",
    "resolve_reference_asset_from_selection",
]

import os
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
from omni.flux.utils.common.omni_url import OmniUrl
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


def normalize_asset_path(path: str) -> str:
    """Return an asset path trimmed and with forward slashes, as the brush stores it."""
    return str(path).strip().replace("\\", "/")


def asset_comparison_key(path: str) -> str:
    """Return the key that decides whether two asset paths mean the same file on this platform.

    Used wherever an asset has to be found again by path: adding to the palette, erasing by asset, and matching a
    placement back to the palette entry it was painted from.

    Args:
        path: Asset path in any form.

    Returns:
        The comparison key.
    """
    return os.path.normcase(os.path.normpath(normalize_asset_path(path)))


def read_authored_asset_up_axis(path: str) -> UpAxis | None:
    """Read the ``upAxis`` metadata a USD file actually authors, without composing it.

    Args:
        path: USD file path.

    Returns:
        The authored up axis, or None when the file cannot be opened or authors none. Callers that need a value
        should use :func:`read_asset_up_axis`; this variant exists so a file that says nothing can be told apart
        from one that says Y, which is what keeps the palette from reporting a mismatch against its own fallback.
    """
    layer = Sdf.Layer.FindOrOpen(path) if path else None
    if layer is None:
        return None
    pseudo_root = layer.pseudoRoot
    if not pseudo_root.HasInfo(UsdGeom.Tokens.upAxis):
        return None
    return UpAxis.Y if str(pseudo_root.GetInfo(UsdGeom.Tokens.upAxis)) == UsdGeom.Tokens.y else UpAxis.Z


def read_asset_up_axis(path: str) -> UpAxis:
    """Read the ``upAxis`` metadata of a USD file without composing it.

    Args:
        path: USD file path.

    Returns:
        The authored up axis; Y when the file cannot be opened or authors none, because that is what USD itself
        falls back to. Answering Z there would claim the asset already stands the way a Z-up stage does and skip the
        up-axis correction entirely, which lays the asset on its side with nothing to point at.
    """
    return read_authored_asset_up_axis(path) or UpAxis.Y


def read_asset_meters_per_unit(path: str) -> float:
    """Read the ``metersPerUnit`` metadata of a USD file without composing it.

    Args:
        path: USD file path.

    Returns:
        The authored unit scale; 1.0 when the file cannot be opened or authors none. The fallback is 1.0 rather than
        USD's own 0.01 so that a file declaring nothing is taken to already be in stage units, which makes the
        conversion a no-op instead of silently shrinking the asset a hundredfold.
    """
    layer = Sdf.Layer.FindOrOpen(path) if path else None
    if layer is None:
        return 1.0
    pseudo_root = layer.pseudoRoot
    if not pseudo_root.HasInfo(UsdGeom.Tokens.metersPerUnit):
        return 1.0
    return float(pseudo_root.GetInfo(UsdGeom.Tokens.metersPerUnit))


def read_asset_natural_extent(path: str) -> tuple[float, float, float]:
    """Measure how large a USD asset's default prim is along each axis, in the asset's own units.

    Transforms authored *below* the default prim are included, because ingestion bakes the source file's unit scale
    into a scale op on the wrapper prim it creates. The default prim's own transform is excluded, because a placement
    authors its whole ``xformOpOrder`` and so replaces anything the default prim declared for itself.

    Args:
        path: USD file path.

    Returns:
        The (x, y, z) extent, or zeros when the file is missing, unreadable, has no default prim, or bounds nothing.
    """
    # Checked before opening: composing a path that is not there logs an error for a case that is simply "the asset is
    # gone", which the palette already reports elsewhere.
    if not path or not OmniUrl(path).exists or not asset_crate_supported(path):
        return (0.0, 0.0, 0.0)
    try:
        stage = Usd.Stage.Open(path)
    except Exception as exc:  # noqa: BLE001 - a malformed asset must not take the palette down
        carb.log_warn(f"[lightspeed.trex.scatter.core] Cannot open '{path}' to measure it: {exc}")
        return (0.0, 0.0, 0.0)
    if stage is None:
        carb.log_warn(f"[lightspeed.trex.scatter.core] Cannot open '{path}' to measure it")
        return (0.0, 0.0, 0.0)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        # Placements reference the asset without a prim path, so a file without a default prim composes to nothing.
        return (0.0, 0.0, 0.0)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_range = bbox_cache.ComputeUntransformedBound(default_prim).ComputeAlignedRange()
    if aligned_range.IsEmpty():
        return (0.0, 0.0, 0.0)
    size = aligned_range.GetSize()
    return (float(size[0]), float(size[1]), float(size[2]))


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
