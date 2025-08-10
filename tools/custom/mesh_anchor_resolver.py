"""
Mesh Anchor Resolver

APIs:
- resolve_or_create_anchor_for_hit(hit_prim_path: str, stage: Usd.Stage | None = None) -> Sdf.Path
- get_active_point_instancer(anchor_path: Sdf.Path | str, asset_set_key: str, stage: Usd.Stage | None = None) -> Sdf.Path

Behavior:
- Finds or creates a per-mesh anchor under /World/ScatterAnchors based on a capture hit prim path.
- Never authors into capture layers; requires the current edit target to be a non-capture (mod/replacement) layer.

Note:
- Designed to work standalone with pxr. If stage is not provided, it will attempt to fetch an omni.usd context if available.
"""

from __future__ import annotations

from typing import Optional
import re

try:  # Optional, only used when no stage is provided
    import omni.usd  # type: ignore
except Exception:  # pragma: no cover - tests pass a stage explicitly
    omni = None  # type: ignore

from pxr import Sdf, Usd, UsdGeom

# Reuse Remix constants for capture naming conventions
from lightspeed.common import constants as _C
from lightspeed.trex.utils.common.asset_utils import is_layer_from_capture as _is_layer_from_capture


_SCATTER_ANCHORS_ROOT = Sdf.Path("/World/ScatterAnchors")


def _get_stage(stage: Optional[Usd.Stage]) -> Usd.Stage:
    if stage:
        return stage
    try:
        import omni.usd as _omni_usd  # type: ignore

        ctx = _omni_usd.get_context()
        return ctx.get_stage()
    except Exception as _e:  # pragma: no cover - tests pass a stage explicitly
        pass
    raise RuntimeError("No stage provided and omni.usd context is not available")


def _ensure_mod_authoring(stage: Usd.Stage) -> None:
    edit_layer = stage.GetEditTarget().GetLayer()
    if edit_layer is None:
        raise RuntimeError("No active edit target layer set. Set a mod/replacement layer as the edit target.")
    # Path-based detection
    is_capture = _is_layer_from_capture(edit_layer.identifier)
    # Custom data-based detection (LayerManagerCore convention)
    try:
        layer_type = (edit_layer.customLayerData or {}).get("layer_type")
        if layer_type in ("capture", "capture_baker"):
            is_capture = True
    except Exception:
        pass
    if is_capture:
        raise RuntimeError("Refusing to author into the capture layer. Set a mod/replacement layer as the edit target.")


def _extract_mesh_hash_from_path(path: str) -> Optional[str]:
    # Try direct mesh path match
    m = re.match(_C.REGEX_MESH_PATH, path)
    if m:
        return m.group(3)  # the 16-char hash captured by the shared regex

    # Try inside-mesh path
    m = re.match(_C.REGEX_IN_MESH_PATH, path)
    if m:
        return m.group(3)

    # Try instance path -> mesh hash
    m = re.match(_C.REGEX_INSTANCE_PATH, path)
    if m:
        return m.group(3)

    m = re.match(_C.REGEX_IN_INSTANCE_PATH, path)
    if m:
        return m.group(3)

    # Fallback: search ancestor tokens for mesh_XXXXXXXXXXXXXXX
    parts = [p for p in path.split("/") if p]
    for token in reversed(parts):
        if token.startswith(_C.MESH_NAME_PREFIX):
            # token like mesh_<hash>
            hash_part = token[len(_C.MESH_NAME_PREFIX) :]
            if re.fullmatch(r"[A-Z0-9]{16}(?:_[0-9]+)?", hash_part):
                return hash_part.split("_")[0]
    return None


def _compute_anchor_path_from_hit(hit_prim_path: str) -> Sdf.Path:
    mesh_hash = _extract_mesh_hash_from_path(hit_prim_path)
    if not mesh_hash:
        # If nothing matches, sanitize name from last token
        last = hit_prim_path.rstrip("/").split("/")[-1]
        name = re.sub(r"[^A-Za-z0-9_]+", "_", last) or "anchor"
        return _SCATTER_ANCHORS_ROOT.AppendChild(name)
    anchor_name = f"{_C.MESH_NAME_PREFIX}{mesh_hash}"
    return _SCATTER_ANCHORS_ROOT.AppendChild(anchor_name)


def _define_anchors_root(stage: Usd.Stage) -> Sdf.Path:
    # Ensure /World exists
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    # Ensure /World/ScatterAnchors exists as Scope
    UsdGeom.Scope.Define(stage, _SCATTER_ANCHORS_ROOT)
    return _SCATTER_ANCHORS_ROOT


def resolve_or_create_anchor_for_hit(hit_prim_path: str, stage: Optional[Usd.Stage] = None) -> Sdf.Path:
    """
    Resolve the anchor prim for a picked hit. Create it if missing under /World/ScatterAnchors.

    Returns the Sdf.Path to the anchor prim.
    """
    stage = _get_stage(stage)
    _ensure_mod_authoring(stage)

    _define_anchors_root(stage)

    anchor_path = _compute_anchor_path_from_hit(hit_prim_path)
    prim = stage.GetPrimAtPath(anchor_path)
    if not prim or not prim.IsValid():
        UsdGeom.Xform.Define(stage, anchor_path)
        prim = stage.GetPrimAtPath(anchor_path)
        # Tag so runtime/tools can detect anchors
        attr = prim.CreateAttribute("remix_anchor", Sdf.ValueTypeNames.Bool, custom=True)
        attr.Set(True)
        # Also store the source mesh path for diagnostics
        prim.CreateAttribute("remix_anchor:source_hit_path", Sdf.ValueTypeNames.String, custom=True).Set(hit_prim_path)

    return anchor_path


def _sanitize_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def get_active_point_instancer(
    anchor_path: Sdf.Path | str, asset_set_key: str, stage: Optional[Usd.Stage] = None
) -> Sdf.Path:
    """
    Find or create a UsdGeomPointInstancer under the given anchor. One PI per asset_set_key.

    Returns the Sdf.Path to the PointInstancer prim.
    """
    stage = _get_stage(stage)
    _ensure_mod_authoring(stage)

    anchor_path = Sdf.Path(anchor_path) if not isinstance(anchor_path, Sdf.Path) else anchor_path
    if not stage.GetPrimAtPath(anchor_path):
        raise RuntimeError(f"Anchor {anchor_path} does not exist. Call resolve_or_create_anchor_for_hit() first.")

    key = _sanitize_token(asset_set_key or "default")
    pi_path = anchor_path.AppendChild(f"PI_{key}")

    prim = stage.GetPrimAtPath(pi_path)
    if not prim or not prim.IsValid():
        # Define PointInstancer
        UsdGeom.PointInstancer.Define(stage, pi_path)
    return pi_path