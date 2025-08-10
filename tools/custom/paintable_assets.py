from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional USD import (available when running inside Kit or a USD-enabled Python)
try:
    from pxr import Gf, Usd, UsdGeom, UsdShade  # type: ignore
    HAVE_USD = True
    # Detect our local pxr stub and treat as no-USD for functionality
    if getattr(Usd, "_FAKE_PXR", False):  # type: ignore[attr-defined]
        HAVE_USD = False
except Exception:  # pragma: no cover - exercised in environments without USD
    HAVE_USD = False


PROTOTYPES_ENV = "OMNISCATTER_PROTOTYPES_DIR"
DEFAULT_RELATIVE_INDEX = Path("extensions/omniscatter/data/prototypes/prototypes_index.json")


@dataclasses.dataclass
class PaintableAsset:
    id: str
    display_name: str
    usd_path: str
    prototype_prim_path: str
    default_material_path: Optional[str]
    thumbnail_path: Optional[str]
    bounding_radius: float
    uses_mdl: bool


def _resolve_prototypes_index() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns a tuple (prototypes_root_dir, prototypes_index_json).
    The root dir contains the USDs and thumbnails. Index JSON is inside it.
    Resolution order:
      1) PROTOTYPES_ENV (file path or directory path)
      2) ./extensions/omniscatter/data/prototypes/prototypes_index.json
      3) First index found under ./_build/**/extensions/omniscatter/data/prototypes/
    """
    # 1) Environment override
    env = os.environ.get(PROTOTYPES_ENV)
    if env:
        p = Path(env)
        if p.is_file():
            return p.parent, p
        elif p.is_dir():
            idx = p / "prototypes_index.json"
            if idx.exists():
                return p, idx
    # 2) Workspace default
    idx = DEFAULT_RELATIVE_INDEX
    if idx.exists():
        return idx.parent, idx
    # 3) Search in _build
    build_root = Path("_build")
    if build_root.exists():
        for sub in build_root.rglob("prototypes_index.json"):
            if sub.parent.name == "prototypes" and "extensions" in sub.as_posix():
                return sub.parent, sub
    return None, None


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _join_root(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (root / rel_or_abs).resolve()


def _usd_find_prototype_and_material(stage: "Usd.Stage") -> Tuple[Optional[str], Optional[str], bool]:
    """
    Best-effort probe to find prototype root prim path and its default bound material path.
    Returns (prototype_prim_path, material_path, uses_mdl)
    """
    proto_path: Optional[str] = None
    mat_path: Optional[str] = None
    uses_mdl = False

    # Prefer well-known path used by our generator
    cand = stage.GetPrimAtPath("/World/proto")
    if cand and cand.IsValid():
        proto_path = str(cand.GetPath())
        # Try direct binding on proto prim
        mat, _ = UsdShade.MaterialBindingAPI(cand).ComputeBoundMaterial()
        if mat:
            mat_path = str(mat.GetPrim().GetPath())
            uses_mdl = _material_looks_like_mdl(mat)
            return proto_path, mat_path, uses_mdl
        # Fallback: search for a mesh child and its binding
        for prim in stage.Traverse():
            if prim.GetPath().pathString.startswith("/World/proto"):
                try:
                    if UsdGeom.Mesh(prim):
                        m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                        if m:
                            proto_path = str(prim.GetPath())
                            mat_path = str(m.GetPrim().GetPath())
                            uses_mdl = _material_looks_like_mdl(m)
                            return proto_path, mat_path, uses_mdl
                except Exception:
                    continue
    # Otherwise, first mesh in the stage + its binding
    for prim in stage.Traverse():
        try:
            if UsdGeom.Mesh(prim):
                proto_path = str(prim.GetPath())
                m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if m:
                    mat_path = str(m.GetPrim().GetPath())
                    uses_mdl = _material_looks_like_mdl(m)
                return proto_path, mat_path, uses_mdl
        except Exception:
            continue

    return proto_path, mat_path, uses_mdl


def _material_looks_like_mdl(material: "UsdShade.Material") -> bool:
    """Heuristic: scan downstream shader ids for indications of MDL usage."""
    try:
        shader = UsdShade.Material(material).ComputeSurfaceSource()
        if not shader:
            return False
        sid = shader.GetIdAttr().Get()
        if isinstance(sid, str) and ("mdl" in sid.lower() or "omniPBR" in sid or "OmniPBR" in sid):
            return True
        # Scan connected shader graph
        visited = set()
        stack = [shader]
        while stack:
            sh = stack.pop()
            if sh in visited:
                continue
            visited.add(sh)
            sid2 = sh.GetIdAttr().Get()
            if isinstance(sid2, str) and "mdl" in sid2.lower():
                return True
            for inp in sh.GetInputs():
                src = inp.GetConnectedSource()
                if src:
                    stack.append(src[0])
    except Exception:
        pass
    return False


_POINTS_RE = re.compile(r"point3f\[\]\s+points\s*=\s*\[(.*?)\]", re.S)
_VEC_RE = re.compile(r"\(([-+0-9\.eE]+)\s*,\s*([-+0-9\.eE]+)\s*,\s*([-+0-9\.eE]+)\)")
_BINDING_RE = re.compile(r"rel\s+material:binding\s*=\s*<([^>]+)>")


def _parse_textual_usda_for_proto_and_material(usda_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to infer prototype prim path and bound material path from ASCII usda snippets.
    Assumes our generator structure where proto is at /World/proto.
    """
    prototype_path = "/World/proto" if "/World/proto" in usda_text else None
    material_path: Optional[str] = None

    # Find binding within proto block if present
    if prototype_path:
        # Narrow to the proto block first for better accuracy
        start = usda_text.find('def Mesh "proto"')
        if start < 0:
            start = usda_text.find('def Xform "proto"')
        if start >= 0:
            end = usda_text.find("\n}", start)
            snippet = usda_text[start : end + 1] if end > start else usda_text[start:]
            m = _BINDING_RE.search(snippet)
            if m:
                material_path = m.group(1)
    else:
        # Global search for any binding
        m = _BINDING_RE.search(usda_text)
        if m:
            material_path = m.group(1)

    return prototype_path, material_path


def _parse_textual_usda_points_radius(usda_text: str) -> Optional[float]:
    """Compute radius from points array if present."""
    m = _POINTS_RE.search(usda_text)
    if not m:
        return None
    vecs = _VEC_RE.findall(m.group(1))
    if not vecs:
        return None
    r2 = 0.0
    for x, y, z in vecs:
        fx, fy, fz = float(x), float(y), float(z)
        r2 = max(r2, fx * fx + fy * fy + fz * fz)
    return math.sqrt(r2)


def _compute_radius_usd(stage: "Usd.Stage", proto_path: Optional[str]) -> Optional[float]:
    try:
        prim = stage.GetPrimAtPath(proto_path) if proto_path else None
        # Prefer exact prim if it is a Mesh or has a Mesh child
        candidates = []
        if prim and prim.IsValid():
            if UsdGeom.Mesh(prim):
                candidates.append(prim)
            for child in stage.Traverse():
                if str(child.GetPath()).startswith(str(prim.GetPath())) and UsdGeom.Mesh(child):
                    candidates.append(child)
        if not candidates:
            for p in stage.Traverse():
                if UsdGeom.Mesh(p):
                    candidates.append(p)
                    break
        for p in candidates:
            try:
                mesh = UsdGeom.Mesh(p)
                # extent is min/max bounds; use diagonal length / 2
                extent_attr = mesh.GetExtentAttr()
                extent = extent_attr.Get()
                if extent and len(extent) == 2:
                    mn = extent[0]
                    mx = extent[1]
                    dx = mx[0] - mn[0]
                    dy = mx[1] - mn[1]
                    dz = mx[2] - mn[2]
                    return 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _entry_looks_paintable(entry: Dict, usd_path: Path, proto_path: Optional[str], mat_path: Optional[str]) -> bool:
    # If textures present and all are .dds, require that a material is bound and a proto path exists
    textures: Dict[str, str] = entry.get("textures") or {}
    if textures:
        exts = {Path(t).suffix.lower() for t in textures.values()}
        if exts and exts.issubset({".dds"}):
            if not (proto_path and mat_path):
                return False
    # Must have a prototype prim or at least a mesh in the file
    if proto_path:
        return True
    # Best-effort textual check for ASCII usda (binary .usd will fail fast)
    try:
        text = usd_path.read_text(encoding="utf-8")
        if 'def Mesh "proto"' in text or 'def Mesh ' in text or 'def Xform "proto"' in text:
            return True
    except Exception:
        pass
    # In environments without USD, accept non-DDS-only entries even if we cannot introspect the USD
    return not HAVE_USD


def list_paintable_assets() -> List[PaintableAsset]:
    root, idx = _resolve_prototypes_index()
    if not idx:
        return []
    data = _read_json(idx)
    assets: List[PaintableAsset] = []

    for ent in data.get("prototypes", []):
        try:
            asset_id = ent.get("uuid") or ent.get("id") or ent.get("name")
            if not asset_id:
                continue
            name = ent.get("name") or asset_id
            usd_path = _join_root(root, ent.get("usd_path"))

            proto_prim_path: Optional[str] = None
            material_path: Optional[str] = None
            uses_mdl = False
            radius: Optional[float] = None

            if HAVE_USD:
                try:
                    stage = Usd.Stage.Open(str(usd_path))
                    if stage:
                        proto_prim_path, material_path, uses_mdl = _usd_find_prototype_and_material(stage)
                        radius = _compute_radius_usd(stage, proto_prim_path)
                except Exception:
                    pass

            if not HAVE_USD or not proto_prim_path or radius is None:
                # textual fallback for ASCII usda
                try:
                    text = usd_path.read_text(encoding="utf-8")
                    if not proto_prim_path or not material_path:
                        pth, mph = _parse_textual_usda_for_proto_and_material(text)
                        proto_prim_path = proto_prim_path or pth
                        material_path = material_path or mph
                    if radius is None:
                        radius = _parse_textual_usda_points_radius(text)
                except Exception:
                    pass

            if not _entry_looks_paintable(ent, usd_path, proto_prim_path, material_path):
                continue

            thumb_rel = ent.get("thumbnail_128") or ent.get("thumbnail")
            thumb_path = _join_root(root, thumb_rel) if thumb_rel else None

            assets.append(
                PaintableAsset(
                    id=asset_id,
                    display_name=name,
                    usd_path=str(usd_path),
                    prototype_prim_path=proto_prim_path or "/World/proto",
                    default_material_path=material_path,
                    thumbnail_path=str(thumb_path) if thumb_path else None,
                    bounding_radius=float(radius) if radius is not None else 0.5,
                    uses_mdl=uses_mdl,
                )
            )
        except Exception:
            # Ignore malformed entries
            continue

    return assets


def resolve_asset_prototype(asset_id: str):  # -> Usd.Prim
    """
    Resolve the USD prototype root prim for a given asset id.
    Requires USD to be available. Raises RuntimeError if USD is not available.
    """
    if not HAVE_USD:
        raise RuntimeError("USD (pxr) is not available in this Python environment")

    root, idx = _resolve_prototypes_index()
    if not idx:
        raise FileNotFoundError("No prototypes_index.json was found")
    data = _read_json(idx)

    ent = next((e for e in data.get("prototypes", []) if (e.get("uuid") or e.get("id") or e.get("name")) == asset_id), None)
    if not ent:
        raise KeyError(f"Unknown asset id: {asset_id}")

    usd_path = _join_root(root, ent.get("usd_path"))
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    proto_prim_path, _, _ = _usd_find_prototype_and_material(stage)
    if not proto_prim_path:
        # Fallback to conventional path
        proto_prim_path = "/World/proto"
    prim = stage.GetPrimAtPath(proto_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prototype prim not found or invalid at path: {proto_prim_path}")
    return prim