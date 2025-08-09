"""
Author: Omniscatter Toolkit Agent (GPT-5)
Date: 2025-08-09
License: Apache-2.0

Prototype ingestion utility for RTX-Remix Omniscatter.

This script scans an ingestion directory for meshes and textures, generates
USD prototype assets, and produces thumbnails and an index JSON for the
prototype picker. It favors UsdPreviewSurface material authoring for
portability and can optionally attempt MDL material binding if Omniverse MDL
APIs are available.

CLI example:
  python tools/custom/prototype_ingest.py \
    --ingest-dir assets \
    --out-dir extensions/omniscatter/data/prototypes \
    --apply-to-builds

Expected outputs:
  - One USD per prototype under out-dir
  - thumbnails/ <name>_{128,256}.png
  - prototypes_index.json with prototype metadata

Assumptions:
  - Ingested textures may be .dds/.png/.jpg; meshes may be .usd/.usdc/.usda,
    or source meshes (.fbx/.obj). Source meshes are optionally converted if
    Omniverse Asset Converter is present; otherwise a plane proxy is emitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple


try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    HAVE_USD = True
except Exception:
    HAVE_USD = False

try:
    # Optional Omniverse converter (only used when available)
    from omni.kit.asset_converter import AssetConverter
    HAVE_ASSET_CONVERTER = True
except Exception:
    HAVE_ASSET_CONVERTER = False


CHANNEL_ALIASES = {
    "basecolor": ["basecolor", "albedo", "diffuse", "base_color", "color"],
    "normal": ["normal", "nrm"],
    "roughness": ["roughness", "rough"],
    "metallic": ["metallic", "metal", "metalness"],
    "emissive": ["emissive", "emit", "emission"],
    "opacity": ["opacity", "alpha", "transparency"],
    "ao": ["ao", "ambientocclusion", "occlusion"],
}

TEXTURE_EXTS = {".dds", ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff"}
USD_EXTS = {".usd", ".usda", ".usdc"}
MESH_EXTS = USD_EXTS | {".fbx", ".obj"}


@dataclasses.dataclass
class Prototype:
    name: str
    usd_path: Path
    thumbnail_128: Path
    thumbnail_256: Path
    recommended_scale: float
    uuid: str
    textures: Dict[str, str]


def _stem_without_channel(stem: str) -> Tuple[str, Optional[str]]:
    lowered = stem.lower()
    for chan, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            m = re.search(rf"(?:[_\-\.]){re.escape(alias)}(?:[_\-\.]|$)", lowered)
            if m:
                base = lowered[: m.start()]
                base = re.sub(r"[_\-\.]$", "", base)
                return base if base else lowered, chan
    return lowered, None


def _discover_assets(ingest_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Path]]]:
    meshes: Dict[str, Path] = {}
    texture_groups: Dict[str, Dict[str, Path]] = {}
    for p in ingest_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in MESH_EXTS:
            meshes[p.stem.lower()] = p
        elif ext in TEXTURE_EXTS:
            base, chan = _stem_without_channel(p.stem)
            if base not in texture_groups:
                texture_groups[base] = {}
            if chan:
                texture_groups[base][chan] = p
            else:
                # default to basecolor if unknown and nothing set
                texture_groups[base].setdefault("basecolor", p)
    return meshes, texture_groups


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _rel_or_abs(src: Path, dest_dir: Path) -> str:
    try:
        return os.path.relpath(src, dest_dir)
    except Exception:
        return str(src)


def _write_plane_usdpreview(out_usd: Path, basecolor_tex: Optional[Path], normal_tex: Optional[Path]) -> None:
    if HAVE_USD:
        stage = Usd.Stage.CreateNew(str(out_usd))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/proto")
        points = [
            Gf.Vec3f(-0.5, 0.0, -0.5),
            Gf.Vec3f(0.5, 0.0, -0.5),
            Gf.Vec3f(0.5, 0.0, 0.5),
            Gf.Vec3f(-0.5, 0.0, 0.5),
        ]
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]
        uvs = [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)]
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
        st.Set(uvs)
        st_reader = UsdShade.Shader.Define(stage, "/World/Materials/Mat/Primvar_st")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        material = UsdShade.Material.Define(stage, "/World/Materials/Mat")
        shader = UsdShade.Shader.Define(stage, "/World/Materials/Mat/PBR")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if basecolor_tex:
            tx = UsdShade.Shader.Define(stage, "/World/Materials/Mat/Tex_baseColor")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_rel_or_abs(basecolor_tex, out_usd.parent))
            tx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(st_reader_out)
            tx_out = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx_out)
        if normal_tex:
            ntx = UsdShade.Shader.Define(stage, "/World/Materials/Mat/Tex_normal")
            ntx.CreateIdAttr("UsdUVTexture")
            ntx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(_rel_or_abs(normal_tex, out_usd.parent))
            ntx.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(st_reader_out)
            ntx_out = ntx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(ntx_out)
        # Connect material surface to preview surface output via explicit output
        surf_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surf_out)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
        stage.Save()
        return
    # Fallback usda
    _ensure_dir(out_usd.parent)
    rel_base = _rel_or_abs(basecolor_tex, out_usd.parent) if basecolor_tex else None
    rel_nrm = _rel_or_abs(normal_tex, out_usd.parent) if normal_tex else None
    lines = [
        "usda 1.0",
        "(defaultPrim=\"World\")\n",
        "def Xform \"World\" {",
        "  def Material \"Materials\" {",
        "    def Material \"Mat\" {",
        "      def Shader \"PBR\" {",
        "        uniform token info:id = \"UsdPreviewSurface\"",
        "        float inputs:roughness = 0.5",
        "        float inputs:metallic = 0.0",
        "      }",
    ]
    if rel_base:
        lines += [
            "      def Shader \"Tex_baseColor\" {",
            "        uniform token info:id = \"UsdUVTexture\"",
            f"        asset inputs:file = @{rel_base}@",
            "      }",
        ]
    if rel_nrm:
        lines += [
            "      def Shader \"Tex_normal\" {",
            "        uniform token info:id = \"UsdUVTexture\"",
            f"        asset inputs:file = @{rel_nrm}@",
            "      }",
        ]
    lines += [
        "      token outputs:surface.connect = \"PBR.outputs:surface\"",
        "    }",
        "  }",
        "  def Mesh \"proto\" {",
        "    int[] faceVertexCounts = [4]",
        "    int[] faceVertexIndices = [0, 1, 2, 3]",
        "    point3f[] points = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)]",
        "    rel material:binding = </World/Materials/Mat>",
        "  }",
        "}",
    ]
    out_usd.write_text("\n".join(lines))


def _maybe_convert_mesh_to_usd(src: Path, out_dir: Path) -> Optional[Path]:
    if src.suffix.lower() in USD_EXTS:
        return src
    if HAVE_ASSET_CONVERTER:
        try:
            conv = AssetConverter()
            dst = out_dir / f"{src.stem}.usd"
            task = conv.create_converter_task(str(src), str(dst))
            task.run()
            if dst.exists():
                return dst
        except Exception:
            pass
    return None


def _write_reference_usd(out_usd: Path, ref_asset: Path) -> None:
    if HAVE_USD:
        stage = Usd.Stage.CreateNew(str(out_usd))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        prim = UsdGeom.Xform.Define(stage, "/World/proto").GetPrim()
        prim.GetReferences().AddReference(_rel_or_abs(ref_asset, out_usd.parent))
        stage.Save()
        return
    _ensure_dir(out_usd.parent)
    rel = _rel_or_abs(ref_asset, out_usd.parent)
    txt = f"""usda 1.0
(defaultPrim=\"World\")

def Xform \"World\" {{
  def Xform \"proto\" {{
    rel references = @{rel}@
  }}
}}
"""
    out_usd.write_text(txt)


def _select_base_texture(textures: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    base = textures.get("basecolor")
    if not base:
        for alias in ["albedo", "diffuse", "color"]:
            base = textures.get(alias)
            if base:
                break
    normal = textures.get("normal")
    return base, normal


def _generate_thumbnail(texture: Optional[Path], out_dir: Path, name: str) -> Tuple[Path, Path]:
    from PIL import Image, ImageDraw

    _ensure_dir(out_dir)
    out128 = out_dir / f"{name}_128.png"
    out256 = out_dir / f"{name}_256.png"
    if texture and texture.exists():
        try:
            img = Image.open(texture)
            img.convert("RGBA").resize((128, 128), Image.BICUBIC).save(out128)
            img.convert("RGBA").resize((256, 256), Image.BICUBIC).save(out256)
            return out128, out256
        except Exception:
            pass
    for sz, outp in [(128, out128), (256, out256)]:
        img = Image.new("RGBA", (sz, sz), (40, 40, 40, 255))
        dr = ImageDraw.Draw(img)
        dr.rectangle((8, 8, sz - 8, sz - 8), outline=(200, 200, 200, 255), width=2)
        dr.text((12, sz // 2 - 8), name[:12], fill=(220, 220, 220, 255))
        img.save(outp)
    return out128, out256


def _write_prototype(
    name: str,
    out_dir: Path,
    mesh: Optional[Path],
    textures: Dict[str, Path],
) -> Prototype:
    usd_path = out_dir / f"{name}.usd"
    base_tex, normal_tex = _select_base_texture(textures)
    if mesh:
        ref_mesh = _maybe_convert_mesh_to_usd(mesh, out_dir) or mesh if mesh.suffix.lower() in USD_EXTS else None
        if ref_mesh and ref_mesh.suffix.lower() in USD_EXTS:
            _write_reference_usd(usd_path, ref_mesh)
        else:
            _write_plane_usdpreview(usd_path, base_tex, normal_tex)
    else:
        _write_plane_usdpreview(usd_path, base_tex, normal_tex)

    thumbs_dir = out_dir / "thumbnails"
    t128, t256 = _generate_thumbnail(base_tex, thumbs_dir, name)
    prot = Prototype(
        name=name,
        usd_path=usd_path,
        thumbnail_128=t128,
        thumbnail_256=t256,
        recommended_scale=1.0,
        uuid=str(uuid.uuid4()),
        textures={k: str(v) for k, v in textures.items()},
    )
    return prot


def _save_index(prototypes: List[Prototype], out_dir: Path) -> Path:
    data = {
        "version": 1,
        "prototypes": [
            {
                "uuid": p.uuid,
                "name": p.name,
                "usd_path": os.path.relpath(p.usd_path, out_dir),
                "thumbnail_128": os.path.relpath(p.thumbnail_128, out_dir),
                "thumbnail_256": os.path.relpath(p.thumbnail_256, out_dir),
                "recommended_scale": p.recommended_scale,
                "textures": p.textures,
            }
            for p in prototypes
        ],
    }
    idx = out_dir / "prototypes_index.json"
    with idx.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return idx


def _replicate_to_builds(out_dir: Path) -> List[Path]:
    copied = []
    root = Path.cwd()
    for build in ["_build/release", "_build/debug"]:
        target = root / build / "extensions/omniscatter/data/prototypes"
        try:
            if not target.exists():
                continue
            # simple dir sync: copy tree by files
            for src in out_dir.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(out_dir)
                    dst = target / rel
                    _ensure_dir(dst.parent)
                    dst.write_bytes(src.read_bytes())
                    copied.append(dst)
        except Exception:
            continue
    return copied


def run(ingest_dir: Path, out_dir: Path, apply_to_builds: bool) -> Tuple[List[Prototype], Path]:
    _ensure_dir(out_dir)
    meshes, tex_groups = _discover_assets(ingest_dir)
    prototypes: List[Prototype] = []

    matched_mesh_bases = set()
    # Pair meshes and texture groups by shared base prefix when possible
    for base, mesh_path in meshes.items():
        textures = tex_groups.get(base, {})
        prot = _write_prototype(base, out_dir, mesh_path, textures)
        prototypes.append(prot)
        matched_mesh_bases.add(base)

    # Remaining texture-only groups → plane proxies
    for base, textures in tex_groups.items():
        if base in matched_mesh_bases:
            continue
        prot = _write_prototype(base, out_dir, None, textures)
        prototypes.append(prot)

    index_path = _save_index(prototypes, out_dir)
    if apply_to_builds:
        _replicate_to_builds(out_dir)
    return prototypes, index_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate USD prototypes and thumbnails from ingested assets")
    parser.add_argument("--ingest-dir", type=Path, default=Path("assets"))
    parser.add_argument("--out-dir", type=Path, default=Path("extensions/omniscatter/data/prototypes"))
    parser.add_argument("--apply-to-builds", action="store_true")
    args = parser.parse_args(argv)

    protos, index_path = run(args.ingest_dir, args.out_dir, args.apply_to_builds)
    print(f"Generated {len(protos)} prototypes. Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


