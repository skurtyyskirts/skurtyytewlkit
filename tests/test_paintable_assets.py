from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.custom.prototype_ingest import run as ingest_run
from tools.custom.paintable_assets import list_paintable_assets, resolve_asset_prototype, HAVE_USD


@pytest.fixture()
def temp_protos(tmp_path: Path):
    ingest = tmp_path / "assets"
    out = tmp_path / "out"
    ingest.mkdir()

    # Create a tiny texture to drive thumbnail generation
    from PIL import Image

    img = Image.new("RGB", (4, 4), (180, 140, 90))
    tex = ingest / "brick_basecolor.png"
    img.save(tex)

    # Create a dummy usda mesh, with material binding as our generator writes
    mesh_usda = ingest / "brick.usda"
    mesh_usda.write_text(
        """
        #usda 1.0
        (
            defaultPrim = "World"
        )
        def Xform "World" {
            def Mesh "proto" {
                rel material:binding = </World/Materials/Mat>
                point3f[] points = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)]
            }
            def Material "Materials" {
                def Material "Mat" {
                    token outputs:surface.connect = "PBR.outputs:surface"
                    def Shader "PBR" {
                        uniform token info:id = "UsdPreviewSurface"
                        float inputs:roughness = 0.5
                        float inputs:metallic = 0.0
                    }
                }
            }
        }
        """,
        encoding="utf-8",
    )

    protos, index_path = ingest_run(ingest, out, apply_to_builds=False)
    # Point the resolver at this temp directory
    os.environ["OMNISCATTER_PROTOTYPES_DIR"] = str(index_path)

    yield out, index_path

    del os.environ["OMNISCATTER_PROTOTYPES_DIR"]


def test_list_paintable_assets_textual(temp_protos):
    out, index_path = temp_protos
    assets = list_paintable_assets()
    assert assets, "Expected at least one paintable asset"

    a = assets[0]
    assert a.display_name
    assert a.usd_path.endswith(".usd")
    assert a.thumbnail_path and Path(a.thumbnail_path).exists()
    assert a.bounding_radius > 0


def test_dds_only_requires_binding(tmp_path: Path):
    # Construct an index with only DDS textures and no material binding => should be filtered out
    root = tmp_path / "protos"
    root.mkdir()
    # Minimal usda without binding
    usd = root / "plane.usda"
    usd.write_text(
        """
        #usda 1.0
        (
            defaultPrim = "World"
        )
        def Xform "World" {
            def Mesh "proto" {
                point3f[] points = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)]
            }
        }
        """,
        encoding="utf-8",
    )
    idx = root / "prototypes_index.json"
    data = {
        "version": 1,
        "prototypes": [
            {
                "uuid": "only-dds",
                "name": "only-dds",
                "usd_path": os.path.relpath(usd, root),
                "thumbnail_128": "thumbnails/only-dds_128.png",
                "thumbnail_256": "thumbnails/only-dds_256.png",
                "recommended_scale": 1.0,
                "textures": {"basecolor": "tex/albedo.dds"},
            }
        ],
    }
    (root / "thumbnails").mkdir()
    (root / "thumbnails/only-dds_128.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "thumbnails/only-dds_256.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    idx.write_text(json.dumps(data), encoding="utf-8")

    os.environ["OMNISCATTER_PROTOTYPES_DIR"] = str(idx)
    try:
        assets = list_paintable_assets()
        assert not assets, "DDS-only entries without material binding must be excluded"
    finally:
        del os.environ["OMNISCATTER_PROTOTYPES_DIR"]


@pytest.mark.skipif(not HAVE_USD, reason="USD not available in test environment")
def test_resolve_asset_prototype(temp_protos):
    out, index_path = temp_protos
    assets = list_paintable_assets()
    target = assets[0]
    prim = resolve_asset_prototype(target.id)
    assert prim
    assert str(prim.GetPath()).startswith("/World/proto")