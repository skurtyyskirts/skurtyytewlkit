"""
Basic tests for prototype_ingest utility.
Run with: pytest -q
"""

from pathlib import Path
import json

from tools.custom.prototype_ingest import run


def test_ingest_texture_only(tmp_path: Path):
    ingest = tmp_path / "assets"
    out = tmp_path / "out"
    ingest.mkdir()
    # create dummy texture
    from PIL import Image

    img = Image.new("RGB", (4, 4), (128, 128, 128))
    tex = ingest / "brick_basecolor.png"
    img.save(tex)

    protos, index_path = run(ingest, out, apply_to_builds=False)
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert data["prototypes"][0]["usd_path"].endswith(".usd")
    assert (out / data["prototypes"][0]["usd_path"]).exists()
    # thumbnails
    thumb = out / data["prototypes"][0]["thumbnail_128"]
    assert thumb.exists()


