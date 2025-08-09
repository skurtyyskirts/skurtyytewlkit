"""
Author: Omniscatter Toolkit Agent (GPT-5)
Date: 2025-08-09
License: Apache-2.0

Thumbnail generator utility.

Given a USD prototype or a texture, create 128x128 and 256x256 PNG thumbnails.
This script prefers simple image resampling (Pillow) as a portable fallback.
If run inside an Omniverse Kit Python with offscreen rendering available, this
script may be extended to render actual USD previews (pseudocode noted).

CLI examples:
  python tools/custom/thumbnail_generator.py --src path/to/texture.dds --out-dir extensions/omniscatter/data/prototypes/thumbnails
  python tools/custom/thumbnail_generator.py --src path/to/prototype.usd --out-dir ...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _write_simple_thumbs(image_path: Path, out_dir: Path, name: str) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    img.convert("RGBA").resize((128, 128), Image.BICUBIC).save(out_dir / f"{name}_128.png")
    img.convert("RGBA").resize((256, 256), Image.BICUBIC).save(out_dir / f"{name}_256.png")


def _render_usd_preview_pseudocode(usdp: Path, out_dir: Path, name: str) -> None:
    # Pseudocode: verify and replace with actual omni.kit.viewport/renderer offscreen calls
    # References to verify:
    # - omni.kit.viewport.utility or omni.kit.viewport_legacy for offscreen rendering
    # - carb.settings to force headless GL context
    # - omni.usd.get_context().open_stage(str(usdp))
    # - viewport.capture_next_frame()
    # For now, we no-op and expect callers to provide a texture path version.
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate prototype thumbnails")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    src = args.src
    name = src.stem
    if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".tga", ".dds"}:
        _write_simple_thumbs(src, args.out_dir, name)
    else:
        _render_usd_preview_pseudocode(src, args.out_dir, name)
    print(f"Thumbnails written for {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


