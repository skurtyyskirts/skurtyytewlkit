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


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

import time
import importlib.util
import pytest

KIT_AVAILABLE = importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None

if KIT_AVAILABLE:
    from pxr import Gf
    from lightspeed.trex.viewports.shared.widget.tools.scatter_brush import ScatterBrush, BrushSettings

    class _MockViewportAPI:
        def __init__(self):
            self.stage = None
            self.usd_context_name = ""
            self.camera_path = None

    def _make_brush():
        vp = _MockViewportAPI()
        brush = ScatterBrush(vp)
        return brush

    def test_spatial_hash_acceptance_micro_benchmark():
        brush = _make_brush()
        # enable spatial hash and min distance
        brush._settings.min_distance = 0.5
        start = time.time()
        accepted = 0
        for i in range(1000):
            pos = Gf.Vec3d(float(i) * 0.1, 0.0, 0.0)
            if brush._should_accept_position(pos):
                brush._spatial_hash_insert(pos)
                accepted += 1
        elapsed_ms = (time.time() - start) * 1000.0
        # sanity: should be fast and accept fewer than all due to min distance
        assert elapsed_ms < 50.0
        assert accepted < 1000

    def test_batch_queue_flushing_micro_benchmark():
        brush = _make_brush()
        # Pre-populate a queue
        for i in range(500):
            brush._queue.append(("", Gf.Vec3d(float(i), 0.0, 0.0)))
        start = time.time()
        brush._flush_queue(force=True)
        elapsed_ms = (time.time() - start) * 1000.0
        # Should be quick even when flushing (mocked viewport/stage avoids heavy USD ops)
        assert elapsed_ms < 30.0
else:
    pytest.skip("Omniverse Kit/pxr not available; skipping scatter brush micro-benchmarks", allow_module_level=True)


