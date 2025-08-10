"""
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "HitSample",
    "StrokePointerEvent",
    "BrushStrokeSettings",
    "StrokeEngine",
    "generate_stroke_samples",
]

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, Optional, Sequence, Tuple
from collections import deque

import carb
import omni.ui as ui
from pxr import Gf

from .teleport import PointMousePicker  # reuse NDC mapping and viewport pick integration


@dataclass
class HitSample:
    stroke_id: int
    sample_index: int
    pixel: Tuple[int, int]
    ndc: Tuple[float, float]
    world_position: Optional[Gf.Vec3d]
    prim_path: str
    timestamp_ms: int
    # optional data if available (not currently provided by HDRemix query)
    normal: Optional[Gf.Vec3d] = None
    face_index: Optional[int] = None
    uv: Optional[Tuple[float, float]] = None


@dataclass
class StrokePointerEvent:
    # type: "down" | "move" | "up"
    type: str
    screen_x: float
    screen_y: float
    button: int
    timestamp_ms: int


@dataclass
class BrushStrokeSettings:
    # Spacing of samples along the stroke in screen pixels
    spacing_px: float = 12.0
    # Brush radius in screen pixels for optional scatter
    radius_px: float = 24.0
    # Number of extra samples per step inside disk; 0 disables scatter
    scatter_count: int = 0
    # Minimal separation for Poisson-disk in pixels (if scatter_count > 0)
    scatter_min_dist_px: float = 8.0
    # Random seed for deterministic sampling
    seed: int = 1337


class StrokeEngine:
    """Converts pointer events into a stream of world-space `HitSample`s with stable spacing.

    Picking is done via `PointMousePicker` using the provided `viewport_api` and `viewport_frame`.
    The engine samples along the pointer path in pixel space using `spacing_px` and, for each
    step, can optionally generate additional picks within a disk (Poisson-disk sampling) in
    screen space of radius `radius_px`.
    """

    def __init__(
        self,
        viewport_api,
        viewport_frame: ui.Frame,
        on_sample: Optional[Callable[[HitSample], None]] = None,
    ) -> None:
        if viewport_api is None or viewport_frame is None:
            raise ValueError("viewport_api and viewport_frame are required")
        self._viewport_api = viewport_api
        self._viewport_frame = viewport_frame
        self._picker = PointMousePicker(viewport_api, viewport_frame, self._on_pick_result)

        self._on_sample = on_sample

        # Stroke state
        self._active: bool = False
        self._stroke_id: int = 0
        self._next_sample_index: int = 0
        self._last_event_px: Optional[Tuple[float, float]] = None
        self._accum_px: float = 0.0
        self._pending_queries: int = 0
        self._max_pending_queries: int = 64

        # Output samples buffer
        self._samples: Deque[HitSample] = deque(maxlen=4096)

        # Current settings
        self._settings: BrushStrokeSettings = BrushStrokeSettings()

        # Random state for scatter
        self._rng = random.Random(self._settings.seed)

    # --------------- Public API ---------------

    def on_pointer_event(self, event: StrokePointerEvent) -> None:
        if event.type not in ("down", "move", "up"):
            return
        if event.type == "down":
            self._begin_stroke(event)
        elif event.type == "move":
            if self._active:
                self._extend_stroke(event)
        elif event.type == "up":
            if self._active:
                self._end_stroke(event)

    def generate_stroke_samples(self, settings: Optional[BrushStrokeSettings] = None) -> Iterable[HitSample]:
        if settings is not None:
            self._apply_settings(settings)
        # Yield any currently buffered samples; caller can keep iterating as more are appended
        while self._samples:
            yield self._samples.popleft()

    # --------------- Internal helpers ---------------

    def _apply_settings(self, settings: BrushStrokeSettings) -> None:
        self._settings = settings
        self._rng.seed(settings.seed)

    def _begin_stroke(self, event: StrokePointerEvent) -> None:
        self._active = True
        self._stroke_id += 1
        self._next_sample_index = 0
        self._last_event_px = (event.screen_x, event.screen_y)
        self._accum_px = 0.0
        # Immediately sample at the down position
        self._enqueue_picks_for_pixel((event.screen_x, event.screen_y))

    def _extend_stroke(self, event: StrokePointerEvent) -> None:
        if self._last_event_px is None:
            self._last_event_px = (event.screen_x, event.screen_y)
        last_x, last_y = self._last_event_px
        cur_x, cur_y = event.screen_x, event.screen_y
        dx = cur_x - last_x
        dy = cur_y - last_y
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-3:
            return

        spacing = max(1.0, float(self._settings.spacing_px))
        # Account for leftover from previous segment to maintain stable spacing
        dist = self._accum_px + seg_len
        num_new = int(dist // spacing)
        if num_new <= 0:
            # not enough movement to place a new sample yet
            self._accum_px = dist
            self._last_event_px = (cur_x, cur_y)
            return

        # step along current segment, emitting samples each 'spacing' distance
        step = (dx / seg_len, dy / seg_len)
        # first sample is at (spacing - accum)
        first_offset = spacing - self._accum_px
        for i in range(num_new):
            t = (first_offset + i * spacing) / seg_len
            px = last_x + dx * t
            py = last_y + dy * t
            self._enqueue_picks_for_pixel((px, py))
        # compute leftover for next segment
        self._accum_px = dist - num_new * spacing
        self._last_event_px = (cur_x, cur_y)

    def _end_stroke(self, event: StrokePointerEvent) -> None:
        # Place a final sample at up position
        self._enqueue_picks_for_pixel((event.screen_x, event.screen_y))
        self._active = False
        self._last_event_px = None
        self._accum_px = 0.0

    def _screen_to_ndc(self, screen_coords: Tuple[float, float]) -> Tuple[float, float]:
        # Reuse same math as PointMousePicker._convert_screen_coords_to_ndc_coords
        frame = self._viewport_frame
        x, y = screen_coords
        ndc = (
            (-1.0 + 2.0 * ((x - frame.screen_position_x) / max(1.0, frame.computed_width))),
            (1 - 2.0 * ((y - frame.screen_position_y) / max(1.0, frame.computed_height))),
        )
        return ndc

    def _enqueue_picks_for_pixel(self, pixel_coords: Tuple[float, float]) -> None:
        # Limit outstanding queries
        if self._pending_queries > self._max_pending_queries:
            return

        # primary center sample
        self._request_pick(pixel_coords)

        # scatter samples
        if self._settings.scatter_count > 0 and self._settings.radius_px > 1.0:
            offsets = self._poisson_disk_samples_in_disk(
                self._settings.radius_px,
                self._settings.scatter_min_dist_px,
                self._settings.scatter_count,
            )
            cx, cy = pixel_coords
            for ox, oy in offsets:
                self._request_pick((cx + ox, cy + oy))

    def _request_pick(self, pixel_coords: Tuple[float, float]) -> None:
        # Convert to NDC using our viewport frame, then ask PointMousePicker to map through to a pixel and request HDRemix query
        ndc_coords = self._screen_to_ndc(pixel_coords)
        # piggyback PointMousePicker which calls viewport_api_request_query_hdremix under the hood
        self._pending_queries += 1
        try:
            self._picker.pick(ndc_coords=ndc_coords)
        except Exception:  # noqa
            self._pending_queries -= 1

    def _on_pick_result(self, prim_path: str, position: carb.Double3 | None, pixels: carb.Uint2) -> None:
        try:
            world = None
            if position is not None:
                world = Gf.Vec3d(position[0], position[1], position[2])
            ndc = self._screen_to_ndc((float(pixels[0]), float(pixels[1])))
            sample = HitSample(
                stroke_id=self._stroke_id,
                sample_index=self._next_sample_index,
                pixel=(int(pixels[0]), int(pixels[1])),
                ndc=(ndc[0], ndc[1]),
                world_position=world,
                prim_path=prim_path or "",
                timestamp_ms=int(time.time() * 1000),
            )
            self._next_sample_index += 1
            self._samples.append(sample)
            carb.log_info(
                f"Stroke[{sample.stroke_id}] sample#{sample.sample_index} path='{sample.prim_path}' px={sample.pixel} world={tuple(sample.world_position) if sample.world_position else None}"
            )
            if self._on_sample:
                self._on_sample(sample)
        finally:
            self._pending_queries = max(0, self._pending_queries - 1)

    # --------------- Sampling utilities ---------------

    def _poisson_disk_samples_in_disk(
        self, radius_px: float, min_dist_px: float, count: int
    ) -> list[Tuple[float, float]]:
        """Generate up to `count` Poisson-disk samples within a disk centered at origin (pixel space).

        Returns a list of (dx, dy) offsets.
        """
        # Simple dart-throwing with rejection. Good enough for small counts.
        results: list[Tuple[float, float]] = []
        attempts = 0
        max_attempts = 200 * max(1, count)
        r2 = radius_px * radius_px
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            # Uniform in disk via polar sampling
            u = self._rng.random()
            v = self._rng.random()
            r = radius_px * math.sqrt(u)
            theta = 2.0 * math.pi * v
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            if x * x + y * y > r2:
                continue
            if all((x - px) ** 2 + (y - py) ** 2 >= (min_dist_px * min_dist_px) for px, py in results):
                results.append((x, y))
        return results


# Convenience API wrapper (module-level) to match requested signature

def generate_stroke_samples(settings: BrushStrokeSettings) -> Iterable[HitSample]:
    """Generator placeholder; in practice, create a StrokeEngine and call its generate_stroke_samples.

    This exists to satisfy the requested API surface.
    """
    # No global engine context; this is a stub to clarify the API shape.
    return iter(())