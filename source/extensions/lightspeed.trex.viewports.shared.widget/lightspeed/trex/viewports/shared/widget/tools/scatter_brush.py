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
    "ScatterBrushButtonGroup",
    "create_button_instance",
    "delete_button_instance",
    "scatter_brush_factory",
]

import asyncio
import time
from dataclasses import dataclass
import contextlib
from typing import TYPE_CHECKING, Callable, Optional

import carb
import carb.settings
import omni.kit.commands
import omni.kit.undo
import omni.ui as ui
import omni.usd
from lightspeed.trex.app.style import style
from lightspeed.trex.hotkeys import TrexHotkeyEvent
from lightspeed.trex.hotkeys import get_global_hotkey_manager as _get_global_hotkey_manager
from pxr import Gf, Sdf, Usd, UsdGeom
from omni.kit.widget.toolbar.widget_group import WidgetGroup

from .teleport import PointMousePicker  # reuse picker and screen->NDC mapping

if TYPE_CHECKING:
    from omni.kit.widget.viewport.api import ViewportAPI


@dataclass
class BrushSettings:
    radius: float = 0.0  # future use (jitter radius)
    min_interval_ms: int = 120  # place at most every N ms when dragging
    group_parent_name: str = "ScatterBrush"
    create_type: str = "Xform"  # default prim type when scattering
    per_frame_max_instances: int = 10
    enable_throttle: bool = True
    use_spatial_hash: bool = True
    min_distance: float = 0.0
    hash_cell_size: float = 0.25
    z_epsilon: float = 0.0005
    offset_along_view: bool = True


_scatter_button_group: ScatterBrushButtonGroup | None = None


class _ToggleModel(ui.AbstractValueModel):
    def __init__(self):
        super().__init__()
        self._value = False

    def get_value_as_bool(self):
        return bool(self._value)

    def set_value(self, v):
        new_val = bool(v)
        if new_val != self._value:
            self._value = new_val
            self._value_changed()


class ScatterBrushButtonGroup(WidgetGroup):
    """Scatter Brush toolbar button (toggle)."""

    name = "scatter_brush"

    def __init__(self):
        super().__init__()
        self._button: Optional[ui.ToolButton] = None
        self._model = _ToggleModel()
        self.__on_toggled: list[Callable[[bool], None]] = []

        # subscribe to model changes to propagate
        self._model.add_value_changed_fn(lambda *_: self._notify(self._model.get_value_as_bool()))

    def _notify(self, value: bool):
        for cb in list(self.__on_toggled):
            try:
                cb(value)
            except Exception:  # noqa
                carb.log_warn("ScatterBrushButtonGroup subscriber raised")

    def subscribe_toggled(self, callback: Callable[[bool], None]):
        self.__on_toggled.append(callback)
        return callback

    def unsubscribe_toggled(self, callback: Callable[[bool], None]):
        with contextlib.suppress(ValueError):
            self.__on_toggled.remove(callback)

    def get_style(self):
        # expects a style key Button.Image::scatter_brush (lowercase to match style key)
        return {f"Button.Image::{self.name}": style.default.get(f"Button.Image::{self.name}", {})}

    def _on_mouse_released(self, _button):
        # toggle state
        self._acquire_toolbar_context()
        if self._is_in_context() and self._button is not None and self._button.enabled:
            current = self._model.get_value_as_bool()
            self._model.set_value(not current)

    def create(self, default_size: ui.Length):
        self._button = ui.ToolButton(
            model=self._model,
            name=self.name,
            identifier=self.name,
            tooltip="Scatter Brush (paint to place objects under the mouse)",
            width=default_size,
            height=default_size,
            mouse_released_fn=lambda x, y, b, _: self._on_mouse_released(b),
        )
        return {self.name: self._button}

    def clean(self):
        super().clean()
        self._button = None
        self.__on_toggled = []


def create_button_instance():
    global _scatter_button_group
    _scatter_button_group = ScatterBrushButtonGroup()
    return _scatter_button_group


def delete_button_instance():
    global _scatter_button_group
    _scatter_button_group = None


class ScatterBrush:
    """Handle scattering while active; intended to be created via factory per viewport."""

    def __init__(self, viewport_api: "ViewportAPI"):
        self._viewport_api = viewport_api
        self._settings = BrushSettings()

        # Settings knobs (exposed via carb.settings)
        self._settings_iface = carb.settings.get_settings()
        self._PERF_MAX_PER_FRAME_KEY = "/app/viewport/scatter/per_frame_max_instances"
        self._THROTTLE_ENABLE_KEY = "/app/viewport/scatter/enable_throttle"
        self._MIN_INTERVAL_MS_KEY = "/app/viewport/scatter/min_interval_ms"
        self._MIN_DISTANCE_KEY = "/app/viewport/scatter/min_distance"
        self._USE_SPATIAL_HASH_KEY = "/app/viewport/scatter/use_spatial_hash"
        self._HASH_CELL_SIZE_KEY = "/app/viewport/scatter/hash_cell_size"
        self._Z_EPSILON_KEY = "/app/viewport/scatter/z_epsilon"
        self._OFFSET_ALONG_VIEW_KEY = "/app/viewport/scatter/offset_along_view"

        # Register defaults so users can tweak in Settings UI or config
        self._settings_iface.set_default(self._PERF_MAX_PER_FRAME_KEY, self._settings.per_frame_max_instances)
        self._settings_iface.set_default(self._THROTTLE_ENABLE_KEY, self._settings.enable_throttle)
        self._settings_iface.set_default(self._MIN_INTERVAL_MS_KEY, self._settings.min_interval_ms)
        self._settings_iface.set_default(self._MIN_DISTANCE_KEY, self._settings.min_distance)
        self._settings_iface.set_default(self._USE_SPATIAL_HASH_KEY, self._settings.use_spatial_hash)
        self._settings_iface.set_default(self._HASH_CELL_SIZE_KEY, self._settings.hash_cell_size)
        self._settings_iface.set_default(self._Z_EPSILON_KEY, self._settings.z_epsilon)
        self._settings_iface.set_default(self._OFFSET_ALONG_VIEW_KEY, self._settings.offset_along_view)

        # internal state
        self._stroke_active: bool = False
        self._queue: list[tuple[str, Gf.Vec3d]] = []  # list of (anchor_path, world_pos)
        self._spatial_hash: dict[tuple[int, int, int], list[Gf.Vec3d]] = {}
        self._last_lmb_down: bool = False
        self._current_anchor: Optional[str] = None

        # overlay frame (same pattern as Teleporter) for coordinate conversions if needed later
        self.__viewport_frame = ui.Frame()

        self._active: bool = False
        self._picker = PointMousePicker(self._viewport_api, self.__viewport_frame, self._on_pick)
        self._last_place_time_ms: int = 0
        self._pending_pick: bool = False

        # Register hotkey to toggle painting (reuse teleport hotkey? no, keep separate if available)
        # Not defining a new TrexHotkeyEvent; toolbar toggle is primary control.
        # Subscribe to toolbar toggle
        if _scatter_button_group:
            _scatter_button_group.subscribe_toggled(self.set_active)

        # Ensure selection of viewport (avoid painting when inactive viewport)
        self._hotkey_subscription = _get_global_hotkey_manager().subscribe_hotkey_event(
            TrexHotkeyEvent.F, lambda: None
        )

        # Background loop to sample mouse while active
        self._task: Optional[asyncio.Task] = asyncio.ensure_future(self._run_loop())

    def destroy(self):
        with contextlib.suppress(Exception):
            if self._task:
                self._task.cancel()

    # Required for compatibility with scene layer
    @property
    def visible(self):
        return True

    @property
    def categories(self):
        return ("tools",)

    @property
    def name(self):
        return "Scatter Brush"

    def set_active(self, value: bool):
        self._active = bool(value)

    async def _run_loop(self):
        # periodic sampler; when active and LMB is pressed, request a pick under mouse
        input_interface = carb.input.acquire_input_interface()
        import omni.appwindow as _appwindow

        app_window = _appwindow.get_default_app_window()
        mouse = app_window.get_mouse()
        while True:
            await asyncio.sleep(0.03)
            if not self._active:
                continue

            # Update stroke state transitions
            lmb_down = input_interface.get_mouse_value(mouse, carb.input.MouseInput.LEFT_BUTTON) > 0
            if lmb_down and not self._last_lmb_down:
                self._begin_stroke()
            if not lmb_down and self._last_lmb_down:
                # end stroke and flush remaining
                self._end_stroke()
            self._last_lmb_down = lmb_down

            # Only act if left mouse button is pressed
            if not lmb_down:
                continue
            # Avoid overlapping pick requests
            if self._pending_pick:
                continue
            self._pending_pick = True
            try:
                # Pick at current mouse position
                self._picker.pick()
            finally:
                # _on_pick will run async context via callback; small delay allows callback to schedule
                await asyncio.sleep(0)
                self._pending_pick = False

            # Try to flush a batch each loop to keep FPS smooth
            self._flush_queue()

    def _on_pick(self, path: str, position: carb.Double3 | None, _pixels: carb.Uint2):
        # place object if rate-limited and valid position
        if position is None:
            return
        now_ms = int(time.time() * 1000)
        min_interval_ms = int(self._settings_iface.get(self._MIN_INTERVAL_MS_KEY) or self._settings.min_interval_ms)
        enable_throttle = bool(self._settings_iface.get(self._THROTTLE_ENABLE_KEY) if self._settings_iface else True)
        if enable_throttle and (now_ms - self._last_place_time_ms < min_interval_ms):
            return
        self._last_place_time_ms = now_ms
        try:
            world_pos = Gf.Vec3d(position[0], position[1], position[2])
            # Queue for batched placement; track anchor path for mid-stroke switching
            anchor_path = path or ""
            # Reset batch on anchor change mid-stroke to avoid mixing groups
            if self._stroke_active and self._current_anchor and anchor_path and anchor_path != self._current_anchor:
                self._flush_queue(force=True)
            self._current_anchor = anchor_path or self._current_anchor
            # Apply optional z-fighting epsilon along view direction
            z_epsilon = float(self._settings_iface.get(self._Z_EPSILON_KEY) or self._settings.z_epsilon)
            if z_epsilon > 0.0 and bool(self._settings_iface.get(self._OFFSET_ALONG_VIEW_KEY) or True):
                world_pos = self._offset_along_view_direction(world_pos, z_epsilon)

            # Spatial hash rejection to avoid overdraw
            if self._should_accept_position(world_pos):
                self._queue.append((self._current_anchor or "", world_pos))
        except Exception:  # noqa
            carb.log_warn("Scatter brush placement failed")

    def _ensure_group_parent(self, stage: Usd.Stage) -> Usd.Prim:
        # try default prim, else /World
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            parent_path = default_prim.GetPath()
        else:
            parent_path = Sdf.Path("/World")
            if not stage.GetPrimAtPath(parent_path).IsValid():
                UsdGeom.Xform.Define(stage, parent_path)
        group_path = omni.usd.get_stage_next_free_path(
            stage, str(parent_path.AppendPath(self._settings.group_parent_name)), False
        )
        prim = stage.GetPrimAtPath(group_path)
        if not prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand",
                prim_path=group_path,
                prim_type="Xform",
                select_new_prim=False,
                context_name=self._viewport_api.usd_context_name,
            )
            prim = stage.GetPrimAtPath(group_path)
        return prim

    def _ensure_group_parent_for_anchor(self, stage: Usd.Stage, anchor_path: str | None) -> Usd.Prim:
        if stage is None:
            return None  # headless path for perf tests
        if not anchor_path:
            return self._ensure_group_parent(stage)
        anchor_prim = stage.GetPrimAtPath(anchor_path)
        if not anchor_prim or not anchor_prim.IsValid():
            return self._ensure_group_parent(stage)
        parent_path = anchor_prim.GetPath()
        group_path = omni.usd.get_stage_next_free_path(
            stage, str(parent_path.AppendPath(self._settings.group_parent_name)), False
        )
        prim = stage.GetPrimAtPath(group_path)
        if not prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand",
                prim_path=group_path,
                prim_type="Xform",
                select_new_prim=False,
                context_name=self._viewport_api.usd_context_name,
            )
            prim = stage.GetPrimAtPath(group_path)
        return prim

    def _flush_queue(self, force: bool = False):
        if not self._queue:
            return
        stage = getattr(self._viewport_api, "stage", None)
        max_per_frame = int(self._settings_iface.get(self._PERF_MAX_PER_FRAME_KEY) or self._settings.per_frame_max_instances)
        budget = len(self._queue) if force else max(1, max_per_frame)
        to_process = min(budget, len(self._queue))
        if to_process <= 0:
            return
        # Headless fast-path: no stage available; simulate by hashing only
        if stage is None:
            for _ in range(to_process):
                _anchor_path, world_pos = self._queue.pop(0)
                self._spatial_hash_insert(world_pos)
            return
        with omni.kit.undo.group():
            for _ in range(to_process):
                anchor_path, world_pos = self._queue.pop(0)
                parent_prim = self._ensure_group_parent_for_anchor(stage, anchor_path)
                parent_to_world = (
                    UsdGeom.Xformable(parent_prim).ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()
                )
                local_translation = parent_to_world.Transform(world_pos)

                # Create child prim under parent and set local translation
                child_path = omni.usd.get_stage_next_free_path(
                    stage, str(parent_prim.GetPath().AppendPath("item")), False
                )
                omni.kit.commands.execute(
                    "CreatePrimCommand",
                    prim_path=child_path,
                    prim_type=self._settings.create_type,
                    select_new_prim=False,
                    context_name=self._viewport_api.usd_context_name,
                )
                omni.kit.commands.execute(
                    "TransformPrimSRT",
                    path=child_path,
                    new_translation=Gf.Vec3d(*local_translation),
                    usd_context_name=self._viewport_api.usd_context_name,
                )
                # update spatial hash with placed position (world space)
                self._spatial_hash_insert(world_pos)

    def _begin_stroke(self):
        self._stroke_active = True
        self._queue.clear()
        self._spatial_hash.clear()
        self._current_anchor = None

    def _end_stroke(self):
        # flush everything left
        self._flush_queue(force=True)
        self._stroke_active = False
        self._queue.clear()
        self._spatial_hash.clear()
        self._current_anchor = None

    # Spatial hash helpers
    def _cell_key(self, pos: Gf.Vec3d) -> tuple[int, int, int]:
        cell = float(self._settings_iface.get(self._HASH_CELL_SIZE_KEY) or self._settings.hash_cell_size)
        if cell <= 0:
            cell = 0.25
        return (int(pos[0] // cell), int(pos[1] // cell), int(pos[2] // cell))

    def _spatial_hash_insert(self, pos: Gf.Vec3d):
        if not bool(self._settings_iface.get(self._USE_SPATIAL_HASH_KEY) or self._settings.use_spatial_hash):
            return
        key = self._cell_key(pos)
        bucket = self._spatial_hash.get(key)
        if bucket is None:
            self._spatial_hash[key] = [pos]
        else:
            bucket.append(pos)

    def _should_accept_position(self, pos: Gf.Vec3d) -> bool:
        min_dist = float(self._settings_iface.get(self._MIN_DISTANCE_KEY) or self._settings.min_distance)
        use_hash = bool(self._settings_iface.get(self._USE_SPATIAL_HASH_KEY) or self._settings.use_spatial_hash)
        if not use_hash or min_dist <= 0.0:
            return True
        # Check neighboring cells for collisions
        cell = self._cell_key(pos)
        min_dist2 = min_dist * min_dist
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = self._spatial_hash.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
                    if not bucket:
                        continue
                    for other in bucket:
                        d2 = (pos[0] - other[0]) ** 2 + (pos[1] - other[1]) ** 2 + (pos[2] - other[2]) ** 2
                        if d2 < min_dist2:
                            return False
        return True

    def _offset_along_view_direction(self, world_pos: Gf.Vec3d, epsilon: float) -> Gf.Vec3d:
        cam_path = self._viewport_api.camera_path
        if not cam_path:
            return world_pos
        cam_prim = UsdGeom.Camera(self._viewport_api.stage.GetPrimAtPath(cam_path))
        if not cam_prim:
            return world_pos
        cam_xf = cam_prim.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # forward in camera space is -Z
        forward = cam_xf.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        try:
            fwd_norm = forward.GetNormalized()
        except Exception:
            fwd_norm = forward
        return Gf.Vec3d(world_pos[0] + fwd_norm[0] * epsilon, world_pos[1] + fwd_norm[1] * epsilon, world_pos[2] + fwd_norm[2] * epsilon)


def scatter_brush_factory(desc: dict[str, object]):
    viewport_api = desc.get("viewport_api")
    return ScatterBrush(viewport_api)
