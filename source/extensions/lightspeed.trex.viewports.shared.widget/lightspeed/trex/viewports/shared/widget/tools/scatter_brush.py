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
import omni.kit.commands
import omni.kit.undo
import omni.ui as ui
from omni.ui import scene as sc
import omni.ui.color as cl
import omni.usd
from lightspeed.trex.app.style import style
from lightspeed.trex.hotkeys import TrexHotkeyEvent
from lightspeed.trex.hotkeys import get_global_hotkey_manager as _get_global_hotkey_manager
from pxr import Gf, Sdf, Usd, UsdGeom
from omni.kit.widget.toolbar.widget_group import WidgetGroup

from .teleport import PointMousePicker  # reuse picker and screen->NDC mapping
from .stroke_engine import BrushStrokeSettings as _BrushStrokeSettings
from .stroke_engine import HitSample as _HitSample
from .stroke_engine import StrokeEngine as _StrokeEngine
from .stroke_engine import StrokePointerEvent as _StrokePointerEvent

if TYPE_CHECKING:
    from omni.kit.widget.viewport.api import ViewportAPI


@dataclass
class BrushSettings:
    radius: float = 0.0  # future use (jitter radius)
    min_interval_ms: int = 120  # place at most every N ms when dragging
    group_parent_name: str = "ScatterBrush"
    create_type: str = "Xform"  # default prim type when scattering


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

        # overlay frame (same pattern as Teleporter) for coordinate conversions and overlay drawing
        self.__viewport_frame = ui.Frame()

        self._active: bool = False
        # Stroke engine converts pointer into samples
        self._stroke_engine = _StrokeEngine(self._viewport_api, self.__viewport_frame, on_sample=self._on_sample)
        # Debug: keep a short history of samples for drawing
        self._recent_samples: list[_HitSample] = []
        # Scene overlay root for debug drawing
        self._overlay_root = sc.Transform()

        # Register hotkey to toggle painting (reuse teleport hotkey? no, keep separate if available)
        # Not defining a new TrexHotkeyEvent; toolbar toggle is primary control.
        # Subscribe to toolbar toggle
        if _scatter_button_group:
            _scatter_button_group.subscribe_toggled(self.set_active)

        # Ensure selection of viewport (avoid painting when inactive viewport)
        self._hotkey_subscription = _get_global_hotkey_manager().subscribe_hotkey_event(
            TrexHotkeyEvent.F, lambda: None
        )

        # Background loop to read mouse and feed stroke engine
        self._task: Optional[asyncio.Task] = asyncio.ensure_future(self._run_loop())

    def destroy(self):
        with contextlib.suppress(Exception):
            if self._task:
                self._task.cancel()
        self._recent_samples.clear()
        if self._overlay_root:
            self._overlay_root.clear()
            self._overlay_root = None

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
        # periodic reader; translates mouse state to pointer events for stroke engine
        input_interface = carb.input.acquire_input_interface()
        import omni.appwindow as _appwindow

        app_window = _appwindow.get_default_app_window()
        mouse = app_window.get_mouse()
        was_down = False
        while True:
            await asyncio.sleep(0.03)
            if not self._active:
                continue
            # Mouse state
            lmb = input_interface.get_mouse_value(mouse, carb.input.MouseInput.LEFT_BUTTON) > 0
            pos_x, pos_y = input_interface.get_mouse_coords_pixel(mouse)
            dpi_scale = ui.Workspace.get_dpi_scale()
            sx, sy = pos_x / dpi_scale, pos_y / dpi_scale
            now_ms = int(time.time() * 1000)
            if lmb and not was_down:
                self._stroke_engine.on_pointer_event(_StrokePointerEvent("down", sx, sy, int(carb.input.MouseInput.LEFT_BUTTON), now_ms))
            elif lmb and was_down:
                self._stroke_engine.on_pointer_event(_StrokePointerEvent("move", sx, sy, int(carb.input.MouseInput.LEFT_BUTTON), now_ms))
            elif (not lmb) and was_down:
                self._stroke_engine.on_pointer_event(_StrokePointerEvent("up", sx, sy, int(carb.input.MouseInput.LEFT_BUTTON), now_ms))
            was_down = lmb

    def _on_sample(self, sample: _HitSample):
        # Debug: capture for overlay; perform placement on valid hits
        self._recent_samples.append(sample)
        if len(self._recent_samples) > 64:
            self._recent_samples = self._recent_samples[-64:]
        self._refresh_overlay()
        if sample.world_position is None:
            return
        try:
            self._place_at_world_position(sample.world_position)
        except Exception:
            carb.log_warn("Scatter brush placement failed")

    def _refresh_overlay(self):
        if not self._overlay_root:
            return
        # Rebuild simple crosses at recent sample positions
        self._overlay_root.clear()
        size = 0.15
        with self._overlay_root:
            for s in self._recent_samples[-32:]:
                if not s.world_position:
                    continue
                p = s.world_position
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                sc.Line((x - size, y, z), (x + size, y, z), color=cl.yellow, thickness=2.0)
                sc.Line((x, y - size, z), (x, y + size, z), color=cl.yellow, thickness=2.0)
                sc.Line((x, y, z - size), (x, y, z + size), color=cl.yellow, thickness=2.0)

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

    def _place_at_world_position(self, world_pos: Gf.Vec3d):
        stage = self._viewport_api.stage
        with omni.kit.undo.group():
            parent_prim = self._ensure_group_parent(stage)
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


def scatter_brush_factory(desc: dict[str, object]):
    viewport_api = desc.get("viewport_api")
    return ScatterBrush(viewport_api)
