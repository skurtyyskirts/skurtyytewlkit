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
    "scatter_brush_factory",
    "ScatterBrushButtonGroup",
    "create_button_instance",
    "delete_button_instance",
]

import math
import random
import time
from typing import Any, Callable, Optional

import carb
import carb.settings
import omni.kit.commands
import omni.kit.undo
import omni.ui as ui
import omni.usd
import omni.client
from lightspeed.hydra.remix.core import RemixRequestQueryType, viewport_api_request_query_hdremix
from lightspeed.trex.app.style import style
from omni.flux.utils.widget.file_pickers import open_file_picker as _open_file_picker
from omni.kit.notification_manager import NotificationStatus, post_notification
from omni.kit.widget.toolbar.widget_group import WidgetGroup
from pxr import Gf, Sdf, Usd, UsdGeom

# Reuse the point picker utilities from the Teleport tool for robust coordinate mapping
from .teleport import PointMousePicker

# Settings path for persistence
_SETTINGS_PREFIX = "/exts/lightspeed.trex.scatter_brush"

# Global singleton toolbar instance
_scatter_button_group: ScatterBrushButtonGroup | None = None


class _Settings:
    def __init__(self):
        self._s = carb.settings.get_settings()

    def _get(self, key: str, default):
        value = self._s.get(f"{_SETTINGS_PREFIX}/{key}")
        return default if value is None else value

    def _set(self, key: str, value):
        self._s.set(f"{_SETTINGS_PREFIX}/{key}", value)

    @property
    def asset_path(self) -> str:
        return self._get("assetPath", "")

    @asset_path.setter
    def asset_path(self, value: str):
        self._set("assetPath", value or "")

    @property
    def radius(self) -> float:
        return float(self._get("radius", 0.5))

    @radius.setter
    def radius(self, value: float):
        self._set("radius", float(max(0.01, value)))

    @property
    def spacing(self) -> float:
        return float(self._get("spacing", 0.25))

    @spacing.setter
    def spacing(self, value: float):
        self._set("spacing", float(max(0.01, value)))

    @property
    def count_per_stamp(self) -> int:
        return int(self._get("countPerStamp", 4))

    @count_per_stamp.setter
    def count_per_stamp(self, value: int):
        self._set("countPerStamp", int(max(1, value)))

    @property
    def scale_min(self) -> float:
        return float(self._get("scaleMin", 1.0))

    @scale_min.setter
    def scale_min(self, value: float):
        self._set("scaleMin", float(max(0.001, value)))

    @property
    def scale_max(self) -> float:
        return float(self._get("scaleMax", 1.0))

    @scale_max.setter
    def scale_max(self, value: float):
        self._set("scaleMax", float(max(0.001, value)))

    @property
    def yaw_jitter_deg(self) -> float:
        return float(self._get("yawJitterDeg", 180.0))

    @yaw_jitter_deg.setter
    def yaw_jitter_deg(self, value: float):
        self._set("yawJitterDeg", float(max(0.0, value)))

    @property
    def world_up_axis(self) -> str:
        # Default to Y up to match Omniverse conventions
        return str(self._get("worldUpAxis", "Y")).upper()

    @world_up_axis.setter
    def world_up_axis(self, value: str):
        v = str(value).upper()
        if v not in ("X", "Y", "Z"):
            v = "Y"
        self._set("worldUpAxis", v)


class _BrushSettingsWindow:
    _window: ui.Window | None = None

    @classmethod
    def show(cls, settings: _Settings):
        if cls._window and not cls._window.visible:
            cls._window = None
        if cls._window is None:
            cls._window = ui.Window("Scatter Brush", width=360, height=260)
        with cls._window.frame:
            cls._window.frame.clear()
            with ui.VStack():
                with ui.HStack(height=0):
                    ui.Label("Asset USD:", width=ui.Pixel(90))
                    asset_field = ui.StringField()
                    asset_field.model.set_value(settings.asset_path)
                    def _browse():
                        _open_file_picker(
                            "Choose USD to scatter",
                            lambda v: asset_field.model.set_value(v or ""),
                            default_dir="${data}",
                            exts=["*.usd", "*.usda", "*.usdc"],
                        )
                    ui.Button("Browse", width=ui.Pixel(80), mouse_pressed_fn=lambda *_: _browse())
                def _apply_asset():
                    settings.asset_path = asset_field.model.get_value_as_string()
                ui.Button("Apply Asset", mouse_pressed_fn=lambda *_: _apply_asset())
                ui.Spacer(height=ui.Pixel(6))
                def _row(label: str, get_val: Callable[[], float], set_val: Callable[[float], None], step=0.1):
                    with ui.HStack(height=0):
                        ui.Label(label, width=ui.Pixel(90))
                        field = ui.FloatField()
                        field.model.set_value(float(get_val()))
                        def _apply():
                            try:
                                set_val(float(field.model.get_value_as_float()))
                            except Exception:
                                pass
                        ui.Button("Set", width=ui.Pixel(60), mouse_pressed_fn=lambda *_: _apply())
                _row("Radius (m)", settings.radius.__float__, lambda v: setattr(settings, "radius", v))
                _row("Spacing (m)", settings.spacing.__float__, lambda v: setattr(settings, "spacing", v))
                _row("Count/Stamp", lambda: float(settings.count_per_stamp), lambda v: setattr(settings, "count_per_stamp", int(v)), step=1.0)
                _row("Scale Min", settings.scale_min.__float__, lambda v: setattr(settings, "scale_min", v))
                _row("Scale Max", settings.scale_max.__float__, lambda v: setattr(settings, "scale_max", v))
                _row("Yaw Jitter", settings.yaw_jitter_deg.__float__, lambda v: setattr(settings, "yaw_jitter_deg", v))
                with ui.HStack(height=0):
                    ui.Label("Up Axis:", width=ui.Pixel(90))
                    up_combo = ui.ComboBox(0, "X", "Y", "Z")
                    idx = {"X": 0, "Y": 1, "Z": 2}.get(settings.world_up_axis, 1)
                    up_combo.model.set_value(idx)
                    def _apply_up():
                        val = ["X", "Y", "Z"][int(up_combo.model.get_value_as_int())]
                        settings.world_up_axis = val
                    ui.Button("Set", width=ui.Pixel(60), mouse_pressed_fn=lambda *_: _apply_up())


class _ScatterStamp:
    def __init__(self, position: Gf.Vec3d):
        self.position = position


class ScatterBrush:
    """Per-viewport controller that handles painting when the tool is active."""

    def __init__(self, viewport_api: "ViewportAPI"):
        self._viewport_api = viewport_api
        self._settings = _Settings()
        self._overlay = ui.Frame()
        self._overlay.opaque_for_mouse_events = False
        self._picker = PointMousePicker(self._viewport_api, self._overlay, self._on_pick)
        self._is_painting = False
        self._last_stamp_worldpos: Optional[Gf.Vec3d] = None
        self._undo_open = False
        # Subscribe to toolbar button state
        if _scatter_button_group:
            self._toggle_sub = _scatter_button_group.subscribe_toggled(self._on_toggle)
        else:
            self._toggle_sub = None

    def _on_toggle(self, enabled: bool):
        # When enabled, capture mouse on the overlay frame
        self._overlay.opaque_for_mouse_events = enabled
        if enabled:
            self._overlay.set_mouse_pressed_fn(self._on_mouse_pressed)
            self._overlay.set_mouse_moved_fn(self._on_mouse_moved)
            self._overlay.set_mouse_released_fn(self._on_mouse_released)
        else:
            self._overlay.set_mouse_pressed_fn(None)
            self._overlay.set_mouse_moved_fn(None)
            self._overlay.set_mouse_released_fn(None)
            self._is_painting = False
            self._last_stamp_worldpos = None

    def _on_mouse_pressed(self, x: float, y: float, button: int, _mods: int):
        if button != 0:  # left click only
            return
        if not self._validate_asset():
            return
        self._is_painting = True
        self._last_stamp_worldpos = None
        self._ensure_undo_group()
        # Initial pick
        self._picker.pick()

    def _on_mouse_moved(self, x: float, y: float, _mods: int):
        if not self._is_painting:
            return
        self._picker.pick()

    def _on_mouse_released(self, x: float, y: float, button: int, _mods: int):
        if button != 0:
            return
        self._is_painting = False
        self._last_stamp_worldpos = None
        self._close_undo_group()

    def _ensure_undo_group(self):
        if not self._undo_open:
            # Begin a stroke group
            self._undo_open = True
            omni.kit.undo.begin_group("Scatter Brush Stroke")

    def _close_undo_group(self):
        if self._undo_open:
            try:
                omni.kit.undo.end_group()
            except Exception:
                pass
            self._undo_open = False

    def _on_pick(self, _path: str, position: carb.Double3 | None, _pixel: carb.Uint2):
        if not self._is_painting or position is None:
            return
        worldpos = Gf.Vec3d(position[0], position[1], position[2])
        # Respect spacing between consecutive stamps
        spacing = max(0.01, self._settings.spacing)
        if self._last_stamp_worldpos is None or (worldpos - self._last_stamp_worldpos).GetLength() >= spacing:
            self._stamp(worldpos)
            self._last_stamp_worldpos = worldpos

    def _validate_asset(self) -> bool:
        if not self._settings.asset_path:
            post_notification("Scatter Brush: Set Asset USD first (right-click the brush icon)", status=NotificationStatus.WARNING)
            return False
        return True

    def _stamp(self, center_worldpos: Gf.Vec3d):
        stage = self._viewport_api.stage
        if not stage:
            return
        parent_path = self._ensure_scatter_group(stage)
        count = max(1, int(self._settings.count_per_stamp))
        radius = max(0.0, float(self._settings.radius))
        # Simple uniform disk sampling
        for _ in range(count):
            dx, dy = _sample_uniform_disk(radius)
            offset = _axis_to_vec(self._settings.world_up_axis).Orthogonalize(Gf.Vec3d(dx, 0.0, dy))
            # Build position by offsetting in plane perpendicular to up-axis
            pos = _offset_in_plane(center_worldpos, self._settings.world_up_axis, dx, dy)
            scale = _rand_float(self._settings.scale_min, self._settings.scale_max)
            yaw_deg = (random.random() * 2.0 - 1.0) * float(self._settings.yaw_jitter_deg)
            self._create_instance(stage, parent_path, pos, scale, yaw_deg)

    def _ensure_scatter_group(self, stage: Usd.Stage) -> Sdf.Path:
        # Create a stable parent if not exists: /World/ScatterBrush
        root = stage.GetPrimAtPath("/World")
        if not root:
            UsdGeom.Xform.Define(stage, "/World")
        group_path = Sdf.Path("/World/ScatterBrush")
        group_prim = stage.GetPrimAtPath(group_path)
        if not group_prim:
            UsdGeom.Xform.Define(stage, group_path)
        return group_path

    def _create_instance(self, stage: Usd.Stage, parent_path: Sdf.Path, worldpos: Gf.Vec3d, scale: float, yaw_deg: float):
        # Create an Xform prim under parent and add a reference to the chosen asset USD
        parent = stage.GetPrimAtPath(parent_path)
        if not parent:
            return
        name = f"scatter_{int(time.time()*1000)}_{random.randint(0, 9999):04d}"
        child_path = parent_path.AppendChild(name)
        xform = UsdGeom.Xform.Define(stage, child_path)
        xform_prim = xform.GetPrim()
        # Add reference to asset
        try:
            ref_asset = carb.tokens.get_tokens_interface().resolve(self._settings.asset_path)
        except Exception:
            ref_asset = self._settings.asset_path
        # Normalize to URL for USD reference stability
        ref_asset = omni.client.normalize_url(ref_asset)
        xform_prim.GetReferences().AddReference(ref_asset)
        # Apply SRT
        api = UsdGeom.XformCommonAPI(xform)
        up_axis = self._settings.world_up_axis
        rot = _rotation_about_axis(up_axis, yaw_deg)
        api.SetRotate(rot)
        api.SetScale(Gf.Vec3f(scale, scale, scale))
        api.SetTranslate(worldpos)

    def destroy(self):
        if self._toggle_sub:
            self._toggle_sub = None
        # keep overlay around until layer is destroyed; clear handlers
        self._on_toggle(False)

    # Required for compatibility with SceneItem wrapper
    @property
    def visible(self):
        return True


def _axis_to_vec(axis: str) -> Gf.Vec3d:
    axis = axis.upper()
    if axis == "X":
        return Gf.Vec3d(1.0, 0.0, 0.0)
    if axis == "Z":
        return Gf.Vec3d(0.0, 0.0, 1.0)
    return Gf.Vec3d(0.0, 1.0, 0.0)


def _offset_in_plane(center: Gf.Vec3d, up_axis: str, dx: float, dy: float) -> Gf.Vec3d:
    up = _axis_to_vec(up_axis)
    # Build an orthonormal basis (u, v) on the plane perpendicular to up
    # Choose arbitrary right vector not parallel to up
    if abs(up[0]) < 0.9:
        right = Gf.Vec3d(1.0, 0.0, 0.0)
    else:
        right = Gf.Vec3d(0.0, 0.0, 1.0)
    u = right ^ up  # cross product
    u.Normalize()
    v = up ^ u
    v.Normalize()
    return center + u * dx + v * dy


def _rotation_about_axis(axis: str, yaw_deg: float) -> Gf.Vec3f:
    yaw = float(yaw_deg)
    if axis.upper() == "X":
        return Gf.Vec3f(yaw, 0.0, 0.0)
    if axis.upper() == "Z":
        return Gf.Vec3f(0.0, 0.0, yaw)
    return Gf.Vec3f(0.0, yaw, 0.0)


def _sample_uniform_disk(radius: float) -> tuple[float, float]:
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2.0 * math.pi
    return r * math.cos(theta), r * math.sin(theta)


def _rand_float(a: float, b: float) -> float:
    low = min(a, b)
    high = max(a, b)
    return low + (high - low) * random.random()


class _ToggleModel(ui.AbstractValueModel):
    def __init__(self):
        super().__init__()
        self._value = False

    def set_value(self, value):
        self._value = bool(value)
        self._value_changed()

    def get_value_as_bool(self):
        return bool(self._value)


class ScatterBrushButtonGroup(WidgetGroup):
    """Scatter brush toolbar button (toggle). Left-click toggles; right-click opens settings."""

    name = "scatter_brush"

    def __init__(self):
        super().__init__()
        self._button: ui.ToolButton | None = None
        self._model = _ToggleModel()
        self.__toggled = []  # list[Callable[[bool], Any]]

    def _on_mouse_released(self, button):
        if self._button is None or not self._button.enabled:
            return
        if button == 0:
            # toggle
            self._model.set_value(not self._model.get_value_as_bool())
            self._notify_toggled()
        elif button == 1:
            # right click -> settings
            _BrushSettingsWindow.show(_Settings())

    def subscribe_toggled(self, callback: Callable[[bool], Any]):
        self.__toggled.append(callback)
        # return a simple unsubscriber
        class _Sub:
            def __init__(self, arr, fn):
                self._arr = arr
                self._fn = fn
            def __del__(self):
                try:
                    self._arr.remove(self._fn)
                except Exception:
                    pass
        return _Sub(self.__toggled, callback)

    def _notify_toggled(self):
        value = self._model.get_value_as_bool()
        for fn in list(self.__toggled):
            try:
                fn(value)
            except Exception:
                carb.log_error("ScatterBrushButtonGroup subscriber error")

    def get_style(self):
        # Provide style mapping if available
        try:
            return {f"Button.Image::{self.name}": style.default.get(f"Button.Image::{self.name}", {})}
        except Exception:
            return {}

    def create(self, default_size: ui.Length):
        self._button = ui.ToolButton(
            model=self._model,
            name=self.name,
            identifier=self.name,
            tooltip="Scatter Brush (RMB for settings)",
            width=default_size,
            height=default_size,
            mouse_released_fn=lambda x, y, b, _: self._on_mouse_released(b),
        )
        return {self.name: self._button}

    def clean(self):
        super().clean()
        self._button = None
        self.__toggled = []


def create_button_instance():
    global _scatter_button_group
    _scatter_button_group = ScatterBrushButtonGroup()
    return _scatter_button_group


def delete_button_instance():
    global _scatter_button_group
    _scatter_button_group = None


def scatter_brush_factory(desc: dict[str, Any]):
    return ScatterBrush(desc.get("viewport_api"))