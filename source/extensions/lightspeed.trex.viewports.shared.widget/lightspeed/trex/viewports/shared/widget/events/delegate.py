"""
* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

__all__ = ["MouseWheelInterceptorSubscription", "ViewportEventDelegate", "register_mouse_wheel_interceptor"]

import contextlib
import math
import traceback
from typing import TYPE_CHECKING
from collections.abc import Callable

import carb
import omni.appwindow
from lightspeed.common.constants import GlobalEventNames
from lightspeed.events_manager import get_instance as _get_event_manager_instance
from lightspeed.trex.viewports.manipulators.zoom import zoom_operation as _zoom_operation

if TYPE_CHECKING:
    from omni.kit.widget.viewport.api import ViewportAPI as _ViewportAPI

# Consulted before any camera handling in ViewportEventDelegate.mouse_wheel so a viewport tool can claim the wheel
# (for example to resize a brush) without the camera zooming underneath it.
_MOUSE_WHEEL_INTERCEPTORS: list[Callable[[_ViewportAPI | None, float, float, int], bool]] = []


class MouseWheelInterceptorSubscription:
    """Keeps a mouse-wheel interceptor registered for as long as the subscription is alive.

    The interceptor is removed when `destroy` is called or when the subscription is garbage collected.
    """

    def __init__(self, fn: Callable[[_ViewportAPI | None, float, float, int], bool]):
        self.__fn = fn

    def __del__(self):
        self.destroy()

    def destroy(self):
        """Unregister the interceptor; calling it again is a no-op."""
        fn, self.__fn = self.__fn, None
        if fn is not None and fn in _MOUSE_WHEEL_INTERCEPTORS:
            _MOUSE_WHEEL_INTERCEPTORS.remove(fn)


def register_mouse_wheel_interceptor(
    fn: Callable[[_ViewportAPI | None, float, float, int], bool],
) -> MouseWheelInterceptorSubscription:
    """Register a callback that is consulted before the viewport handles a mouse-wheel event.

    Args:
        fn: Called as ``fn(viewport_api, x, y, modifiers)`` with the raw wheel deltas. Return True to consume the
            event, which skips the flight-speed adjustment and the camera zoom.

    Returns:
        The subscription that keeps the interceptor registered until it is destroyed.
    """
    _MOUSE_WHEEL_INTERCEPTORS.append(fn)
    return MouseWheelInterceptorSubscription(fn)


def _limit_camera_velocity(value: float, settings: carb.settings.ISettings, context_name: str):
    cam_limit = settings.get("/exts/omni.kit.viewport.window/cameraSpeedLimit")
    if context_name in cam_limit:
        vel_min = settings.get("/persistent/app/viewport/camVelocityMin")
        if vel_min is not None:
            value = max(vel_min, value)
        vel_max = settings.get("/persistent/app/viewport/camVelocityMax")
        if vel_max is not None:
            value = min(vel_max, value)
    return value


def _regular_keyboard_inputs():
    # Kit orders regular keyboard inputs before modifier keys; stop before LEFT_SHIFT and skip ESC.
    for key_index in range(int(carb.input.KeyboardInput.LEFT_SHIFT)):
        if key_index == int(carb.input.KeyboardInput.ESCAPE):
            continue
        with contextlib.suppress(ValueError, TypeError):
            yield carb.input.KeyboardInput(key_index)


_REGULAR_KEYBOARD_INPUTS = tuple(_regular_keyboard_inputs())


class ViewportEventDelegate:
    """Routes the mouse-wheel, keyboard, and drag-and-drop events of one viewport scene view."""

    def __init__(self, scene_view, viewport_api):
        self.__scene_view = scene_view
        self.__viewport_api = viewport_api
        scene_view.set_mouse_wheel_fn(self.mouse_wheel)
        scene_view.set_key_pressed_fn(self.key_pressed)
        scene_view.set_accept_drop_fn(self.drop_accept)
        scene_view.set_drop_fn(self.drop)
        scene_view.scroll_only_window_hovered = True
        self.__dd_handler = None
        self.__key_down = set()

    def destroy(self):
        scene_view = self.scene_view
        if scene_view:
            scene_view.set_mouse_wheel_fn(None)
            scene_view.set_key_pressed_fn(None)
            scene_view.set_accept_drop_fn(None)
            scene_view.set_drop_fn(None)
            self.__scene_view = None

    @property
    def scene_view(self):
        with contextlib.suppress(ReferenceError):
            if self.__scene_view:
                return self.__scene_view
        return None

    @property
    def viewport_api(self):
        with contextlib.suppress(ReferenceError):
            if self.__viewport_api:
                return self.__viewport_api
        return None

    @property
    def drag_drop_handler(self):
        return self.__dd_handler

    def adjust_flight_speed(self, x: float, y: float):
        try:
            iinput = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            mouse = app_window.get_mouse()
            mouse_value = iinput.get_mouse_value(mouse, carb.input.MouseInput.RIGHT_BUTTON)
            if mouse_value > 0:
                settings = carb.settings.get_settings()
                value = settings.get("/persistent/app/viewport/camMoveVelocity") or 1
                scaler = settings.get("/persistent/app/viewport/camVelocityScalerMultAmount") or 1.1
                scaler = 1.0 + (max(scaler, 1.0 + 1e-8) - 1.0) * abs(y)
                if y < 0:
                    value /= scaler
                elif y > 0:
                    value *= scaler
                if math.isfinite(value) and (value > 1e-8):
                    value = _limit_camera_velocity(value, settings, "scroll")
                    settings.set("/persistent/app/viewport/camMoveVelocity", value)
                return True

            # OM-58310: orbit + scroll does not behave well together, but when scroll is moved to omni.ui.scene
            # they cannot both exists anyway, so disable this possibility for now by returning True if any button down
            return iinput.get_mouse_value(mouse, carb.input.MouseInput.LEFT_BUTTON) or iinput.get_mouse_value(
                mouse, carb.input.MouseInput.MIDDLE_BUTTON
            )

        except Exception:  # noqa: BLE001
            carb.log_error(f"Traceback:\n{traceback.format_exc()}")

        return False

    def mouse_wheel(self, x: float, y: float, modifiers: int):
        """Zoom the camera unless an interceptor, the flight-speed adjustment, or a held key claims the wheel."""
        if self.__intercept_mouse_wheel(x, y, modifiers):
            return
        # Do not use horizontal scroll at all (do we want to hide this behind a setting, or allow it for speed
        # but not zoom)
        x = 0
        # Try to apply flight speed first (should be applied when flight-mode key is active)
        if self.adjust_flight_speed(x, y):
            return
        # If a key is down, ignore the wheel-event (i.e. don't zoom on paint b+scroll event)
        if self.__regular_key_is_down():
            return

        try:
            settings = carb.settings.get_settings()
            cam_velocity = settings.get("/persistent/app/viewport/camMoveVelocity") or 5.0
            default_velocity = 5.0
            speed_scale = cam_velocity / default_velocity
            _zoom_operation(x, y * speed_scale, self.viewport_api)
        except Exception:  # noqa: BLE001
            carb.log_error(f"Traceback:\n{traceback.format_exc()}")

    def __intercept_mouse_wheel(self, x: float, y: float, modifiers: int) -> bool:
        # Iterate over a snapshot: an interceptor may unregister itself while it is being called.
        for interceptor in tuple(_MOUSE_WHEEL_INTERCEPTORS):
            try:
                if interceptor(self.viewport_api, x, y, modifiers):
                    return True
            except Exception:  # noqa: BLE001
                carb.log_error(f"Traceback:\n{traceback.format_exc()}")
        return False

    def __regular_key_is_down(self):
        app_window = omni.appwindow.get_default_app_window()
        key_input = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()
        pressed_keys = set()
        for key in {*self.__key_down, *_REGULAR_KEYBOARD_INPUTS}:
            if key_input.get_keyboard_value(keyboard, key):
                pressed_keys.add(key)

        self.__key_down = pressed_keys
        return bool(pressed_keys)

    def key_pressed(self, key_index: int, modifiers: int, is_down: bool):
        if key_index in (int(carb.input.KeyboardInput.DEL), int(carb.input.KeyboardInput.NUMPAD_DEL)):
            if not is_down and not modifiers and self.viewport_api:
                _get_event_manager_instance().call_global_custom_event(
                    GlobalEventNames.VIEWPORT_DELETE_SELECTION_REQUEST.value,
                    self.viewport_api.usd_context_name,
                )
            return

        # Ignore all key-modifier up/down events, only care about escape or blocking scroll with real-key
        if key_index >= int(carb.input.KeyboardInput.LEFT_SHIFT):
            return
        if key_index == int(carb.input.KeyboardInput.ESCAPE):
            self.stop_drag_drop()
            return
        if is_down:
            self.__key_down.add(carb.input.KeyboardInput(key_index))
        else:
            self.__key_down.discard(carb.input.KeyboardInput(key_index))

    def mouse_moved(self, x: float, y: float, modifiers: int, is_pressed: bool, *args):
        if self.__dd_handler:
            self.__dd_handler._perform_query(self.__scene_view, (x, y))  # noqa: SLF001

    def drop_accept(self, url: str):
        return False

    def drop(self, data):
        dd_handler = self.stop_drag_drop(False)
        if dd_handler:
            dd_handler.dropped(self.__scene_view, data)

    def mouse_hovered(self, value: bool, *args):
        if not value and self.__dd_handler:
            self.stop_drag_drop()

    def stop_drag_drop(self, cancel: bool = True):
        dd_handler, self.__dd_handler = self.__dd_handler, None
        self.__scene_view.set_mouse_moved_fn(None)
        self.__scene_view.set_mouse_hovered_fn(None)
        if dd_handler and cancel:
            dd_handler.cancel(self)
        return dd_handler
