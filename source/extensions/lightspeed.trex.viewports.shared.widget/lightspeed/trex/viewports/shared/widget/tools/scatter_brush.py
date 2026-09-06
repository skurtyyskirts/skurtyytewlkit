"""
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "ScatterBrushTool",
    "brush_cursor_matrix",
    "create_button_instance",
    "delete_button_instance",
    "scatter_brush_factory",
]

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable

import carb
import carb.input
import numpy as np
import omni.appwindow
import omni.kit.app
import omni.ui as ui
import omni.usd
from lightspeed.trex.app.style import style
from lightspeed.trex.hotkeys import TrexHotkeyEvent
from lightspeed.trex.hotkeys import get_global_hotkey_manager as _get_global_hotkey_manager
from lightspeed.trex.scatter.core import (
    ApplyTo,
    MeshSurfaceCache,
    ScatterMode,
    StrokeSession,
    SurfaceSample,
    get_scatter_brush_controller,
    resolve_target,
    selected_prototypes,
    stage_up_axis,
    tangent_basis,
    up_axis_vector,
)
from lightspeed.ui_scene.light_manipulator.gesture import disable_other_drag_gestures
from omni.kit.notification_manager import NotificationStatus, post_notification
from omni.kit.widget.toolbar.widget_group import WidgetGroup
from omni.ui import scene as sc
from pxr import Gf

from ..events import register_mouse_wheel_interceptor
from ..scene.utils import flatten_matrix

if TYPE_CHECKING:
    from lightspeed.trex.scatter.core import ScatterBrushSettings, ScatterTarget, SurfaceHit, SurfacePicker
    from omni.kit.widget.viewport.api import ViewportAPI
    from pxr import Sdf, Usd

_LOG_PREFIX = "[lightspeed.trex.viewports.shared.widget]"
_TOOLTIP = (
    "Scatter brush: paint ingested assets onto meshes. Ctrl+B toggles, Shift+drag erases, hold B and scroll to resize."
)
_NO_ASSETS_MESSAGE = "Add at least one ingested asset to the scatter brush"
_HOVER_SUBSCRIPTION_NAME = "ScatterBrushHover"

_PAINT_COLOR = (0.25, 0.9, 0.35, 1.0)
_ERASE_COLOR = (0.95, 0.3, 0.25, 1.0)
_NOT_APPLICABLE_COLOR = (0.6, 0.6, 0.6, 0.6)
# Length of the normal tick relative to the brush radius.
_NORMAL_TICK_LENGTH = 0.35
# One wheel notch while B is held multiplies (or divides) the radius by this factor, within the settings bounds.
_RADIUS_WHEEL_FACTOR = 1.1
_RADIUS_MIN = 1.0
_RADIUS_MAX = 10000.0

_ALT_KEYS = (carb.input.KeyboardInput.LEFT_ALT, carb.input.KeyboardInput.RIGHT_ALT)
_SHIFT_KEYS = (carb.input.KeyboardInput.LEFT_SHIFT, carb.input.KeyboardInput.RIGHT_SHIFT)
_RESIZE_KEY = carb.input.KeyboardInput.B

_scatter_button_group: ScatterBrushButtonGroup | None = None


def create_button_instance() -> ScatterBrushButtonGroup:
    """Create the single toolbar button group shared by every viewport."""
    global _scatter_button_group
    _scatter_button_group = ScatterBrushButtonGroup()
    return _scatter_button_group


def delete_button_instance() -> None:
    """Forget the shared toolbar button group; the extension cleans it before calling this."""
    global _scatter_button_group
    _scatter_button_group = None


def _is_key_down(key: carb.input.KeyboardInput) -> bool:
    """Return whether a keyboard key is currently held on the default application window."""
    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    return bool(input_interface.get_keyboard_value(app_window.get_keyboard(), key))


def _any_key_down(keys: Iterable[carb.input.KeyboardInput]) -> bool:
    """Return whether at least one of the keys is currently held."""
    return any(_is_key_down(key) for key in keys)


def _mouse_ndc(frame: ui.Frame) -> tuple[float, float] | None:
    """Return the mouse position as normalized device coordinates of ``frame``.

    Args:
        frame: Widget overlaying the viewport; its screen rectangle defines the [-1, 1] square (y up).

    Returns:
        The coordinates, possibly outside [-1, 1] when the mouse is elsewhere, or None while the frame has no size.
    """
    width, height = frame.computed_width, frame.computed_height
    if width <= 0.0 or height <= 0.0:
        return None
    app_window = omni.appwindow.get_default_app_window()
    dpi_scale = ui.Workspace.get_dpi_scale()
    pixel_x, pixel_y = carb.input.acquire_input_interface().get_mouse_coords_pixel(app_window.get_mouse())
    x = pixel_x / dpi_scale
    y = pixel_y / dpi_scale
    return (
        -1.0 + 2.0 * ((x - frame.screen_position_x) / width),
        1.0 - 2.0 * ((y - frame.screen_position_y) / height),
    )


def _inside_viewport(ndc: tuple[float, float]) -> bool:
    """Return whether normalized device coordinates fall inside the viewport square."""
    return -1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0


def _stage_for(usd_context_name: str) -> Usd.Stage | None:
    """Return the stage of a USD context, or None when the context no longer exists or has no stage open."""
    context = omni.usd.get_context(usd_context_name)
    return context.get_stage() if context is not None else None


def brush_cursor_matrix(position: Any, normal: Any, radius: float) -> Gf.Matrix4d:
    """Build the row-major transform that places a unit brush disk on a surface sample.

    Local X and Y map onto the surface tangent and bitangent scaled by ``radius``, local Z onto the unit normal
    scaled the same way, and the origin onto ``position``; ``Gf.Matrix4d.Transform((1, 0, 0))`` therefore lands at
    ``position + tangent * radius``. Flattened row by row, the matrix is what ``omni.ui.scene`` expects.

    Args:
        position: World-space center of the brush (any 3-component indexable).
        normal: World-space surface normal (any 3-component indexable, not necessarily unit length).
        radius: Brush radius in world units.

    Returns:
        The transform of the cursor, with the translation in the last row.
    """
    tangent, bitangent = tangent_basis(np.asarray(normal, dtype=np.float64))
    unit_normal = np.cross(tangent, bitangent)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRow(0, Gf.Vec4d(*(tangent * radius), 0.0))
    matrix.SetRow(1, Gf.Vec4d(*(bitangent * radius), 0.0))
    matrix.SetRow(2, Gf.Vec4d(*(unit_normal * radius), 0.0))
    matrix.SetRow(3, Gf.Vec4d(float(position[0]), float(position[1]), float(position[2]), 1.0))
    return matrix


class _BrushGestureManager(sc.GestureManager):
    """Keeps the brush drag out of the default gesture arbitration.

    The brush Screen covers the whole viewport, so with the default manager the drag would fight the camera and
    selection gestures over every press. Conflicting layers are hidden for the duration of a stroke instead.
    """

    def can_be_prevented(self, gesture: sc.AbstractGesture) -> bool:
        """Report that the brush drag can never be prevented by another gesture."""
        return False

    def should_prevent(self, gesture: sc.AbstractGesture, preventer: sc.AbstractGesture) -> bool:
        """Report that the brush drag never prevents another gesture."""
        return False


class ScatterBrushButtonGroup(WidgetGroup):
    """Toolbar toggle that turns the scatter brush on and off.

    The toggle mirrors the controller mode (checked while painting or erasing) and switches the controller between
    ``PAINT`` and ``OFF`` when the user clicks it. Create it once through ``create_button_instance``.
    """

    name = "scatter_brush"

    def __init__(self):
        super().__init__()
        self._button: ui.ToolButton | None = None
        self._syncing = False
        self._model: ui.SimpleBoolModel | None = ui.SimpleBoolModel(False)
        self._model_subscription = self._model.subscribe_value_changed_fn(self._on_model_value_changed)
        controller = get_scatter_brush_controller()
        self._mode_subscription = controller.subscribe_mode_changed(self._on_mode_changed)
        self._on_mode_changed(controller.mode)

    def get_style(self) -> dict:
        """Return the toolbar style entries of the button that the application style defines."""
        base_key = f"Button.Image::{self.name}"
        keys = (base_key, f"{base_key}:checked", f"{base_key}:hovered")
        return {key: style.default[key] for key in keys if key in style.default}

    def create(self, default_size: ui.Length) -> dict[str, ui.Widget]:
        """Build the toggle button and return it under the group name."""
        self._button = ui.ToolButton(
            model=self._model,
            name=self.name,
            identifier=self.name,
            tooltip=_TOOLTIP,
            width=default_size,
            height=default_size,
        )
        return {self.name: self._button}

    def clean(self):
        """Drop the controller and model subscriptions together with the button."""
        super().clean()
        self._mode_subscription = None
        self._model_subscription = None
        self._button = None
        self._model = None

    def _on_mode_changed(self, mode: ScatterMode) -> None:
        """Check the button whenever the brush is painting or erasing, without echoing the change back."""
        if self._model is None:
            return
        self._syncing = True
        try:
            self._model.set_value(mode != ScatterMode.OFF)
        finally:
            self._syncing = False

    def _on_model_value_changed(self, model: ui.AbstractValueModel) -> None:
        """Switch the brush between paint and off when the user clicks the button."""
        if self._syncing:
            return
        controller = get_scatter_brush_controller()
        if not self._is_in_context():
            # Another tool owns the toolbar: the click is ignored, so undo the visual toggle.
            self._on_mode_changed(controller.mode)
            return
        controller.set_mode(ScatterMode.PAINT if model.as_bool else ScatterMode.OFF)


class ScatterBrushTool:
    """Viewport half of the scatter brush for one viewport: cursor, gestures, hover picking and stroke lifecycle.

    Built by ``scatter_brush_factory`` inside the scene layer's transform, so every ``omni.ui.scene`` item it creates
    belongs to that layer. The shared ``ScatterBrushController`` owns the mode, settings and assets; this class turns
    mouse input into ``StrokeSession`` calls and draws the brush footprint under the cursor. The picker is created on
    first use so the renderer support check has had time to settle.
    """

    def __init__(
        self,
        viewport_api: ViewportAPI,
        layer_provider: Any,
        is_active_fn: Callable[[], bool] | None,
        usd_context_name: str | None,
    ):
        """
        Args:
            viewport_api: Viewport the brush draws in and picks through.
            layer_provider: Layer collection of the viewport, used to hide the selection and transform manipulators
                during a stroke.
            is_active_fn: Returns whether this viewport is the active one; the brush stays idle otherwise.
            usd_context_name: USD context the strokes are authored in; defaults to the viewport's context.
        """
        self._viewport_api = viewport_api
        self._layer_provider = layer_provider
        self._is_active_fn = is_active_fn
        self._usd_context_name = viewport_api.usd_context_name if usd_context_name is None else usd_context_name

        self._visible = True
        self._enabled = False
        self._session: StrokeSession | None = None
        self._layer_guard = None
        self._update_subscription = None
        self._picker: SurfacePicker | None = None
        self._last_hover: tuple[SurfaceHit, ScatterTarget, SurfaceSample] | None = None
        self._last_pick_ndc: tuple[float, float] | None = None
        self._last_pick_camera: Gf.Matrix4d | None = None
        self._force_pick = False
        self._stroke_prototypes: dict[Sdf.Path, int] = {}
        self._warned_prototypes: set[Sdf.Path] = set()

        # Created in the factory context, so it overlays the viewport and gives the screen-to-NDC mapping.
        self._frame: ui.Frame | None = ui.Frame()
        self._cache: MeshSurfaceCache | None = MeshSurfaceCache(lambda: _stage_for(self._usd_context_name))

        self._gesture_manager = _BrushGestureManager()
        self._drag = sc.DragGesture(
            mouse_button=0,
            check_mouse_moved=True,
            on_began_fn=self._on_drag_began,
            on_changed_fn=self._on_drag_changed,
            on_ended_fn=self._on_drag_ended,
            manager=self._gesture_manager,
        )
        self._screen: sc.Screen | None = sc.Screen(gestures=[self._drag], visible=False)
        self._cursor: sc.Transform | None = sc.Transform(visible=False)
        color = list(_PAINT_COLOR)
        with self._cursor:
            self._outer_ring = sc.Arc(1.0, axis=2, wireframe=True, tesselation=64, thickness=2, color=color)
            self._inner_ring = sc.Arc(0.5, axis=2, wireframe=True, tesselation=48, thickness=1, color=color)
            self._normal_tick = sc.Line([0.0, 0.0, 0.0], [0.0, 0.0, _NORMAL_TICK_LENGTH], thickness=2, color=color)
        self._cursor_color: tuple[float, float, float, float] = _PAINT_COLOR

        self._hotkey_subscription = _get_global_hotkey_manager().subscribe_hotkey_event(
            TrexHotkeyEvent.CTRL_B, self._on_toggle_hotkey, enable_fn=self._is_viewport_active
        )
        self._wheel_subscription = register_mouse_wheel_interceptor(self._on_wheel)
        controller = get_scatter_brush_controller()
        self._mode_subscription = controller.subscribe_mode_changed(self._on_mode_changed)
        self._settings_subscription = controller.subscribe_settings_changed(self._on_settings_changed)
        self._refresh_enabled()

    @property
    def name(self) -> str:
        """Layer name shown by the viewport layer system."""
        return "Scatter Brush"

    @property
    def categories(self) -> list[str]:
        """Layer categories used by ``find_viewport_layer``."""
        return ["tools"]

    @property
    def visible(self) -> bool:
        """Whether the layer system shows this tool; hiding it also ends any stroke."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        if not self._visible:
            self._hide_cursor()
        self._refresh_enabled()

    @property
    def enabled(self) -> bool:
        """Whether the brush currently listens to the mouse (mode is not off and the tool is visible)."""
        return self._enabled

    def destroy(self):
        """End any stroke, release the viewport layers and drop every subscription and scene item."""
        self._end_stroke()
        self._update_subscription = None
        self._hotkey_subscription = None
        self._wheel_subscription = None
        self._mode_subscription = None
        self._settings_subscription = None
        self._enabled = False
        if self._picker is not None:
            self._picker.cancel()
        self._picker = None
        if self._cache is not None:
            self._cache.destroy()
        self._cache = None
        for item in (self._screen, self._cursor):
            if item is not None:
                item.visible = False
        self._outer_ring = None
        self._inner_ring = None
        self._normal_tick = None
        self._cursor = None
        self._screen = None
        self._drag = None
        self._gesture_manager = None
        self._frame = None
        self._last_hover = None

    def _is_viewport_active(self) -> bool:
        """Return whether this viewport renders and is the active one, which gates every mouse and hotkey callback."""
        return (
            bool(self._viewport_api.updates_enabled) and self._is_active_fn is not None and bool(self._is_active_fn())
        )

    def _refresh_enabled(self) -> None:
        """Listen to the mouse exactly when the brush mode is on and the tool is visible."""
        mode = get_scatter_brush_controller().mode
        self._set_enabled(self._visible and mode != ScatterMode.OFF)

    def _set_enabled(self, enabled: bool) -> None:
        """Show or hide the gesture screen and start or stop the per-frame hover poll."""
        if enabled == self._enabled or self._screen is None:
            return
        self._enabled = enabled
        self._screen.visible = enabled
        if enabled:
            if self._picker is None:
                self._picker = get_scatter_brush_controller().create_picker(self._viewport_api, self._cache)
            self._force_pick = True
            self._update_subscription = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(self._on_update, name=_HOVER_SUBSCRIPTION_NAME)
            )
            return
        self._update_subscription = None
        self._end_stroke()
        if self._picker is not None:
            self._picker.cancel()
        self._last_pick_ndc = None
        self._hide_cursor()

    def _on_mode_changed(self, _mode: ScatterMode) -> None:
        """Follow the controller mode and refresh the cursor colour on the next pick."""
        self._force_pick = True
        self._refresh_enabled()

    def _on_settings_changed(self, _settings: ScatterBrushSettings) -> None:
        """Re-pick on the next frame so the cursor reflects the new radius and target settings."""
        self._force_pick = True

    def _on_toggle_hotkey(self) -> None:
        """Toggle the brush between off and paint (Ctrl+B)."""
        get_scatter_brush_controller().toggle_paint()

    def _on_wheel(self, viewport_api: ViewportAPI | None, _x: float, y: float, _modifiers: int) -> bool:
        """Resize the brush radius while B is held; returns True when the wheel event was consumed."""
        if not self._enabled or not self._is_viewport_active():
            return False
        if viewport_api is not None and viewport_api is not self._viewport_api:
            return False
        if y == 0 or not _is_key_down(_RESIZE_KEY):
            return False
        controller = get_scatter_brush_controller()
        factor = _RADIUS_WHEEL_FACTOR if y > 0 else 1.0 / _RADIUS_WHEEL_FACTOR
        radius = min(_RADIUS_MAX, max(_RADIUS_MIN, controller.settings.radius * factor))
        controller.update_settings(radius=radius)
        return True

    def _on_update(self, _event) -> None:
        """Poll the mouse once per frame and request the surface pick that drives the cursor and the stroke.

        The picker alone decides whether a request goes out: it refuses one while another is in flight and drops a
        request whose completion never arrived, so the poll keeps asking instead of gating on ``in_flight`` itself.
        """
        if self._picker is None or not self._is_viewport_active():
            self._hide_cursor()
            return
        ndc = _mouse_ndc(self._frame)
        if ndc is None or not _inside_viewport(ndc):
            self._last_pick_ndc = None
            self._hide_cursor()
            return
        camera = self._viewport_api.transform
        camera_unchanged = self._last_pick_camera is not None and camera == self._last_pick_camera
        if not self._force_pick and ndc == self._last_pick_ndc and camera_unchanged:
            return
        if self._picker.pick(ndc, self._on_hit):
            self._force_pick = False
            self._last_pick_ndc = ndc
            self._last_pick_camera = Gf.Matrix4d(camera)

    def _on_hit(self, hit: SurfaceHit | None) -> None:
        """Handle a pick result, degrading to a hidden cursor when the handling fails.

        The CPU picker calls back from inside the per-frame poll, so an exception here would otherwise escape into the
        update subscription and repeat every frame until the brush is toggled.
        """
        try:
            self._handle_hit(hit)
        except Exception as exc:  # noqa: BLE001 - runs from the per-frame poll; a bad frame must not become a storm.
            carb.log_error(f"{_LOG_PREFIX} Scatter brush hover update failed: {exc}")
            self._hide_cursor()

    def _handle_hit(self, hit: SurfaceHit | None) -> None:
        """Resolve a pick result into a target and sample, move the cursor and feed the active stroke."""
        if self._cursor is None:
            return
        stage = self._viewport_api.stage if hit is not None else None
        if stage is None:
            self._hide_cursor()
            return
        controller = get_scatter_brush_controller()
        settings = controller.settings
        target = resolve_target(stage, hit, settings.target_mode, settings.anchor_prototype_path or None)
        if target is None:
            self._hide_cursor()
            return
        applicable = settings.apply_to != ApplyTo.SELECTED or target.prototype_root in selected_prototypes(
            omni.usd.get_context(self._usd_context_name)
        )
        sample = self._cache.closest_point(target.mesh_path, hit.world_position, settings.radius)
        if sample is None:
            sample = self._fallback_sample(stage, hit)
        session = self._session
        erase = session.erase if session is not None else self._erase_intent(controller.mode)
        self._show_cursor(sample, settings.radius, self._cursor_color_for(applicable, erase))
        if not applicable:
            self._last_hover = None
            return
        self._last_hover = (hit, target, sample)
        if session is not None:
            self._advance_stroke(session, hit, target, sample)

    def _on_drag_began(self, _sender: sc.AbstractGesture) -> None:
        """Start a paint or erase stroke unless the press belongs to the camera or nothing can be painted."""
        if not self._enabled or not self._is_viewport_active() or _any_key_down(_ALT_KEYS):
            return
        stage = self._viewport_api.stage
        if stage is None:
            return
        controller = get_scatter_brush_controller()
        erase = controller.mode == ScatterMode.ERASE or _any_key_down(_SHIFT_KEYS)
        if not erase and not controller.enabled_assets():
            controller.post_status(_NO_ASSETS_MESSAGE, True)
            return
        self._end_stroke()
        seed, stroke_index = controller.next_stroke()
        self._session = StrokeSession(
            self._usd_context_name,
            controller.settings,
            controller.assets,
            self._cache,
            erase,
            stage_up_axis(stage),
            seed,
            stroke_index,
        )
        self._layer_guard = disable_other_drag_gestures(self._layer_provider)
        self._force_pick = True
        if self._last_hover is not None:
            hit, target, sample = self._last_hover
            self._advance_stroke(self._session, hit, target, sample)

    def _on_drag_changed(self, _sender: sc.AbstractGesture) -> None:
        """Ask the hover poll for a fresh pick; the poll, not the gesture, advances the stroke."""
        self._force_pick = True

    def _on_drag_ended(self, _sender: sc.AbstractGesture) -> None:
        """Commit the stroke and restore the viewport layers."""
        self._end_stroke()

    def _advance_stroke(
        self, session: StrokeSession, hit: SurfaceHit, target: ScatterTarget, sample: SurfaceSample
    ) -> None:
        """Begin the stroke at the sample or stamp along the way to it; a failure ends the stroke."""
        self._stroke_prototypes[target.prototype_root] = target.instance_count
        try:
            if session.active:
                session.update(hit, target, sample)
            else:
                session.begin(hit, target, sample)
        except Exception as exc:  # noqa: BLE001 - runs from a pick callback; the stroke must end instead of leaking.
            carb.log_error(f"{_LOG_PREFIX} Scatter stroke failed and was ended: {exc}")
            self._end_stroke()

    def _end_stroke(self) -> None:
        """Commit whatever the current stroke authored, release the layer guard and notify listeners."""
        session, self._session = self._session, None
        self._layer_guard = None
        touched, self._stroke_prototypes = self._stroke_prototypes, {}
        if session is None:
            return
        try:
            count = session.end()
        except Exception as exc:  # noqa: BLE001 - runs from gesture and teardown paths that must not propagate.
            carb.log_error(f"{_LOG_PREFIX} Scatter stroke commit failed: {exc}")
            return
        get_scatter_brush_controller().notify_stroke_committed(count, session.erase)
        if count and not session.erase:
            self._warn_replicated_prototypes(touched)

    def _warn_replicated_prototypes(self, touched: dict[Sdf.Path, int]) -> None:
        """Tell the user once per prototype that placements under it show up on every instance of its hash."""
        for prototype_root, count in touched.items():
            if count <= 1 or prototype_root in self._warned_prototypes:
                continue
            self._warned_prototypes.add(prototype_root)
            post_notification(
                f"Scatter: placements under {prototype_root.name} replicate onto {count} instances",
                status=NotificationStatus.WARNING,
            )

    def _fallback_sample(self, stage: Usd.Stage, hit: SurfaceHit) -> SurfaceSample:
        """Build a sample at the hit position facing the stage up axis when the mesh cache cannot project it."""
        up = up_axis_vector(stage_up_axis(stage))
        return SurfaceSample(
            position=np.array(hit.world_position, dtype=np.float64),
            normal=np.array(up, dtype=np.float64),
            triangle_index=-1,
            distance=0.0,
        )

    def _erase_intent(self, mode: ScatterMode) -> bool:
        """Return whether a stroke started now would erase (erase mode or Shift held)."""
        return mode == ScatterMode.ERASE or _any_key_down(_SHIFT_KEYS)

    @staticmethod
    def _cursor_color_for(applicable: bool, erase: bool) -> tuple[float, float, float, float]:
        """Return the cursor colour: grey when the target is excluded, red when erasing, green when painting."""
        if not applicable:
            return _NOT_APPLICABLE_COLOR
        return _ERASE_COLOR if erase else _PAINT_COLOR

    def _show_cursor(self, sample: SurfaceSample, radius: float, color: tuple[float, float, float, float]) -> None:
        """Place the cursor on the sample scaled to the brush radius and recolour it when needed."""
        matrix = brush_cursor_matrix(sample.position, sample.normal, radius)
        self._cursor.transform = sc.Matrix44(*flatten_matrix(matrix))
        if color != self._cursor_color:
            self._cursor_color = color
            for shape in (self._outer_ring, self._inner_ring, self._normal_tick):
                shape.color = list(color)
        self._cursor.visible = True

    def _hide_cursor(self) -> None:
        """Hide the cursor and forget the hover it showed, so a press cannot start a stroke from a stale pick."""
        self._last_hover = None
        if self._cursor is not None:
            self._cursor.visible = False


def scatter_brush_factory(desc: dict[str, Any]) -> ScatterBrushTool:
    """Create the brush tool for a viewport from the scene layer factory arguments."""
    return ScatterBrushTool(
        desc.get("viewport_api"),
        desc.get("layer_provider"),
        desc.get("is_active_fn"),
        desc.get("usd_context_name"),
    )
