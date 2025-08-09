"""
Scatter Brush tool (scaffold)
"""
from __future__ import annotations

from typing import Any, Callable

import carb
import omni.ui as ui
from lightspeed.trex.app.style import style
from omni.flux.utils.common import Event as _Event
from omni.flux.utils.common import EventSubscription as _EventSubscription
from omni.kit.widget.toolbar.widget_group import WidgetGroup

# Reuse picker from teleport tool to get world position under mouse
from .teleport import PointMousePicker

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from omni.kit.widget.viewport.api import ViewportAPI
except Exception:  # noqa: PLW0718
    TYPE_CHECKING = False


__all__ = [
    "scatter_brush_factory",
    "ScatterBrushButtonGroup",
    "create_button_instance",
    "delete_button_instance",
]


_scatter_button_group: ScatterBrushButtonGroup | None = None


class ScatterBrushButtonGroup(WidgetGroup):
    name = "scatter"

    def __init__(self):
        super().__init__()
        self._button: ui.ToolButton | None = None
        self.__button_pressed = _Event()

    def get_style(self):
        return {f"Button.Image::{self.name}": style.default.get(f"Button.Image::{self.name}", {})}

    def _on_mouse_released(self, button):
        self._acquire_toolbar_context()
        if self._is_in_context() and self._button is not None and self._button.enabled:
            self.__button_pressed()

    def subscribe_button_pressed(self, callback: Callable[[bool], Any]):
        return _EventSubscription(self.__button_pressed, callback)

    def create(self, default_size: ui.Length):
        self._button = ui.ToolButton(
            model=None,
            name=self.name,
            identifier=self.name,
            tooltip="Scatter Brush (placeholder)",
            width=default_size,
            height=default_size,
            mouse_released_fn=lambda x, y, b, _: self._on_mouse_released(b),
        )
        return {self.name: self._button}

    def clean(self):
        super().clean()
        self._button = None
        self.__button_pressed = None


def create_button_instance():
    global _scatter_button_group
    _scatter_button_group = ScatterBrushButtonGroup()
    return _scatter_button_group


def delete_button_instance():
    global _scatter_button_group
    _scatter_button_group = None


class ScatterBrush:
    """Viewport-connected Scatter Brush (placeholder implementation)."""

    def __init__(self, viewport_api: "ViewportAPI"):
        self._viewport_api = viewport_api
        # this overlays the viewport; required for screen->NDC conversion in PointMousePicker
        self.__viewport_frame = ui.Frame()
        # subscribe to toolbar button
        if _scatter_button_group is not None:
            self._button_pressed_subscription = _scatter_button_group.subscribe_button_pressed(self.on_scatter_button)
        else:
            self._button_pressed_subscription = None

    def get_picker(self):
        def pick_callback(prim_path: str, position: carb.Double3 | None, _pixels: carb.Uint2):
            carb.log_info(f"[ScatterBrush] Picked at {position} on {prim_path} (placeholder)")
            # TODO: Developer Agent - implement scatter placement here
        return PointMousePicker(self._viewport_api, self.__viewport_frame, point_picked_callback_fn=pick_callback)

    def on_scatter_button(self):
        # For now, pick center of viewport
        self.get_picker().pick(ndc_coords=(0, 0))

    def destroy(self):
        self._button_pressed_subscription = None

    @property
    def visible(self):
        return True


def scatter_brush_factory(desc: dict[str, Any]):
    viewport_api = desc.get("viewport_api")
    return ScatterBrush(viewport_api)
