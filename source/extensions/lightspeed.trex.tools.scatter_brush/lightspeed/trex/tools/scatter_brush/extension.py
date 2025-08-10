# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import carb
import carb.input
import carb.settings
import omni.ext
from omni.kit.hotkeys.core import KeyCombination

from lightspeed.trex.hotkeys.hotkey import AppHotkey
from lightspeed.events_manager import get_instance as _get_event_manager_instance

from .model import get_model
from .ui import ScatterBrushPane


_SETTINGS_CHANGED_EVENT = "lightspeed.trex.tools.scatter_brush.settings_changed"
_TOGGLED_EVENT = "lightspeed.trex.tools.scatter_brush.toggled"


class ScatterBrushExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self._ui: ScatterBrushPane | None = None
        self._hotkey: AppHotkey | None = None
        self._settings_sub: int | None = None
        self._subs = []

    def on_startup(self, ext_id):
        carb.log_info("[lightspeed.trex.tools.scatter_brush] Startup")
        self._ui = ScatterBrushPane()

        # Register events
        evt_mgr = _get_event_manager_instance()
        evt_mgr.register_global_custom_event(_SETTINGS_CHANGED_EVENT)
        evt_mgr.register_global_custom_event(_TOGGLED_EVENT)

        # Bridge model events to global events
        self._subs.append(get_model().subscribe_changed(lambda data: evt_mgr.call_global_custom_event(_SETTINGS_CHANGED_EVENT, data)))
        self._subs.append(get_model().subscribe_toggle(lambda enabled: evt_mgr.call_global_custom_event(_TOGGLED_EVENT, enabled)))

        # Register 'B' hotkey to toggle brush enabled state and try to toggle toolbar button if present
        def _on_hotkey():
            model = get_model()
            try:
                model.set_enabled(not model.enabled)
            except Exception:
                pass
            # Best-effort toggle the existing shared scatter toolbar button if present
            try:
                from lightspeed.trex.viewports.shared.widget.tools import scatter_brush as _scatter_mod
                btn = getattr(_scatter_mod, "_scatter_button_group", None)
                if btn is not None and hasattr(btn, "_model"):
                    current = bool(btn._model.get_value_as_bool())
                    btn._model.set_value(not current)
            except Exception:
                # okay if toolbar not present or API changed
                pass

        self._hotkey = AppHotkey(
            action_id="trex::Scatter Brush Toggle",
            key=KeyCombination(carb.input.KeyboardInput.B),
            action=_on_hotkey,
            display_name="Scatter Brush Toggle (B)",
            description="Toggle Scatter Brush mode",
        )

        # Listen to settings changes to reflect external toggles (e.g., toolbar) back into model
        settings = carb.settings.get_settings()
        path = 'exts."lightspeed.trex.tools.scatter_brush".enabled'
        def _on_setting_changed(path_, value):
            try:
                if path_ == path:
                    get_model().set_enabled(bool(value))
            except Exception:
                pass
        self._settings_sub = settings.subscribe_to_node_change_events(_on_setting_changed)

    def on_shutdown(self):
        carb.log_info("[lightspeed.trex.tools.scatter_brush] Shutdown")
        if self._hotkey:
            self._hotkey.destroy()
            self._hotkey = None
        if self._ui:
            self._ui.destroy()
            self._ui = None
        if self._settings_sub is not None:
            try:
                carb.settings.get_settings().unsubscribe_to_change_events(self._settings_sub)
            except Exception:
                pass
            self._settings_sub = None
        self._subs = []