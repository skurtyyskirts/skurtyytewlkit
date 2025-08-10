"""
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import carb
import carb.settings
import omni.ext
from omni.kit.widget.toolbar import get_instance as _get_toolbar_instance

from omni.flux.feature_flags.core import FeatureFlagsCore as _FeatureFlagsCore
from .window import PackagingRolloutWindow, create_toolbar_widget, delete_toolbar_widget


_INSTANCE: PackagingRolloutWindow | None = None


class PackagingRolloutExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self._settings = carb.settings.get_settings()
        self._flag_key = "/exts/omni.flux.feature_flags.core/flags/packaging_rollout/value"
        self._subscriptions = []

    def _is_enabled(self) -> bool:
        return bool(self._settings.get(self._flag_key) or False)

    def _apply_flag_state(self):
        global _INSTANCE
        enabled = self._is_enabled()
        toolbar = _get_toolbar_instance()
        if enabled and _INSTANCE is None:
            _INSTANCE = PackagingRolloutWindow()
            toolbar_widget = create_toolbar_widget(_INSTANCE)
            if toolbar and toolbar_widget:
                toolbar.add_widget(toolbar_widget, 12)
        elif not enabled and _INSTANCE is not None:
            delete_toolbar_widget()
            _INSTANCE.destroy()
            _INSTANCE = None

    def on_startup(self, ext_id):
        carb.log_info("[lightspeed.trex.packaging_rollout.window] Startup")

        # Apply current flag state and subscribe to changes
        self._apply_flag_state()
        core = _FeatureFlagsCore()
        self._subscriptions = core.subscribe_feature_flags_changed(lambda *_: self._apply_flag_state())

    def on_shutdown(self):
        global _INSTANCE
        carb.log_info("[lightspeed.trex.packaging_rollout.window] Shutdown")

        # Unsubscribe
        if self._subscriptions:
            core = _FeatureFlagsCore()
            core.unsubscribe_feature_flags_changed(self._subscriptions)
            self._subscriptions = []

        delete_toolbar_widget()

        if _INSTANCE:
            _INSTANCE.destroy()
        _INSTANCE = None