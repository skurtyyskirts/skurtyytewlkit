"""
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from typing import Optional

from omni import ui
from omni.flux.utils.common import reset_default_attrs
from omni.kit.widget.toolbar.widget_group import WidgetGroup


_TOOLBAR_GROUP: _PackagingToolbar | None = None  # type: ignore[name-defined]


class _PackagingToolbar(WidgetGroup):
    name = "packaging_rollout"

    def __init__(self, window: "PackagingRolloutWindow"):
        super().__init__()
        self._button: Optional[ui.ToolButton] = None
        self._window = window

    def create(self, default_size: ui.Length):
        self._button = ui.ToolButton(
            name=self.name,
            identifier=self.name,
            tooltip="Open Packaging & Rollout",
            width=default_size,
            height=default_size,
            mouse_released_fn=lambda *_: self._window.show(),
        )
        return {self.name: self._button}

    def clean(self):
        super().clean()
        self._button = None


class PackagingRolloutWindow:
    def __init__(self):
        self._default_attr = {
            "window": None,
        }
        for attr, value in self._default_attr.items():
            setattr(self, attr, value)

        self.window = ui.Window(
            "Packaging & Rollout",
            width=900,
            height=700,
            visible=False,
            dockPreference=ui.DockPreference.DISABLED,
        )

        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="WorkspaceBackground")
                with ui.VStack(spacing=8):
                    ui.Spacer(height=0)
                    ui.Label("Packaging & Rollout", style_type_name_override="Label::WizardTitle")
                    ui.Separator()
                    with ui.HStack(height=0):
                        ui.Label("Package your mod and roll out updates.")
                    ui.Spacer(height=0)

    def show(self):
        self.window.visible = True

    def destroy(self):
        reset_default_attrs(self)


def create_toolbar_widget(window: PackagingRolloutWindow):
    global _TOOLBAR_GROUP
    _TOOLBAR_GROUP = _PackagingToolbar(window)
    return _TOOLBAR_GROUP


def delete_toolbar_widget():
    global _TOOLBAR_GROUP
    if _TOOLBAR_GROUP:
        _TOOLBAR_GROUP.clean()
    _TOOLBAR_GROUP = None