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

__all__ = ["PresetsWidget"]

import asyncio
from collections.abc import Callable

import omni.kit.app
import omni.usd
from lightspeed.trex.scatter.core import ScatterBrushSettings, get_scatter_brush_controller
from lightspeed.trex.utils.widget import TrexMessageDialog
from omni import ui

from .combo_model import ChoicesComboModel


@omni.usd.handle_exception
async def _destroy_window_next_frame(window: ui.Window) -> None:
    """Destroy a window after the current frame; omni.ui cannot destroy a container from inside its own callbacks."""
    await omni.kit.app.get_app().next_update_async()
    window.destroy()


class _NamePrompt:
    """Modal window asking for a preset name; OK stays disabled while the name is blank.

    The OK and Cancel buttons dismiss the prompt from inside their click callbacks, so ``destroy`` hides the window
    right away and destroys it on the next frame.
    """

    _WIDTH = 360
    _HEIGHT = 120
    _BUTTON_WIDTH = ui.Pixel(80)
    _ROW_HEIGHT = ui.Pixel(24)
    _SPACING = ui.Pixel(8)

    def __init__(self, title: str, message: str, initial_name: str, accept_fn: Callable[[str], None]):
        self._accept_fn = accept_fn
        self._field: ui.StringField | None = None
        self._ok_button: ui.Button | None = None
        self._destroy_task: asyncio.Task | None = None
        self._window: ui.Window | None = ui.Window(
            title,
            width=self._WIDTH,
            height=self._HEIGHT,
            dockPreference=ui.DockPreference.DISABLED,
            flags=(
                ui.WINDOW_FLAGS_NO_COLLAPSE
                | ui.WINDOW_FLAGS_NO_RESIZE
                | ui.WINDOW_FLAGS_NO_SCROLLBAR
                | ui.WINDOW_FLAGS_NO_DOCKING
                | ui.WINDOW_FLAGS_MODAL
            ),
        )
        with self._window.frame:
            with ui.VStack(spacing=self._SPACING):
                ui.Label(message, height=0)
                self._field = ui.StringField(height=self._ROW_HEIGHT, identifier="scatter_preset_name_field")
                self._field.model.set_value(initial_name)
                self._field_sub = self._field.model.subscribe_value_changed_fn(self._on_name_changed)
                with ui.HStack(height=self._ROW_HEIGHT, spacing=self._SPACING):
                    ui.Spacer()
                    self._ok_button = ui.Button(
                        "OK",
                        width=self._BUTTON_WIDTH,
                        clicked_fn=self._accept,
                        enabled=bool(initial_name.strip()),
                        identifier="scatter_preset_name_ok",
                    )
                    ui.Button(
                        "Cancel",
                        width=self._BUTTON_WIDTH,
                        clicked_fn=self.destroy,
                        identifier="scatter_preset_name_cancel",
                    )

    @property
    def name(self) -> str:
        """Trimmed text of the name field."""
        return self._field.model.as_string.strip() if self._field is not None else ""

    def destroy(self) -> None:
        """Hide the window now, release its widgets and destroy the window on the next frame."""
        self._field_sub = None
        self._field = None
        self._ok_button = None
        window, self._window = self._window, None
        if window is None:
            return
        window.visible = False
        self._destroy_task = asyncio.ensure_future(_destroy_window_next_frame(window))

    def _accept(self) -> None:
        """Close the prompt and hand the typed name to the owner."""
        name = self.name
        if not name:
            return
        self.destroy()
        self._accept_fn(name)

    def _on_name_changed(self, model: ui.AbstractValueModel) -> None:
        """Keep OK disabled while the name is blank."""
        self._ok_button.enabled = bool(model.as_string.strip())


class PresetsWidget:
    """Preset picker and Save, Save As, Rename, Clone and Delete actions bound to ``ScatterBrushController``.

    The combo box lists the stored presets; when the current ``preset_name`` has no file yet it is shown first so the
    box always displays the active preset. Rename, Clone and Delete are only enabled for stored presets.
    """

    _ROW_HEIGHT = ui.Pixel(24)
    _SPACING = ui.Pixel(8)
    _BUTTON_SPACING = ui.Pixel(4)

    def __init__(self):
        self._controller = get_scatter_brush_controller()
        self._refreshing = False
        self._name_prompt: _NamePrompt | None = None
        self._names_model = ChoicesComboModel()
        self._rename_button: ui.Button | None = None
        self._clone_button: ui.Button | None = None
        self._delete_button: ui.Button | None = None
        self._index_sub = self._names_model.index_model.subscribe_value_changed_fn(self._on_preset_selected)
        self._settings_sub = self._controller.subscribe_settings_changed(self._on_settings_changed)
        self._root = ui.VStack(height=0, spacing=self._SPACING)
        self.__build_ui()
        self.refresh()

    @property
    def names_model(self) -> ChoicesComboModel:
        """Combo box model listing the preset names."""
        return self._names_model

    def refresh(self) -> None:
        """Re-list the stored presets, select the current one and update which actions apply."""
        names = self._controller.preset_names()
        current = self._controller.settings.preset_name
        stored = current in names
        choices = names if stored else [current, *names]
        self._refreshing = True
        try:
            self._names_model.set_choices(choices, choices.index(current))
        finally:
            self._refreshing = False
        for button in (self._rename_button, self._clone_button, self._delete_button):
            button.enabled = stored

    def destroy(self) -> None:
        """Release subscriptions, the prompt and the widgets."""
        self._settings_sub = None
        self._index_sub = None
        self._close_prompt()
        self._names_model.destroy()
        self._rename_button = None
        self._clone_button = None
        self._delete_button = None
        self._root = None

    def __build_ui(self):
        with self._root:
            ui.ComboBox(
                self._names_model,
                height=self._ROW_HEIGHT,
                tooltip="Brush preset to load",
                identifier="scatter_preset_combo",
            )
            with ui.HStack(height=self._ROW_HEIGHT, spacing=self._BUTTON_SPACING):
                ui.Button(
                    "Save",
                    clicked_fn=self._save,
                    tooltip="Store the current settings in the selected preset",
                    identifier="scatter_preset_save",
                )
                ui.Button(
                    "Save As",
                    clicked_fn=self._save_as,
                    tooltip="Store the current settings as a new preset",
                    identifier="scatter_preset_save_as",
                )
                self._rename_button = ui.Button(
                    "Rename",
                    clicked_fn=self._rename,
                    tooltip="Rename the selected preset",
                    identifier="scatter_preset_rename",
                )
                self._clone_button = ui.Button(
                    "Clone",
                    clicked_fn=self._clone,
                    tooltip="Duplicate the selected preset under a new name",
                    identifier="scatter_preset_clone",
                )
                self._delete_button = ui.Button(
                    "Delete",
                    clicked_fn=self._delete,
                    tooltip="Delete the selected preset file",
                    identifier="scatter_preset_delete",
                )

    def _save(self) -> None:
        """Store the current settings under the current preset name."""
        self._controller.save_preset()
        self.refresh()

    def _save_as(self) -> None:
        """Ask for a name and store the current settings under it."""
        self._prompt(
            "Save Preset As", "Name of the new preset", self._controller.settings.preset_name, self._save_as_named
        )

    def _rename(self) -> None:
        """Ask for a name and rename the current preset to it."""
        self._prompt("Rename Preset", "New name of the preset", self._controller.settings.preset_name, self._rename_to)

    def _clone(self) -> None:
        """Ask for a name and duplicate the current preset under it."""
        current = self._controller.settings.preset_name
        self._prompt("Clone Preset", "Name of the copy", f"{current} Copy", self._clone_to)

    def _delete(self) -> None:
        """Ask for confirmation, then delete the current preset."""
        name = self._controller.settings.preset_name
        TrexMessageDialog(
            title="Delete Preset",
            message=f"Delete the preset '{name}'? This cannot be undone.",
            ok_label="Delete",
            ok_handler=self._delete_current,
        )

    def _save_as_named(self, name: str) -> None:
        """Store the current settings under a new name."""
        self._controller.save_preset(name)
        self.refresh()

    def _rename_to(self, name: str) -> None:
        """Rename the current preset."""
        self._controller.rename_preset(self._controller.settings.preset_name, name)
        self.refresh()

    def _clone_to(self, name: str) -> None:
        """Duplicate the current preset."""
        self._controller.clone_preset(self._controller.settings.preset_name, name)
        self.refresh()

    def _delete_current(self) -> None:
        """Delete the current preset file."""
        self._controller.delete_preset(self._controller.settings.preset_name)
        self.refresh()

    def _prompt(self, title: str, message: str, initial_name: str, accept_fn: Callable[[str], None]) -> None:
        """Replace any open name prompt with a new one."""
        self._close_prompt()
        self._name_prompt = _NamePrompt(title, message, initial_name, accept_fn)

    def _close_prompt(self) -> None:
        """Close the name prompt when one is open."""
        if self._name_prompt is not None:
            self._name_prompt.destroy()
        self._name_prompt = None

    def _on_preset_selected(self, model: ui.AbstractValueModel) -> None:
        """Apply the preset the user picked; a failed load falls back to the current selection."""
        if self._refreshing:
            return
        name = self._names_model.choices[model.as_int]
        if name == self._controller.settings.preset_name:
            return
        if not self._controller.apply_preset(name):
            self.refresh()

    def _on_settings_changed(self, settings: ScatterBrushSettings) -> None:
        """Follow the preset name of the new settings."""
        if settings.preset_name != self._names_model.current_choice:
            self.refresh()
