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

__all__ = ["AssetUpAxisWidget", "file_axis_text"]

import functools
from collections.abc import Sequence
from pathlib import PurePosixPath

from lightspeed.trex.scatter.core import (
    ScatterAssetEntry,
    UpAxis,
    get_scatter_brush_controller,
    read_authored_asset_up_axis,
)
from omni import ui

from .combo_model import ChoicesComboModel

_UP_AXES = tuple(UpAxis)
_AXIS_CHOICES = [f"{axis.value}-up" for axis in _UP_AXES]
_ROW_HEIGHT = ui.Pixel(24)
_SPACING = ui.Pixel(8)
_AXIS_WIDTH = ui.Pixel(72)
_TEXT_STYLE = "PropertiesPaneSectionTreeItem"
_WARNING_STYLE = "Warning"
_EMPTY_TEXT = "Add assets under ASSETS to see which way each one stands"
_TOOLTIP = (
    "Which axis each asset was modelled standing along, so the brush can stand it straight up on the stage.\n\n"
    "- File is what the asset file itself says\n"
    "- The choice beside it is what the brush uses; set it to match the file\n"
    "- A choice that disagrees with the file scatters the asset lying on its side\n"
    "- A file that says nothing is usually Y-up; switch to Z-up if the asset still lies down"
)


def file_axis_text(entry: ScatterAssetEntry, file_axis: UpAxis | None) -> str:
    """Return what a row says about the axis the asset's file authors.

    Args:
        entry: Palette entry the row shows.
        file_axis: Up axis the asset file authors, or None when it authors none.

    Returns:
        The file's axis, followed by the fix when the brush disagrees with it.
    """
    if file_axis is None:
        return "File: not set"
    if UpAxis(entry.up_axis) == file_axis:
        return f"File: {file_axis.value}-up"
    return f"File: {file_axis.value}-up - set {file_axis.value} or it lies on its side"


class _AxisRow:
    """One palette asset: its name, what its file says and the up axis the brush uses for it."""

    def __init__(self, entry: ScatterAssetEntry, file_axis: UpAxis | None, index: int, label_width: ui.Length):
        self.path = entry.path
        self.file_axis = file_axis
        self.model = ChoicesComboModel(_AXIS_CHOICES, _UP_AXES.index(UpAxis(entry.up_axis)))
        with ui.HStack(height=_ROW_HEIGHT, spacing=_SPACING):
            ui.Label(
                PurePosixPath(entry.path).name,
                name=_TEXT_STYLE,
                width=label_width,
                elided_text=True,
                tooltip=entry.path,
                identifier=f"scatter_up_axis_row_{index}_name",
            )
            self.file_label = ui.Label(
                "", elided_text=True, tooltip=_TOOLTIP, identifier=f"scatter_up_axis_row_{index}_file"
            )
            ui.ComboBox(self.model, width=_AXIS_WIDTH, tooltip=_TOOLTIP, identifier=f"scatter_up_axis_row_{index}_axis")
        self.show(entry)

    def show(self, entry: ScatterAssetEntry) -> None:
        """Show an entry's up axis and flag it when it disagrees with the file."""
        index = _UP_AXES.index(UpAxis(entry.up_axis))
        if self.model.index_model.as_int != index:
            self.model.index_model.set_value(index)
        self.file_label.text = file_axis_text(entry, self.file_axis)
        mismatched = self.file_axis is not None and UpAxis(entry.up_axis) != self.file_axis
        self.file_label.name = _WARNING_STYLE if mismatched else _TEXT_STYLE

    def destroy(self) -> None:
        self.model.destroy()


class AssetUpAxisWidget:
    """Rows telling which way each palette asset was modelled, beside the up axis the brush stands it up by.

    Picking the wrong axis is the usual reason an asset is scattered lying down, and the choice otherwise sits at the
    far edge of the asset palette, so the Placement section shows it where orientation is tuned. Each row edits the
    same palette field as the palette's own Up column.
    """

    def __init__(self, label_width: ui.Length):
        """
        Args:
            label_width: Width of the asset name column, matching the label column of the surrounding rows.
        """
        self._controller = get_scatter_brush_controller()
        self._label_width = label_width
        self._rows: list[_AxisRow] = []
        self._row_subs: list = []
        # What each asset file authors, read once per path: a weight drag changes the palette at every step, and the
        # rows must not reopen asset files for it.
        self._file_axes: dict[str, UpAxis | None] = {}
        self._pushing = False
        self._root = ui.VStack(height=0, spacing=ui.Pixel(4))
        with self._root:
            self._empty_label = ui.Label(
                _EMPTY_TEXT, name=_TEXT_STYLE, height=_ROW_HEIGHT, identifier="scatter_up_axis_empty"
            )
            self._rows_frame = ui.Frame(height=0)
        self._assets_sub = self._controller.subscribe_assets_changed(self._on_assets_changed)
        self._on_assets_changed(self._controller.assets)

    def destroy(self) -> None:
        """Drop the controller subscription and the rows."""
        self._assets_sub = None
        self._clear_rows()
        self._root = None

    def _on_assets_changed(self, entries: Sequence[ScatterAssetEntry]) -> None:
        """Rebuild the rows when the palette gained or lost assets, otherwise update them in place."""
        if self._root is None:
            return
        if [entry.path for entry in entries] == [row.path for row in self._rows]:
            self._pushing = True
            try:
                for row, entry in zip(self._rows, entries, strict=True):
                    row.show(entry)
            finally:
                self._pushing = False
            return
        self._clear_rows()
        self._empty_label.visible = not entries
        self._rows_frame.clear()
        with self._rows_frame:
            with ui.VStack(height=0, spacing=ui.Pixel(4)):
                for index, entry in enumerate(entries):
                    row = _AxisRow(entry, self._file_axis(entry.path), index, self._label_width)
                    self._rows.append(row)
                    self._row_subs.append(
                        row.model.index_model.subscribe_value_changed_fn(functools.partial(self._on_axis_chosen, row))
                    )

    def _file_axis(self, path: str) -> UpAxis | None:
        """Return the up axis the asset file authors, reading the file only the first time."""
        if path not in self._file_axes:
            self._file_axes[path] = read_authored_asset_up_axis(path)
        return self._file_axes[path]

    def _on_axis_chosen(self, row: _AxisRow, model: ui.AbstractValueModel) -> None:
        """Hand an axis picked in a row to the palette; a refused one is put back."""
        if self._pushing:
            return
        if self._controller.set_asset_up_axis(row.path, _UP_AXES[model.as_int]):
            return
        entry = next((entry for entry in self._controller.assets if entry.path == row.path), None)
        if entry is not None:
            row.show(entry)

    def _clear_rows(self) -> None:
        self._row_subs.clear()
        for row in self._rows:
            row.destroy()
        self._rows = []
