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

__all__ = ["ChoicesComboModel"]

from collections.abc import Sequence

from omni import ui


class _ChoiceItem(ui.AbstractItem):
    """One selectable entry of a ``ChoicesComboModel``."""

    def __init__(self, text: str):
        super().__init__()
        self.model = ui.SimpleStringModel(text)


class ChoicesComboModel(ui.AbstractItemModel):
    """``ui.ComboBox`` model over a list of strings that can be replaced while the widget is alive.

    The selected index lives in ``index_model``; subscribe to it to react to user selections and set it to change the
    selection programmatically. ``set_choices`` replaces the entries in place so the combo box keeps its identity and
    subscriptions.
    """

    def __init__(self, choices: Sequence[str] = (), index: int = 0):
        super().__init__()
        self._items: list[_ChoiceItem] = []
        self._index = ui.SimpleIntModel(0)
        # The combo box caches its displayed entry; a root change notification makes it re-read the index.
        self._index_sub = self._index.subscribe_value_changed_fn(lambda _model: self._item_changed(None))
        self.set_choices(choices, index)

    @property
    def index_model(self) -> ui.SimpleIntModel:
        """Model holding the selected index."""
        return self._index

    @property
    def choices(self) -> list[str]:
        """Current entries in display order."""
        return [item.model.as_string for item in self._items]

    @property
    def current_choice(self) -> str | None:
        """Selected entry, or None when there are no entries."""
        if not self._items:
            return None
        return self._items[self._index.as_int].model.as_string

    def set_choices(self, choices: Sequence[str], index: int = 0) -> None:
        """Replace the entries and select ``index``, clamped to the new range."""
        self._items = [_ChoiceItem(text) for text in choices]
        self._index.set_value(max(0, min(index, len(self._items) - 1)) if self._items else 0)
        self._item_changed(None)

    def get_item_children(self, item: ui.AbstractItem | None = None) -> list[_ChoiceItem]:
        """Return the entries for the root and nothing for an entry."""
        return list(self._items) if item is None else []

    def get_item_value_model(self, item: ui.AbstractItem | None = None, column_id: int = 0) -> ui.AbstractValueModel:
        """Return the selected-index model for the root and the label model for an entry."""
        return self._index if item is None else item.model

    def get_item_value_model_count(self, item: ui.AbstractItem | None = None) -> int:
        """Return the single label column."""
        return 1

    def destroy(self) -> None:
        """Drop the entries and the index subscription."""
        self._index_sub = None
        self._items = []
