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

__all__ = ["DragValueModel", "field_bounds"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from omni import ui

if TYPE_CHECKING:
    from pydantic import BaseModel


def field_bounds(model_type: type[BaseModel], field_name: str) -> tuple[float | int | None, float | int | None]:
    """Return the inclusive ``(low, high)`` bounds a pydantic model declares on a field, None where it has none.

    The bounded drag widgets use them as hard clamps so a field can never hold a value the model would reject.

    Args:
        model_type: Pydantic model class declaring the field.
        field_name: Name of the field whose ``ge`` / ``le`` constraints are read.
    """
    low = high = None
    for constraint in model_type.model_fields[field_name].metadata:
        low = getattr(constraint, "ge", low)
        high = getattr(constraint, "le", high)
    return low, high


class DragValueModel(ui.AbstractValueModel):
    """Numeric value model that ``FloatBoundedDrag`` and ``IntBoundedDrag`` can clamp.

    The bounded drag widgets install their hard clamp through ``set_callback_pre_set_value``, which
    ``omni.ui.SimpleFloatModel`` and ``omni.ui.SimpleIntModel`` do not offer. This model keeps one plain number and
    routes every ``set_value`` through that hook. It does not batch edits, so the drag widgets apply each edit
    directly.
    """

    def __init__(self, value: float | int):
        """Create the model with its initial number.

        Args:
            value: Initial value; callers keep its type (int or float) for every later ``set_value``.
        """
        super().__init__()
        self._value = value
        self._pre_set_callback: Callable[[Callable[[Any], None], Any], None] | None = None

    @property
    def supports_batch_edit(self) -> bool:
        """Whether edits can be grouped; this model applies every edit directly."""
        return False

    @property
    def is_batch_editing(self) -> bool:
        """Whether a batch edit is open; never, since batching is unsupported."""
        return False

    def begin_batch_edit(self) -> None:
        """Batching is unsupported, so there is nothing to open."""

    def end_batch_edit(self) -> None:
        """Batching is unsupported, so there is nothing to close."""

    def set_callback_pre_set_value(self, callback: Callable[[Callable[[Any], None], Any], None] | None) -> None:
        """Install the hook that decides how a requested value reaches the model.

        Args:
            callback: Receives the function that stores a value and the requested value, and calls the former with
                the value it accepts. None removes the hook.
        """
        self._pre_set_callback = callback

    def get_value_as_float(self) -> float:
        """Return the value as a float."""
        return float(self._value)

    def get_value_as_int(self) -> int:
        """Return the value as an int."""
        return int(self._value)

    def get_value_as_bool(self) -> bool:
        """Return whether the value is not zero."""
        return bool(self._value)

    def get_value_as_string(self) -> str:
        """Return the value as text."""
        return str(self._value)

    def set_value(self, value: float | int) -> None:
        """Store a value, through the pre-set hook when one is installed.

        Args:
            value: Requested value.
        """
        if self._pre_set_callback is None:
            self._store_value(value)
        else:
            self._pre_set_callback(self._store_value, value)

    def _store_value(self, value: float | int) -> None:
        """Store the value and notify subscribers when it differs from the current one."""
        if value == self._value:
            return
        self._value = value
        self._value_changed()
