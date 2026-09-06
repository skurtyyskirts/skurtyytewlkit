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

__all__ = ["TestDragValueModel", "TestFieldBounds"]

from unittest import mock

import omni.kit.app
import omni.kit.test
from lightspeed.trex.scatter.core import ScatterAssetEntry, ScatterBrushSettings
from lightspeed.trex.scatter.widget.value_model import DragValueModel, field_bounds
from omni import ui
from omni.flux.utils.widget.drag_field import FloatBoundedDrag, IntBoundedDrag


class TestFieldBounds(omni.kit.test.AsyncTestCase):
    """Tests the pydantic bound lookup the drag fields clamp with."""

    async def test_field_bounds_returns_declared_ge_and_le(self):
        # Arrange
        field_name = "weight"

        # Act
        bounds = field_bounds(ScatterAssetEntry, field_name)

        # Assert
        self.assertEqual(bounds, (0.0, 100.0))

    async def test_field_bounds_for_unconstrained_field_returns_none_pair(self):
        # Arrange
        field_name = "preset_name"

        # Act
        bounds = field_bounds(ScatterBrushSettings, field_name)

        # Assert
        self.assertEqual(bounds, (None, None))


class TestDragValueModel(omni.kit.test.AsyncTestCase):
    """Tests the numeric value model behind the pane's bounded drag fields."""

    async def test_set_value_without_hook_stores_value_and_notifies(self):
        # Arrange
        model = DragValueModel(1.0)
        callback = mock.Mock()
        subscription = model.subscribe_value_changed_fn(callback)

        # Act
        model.set_value(2.5)

        # Assert
        self.assertEqual(model.as_float, 2.5)
        callback.assert_called_once()
        self.assertIsNotNone(subscription)

    async def test_set_value_with_same_value_does_not_notify(self):
        # Arrange
        model = DragValueModel(1.0)
        callback = mock.Mock()
        subscription = model.subscribe_value_changed_fn(callback)

        # Act
        model.set_value(1.0)

        # Assert
        callback.assert_not_called()
        self.assertIsNotNone(subscription)

    async def test_set_value_with_hook_stores_value_the_hook_passes_on(self):
        # Arrange
        model = DragValueModel(1.0)
        model.set_callback_pre_set_value(lambda store, value: store(min(value, 5.0)))

        # Act
        model.set_value(9.0)

        # Assert
        self.assertEqual(model.as_float, 5.0)

    async def test_set_value_with_hook_that_drops_value_keeps_current_value(self):
        # Arrange
        model = DragValueModel(1.0)
        model.set_callback_pre_set_value(lambda store, value: None)

        # Act
        model.set_value(9.0)

        # Assert
        self.assertEqual(model.as_float, 1.0)

    async def test_set_callback_pre_set_value_none_removes_hook(self):
        # Arrange
        model = DragValueModel(1.0)
        model.set_callback_pre_set_value(lambda store, value: None)
        model.set_callback_pre_set_value(None)

        # Act
        model.set_value(9.0)

        # Assert
        self.assertEqual(model.as_float, 9.0)

    async def test_get_value_as_int_truncates_float_value(self):
        # Arrange
        model = DragValueModel(2.75)

        # Act
        value = model.as_int

        # Assert
        self.assertEqual(value, 2)

    async def test_get_value_as_string_returns_number_text(self):
        # Arrange
        model = DragValueModel(42)

        # Act
        value = model.as_string

        # Assert
        self.assertEqual(value, "42")

    async def test_get_value_as_bool_is_false_for_zero(self):
        # Arrange
        model = DragValueModel(0.0)

        # Act
        value = model.as_bool

        # Assert
        self.assertFalse(value)

    async def test_supports_batch_edit_is_false(self):
        # Arrange
        model = DragValueModel(1.0)

        # Act
        supported = model.supports_batch_edit

        # Assert
        self.assertFalse(supported)
        self.assertFalse(model.is_batch_editing)

    async def test_float_bounded_drag_clamps_value_to_hard_bounds(self):
        # Arrange
        window = ui.Window("DragValueModelTest_float", width=200, height=100)
        model = DragValueModel(5.0)
        with window.frame:
            drag = FloatBoundedDrag(model=model, hard_min_value=1.0, hard_max_value=10.0, enable_batch_edit=False)
        await omni.kit.app.get_app().next_update_async()

        try:
            # Act
            model.set_value(50.0)

            # Assert
            self.assertEqual(model.as_float, 10.0)
        finally:
            drag.destroy()
            window.destroy()

    async def test_int_bounded_drag_clamps_value_to_hard_bounds(self):
        # Arrange
        window = ui.Window("DragValueModelTest_int", width=200, height=100)
        model = DragValueModel(5)
        with window.frame:
            drag = IntBoundedDrag(model=model, hard_min_value=0, hard_max_value=100, enable_batch_edit=False)
        await omni.kit.app.get_app().next_update_async()

        try:
            # Act
            model.set_value(-7)

            # Assert
            self.assertEqual(model.as_int, 0)
        finally:
            drag.destroy()
            window.destroy()
