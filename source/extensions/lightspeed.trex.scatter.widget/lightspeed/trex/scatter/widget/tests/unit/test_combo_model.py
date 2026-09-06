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

__all__ = ["TestChoicesComboModel"]

from unittest import mock

import omni.kit.test
from lightspeed.trex.scatter.widget.combo_model import ChoicesComboModel


class TestChoicesComboModel(omni.kit.test.AsyncTestCase):
    """Tests the refreshable combo box model."""

    async def test_init_with_choices_exposes_entries_and_selected_index(self):
        # Arrange
        choices = ["Grass", "Rocks", "Pebbles"]

        # Act
        model = ChoicesComboModel(choices, 1)

        # Assert
        self.assertEqual(model.choices, choices)
        self.assertEqual(model.index_model.as_int, 1)
        self.assertEqual(model.current_choice, "Rocks")
        self.assertEqual(len(model.get_item_children(None)), 3)

    async def test_set_choices_replaces_entries_and_notifies_root_change(self):
        # Arrange
        model = ChoicesComboModel(["Grass"])
        item_changed = mock.Mock()
        subscription = model.subscribe_item_changed_fn(item_changed)

        # Act
        model.set_choices(["Rocks", "Pebbles"], 1)

        # Assert
        self.assertEqual(model.choices, ["Rocks", "Pebbles"])
        self.assertEqual(model.current_choice, "Pebbles")
        item_changed.assert_called()
        self.assertIsNotNone(subscription)

    async def test_set_choices_when_index_out_of_range_clamps_to_last_entry(self):
        # Arrange
        model = ChoicesComboModel()

        # Act
        model.set_choices(["Grass", "Rocks"], 5)

        # Assert
        self.assertEqual(model.index_model.as_int, 1)
        self.assertEqual(model.current_choice, "Rocks")

    async def test_set_choices_with_no_entries_resets_index_and_current_choice(self):
        # Arrange
        model = ChoicesComboModel(["Grass", "Rocks"], 1)

        # Act
        model.set_choices([])

        # Assert
        self.assertEqual(model.choices, [])
        self.assertEqual(model.index_model.as_int, 0)
        self.assertIsNone(model.current_choice)

    async def test_get_item_children_for_entry_returns_no_children(self):
        # Arrange
        model = ChoicesComboModel(["Grass"])
        entry = model.get_item_children(None)[0]

        # Act
        children = model.get_item_children(entry)

        # Assert
        self.assertEqual(children, [])

    async def test_get_item_value_model_returns_index_for_root_and_label_for_entry(self):
        # Arrange
        model = ChoicesComboModel(["Grass", "Rocks"])
        entry = model.get_item_children(None)[1]

        # Act
        root_model = model.get_item_value_model(None, 0)
        entry_model = model.get_item_value_model(entry, 0)

        # Assert
        self.assertIs(root_model, model.index_model)
        self.assertEqual(entry_model.as_string, "Rocks")
        self.assertEqual(model.get_item_value_model_count(None), 1)

    async def test_index_change_notifies_root_change_so_combo_redraws(self):
        # Arrange
        model = ChoicesComboModel(["Grass", "Rocks"])
        item_changed = mock.Mock()
        subscription = model.subscribe_item_changed_fn(item_changed)

        # Act
        model.index_model.set_value(1)

        # Assert
        item_changed.assert_called_once()
        self.assertEqual(model.current_choice, "Rocks")
        self.assertIsNotNone(subscription)

    async def test_destroy_drops_entries(self):
        # Arrange
        model = ChoicesComboModel(["Grass", "Rocks"])

        # Act
        model.destroy()

        # Assert
        self.assertEqual(model.choices, [])
        self.assertIsNone(model.current_choice)
