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

from unittest import mock

import omni.kit.test
from lightspeed.common.constants import WindowNames
from lightspeed.trex.scatter.widget import workspace as _workspace_module
from omni import ui

_CONTEXT_NAME = "scatter_workspace_test_context"


class TestScatterWindow(omni.kit.test.AsyncTestCase):
    """Tests the Scatter workspace window definition."""

    async def test_title_returns_scatter_window_name(self):
        """The window registers under the shared Scatter window name."""
        # Arrange
        window = _workspace_module.ScatterWindow(_CONTEXT_NAME)

        # Act
        title = window.title

        # Assert
        self.assertEqual(title, WindowNames.SCATTER)
        self.assertEqual(title, "Scatter")

    async def test_flags_include_no_scrollbar_and_no_collapse(self):
        """The window lets the pane own scrolling and cannot be collapsed."""
        # Arrange
        window = _workspace_module.ScatterWindow(_CONTEXT_NAME)

        # Act
        flags = window.flags

        # Assert
        self.assertEqual(flags & ui.WINDOW_FLAGS_NO_SCROLLBAR, ui.WINDOW_FLAGS_NO_SCROLLBAR)
        self.assertEqual(flags & ui.WINDOW_FLAGS_NO_COLLAPSE, ui.WINDOW_FLAGS_NO_COLLAPSE)

    async def test_create_window_ui_returns_scatter_pane_for_context(self):
        """The window content is a Scatter pane bound to the window's USD context."""
        # Arrange
        window = _workspace_module.ScatterWindow(_CONTEXT_NAME)

        with mock.patch.object(_workspace_module, "_ScatterPane") as pane_class:
            # Act
            content = window._create_window_ui()

        # Assert
        pane_class.assert_called_once_with(_CONTEXT_NAME)
        self.assertIs(content, pane_class.return_value)
