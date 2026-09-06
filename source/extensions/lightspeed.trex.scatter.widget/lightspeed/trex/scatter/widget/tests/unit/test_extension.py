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

import contextlib
import re
import tomllib
from pathlib import Path
from unittest import mock

import omni.kit.test
from lightspeed.trex.contexts.setup import Contexts as _TrexContexts
from lightspeed.trex.scatter.widget import extension as _extension_module

_EXTENSION_ID = "lightspeed.trex.scatter.widget-1.0.0"
_WINDOW_TITLE = "Scatter"
_PACKAGE_DIR = Path(_extension_module.__file__).parent
_EXTENSION_TOML = _PACKAGE_DIR.parents[3] / "config" / "extension.toml"


class TestTrexScatterWindowExtension(omni.kit.test.AsyncTestCase):
    """Tests the Scatter window extension lifecycle."""

    async def setUp(self):
        self._window = mock.MagicMock(title=_WINDOW_TITLE)
        self._patches = contextlib.ExitStack()
        self._window_class = self._patches.enter_context(
            mock.patch.object(_extension_module, "_ScatterWindow", return_value=self._window)
        )
        self._set_show_window_fn = self._patches.enter_context(
            mock.patch.object(_extension_module.omni.ui.Workspace, "set_show_window_fn")
        )
        self._extension = _extension_module.TrexScatterWindowExtension()

    async def tearDown(self):
        self._patches.close()

    async def test_on_startup_creates_window_for_stagecraft_context_and_registers_show_fn(self):
        """Startup builds the Scatter window for StageCraft and lets the workspace show it."""
        # Arrange

        # Act
        self._extension.on_startup(_EXTENSION_ID)

        # Assert
        self._window_class.assert_called_once_with(_TrexContexts.STAGE_CRAFT.value)
        self._window.create_window.assert_called_once_with()
        self._set_show_window_fn.assert_called_once_with(_WINDOW_TITLE, self._window.show_window_fn)
        self.assertIs(self._extension._workspace_window, self._window)

    async def test_on_shutdown_after_startup_cleans_up_window_and_resets_show_fn(self):
        """Shutdown releases the window and replaces the workspace show function with a no-op."""
        # Arrange
        self._extension.on_startup(_EXTENSION_ID)
        self._set_show_window_fn.reset_mock()

        # Act
        self._extension.on_shutdown()

        # Assert
        self._window.cleanup.assert_called_once_with()
        self._set_show_window_fn.assert_called_once()
        title, show_fn = self._set_show_window_fn.call_args.args
        self.assertEqual(title, _WINDOW_TITLE)
        self.assertIsNot(show_fn, self._window.show_window_fn)
        self.assertTrue(callable(show_fn))
        self.assertIsNone(self._extension._workspace_window)

    async def test_on_shutdown_called_twice_cleans_up_window_once(self):
        """A repeated shutdown finds no window and leaves the workspace registration alone."""
        # Arrange
        self._extension.on_startup(_EXTENSION_ID)
        self._extension.on_shutdown()
        self._set_show_window_fn.reset_mock()

        # Act
        self._extension.on_shutdown()

        # Assert
        self._window.cleanup.assert_called_once_with()
        self._set_show_window_fn.assert_not_called()
        self.assertIsNone(self._extension._workspace_window)

    async def test_declared_dependencies_are_imported_by_the_extension_source(self):
        """Every extension under [dependencies] is imported by a module outside tests, so the list carries no leftovers."""
        # Arrange
        with _EXTENSION_TOML.open("rb") as toml_file:
            dependencies = tomllib.load(toml_file)["dependencies"]
        sources = "\n".join(path.read_text(encoding="utf-8") for path in _PACKAGE_DIR.glob("*.py"))

        # Act
        unused = [
            name
            for name in dependencies
            if re.search(rf"^\s*(?:from|import)\s+{re.escape(name)}(?:[.\s]|$)", sources, re.MULTILINE) is None
        ]

        # Assert
        self.assertEqual(unused, [])
