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

__all__ = ["ScatterWindow"]

from lightspeed.common.constants import WindowNames as _WindowNames
from lightspeed.trex.utils.widget.workspace import WorkspaceWindowBase as _WorkspaceWindowBase
from omni import ui

from .setup_ui import ScatterPane as _ScatterPane


class ScatterWindow(_WorkspaceWindowBase):
    """Scatter brush window manager."""

    @property
    def title(self) -> str:
        return _WindowNames.SCATTER

    @property
    def flags(self) -> int:
        return ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_COLLAPSE

    def _create_window_ui(self):
        return _ScatterPane(self._usd_context_name)
