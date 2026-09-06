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

__all__ = ["ScatterCoreExtension"]

import carb
import omni.ext
import omni.kit.commands

from . import commands as _commands
from .controller import destroy_scatter_brush_controller, get_scatter_brush_controller


class ScatterCoreExtension(omni.ext.IExt):
    """Registers the scatter commands and owns the brush controller singleton."""

    def on_startup(self, ext_id: str):
        carb.log_info("[lightspeed.trex.scatter.core] Startup")
        omni.kit.commands.register_all_commands_in_module(_commands)
        get_scatter_brush_controller()

    def on_shutdown(self):
        carb.log_info("[lightspeed.trex.scatter.core] Shutdown")
        destroy_scatter_brush_controller()
        omni.kit.commands.unregister_module_commands(_commands)
