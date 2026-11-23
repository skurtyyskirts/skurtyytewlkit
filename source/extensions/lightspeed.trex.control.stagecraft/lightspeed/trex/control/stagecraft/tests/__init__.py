"""
* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .e2e.test_capture_repair_on_open import TestCaptureRepairOnOpen
from .e2e.test_capture_swap_undo import TestCaptureSwapUndo
from .e2e.test_hotkeys import TestHotkeys
from .e2e.test_lighting import TestLightingRigSwap, TestModSkyLightRigs
from .e2e.test_stage_manager import TestStageManagerPropertiesInteraction
from .e2e.test_startup_chain import TestStartupChain
from .unit.test_commands import TestCommands
from .unit.test_extension import TestExtensionLifecycle
from .unit.test_light_rigs import TestLightRigMenu
from .unit.test_mod_skies import TestModSky, TestModSkyRigs, TestReadModSkies, TestWriteModSkyRig
from .unit.test_setup import TestSetup as TestSetupUnit
from .unit.test_stage_manager_schema import TestStageManagerSchema

__all__ = [
    "TestCaptureRepairOnOpen",
    "TestCaptureSwapUndo",
    "TestCommands",
    "TestExtensionLifecycle",
    "TestHotkeys",
    "TestLightRigMenu",
    "TestLightingRigSwap",
    "TestModSky",
    "TestModSkyLightRigs",
    "TestModSkyRigs",
    "TestReadModSkies",
    "TestSetupUnit",
    "TestStageManagerPropertiesInteraction",
    "TestStageManagerSchema",
    "TestStartupChain",
    "TestWriteModSkyRig",
]
