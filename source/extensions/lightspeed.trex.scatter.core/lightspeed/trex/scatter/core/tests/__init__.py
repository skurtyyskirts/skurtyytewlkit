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

from .e2e.test_commands_project import TestScatterCommandsOnProjectExample
from .unit.test_assets import TestAssets
from .unit.test_commands import TestScatterFloodCommand, TestScatterStrokeCommand
from .unit.test_controller import TestScatterBrushController, TestScatterBrushControllerSingleton
from .unit.test_geometry import TestGeometryFunctions, TestMeshGeometry, TestMeshSurfaceCache
from .unit.test_picking import (
    TestCameraRayFromNdc,
    TestCpuRaySurfacePicker,
    TestCreateSurfacePicker,
    TestHdRemixSurfacePicker,
)
from .unit.test_placement import TestPlacementFunctions, TestPlacementRecord
from .unit.test_presets import TestGetDefaultPresetsDirectory, TestPresetStore
from .unit.test_sampling import TestPaddingIndex, TestSamplingFunctions
from .unit.test_settings import TestAssetsCarbPersistence, TestScatterAssetEntry, TestScatterBrushSettings
from .unit.test_stroke import TestStrokeSession
from .unit.test_targets import TestTargets

__all__ = (
    "TestAssets",
    "TestAssetsCarbPersistence",
    "TestCameraRayFromNdc",
    "TestCpuRaySurfacePicker",
    "TestCreateSurfacePicker",
    "TestGeometryFunctions",
    "TestGetDefaultPresetsDirectory",
    "TestHdRemixSurfacePicker",
    "TestMeshGeometry",
    "TestMeshSurfaceCache",
    "TestPaddingIndex",
    "TestPlacementFunctions",
    "TestPlacementRecord",
    "TestPresetStore",
    "TestSamplingFunctions",
    "TestScatterAssetEntry",
    "TestScatterBrushController",
    "TestScatterBrushControllerSingleton",
    "TestScatterCommandsOnProjectExample",
    "TestScatterFloodCommand",
    "TestScatterStrokeCommand",
    "TestStrokeSession",
    "TestTargets",
)
