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

__all__ = [
    "ASSETS_SETTING_PATH",
    "BRUSH_SETTINGS_PATH",
    "CONTAINER_PREFIX",
    "DEFAULT_MAX_PICK_DISTANCE",
    "DEFAULT_PRESET_NAME",
    "EXTENSION_NAME",
    "FORCE_CPU_PICKER_SETTING",
    "IS_REMIX_SCATTER_ATTR",
    "MAX_PICK_DISTANCE_SETTING",
    "PLACEMENT_PREFIX",
    "PRESETS_DIR_SETTING",
    "PRESETS_SUBDIRECTORY",
    "SCATTER_ASSET_ATTR",
    "SCATTER_BRUSH_ID_ATTR",
    "SCENE_FACTORY_ID",
    "TOOLBAR_PRIORITY",
]

EXTENSION_NAME = "lightspeed.trex.scatter.core"

# Prim attributes authored on scattered prims. The container carries the marker only; placements carry all three.
IS_REMIX_SCATTER_ATTR = "IsRemixScatter"
SCATTER_BRUSH_ID_ATTR = "remixScatterBrushId"
SCATTER_ASSET_ATTR = "remixScatterAsset"

# Prim naming: <prototype>/scatter_<preset slug>/s_<uuid>
CONTAINER_PREFIX = "scatter_"
PLACEMENT_PREFIX = "s_"

# Viewport integration constants shared with lightspeed.trex.viewports.shared.widget
SCENE_FACTORY_ID = "omni.kit.lss.viewport.tools.scatter_brush"
TOOLBAR_PRIORITY = 13

# Carb settings
FORCE_CPU_PICKER_SETTING = f"/exts/{EXTENSION_NAME}/forceCpuPicker"
PRESETS_DIR_SETTING = f"/exts/{EXTENSION_NAME}/presetsDirectory"
BRUSH_SETTINGS_PATH = f"/persistent/exts/{EXTENSION_NAME}/brushSettings"
ASSETS_SETTING_PATH = f"/persistent/exts/{EXTENSION_NAME}/assets"
# Shared with the teleport tool: picks farther than this from the camera are treated as sky-dome misses.
MAX_PICK_DISTANCE_SETTING = "/app/viewport/teleport/max_pick_distance_from_camera"
DEFAULT_MAX_PICK_DISTANCE = 1.0e5

DEFAULT_PRESET_NAME = "Default"
PRESETS_SUBDIRECTORY = "rtx-remix/scatter_presets"
