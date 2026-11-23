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
    "GIZMO_SCENE_FACTORY_ID",
    "GROUP_PRESET_PREFIX",
    "INSTANCED_KEYS_ATTR",
    "INSTANCED_ORIGINS_ATTR",
    "INSTANCED_ROTATIONS_ATTR",
    "INSTANCED_TRANSFORM_KEYS_ATTR",
    "INSTANCED_TRANSLATES_ATTR",
    "INSTANCED_TURNS_ATTR",
    "INSTANCED_YAWS_ATTR",
    "INSTANCER_PREFIX",
    "IS_REMIX_SCATTER_ATTR",
    "MAX_PICK_DISTANCE_SETTING",
    "PLACEMENT_PREFIX",
    "PRESETS_DIR_SETTING",
    "PRESETS_SUBDIRECTORY",
    "PREVIEW_ENTRY_PREFIX",
    "PREVIEW_SUFFIX",
    "PROTOTYPES_SCOPE_NAME",
    "PROTOTYPE_PREFIX",
    "SCATTER_ASSET_ATTR",
    "SCATTER_BRUSH_ID_ATTR",
    "SCATTER_CONTAINER_ATTR",
    "SCATTER_GROUPS_LAYER_DATA_KEY",
    "SCATTER_KEY_ATTR",
    "SCATTER_ORIGIN_ATTR",
    "SCATTER_PREVIEW_OF_ATTR",
    "SCATTER_TRANSFORM_KEY_ATTR",
    "SCATTER_YAW_ATTR",
    "SCENE_FACTORY_ID",
    "SURFACE_PREVIEWS_SUBDIRECTORY",
    "TOOLBAR_PRIORITY",
]

EXTENSION_NAME = "lightspeed.trex.scatter.core"

# Prim attributes authored on scattered prims. The container carries the marker and the brush id; placements carry
# the marker, the asset path and the three attributes a re-apply needs to recompose them.
IS_REMIX_SCATTER_ATTR = "IsRemixScatter"
SCATTER_BRUSH_ID_ATTR = "remixScatterBrushId"
SCATTER_ASSET_ATTR = "remixScatterAsset"
# Parent-space surface point, surface normal and, when the stamp had one, the stroke heading: 6 or 9 doubles.
SCATTER_ORIGIN_ATTR = "remixScatterOrigin"
# The placement's own random key, so its rotation and scale draws can be reproduced without replaying its stamp.
SCATTER_KEY_ATTR = "remixScatterKey"
# Checksum of the transform the brush authored, so a transform the user changed by hand can be left alone.
SCATTER_TRANSFORM_KEY_ATTR = "remixScatterTransformKey"
# Degrees about the stage up axis the move gizmo has turned this placement by, added to the yaw it samples. Kept
# because a recomposition reproduces the position and the surface it settled on, but not a turn the user dragged.
SCATTER_YAW_ATTR = "remixScatterYaw"
# The whole turn the move gizmo and the Scatter window's turn fields have given a placement, as X, Y, Z degrees
# in the space of the instance that parents it. Recorded beside the yaw rather than replacing it so that scatter
# painted before this existed recomposes to exactly what it does today.
SCATTER_TURN_ATTR = "remixScatterTurn"

# Prim naming: <prototype>/scatter_<preset slug>/s_<uuid>
CONTAINER_PREFIX = "scatter_"
PLACEMENT_PREFIX = "s_"

# Instanced form of a patch, a sibling of its container: <prototype>/instanced_<preset slug>, a PointInstancer whose
# prototypes live in its own Prototypes scope as p_<signature hash>. Kept beside the container rather than inside it
# so a container still holds nothing but loose placements, which is what lets a stroke be removed or restored in one
# edit of its container.
INSTANCER_PREFIX = "instanced_"
# Name of the container an instancer's entries split back into, authored on the instancer.
SCATTER_CONTAINER_ATTR = "remixScatterContainer"
# The toolkit viewport's stand-in for an instancer, authored only in the session layer: <instancer name>_viewport,
# holding one e_<id> prim per entry. HdRemix draws a PointInstancer's prototypes once each and none of its instances,
# so the toolkit hides the instancer there and shows these instead; the runtime that plays the mod never sees them.
PREVIEW_SUFFIX = "_viewport"
PREVIEW_ENTRY_PREFIX = "e_"
# Path of the instancer a viewport stand-in shows, authored on the stand-in's container.
SCATTER_PREVIEW_OF_ATTR = "remixScatterPreviewOf"
PROTOTYPES_SCOPE_NAME = "Prototypes"
PROTOTYPE_PREFIX = "p_"
# Per-entry arrays an instancer keeps beside the ones it renders with, so that splitting an entry back into a
# placement gives back exactly what the placement recorded: the authored ops at full precision, the surface, the
# random key, the transform checksum and the gizmo turns. Missing values are NaN, or 0 for the two keys.
INSTANCED_TRANSLATES_ATTR = "remixScatter:translates"
INSTANCED_ROTATIONS_ATTR = "remixScatter:rotations"
INSTANCED_ORIGINS_ATTR = "remixScatter:origins"
INSTANCED_KEYS_ATTR = "remixScatter:keys"
INSTANCED_TRANSFORM_KEYS_ATTR = "remixScatter:transformKeys"
INSTANCED_YAWS_ATTR = "remixScatter:yaws"
INSTANCED_TURNS_ATTR = "remixScatter:turns"

# Viewport integration constants shared with lightspeed.trex.viewports.shared.widget
SCENE_FACTORY_ID = "omni.kit.lss.viewport.tools.scatter_brush"
GIZMO_SCENE_FACTORY_ID = "omni.kit.lss.viewport.tools.scatter_gizmo"
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

# Scatter groups: named floods an agent creates and retunes through the REST API. Each group records its request in
# the custom layer data of the layer it authors, so it can be retuned in a later session, and floods under its own
# preset name so its containers and instancers never mix with scatter painted by hand.
SCATTER_GROUPS_LAYER_DATA_KEY = "remixScatterGroups"
GROUP_PRESET_PREFIX = "group "
# Texture previews and contact sheets for identifying surfaces, written under the system temporary directory.
SURFACE_PREVIEWS_SUBDIRECTORY = "rtx-remix/scatter_surfaces"
