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
    "ApplyTo",
    "CpuRaySurfacePicker",
    "EraseScope",
    "Falloff",
    "HdRemixSurfacePicker",
    "MeshGeometry",
    "MeshSurfaceCache",
    "PaddingIndex",
    "PlacementRecord",
    "PresetStore",
    "ScatterAssetEntry",
    "ScatterBrushController",
    "ScatterBrushSettings",
    "ScatterCoreExtension",
    "ScatterFloodCommand",
    "ScatterMode",
    "ScatterStrokeCommand",
    "ScatterStrokeKind",
    "ScatterTarget",
    "StrokeSession",
    "SurfaceHit",
    "SurfacePicker",
    "SurfaceSample",
    "TargetMode",
    "UpAxis",
    "area_weighted_triangle_samples",
    "asset_crate_supported",
    "asset_default_prim_name",
    "asset_up_correction",
    "author_placements",
    "build_mesh_geometry",
    "camera_ray_from_ndc",
    "canonical_instance",
    "choose_asset",
    "closest_point_on_triangles",
    "closest_points",
    "compose_parent_space_transform",
    "container_path_for",
    "create_surface_picker",
    "destroy_scatter_brush_controller",
    "erase_candidates",
    "existing_placement_points",
    "falloff_weight",
    "find_capture_mesh",
    "generate_flood",
    "generate_stamp",
    "get_default_presets_directory",
    "get_instance_root",
    "get_prototype_root",
    "get_scatter_brush_controller",
    "has_scatter_ancestor",
    "instance_count",
    "is_scatter_prim",
    "list_ingested_models",
    "load_assets_from_carb",
    "make_relative_asset_path",
    "new_placement_name",
    "raycast",
    "read_asset_up_axis",
    "remove_placements",
    "resolve_reference_asset_from_selection",
    "resolve_target",
    "restore_prims",
    "rotation_to_xyz_degrees",
    "sample_disk",
    "sample_rotation_degrees",
    "sample_scale",
    "save_assets_to_carb",
    "selected_prototypes",
    "snapshot_prims",
    "stage_up_axis",
    "stamp_rng",
    "tangent_basis",
    "triangulate_faces",
    "up_axis_vector",
    "validated_anchor_prototype",
]

from .assets import (
    asset_crate_supported,
    asset_default_prim_name,
    list_ingested_models,
    read_asset_up_axis,
    resolve_reference_asset_from_selection,
)
from .commands import ScatterFloodCommand, ScatterStrokeCommand, ScatterStrokeKind
from .controller import (
    ScatterBrushController,
    ScatterMode,
    destroy_scatter_brush_controller,
    get_scatter_brush_controller,
)
from .extension import ScatterCoreExtension
from .geometry import (
    MeshGeometry,
    MeshSurfaceCache,
    SurfaceSample,
    area_weighted_triangle_samples,
    build_mesh_geometry,
    closest_point_on_triangles,
    closest_points,
    raycast,
    triangulate_faces,
)
from .picking import (
    CpuRaySurfacePicker,
    HdRemixSurfacePicker,
    SurfaceHit,
    SurfacePicker,
    camera_ray_from_ndc,
    create_surface_picker,
)
from .placement import (
    PlacementRecord,
    author_placements,
    container_path_for,
    erase_candidates,
    existing_placement_points,
    generate_flood,
    generate_stamp,
    make_relative_asset_path,
    new_placement_name,
    remove_placements,
    restore_prims,
    snapshot_prims,
)
from .presets import PresetStore, get_default_presets_directory
from .sampling import (
    PaddingIndex,
    asset_up_correction,
    choose_asset,
    compose_parent_space_transform,
    falloff_weight,
    rotation_to_xyz_degrees,
    sample_disk,
    sample_rotation_degrees,
    sample_scale,
    stage_up_axis,
    stamp_rng,
    tangent_basis,
    up_axis_vector,
)
from .settings import (
    ApplyTo,
    EraseScope,
    Falloff,
    ScatterAssetEntry,
    ScatterBrushSettings,
    TargetMode,
    UpAxis,
    load_assets_from_carb,
    save_assets_to_carb,
)
from .stroke import StrokeSession
from .targets import (
    ScatterTarget,
    canonical_instance,
    find_capture_mesh,
    get_instance_root,
    get_prototype_root,
    has_scatter_ancestor,
    instance_count,
    is_scatter_prim,
    resolve_target,
    selected_prototypes,
    validated_anchor_prototype,
)
