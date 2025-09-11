# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Dict, Optional

import carb
import omni.kit.commands
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdShade

from lightspeed.common import constants as _C


def _as_asset_path(url_or_path: str) -> Sdf.AssetPath:
    try:
        import omni.client as _oc  # type: ignore

        br = _oc.break_url(url_or_path)
        # Keep full URL if present; AssetPath accepts either
        if br.scheme:
            return Sdf.AssetPath(url_or_path)
        if br.path:
            return Sdf.AssetPath(br.path)
    except Exception:
        pass
    return Sdf.AssetPath(url_or_path)


def _get_mdl_shader_from_material(material_prim: Usd.Prim) -> Optional[Usd.Prim]:
    try:
        shader = omni.usd.get_shader_from_material(material_prim, get_prim=True)
        return shader if shader and shader.IsValid() else None
    except Exception:
        return None


_DEFAULT_MDL_URL = _C.SHADER_NAME_OPAQUE  # e.g. "AperturePBR_Opacity.mdl"
_DEFAULT_MDL_NAME = "AperturePBR_Opacity"

# Map loose keys to material input names
_TEXTURE_INPUT_MAP = {
    "diffuse": _C.MATERIAL_INPUTS_DIFFUSE_TEXTURE,
    "basecolor": _C.MATERIAL_INPUTS_DIFFUSE_TEXTURE,
    "albedo": _C.MATERIAL_INPUTS_DIFFUSE_TEXTURE,
    "normal": _C.MATERIAL_INPUTS_NORMALMAP_TEXTURE,
    "normalmap": _C.MATERIAL_INPUTS_NORMALMAP_TEXTURE,
    "roughness": _C.MATERIAL_INPUTS_REFLECTIONROUGHNESS_TEXTURE,
    "metallic": _C.MATERIAL_INPUTS_METALLIC_TEXTURE,
    "emissive_mask": _C.MATERIAL_INPUTS_EMISSIVE_MASK_TEXTURE,
}


def _assign_textures_to_shader(shader_prim: Usd.Prim, textures: Dict[str, str]) -> None:
    shader = UsdShade.Shader(shader_prim)
    for key, path in (textures or {}).items():
        # Allow both full input names and loose keys
        input_name = key if key.startswith("inputs:") else _TEXTURE_INPUT_MAP.get(key)
        if not input_name:
            continue
        try:
            inp = shader.GetInput(input_name)
            if not inp:
                # If the input doesn't exist on the shader, create it as an asset type
                inp = shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset)
            inp.Set(_as_asset_path(path))
        except Exception:
            carb.log_warn(f"Failed setting texture '{key}' on shader {shader_prim.GetPath()}")


def instantiate_prim_with_mdl_from_dds(
    parent_prim_path: str,
    prim_name: str,
    textures: Dict[str, str],
    *,
    mdl_url: str | None = None,
    mdl_name: str | None = None,
    prim_type: str = "Xform",
    usd_context_name: str = "",
) -> tuple[Sdf.Path, Sdf.Path]:
    """
    Create a prim under `parent_prim_path`, create an MDL material, assign DDS textures, and bind it.

    Returns (created_prim_path, material_prim_path).
    """
    ctx = omni.usd.get_context(usd_context_name)
    stage = ctx.get_stage()

    # Ensure parent exists
    parent = stage.GetPrimAtPath(parent_prim_path)
    if not parent or not parent.IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(parent_prim_path))
        parent = stage.GetPrimAtPath(parent_prim_path)
    # Create prim container
    prim_path = Sdf.Path(parent_prim_path).AppendChild(prim_name)
    if prim_type and prim_type != "Xform":
        omni.kit.commands.execute(
            "CreatePrimCommand",
            prim_path=str(prim_path),
            prim_type=str(prim_type),
            select_new_prim=False,
            context_name=usd_context_name,
        )
    else:
        UsdGeom.Xform.Define(stage, prim_path)

    # Create MDL material
    mdl_url = mdl_url or _DEFAULT_MDL_URL
    mdl_name = mdl_name or _DEFAULT_MDL_NAME
    mtl_path = omni.usd.get_stage_next_free_path(stage, f"/Looks/{prim_name}", False)
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url=mdl_url,
        mtl_name=mdl_name,
        mtl_path=mtl_path,
        stage=stage,
    )
    material_prim = stage.GetPrimAtPath(mtl_path)

    # Set textures on shader inputs
    shader_prim = _get_mdl_shader_from_material(material_prim)
    if shader_prim:
        _assign_textures_to_shader(shader_prim, textures)

    # Bind material to created prim
    try:
        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=prim_path,
            material_path=material_prim.GetPath(),
            strength=UsdShade.Tokens.strongerThanDescendants,
            stage=stage,
        )
    except Exception:
        # Fallback to MaterialBindingAPI
        try:
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(UsdShade.Material(material_prim))
        except Exception:
            carb.log_warn(f"Failed to bind material {mtl_path} to {prim_path}")

    return prim_path, material_prim.GetPath()