from pxr import UsdGeom, Gf, Usd, Sdf, Kind
import hashlib
import carb
import carb.settings
import numpy as np
import re
from pathlib import Path
from .constant import *
from omni.paint.system.ui import get_thumbnail_path


def get_ext_path(folder_name: str):
    target_dir = Path(__file__).parent
    count = 7
    while count > 0:
        # print(target_dir.joinpath("data/icons"))
        if Path.exists(target_dir.joinpath(folder_name)):
            return target_dir.joinpath(folder_name).as_posix()
        target_dir = target_dir.parent
        count = count - 1

    carb.log_error(f"Can not find path {folder_name} in extension directory!")
    return ""


def get_icon_path(icon_file):
    icon_path = get_ext_path("icons")
    return f"{icon_path}/{icon_file}"


def get_default_root(stage):
    if stage.HasDefaultPrim():
        defaultPrim = stage.GetDefaultPrim()
        if defaultPrim:
            return defaultPrim.GetPath().pathString

    return ""


def get_asset_name(asset_url):
    filename = asset_url.split("/")[-1].split(".")[0]
    new_name = re.sub(r"[\s():;,~`!@#$&%^+{}=\[\]\-\']", "", filename)
    if new_name[0].isdigit():
        return '_' + new_name
    else:
        return new_name


def get_url_hash(asset_url):
    hash_val = hashlib.md5(asset_url.encode("utf-8")).hexdigest()
    return hash_val[0:5]


def get_paint_root(stage, asset_url):
    paint_root = get_default_root(stage) + PAINT_TOOL_ROOT_KIT
    asset_name = get_asset_name(asset_url)
    url_hash = get_url_hash(asset_url)
    asset_paint_root = paint_root + f"/{asset_name}_{url_hash}"
    return asset_paint_root


def get_paint_prims_path(stage, asset_url):
    asset_instance_path = get_paint_root(stage, asset_url) + "/instances"
    return asset_instance_path


def get_paint_pointInstancer_path(stage, asset_url):
    asset_pointInstancer_path = get_paint_root(stage, asset_url) + "/pointInstancer"
    return asset_pointInstancer_path


def get_paint_asset_path(stage, asset_url):
    # don't do that, the asset_prim have additional attribute, not same as origin prim
    # if stage.GetPrimAtPath(asset_url):
    #    return asset_url
    # else:
    asset_prim_path = get_paint_pointInstancer_path(stage, asset_url) + "/asset"
    return asset_prim_path


def get_paint_asset_ref_path(asset_prim_path):
    return asset_prim_path + "/ref"


def get_stage_up(stage):
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == "X":
        return Gf.Vec3d.XAxis()
    elif up_axis == "Y":
        return Gf.Vec3d.YAxis()
    else:
        return Gf.Vec3d.ZAxis()


class PointInstancePainter:
    def __init__(self, stage, asset_url):
        self._point_instancer = None

        if stage is None:
            return

        asset_pointInstancer_path = get_paint_pointInstancer_path(stage, asset_url)

        asset_pointInstancer_prim = stage.GetPrimAtPath(asset_pointInstancer_path)
        if asset_pointInstancer_prim:
            pointInstancer = UsdGeom.PointInstancer(asset_pointInstancer_prim)
            self._point_instancer = pointInstancer
            return

        carb.log_info(f"[PaintTool] Create pointer instancer for {asset_url}")
        asset_pointInstancer_prim = stage.DefinePrim(asset_pointInstancer_path, "PointInstancer")
        pointInstancer = UsdGeom.PointInstancer(asset_pointInstancer_prim)

        asset_prim_path = get_paint_asset_path(stage, asset_url)
        asset_prim = create_ref_prim(stage, asset_prim_path, asset_url)
        if not asset_prim:
            return

        rot_axis = get_up_rot_axis(stage, asset_url)
        if rot_axis:
            asset_prim.CreateAttribute("up_rot_axis", Sdf.ValueTypeNames.Double3, False).Set(rot_axis)

        proto = pointInstancer.CreatePrototypesRel()
        proto.AddTarget(asset_prim_path)
        pointInstancer.CreatePositionsAttr([])
        pointInstancer.CreateOrientationsAttr([])
        pointInstancer.CreateScalesAttr([])
        pointInstancer.CreateProtoIndicesAttr([])
        self._point_instancer = pointInstancer

    def is_valid(self):
        return self._point_instancer is not None

    def add_points(self, paint_positions, paint_scales, paint_orients, reshape_scale=True):
        new_indices = [0] * len(paint_positions)
        indices = self.get_indices()
        indices.extend(new_indices)
        self.set_indices(indices)

        positions = self.get_positions()
        positions.extend(paint_positions)
        self.set_positions(positions)

        orients = self.get_orients()
        orients.extend(paint_orients)
        self.set_orients(orients)

        scales = self.get_scales()
        if reshape_scale:
            scale_list_v3 = np.repeat(paint_scales, 3).reshape(len(paint_scales), 3)
            scales.extend(scale_list_v3)
        else:
            scales.extend(paint_scales)
        self.set_scales(scales)

    def filter(self, filterList, erased_data):
        indices = self.get_indices()
        new_indices = [i for j, i in enumerate(indices) if filterList[j]]
        erased_data["indices"] = [i for j, i in enumerate(indices) if not filterList[j]]
        self.set_indices(new_indices)

        positions = self.get_positions()
        new_positions = [i for j, i in enumerate(positions) if filterList[j]]
        erased_data["positions"] = [i for j, i in enumerate(positions) if not filterList[j]]
        self.set_positions(new_positions)

        orients = self.get_orients()
        new_orients = [i for j, i in enumerate(orients) if filterList[j]]
        erased_data["orients"] = [i for j, i in enumerate(orients) if not filterList[j]]
        self.set_orients(new_orients)

        scales = self.get_scales()
        new_scales = [i for j, i in enumerate(scales) if filterList[j]]
        erased_data["scales"] = [i for j, i in enumerate(scales) if not filterList[j]]
        self.set_scales(new_scales)

    def restore(self, erased_data):
        self.add_points(erased_data["positions"], erased_data["scales"], erased_data["orients"], reshape_scale=False)

    def delete_last(self, count):
        indices = self.get_indices()
        indices = indices[: len(indices) - count]
        self.set_indices(indices)

        positions = self.get_positions()
        positions = positions[: len(positions) - count]
        self.set_positions(positions)

        orients = self.get_orients()
        orients = orients[: len(orients) - count]
        self.set_orients(orients)

        scales = self.get_scales()
        scales = scales[: len(scales) - count]
        self.set_scales(scales)

    def get_positions(self):
        return list(self._point_instancer.GetPositionsAttr().Get())

    def set_positions(self, positions):
        self._point_instancer.GetPositionsAttr().Set(positions)

    def get_orients(self):
        return list(self._point_instancer.GetOrientationsAttr().Get())

    def set_orients(self, orients):
        self._point_instancer.GetOrientationsAttr().Set(orients)

    def get_scales(self):
        return list(self._point_instancer.GetScalesAttr().Get())

    def set_scales(self, scales):
        self._point_instancer.GetScalesAttr().Set(scales)

    def get_indices(self):
        return list(self._point_instancer.GetProtoIndicesAttr().Get())

    def set_indices(self, indices):
        self._point_instancer.GetProtoIndicesAttr().Set(indices)

    def get_transfrom(self):
        return UsdGeom.Xformable(self._point_instancer).ComputeLocalToWorldTransform(0)


def get_up_rot_axis(stage, asset_url):
    if Sdf.Path().IsValidPathString(asset_url):
        return None

    try:
        ref_stage = Usd.Stage.Open(stage.GetRootLayer().ComputeAbsolutePath(asset_url), Usd.Stage.LoadNone)
    except Exception:
        carb.log_error(f"Fail to open layer {asset_url}")
        return None

    if ref_stage:
        ref_up = UsdGeom.GetStageUpAxis(ref_stage)
        curr_up = UsdGeom.GetStageUpAxis(stage)
        if ref_up != curr_up:
            if ref_up == "Y":
                return Gf.Vec3d.YAxis()
            elif ref_up == "Z":
                return Gf.Vec3d.ZAxis()
    return None


def get_prim_bound(bboxcache, prim):
    bound = bboxcache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bound.IsEmpty():
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Boundable):
                sub_bound = bboxcache.ComputeWorldBound(child).ComputeAlignedRange()
                bound.UnionWith(sub_bound)
    return bound


def get_brush_asset_bound(stage, bboxcache, asset_url):
    asset_prim_path = get_paint_asset_path(stage, asset_url)
    asset_prim = stage.GetPrimAtPath(asset_prim_path)
    if asset_prim:
        return get_prim_bound(bboxcache, asset_prim)
    else:
        carb.log_error(f"[Scatter Brush] Can not get asset {asset_prim_path}")
        return Gf.Range3d()


def create_ref_prim(stage, asset_prim_path, asset_url):
    if Sdf.Path().IsValidPathString(asset_url):
        prim = stage.GetPrimAtPath(asset_url)
        if not prim.IsA(UsdGeom.Xformable):
            carb.log_error(f"Can not paint with asset {asset_url} because it isn't Xformable")
            return None
        asset_prim = stage.DefinePrim(asset_prim_path, "Xform")
        Usd.ModelAPI(asset_prim).SetKind(Kind.Tokens.component)
        asset_ref_path = get_paint_asset_ref_path(asset_prim_path)
        asset_ref = stage.DefinePrim(asset_ref_path, stage.GetPrimAtPath(asset_url).GetTypeName())
        Usd.ModelAPI(asset_ref).SetKind(Kind.Tokens.subcomponent)
        # om-31067
        asset_ref.GetInherits().AddInherit(asset_url)
        asset_ref.SetInstanceable(False)

        # ignore the position of the prototype objects only but keep orientation and scale.
        asset_ref.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(0))
    else:
        asset_prim = stage.DefinePrim(asset_prim_path, "Xform")
        Usd.ModelAPI(asset_prim).SetKind(Kind.Tokens.component)
        asset_prim.GetReferences().AddReference(asset_url)

    return asset_prim
