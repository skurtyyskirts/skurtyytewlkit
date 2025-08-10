import carb
from pxr import Gf, UsdGeom
from .utils import PointInstancePainter, get_paint_prims_path


def getPointToSegmentDistance(point, line_start, segment):
    ap = point - line_start
    # project on the line
    k = (ap * segment) / (segment * segment)
    # clamp to segment
    t = max(min(k, 1), 0)
    # nearest point on segment
    p = line_start + segment * t
    return (p - point).GetLength()


class AssetEraser:
    def __init__(self, stage, asset_url):
        self._asset_url = asset_url

        # point instance
        self._pointInstancer = PointInstancePainter(stage, asset_url)

        # prim instance
        asset_instances_path = get_paint_prims_path(stage, self._asset_url)
        self._instances_prim = stage.GetPrimAtPath(asset_instances_path)

    def prepare_erase(self, position, **kwargs):
        if kwargs.get("erase_all", False):
            erase_point_instance = self._prepare_erase_point_instance_all()
            erase_prim = self._prepare_erase_prims_all()
        else:
            aabb = kwargs.get("erase_bound", None)
            if aabb:
                erase_point_instance = self._prepare_erase_point_instance_aabb(aabb)
                erase_prim = self._prepare_erase_prims_aabb(aabb)
            else:
                start = position + kwargs.get("height", 0)
                radius= kwargs.get("radius", 100)
                erase_point_instance = self._prepare_erase_point_instance(start, position, radius)
                erase_prim = self._prepare_erase_prims(start, position, radius)

        return erase_point_instance or erase_prim


    def _prepare_erase_point_instance_all(self):
        if not self._pointInstancer.is_valid():
            return False

        all_pos = self._pointInstancer.get_positions()
        self._point_instance_erase_count = len(all_pos)
        self._point_instance_filter = [False for i in range(self._point_instance_erase_count)]

        return self._point_instance_erase_count > 0

    def _prepare_erase_point_instance(self, line_start, line_end, radius):
        if not self._pointInstancer.is_valid():
            return False

        self._point_instance_filter = []
        self._point_instance_erase_count = 0

        # OM-66823: if vertical_offset not 0, will erase volume is a capsule, otherwise a sphere
        segment = line_end -line_start
        if segment.GetLength() < 0.001:
            dist_fn = lambda p0, p1, segment: (p0-p1).GetLength()
        else:
            dist_fn = getPointToSegmentDistance

        mat = self._pointInstancer.get_transfrom()
        for point in self._pointInstancer.get_positions():
            world_pos = mat.Transform(point)
            dist = dist_fn(Gf.Vec3d(world_pos), line_start, segment)
            erasing = dist < radius
            self._point_instance_erase_count += 1 if erasing else 0
            self._point_instance_filter.append(not erasing)

        return self._point_instance_erase_count > 0

    def _prepare_erase_point_instance_aabb(self, aabb: Gf.Range3d):
        if not self._pointInstancer.is_valid():
            return False

        self._point_instance_filter = []
        self._point_instance_erase_count = 0

        mat = self._pointInstancer.get_transfrom()
        for point in self._pointInstancer.get_positions():
            pos = mat.Transform(point)
            erasing = aabb.Contains(Gf.Vec3d(pos))
            self._point_instance_filter.append(not erasing)
            if erasing:
                self._point_instance_erase_count += 1

        return self._point_instance_erase_count > 0

    def _erase_point_instance(self):
        erased_data = {"indices": None, "positions": None, "orients": None, "scales": None}

        if self._point_instance_erase_count > 0:
            self._pointInstancer.filter(self._point_instance_filter, erased_data)

            carb.log_info(f"[PaintTool] {len(erased_data['positions'])} points erased!")

        return erased_data

    def erase(self):
        point_instance_erased_data = self._erase_point_instance()
        return {"pointer_instance": point_instance_erased_data, "prim": self._prim_erased_data}

    def restore(self, erased_data):
        erased_point_instance = erased_data["pointer_instance"]
        self._restore_pointer_instance(erased_point_instance)

    def _restore_pointer_instance(self, erased_data):
        if not self._pointInstancer.is_valid():
            return
        if erased_data["positions"] is None:
            return

        carb.log_info(f"[PaintTool] {len(erased_data['positions'])} points restored!")
        self._pointInstancer.restore(erased_data)

    def _prepare_erase_prims_all(self):
        self._prim_erased_data = {"aabbs": None}

        if not self._instances_prim:
            return False

        # remove the parent
        self._prim_erased_data["aabbs"] = [self._instances_prim.GetPath().pathString]
        return True

    def _prepare_erase_prims(self, line_start, line_end, radius):
        self._prim_erased_data = {"aabbs": None}
        erased_prim = []

        if not self._instances_prim:
            return False

        for inst in self._instances_prim.GetChildren():
            # compute ref position due to the object may move(physics drop)
            ref_prim = inst.GetChild("ref")
            if ref_prim:
                pos = UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            else:
                pos = UsdGeom.Xformable(inst).ComputeLocalToWorldTransform(0).ExtractTranslation()

            # erase all the prims between the camera to brush or only the prims in the brush radius?
            # if getPointToSegmentDistance(pos, line_start, line_end) < radius:
            if (line_end - pos).GetLength() < radius:
                erased_prim.append(inst.GetPath().pathString)

        self._prim_erased_data["aabbs"] = erased_prim
        return len(erased_prim) > 0

    def _prepare_erase_prims_aabb(self, aabb: Gf.Range3d):
        self._prim_erased_data = {"aabbs": None}

        if not self._instances_prim:
            return False

        erased_prim = []
        for inst in self._instances_prim.GetChildren():
            ref_prim = inst.GetChild("ref")
            if ref_prim:
                pos = UsdGeom.Xformable(ref_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            else:
                pos = UsdGeom.Xformable(inst).ComputeLocalToWorldTransform(0).ExtractTranslation()

            if aabb.Contains(pos):
                erased_prim.append(inst.GetPath().pathString)

        self._prim_erased_data["aabbs"] = erased_prim
        return len(erased_prim) > 0
