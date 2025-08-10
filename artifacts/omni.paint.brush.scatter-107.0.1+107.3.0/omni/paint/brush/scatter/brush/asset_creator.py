import numpy as np
import random
import math

import carb
from pxr import Usd, UsdGeom, Gf

from .scale_distribution import get_scale_data
from .utils import get_brush_asset_bound, get_paint_asset_path
from .quadtree import QuadTree

from omni.paint.system.ui import get_falloff_data


MAX_CONFILCT_TRY = 4
MAX_CONTINUE_CONFILCT_TRY = 40
MAX_ASSERTS = 100000


class AssetCandicator:
    def __init__(self, index=-1, x=0.0, y=0.0, scale=1.0, rotation=0):
        self.index = index
        self.position = Gf.Vec3d(x, y, 0)
        self.scale = scale
        self.rotation = rotation


class AssetCreator:
    def __init__(self):
        self._bboxcache = None
        self._asset_bound_cache = {}

    def shutdown(self):
        self._bboxcache = None

    def clearCache(self):
        if self._bboxcache:
            self._bboxcache.Clear()
        self._asset_bound_cache.clear()

    def get_predict_count(self, brush, radius):
        if radius > 0:
            scale = radius / brush["size"]
            count = min(MAX_ASSERTS, int(brush["density"] * scale * scale))
        else:
            count = int(brush["density"])
        return count

    def generate_asset_parameters(self, brush, count):
        # Asset index list
        asset_indexes = self._get_asset_indices(brush, count)
        if not asset_indexes:
            return None, None, None

        # Scale data list
        if brush["scale"]["enabled"]:
            scales = get_scale_data(brush["scale"], count)
        else:
            scales = [1.0] * count

        # Rotation data list
        rotations = np.random.uniform(brush["rotation"]["min"], brush["rotation"]["max"], count)

        return asset_indexes, scales, rotations

    def generate_candicators(self, stage, brush, radius):
        assets_candicators = []
        if radius > 0:
            painting_size = radius
            scale = radius / brush["size"]
            count = min(MAX_ASSERTS, int(brush["density"] * scale * scale))
        else:
            painting_size = brush["size"]
            count = int(brush["density"])

        asset_indexes, scales, rotations = self.generate_asset_parameters(brush, count)
        if not asset_indexes:
            return assets_candicators

        # Generate all position based on falloff
        # Here position means normalized distance from brush center
        distances = get_falloff_data(brush["falloff"], count)

        # aabb tree to check position conflicts
        rect = Gf.Range2f(Gf.Vec2f(-painting_size, -painting_size), Gf.Vec2f(painting_size, painting_size))
        aabb_tree = QuadTree(rect, [])

        pad = brush["object_padding"] if brush["object_padding_enabled"] else 0
        # Try to generate candicators one by one
        continueConflict = 0
        for i in range(count):
            asset_url = brush["assets"][asset_indexes[i]]["path"]
            radius, unused_up_offset = self._getCircleBound(stage, asset_url, scales[i], pad)
            if radius > 0:
                # Try to get an available position
                for t in range(MAX_CONFILCT_TRY):
                    pos = self._generate_pos(brush, distances[i], painting_size)
                    aabb = Gf.Range2f(Gf.Vec2f(pos) - Gf.Vec2f(radius), Gf.Vec2f(pos) + Gf.Vec2f(radius))
                    if aabb_tree.add_aabb(aabb):
                        # Add as a candicator
                        assets_candicators.append(
                            AssetCandicator(asset_indexes[i], pos[0], pos[1], scales[i], rotations[i])
                        )
                        continueConflict = 0
                        break
                    continueConflict += 1
                if continueConflict > MAX_CONTINUE_CONFILCT_TRY:
                    break
            else:
                # aabb less than 0(negtive padding)
                pos = self._generate_pos(brush, distances[i], painting_size)
                assets_candicators.append(AssetCandicator(asset_indexes[i], pos[0], pos[1], scales[i], rotations[i]))

        # aabb_tree.print_info()
        carb.log_info(f"[PaintTool] {len(assets_candicators)}/{count} assets prepared")
        if len(assets_candicators) == 0:
            carb.log_warn(f"[PaintTool] No available assets, the brush size ({brush['size']}) may be too small!")
        return assets_candicators

    def _get_bboxcache(self):
        if self._bboxcache is None:
            purposes = [UsdGeom.Tokens.default_]
            self._bboxcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
        return self._bboxcache

    def _generate_pos(self, brush, distance, scale):
        angel = np.random.ranf() * 2 * math.pi
        x = scale * distance * math.cos(angel)
        y = scale * distance * math.sin(angel)
        return x, y

    def _get_asset_indices(self, brush, num):
        if len(brush["assets"]) > 0:
            population = []
            weights = []

            for i in range(len(brush["assets"])):
                asset = brush["assets"][i]
                if asset["enabled"]:
                    population.append(i)
                    weights.append(asset["weight"])
            if len(population) > 0:
                return random.choices(population, weights=weights, k=num)
            else:
                carb.log_warn("[PaintTool] Brush has assets, but all disabled")
                return None
        else:
            carb.log_warn("[PaintTool] Brush has no assets")
            return None

    def _get_asset_bound(self, stage, asset_url):
        if asset_url in self._asset_bound_cache:
            return self._asset_bound_cache[asset_url]
        else:
            bound = get_brush_asset_bound(stage, self._get_bboxcache(), asset_url)
            ref_prim_path = get_paint_asset_path(stage, asset_url)
            ref_asset_prim = stage.GetPrimAtPath(ref_prim_path)
            up_attr = ref_asset_prim.GetAttribute("up_rot_axis")

            up_index = 1
            if up_attr:
                asset_up_axis = up_attr.Get()
                if asset_up_axis[2] > 0:
                    up_index = 2
            else:
                stage_up_axis = UsdGeom.GetStageUpAxis(stage)
                if stage_up_axis == "Z":
                    up_index = 2

            self._asset_bound_cache[asset_url] = (bound, up_index)
            return bound, up_index

    def _getCircleBound(self, stage, asset_url, scale, padding):
        asset_box3d, up_index = self._get_asset_bound(stage, asset_url)
        box3d = asset_box3d * scale

        if up_index == 1:
            right_index = 2
        else:
            right_index = 1

        size3d = box3d.GetSize()
        a = size3d[0] + padding
        b = size3d[right_index] + padding

        if a > 0 and b > 0:
            radius = math.sqrt(a * a + b * b) * 0.5
        else:
            radius = -1

        return radius, box3d.GetMin()[up_index]
