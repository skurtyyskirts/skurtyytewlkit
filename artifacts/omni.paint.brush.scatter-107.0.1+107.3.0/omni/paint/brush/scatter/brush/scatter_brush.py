import math
import os
import carb
import omni.client
import omni.kit.commands
import omni.kit.notification_manager as nm
import omni.usd
from omni.paint.system.core import PainterEventType, PAINT_MODES, get_paintable_meshes
from omni.paint.system.ui import BrushBase
from pxr import Gf, UsdGeom, Usd

from .asset_creator import AssetCreator
from .brush import INSTANCING, update_brush_version, BRUSH_FORMAT_VERSION
from .constant import (
    PAINT_TOOL_ROOT_KIT,
    SETTING_TRANSFORM_GIZMO_ENABLED,
    SETTING_PLACEMENT_MODE_ALLOW_DROP_ON_TRANSFORM_GIZMO_CHANGES,
)
from .erase import AssetEraser
from .utils import (
    PointInstancePainter,
    get_stage_up,
    get_paint_asset_path,
    get_prim_bound,
    get_paint_root,
    get_default_root
)
from .octree import OcTree
from omni.physx import get_physx_scene_query_interface
from .ui import ParamsUi


WARN_NUMBER_POINTER = 10000
WARN_NUMBER_OTHER = 500
REDUNDANCY = 1.2
BRUSH_TYPE = "Scatter"


class ScatterBrush(BrushBase, AssetCreator):
    def __init__(self):
        AssetCreator.__init__(self)
        BrushBase.__init__(self)
        self._brush_version = BRUSH_FORMAT_VERSION
        self._usd_context = omni.usd.get_context()
        self._last_painting_pos = None
        self._showing_msgbox = False
        self._stroke_asset_prims = []
        self._property_window_was_paused = False
        self._params_ui = None
        self._blocks = []

        self._physx_scene_query = get_physx_scene_query_interface()

    def __del__(self):
        AssetCreator.shutdown(self)
        BrushBase.__del__(self)

    @classmethod
    def get_type(self) -> str:
        # return current brush type
        return BRUSH_TYPE

    # for create new bursh and update old version brush file
    def fill_default_brush(self, brush_parameters: dict):
        update_brush_version(brush_parameters)

    # called once at the beginning of this brush's session, constructor
    def begin_brush(self, brush: dict, *args, **kwargs):
        carb.log_info(f"[ScatterBrush] Begin brush {brush['name']}")

        super().begin_brush(brush)

        full_path = kwargs.get("full_path", None)
        # TODO
        # This call changes asset["path"] and dirties the state, causing Save icon to turn blue
        self._uniform_assets_path(full_path)
        self._brush_widget = self._painter.get_brush_widget()

    # called when this brush session ends, cleans up and restores states (if necessary)
    def end_brush(self, *args, **kwargs):
        self._brush_widget = None
        self._usd_context = None

    # called once at the beginning of a stroke, setup anything specific to the scripted brush functionality
    # return True if brush is valid, otherwise return False
    def begin_stroke(self, *args, **kwargs):
        stage = self._usd_context.get_stage()
        if not stage:
            return False

        # check brush has valid assets
        if self._brush is None:
            return False
        if len(self._brush["assets"]) == 0:
            self._show_message_box(
                "You must have at least one asset\nloaded onto the brush to paint.\n\nAdd an Asset on the Assets section of the Paint Tool."
            )
            return False
        valid = False
        for asset in self._brush["assets"]:
            # at least have one enabled asset
            if asset["enabled"]:
                if not (
                    asset["path"].startswith("omniverse:")
                    or asset["path"].startswith("http")
                    or asset["path"].startswith("\\\\")
                    or os.path.isfile(asset["path"])
                    or stage.GetPrimAtPath(asset["path"])
                ):
                    asset["enabled"] = False
                else:
                    valid = True

        if not valid:
            self._show_message_box("At least one asset on the current brush must\nbe enabled/valid to paint.")
            return False

        if not super().begin_stroke():
            return False

        # in case assets changed
        self._preload_assets(stage)

        # settings during mouse drag painting
        self._origin_allow_drop = self._settings.get_as_bool(
            SETTING_PLACEMENT_MODE_ALLOW_DROP_ON_TRANSFORM_GIZMO_CHANGES
        )
        self._settings.set_bool(SETTING_PLACEMENT_MODE_ALLOW_DROP_ON_TRANSFORM_GIZMO_CHANGES, True)

        #  _stroke_asset_prims saved all assets painted in a stroke,  clear it at the begin of stroke
        self._stroke_asset_prims.clear()

        return True

    # called once at the end of a stroke
    def end_stroke(self, *args, **kwargs):
        if self._stroke_started:
            self._last_painting_pos = None

            # always drop a while, to close to the surface
            if self._brush["physics"]:
                stage = self._usd_context.get_stage()

        super().end_stroke()

    # repetitively called for each stamp affected by the stroke during swipe
    # @param flood_size size of flood mode, > 0 means available
    # @param ids array of object id, will set back in set_data in hit_ids
    # @param positions array of object position, related to center of brush
    # @param rotations array of object rotation
    # @param scales arrya of object scale
    # return number of objects to be painted
    def get_data(self, flood_size, ids: list, positions: list, rotations: list, scales: list, *args, **kwargs):
        stage = self._usd_context.get_stage()
        candicators = self.generate_candicators(stage, self._brush, flood_size)

        for candicator in candicators:
            ids.append(candicator.index)
            positions.append(candicator.position)
            scales.append(candicator.scale)
            rotations.append(candicator.rotation)

        return len(candicators)

    # repetitively called for each stamp affected by the stroke during swipe
    # @param hit_ids array of hitted object ids
    # @param hit_positions array of hitted object position, where asset to be painted
    # @param hit_rotations array of hitted object rotation
    # @param hit_scales arrya of hitted object scale
    # @param hit_normals array of hit normal
    # @param hit_meshes array of hitted mesh
    # return True if painted, otherwise False
    def set_data(
        self,
        hit_ids: list,
        hit_positions: list,
        hit_rotations: list,
        hit_scales: list,
        hit_normals: list,
        hit_meshes: list,
        *arg,
        **kwargs,
    ):
        self._paint_candicator = {}
        size = len(hit_ids)
        if size == 0:
            return False

        paint_candicator = {}
        asset_parameters = {}
        stage = self._usd_context.get_stage()
        stage_up_axis = get_stage_up(stage)

        for index in range(size):
            asset_index = hit_ids[index]
            asset_url = self._brush["assets"][asset_index]["path"]
            if asset_url not in paint_candicator:
                paint_candicator[asset_url] = []
                parent_transform, asset_up_axis = self._get_asset_parameters(stage, asset_url)
                asset_parameters[asset_url] = (parent_transform, asset_up_axis)
            else:
                parent_transform, asset_up_axis = asset_parameters[asset_url]

            if asset_up_axis:
                rotation = Gf.Rotation(asset_up_axis, stage_up_axis) * hit_rotations[index]
            else:
                rotation = hit_rotations[index]

            world_position = hit_positions[index] + stage_up_axis * self._brush["vertical_offset"]
            transform_local = Gf.Matrix4d(rotation, world_position) * parent_transform

            factor = transform_local.Factor()
            rotMat = factor[3]
            rotMat.Orthonormalize(False)
            local_rotation = rotMat.ExtractRotation()
            local_position = factor[4]

            object_to_paint = {
                "asset": asset_url,
                "position": local_position,
                "scale": hit_scales[index],
                "rotation": local_rotation,
            }

            paint_candicator[asset_url].append(object_to_paint.copy())

        stamp_asset_prims = []

        omni.kit.commands.execute(
            "ScatterBrushPaintCommand",
            instancing_type=self._brush["instancing"],
            paint_candicator=paint_candicator,
            out_paint_asset_prims=stamp_asset_prims,
        )

        self._stroke_asset_prims.extend(stamp_asset_prims)

        return True

    def _get_asset_parameters(self, stage, asset_url):
        asset_prim_path = get_paint_asset_path(stage, asset_url)
        asset_prim = stage.GetPrimAtPath(asset_prim_path)
        xformable = UsdGeom.Xformable(asset_prim)
        parent_transform = xformable.ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse()

        up_attr = asset_prim.GetAttribute("up_rot_axis")
        asset_up_axis = up_attr.Get() if up_attr else None

        return (parent_transform, asset_up_axis)

    # if erase_bound is None, will erase assets in brush radius, otherwise erase assets in erase_bound
    def erase(self, position, *arg, **kwargs):
        stage = self._usd_context.get_stage()
        height = self._painter.get_upAxis() * self._brush["vertical_offset"]
        radius = self._brush["size"]
        active_erasers = {}
        for asset in self._brush["assets"]:
            if asset["enabled"]:
                erase = AssetEraser(stage, asset["path"])
                if erase.prepare_erase(position, height=height, radius=radius, **kwargs):
                    active_erasers[asset["path"]] = erase

        if len(active_erasers) > 0:
            omni.kit.commands.execute("ScatterBrushEraseCommand", erasers=active_erasers)
            return True
        return False

    # flooding paint on meshes
    def flood(self, paint_mode):
        locked_meshes = self._painter.get_locked_meshes()
        stage = self._usd_context.get_stage()

        # in case assets changed
        self._preload_assets(stage)

        if len(locked_meshes) == 0:
            if paint_mode == PAINT_MODES.ERASE:
                self.erase(None, erase_all=True)
                return True
            else:
                locked_meshes = []
                meshes = get_paintable_meshes(stage.GetPseudoRoot(), self.is_paintable_mesh, out_sdfpath=True)
                for sdfpath in meshes:
                    locked_meshes.append(sdfpath)

        if paint_mode == PAINT_MODES.ERASE:
            if self.begin_stroke():
                for mesh_path in locked_meshes:
                    prim = stage.GetPrimAtPath(mesh_path)
                    center, radius, flood_bound = self._get_flood_data(prim)
                    self.erase(center, erase_bound=flood_bound)
                self.end_stroke()
        else:
            r = self._brush["size"]
            densityPerUnit = self._brush["density"] / (3.14159 * r * r)
            positions, normals = self._painter.get_flood_data(locked_meshes, densityPerUnit * REDUNDANCY)
            candicators_count = len(positions)

            rect = Gf.Range3d()
            bboxcache = self._get_bboxcache()
            for mesh_path in locked_meshes:
                mesh_prim = stage.GetPrimAtPath(mesh_path)
                rect.UnionWith(get_prim_bound(bboxcache, mesh_prim))

            warn_num = WARN_NUMBER_POINTER if self._brush["instancing"] == INSTANCING.POINT else WARN_NUMBER_OTHER
            if self._showing_msgbox or candicators_count > warn_num:
                self._show_warn_box_flood(rect, positions, normals)
                return False

            self._flood_stamp(rect, positions, normals)

        return True

    def create_params_ui(self, on_param_changed_fn, *args, **kwargs):
        """
        * call once for paint tool window to create ui for brush properties
        * use omni.paint.system.ui.create_standard_parameter_panel
        * or create custom ui by omni.ui
        * @ param on_param_changed_fn used to notify paint tool what brush parameter is changed:
        ** on_param_changed(property_name, property_value)
        """
        parent_window = kwargs.get("parent_window", None)
        self._params_ui = ParamsUi(self._brush, on_param_changed_fn, parent_window=parent_window)
        return self._params_ui

    # call when paint tool to select mesh to paint on.
    # return False if current brush does NOT want to paint on the mesh, usually because of the mesh is created by current brush
    def is_paintable_mesh(self, mesh_path, *args, **kwargs):
        return not (PAINT_TOOL_ROOT_KIT in mesh_path)

    def _uniform_assets_path(self, brush_full_path):
        for asset in self._brush["assets"]:
            if not asset["path"]:
                # Empty path
                continue

            broken_url = omni.client.break_url(asset["path"])
            path = broken_url.path

            if broken_url.scheme not in ["omniverse", "http", "https"]:
                if os.path.isabs(path) and os.path.isfile(path):
                    continue

                if "relative_path" in asset:
                    # relative to kit flolder
                    root_path = carb.tokens.get_tokens_interface().resolve("${app}/../")
                    re_path = os.path.join(root_path, asset["relative_path"])
                    asset["path"] = omni.client.break_url(re_path).path
                elif not os.path.isabs(path):
                    # Relative to brush folder
                    re_path = os.path.join(os.path.dirname(brush_full_path), path)
                    asset["path"] = omni.client.break_url(re_path).path

    def _preload_assets(self, stage):
        if stage:
            paint_root = get_default_root(stage) + PAINT_TOOL_ROOT_KIT
            stage.DefinePrim(paint_root, "Xform")
            for asset in self._brush["assets"]:
                if asset["enabled"]:
                    asset_url = asset["path"]
                    asset_paint_root = get_paint_root(stage, asset_url)
                    stage.DefinePrim(asset_paint_root, "Xform")
                    PointInstancePainter(stage, asset_url)

    def _dlg_close(self):
        self._showing_msgbox = False

    def _show_message_box(self, message):

        if not self._showing_msgbox:
            self._showing_msgbox = True
            ok_button = nm.NotificationButtonInfo("OK", on_complete=self._dlg_close)
            nm.post_notification(
                message,
                hide_after_timeout=False,
                duration=0,
                status=nm.NotificationStatus.WARNING,
                button_infos=[ok_button],
            )

    def _show_warn_box(self, center, radius, targets):
        def continue_stamp():
            omni.kit.undo.begin_group()
            for block in self._blocks:
                ret = self.do_stamp(block[0], block[1], block[2])
                if ret:
                    self._last_painting_pos = center
            omni.kit.undo.end_group()

            self._showing_msgbox = False

        if not self._showing_msgbox:
            self._showing_msgbox = True
            self._blocks.clear()
            self._blocks.append([center, radius, targets])

            message = "Potentially large amount of assets will be created.\nThis may require long time to process.\nDo you want to continue ?"
            ok_button = nm.NotificationButtonInfo("OK", on_complete=continue_stamp)
            cancel_button = nm.NotificationButtonInfo("CANCEL", on_complete=self._dlg_close)
            nm.post_notification(
                message,
                hide_after_timeout=False,
                duration=0,
                status=nm.NotificationStatus.WARNING,
                button_infos=[ok_button, cancel_button],
            )
        else:
            self._blocks.append([center, radius, targets])

    def _show_warn_box_flood(self, rect, positions, normals):
        def continue_flood():
            if self.begin_stroke():
                for block in self._blocks:
                    self._flood_stamp(block[0], block[1], block[2])
                self.end_stroke()
            self._showing_msgbox = False

        if not self._showing_msgbox:
            self._showing_msgbox = True
            self._blocks = []
            self._blocks.append([rect, positions, normals])

            message = "Potentially large amount of assets will be created.\nThis may require long time to process.\nDo you want to continue ?"
            ok_button = nm.NotificationButtonInfo("OK", on_complete=continue_flood)
            cancel_button = nm.NotificationButtonInfo("CANCEL", on_complete=self._dlg_close)
            nm.post_notification(
                message,
                hide_after_timeout=False,
                duration=0,
                status=nm.NotificationStatus.WARNING,
                button_infos=[ok_button, cancel_button],
            )
        else:
            self._blocks.append([rect, positions, normals])

    def _flood_stamp(self, rect, positions, normals):
        candicators_count = len(positions)
        asset_indexes, scales, rotations = self.generate_asset_parameters(self._brush, candicators_count)
        if not asset_indexes:
            return False

        f_indexes = []
        f_rotations = []
        f_scales = []
        f_positions = []
        f_normals = []
        aabb_tree = OcTree(rect, [])
        pad = self._brush["object_padding"] if self._brush["object_padding_enabled"] else 0
        stage = self._usd_context.get_stage()
        stage_up_axis = get_stage_up(stage)
        avoid_penetration = 10 * stage_up_axis

        for i in range(candicators_count):
            asset_url = self._brush["assets"][asset_indexes[i]]["path"]
            radius, up_offset = self._getCircleBound(stage, asset_url, scales[i], pad)
            pos = positions[i]
            if radius <= 0 or aabb_tree.add_aabb(
                Gf.Range3d(
                    Gf.Vec3d(pos.x - radius, pos.y - radius, pos.z - radius),
                    Gf.Vec3d(pos.x + radius, pos.y + radius, pos.z + radius),
                )
            ):
                # random rotation (by hit position/normal)
                normal = Gf.Vec3d(normals[i].x, normals[i].y, normals[i].z)
                if self._brush["conform_to_surface"]:
                    random_rot = Gf.Rotation(stage_up_axis, rotations[i]) * Gf.Rotation(stage_up_axis, normal)
                else:
                    random_rot = Gf.Rotation(stage_up_axis, rotations[i])

                f_indexes.append(asset_indexes[i])
                f_rotations.append(random_rot)
                f_scales.append(scales[i])

                if self._brush["physics"]:
                    f_positions.append(Gf.Vec3d(pos.x, pos.y, pos.z) + avoid_penetration - stage_up_axis * up_offset)
                else:
                    f_positions.append(Gf.Vec3d(pos.x, pos.y, pos.z))

                f_normals.append(normal)

        carb.log_info(f"[PaintTool] {len(f_indexes)}/{candicators_count} assets prepared")
        self.set_data(f_indexes, f_positions, f_rotations, f_scales, f_normals, None)

    def _get_flood_data(self, mesh):
        if mesh is None:
            return 0, 0, None

        box3d = self._get_bboxcache().ComputeWorldBound(mesh).ComputeAlignedRange()
        halfSize = box3d.GetSize() * 0.5
        stage = self._usd_context.get_stage()
        up = UsdGeom.GetStageUpAxis(stage)
        if up == "Z":
            flood_radius = math.sqrt(halfSize[0] * halfSize[0] + halfSize[1] * halfSize[1])
        else:
            flood_radius = math.sqrt(halfSize[0] * halfSize[0] + halfSize[2] * halfSize[2])

        if self._brush["conform_to_surface"]:
            a = max(halfSize[0], halfSize[1])
            b = max(min(halfSize[0], halfSize[1]), halfSize[2])
            flood_radius = math.sqrt(a * a + b * b)

        box3d_ext = box3d + Gf.Range3d(Gf.Vec3d(-self._brush["size"]), Gf.Vec3d(self._brush["size"]))
        center = box3d.GetMidpoint()
        return center, flood_radius, box3d_ext

    def on_painter_event(self, event: carb.events.IEvent):
        if event.type == PainterEventType.BRUSH_INACTIVE:
            self.clearCache()

        super().on_painter_event(event)

    def do_stamp(self, position, radius, target_meshes):
        # carb.log_info("[PaintTool] scatter painting starts")
        ids = []
        positions = []
        rotations = []
        scales = []
        hit_ids = []
        hit_rotations = []
        hit_scales = []
        hit_meshes = []
        hit_positions = []
        hit_normals = []
        objects_ready = 0
        objects_prepared = self.get_data(radius, ids, positions, rotations, scales)
        if objects_prepared > 0:
            brush_up_axis = self._painter.get_upAxis()
            camera_pos = self._painter.get_camera_pos()
            ray_axis = (position - camera_pos).GetNormalized()
            ray_dist = (position - camera_pos).GetLength()
            ray_dir = carb.Float3(ray_axis[0], ray_axis[1], ray_axis[2])

            rotation = Gf.Rotation(Gf.Vec3d.ZAxis(), brush_up_axis)
            transform = Gf.Matrix4d(rotation, camera_pos)

            # stroke direction rotation
            if self._last_painting_pos:
                stroke_dir_normalized = (position - self._last_painting_pos).GetNormalized() * 1000
                center_point = position + stroke_dir_normalized
                plane = Gf.Plane(brush_up_axis, position)
                center_point_projected = plane.Project(center_point)
                matrix = Gf.Matrix4d()
                matrix = matrix.SetIdentity()
                matrix.SetLookAt(position, center_point_projected, brush_up_axis)
                stroke_rot = matrix.GetInverse().ExtractRotation()
                align_value = self._brush["align_to_stroke"]
            else:
                stroke_rot = rotation
                align_value = 0

            stage = self._usd_context.get_stage()
            stage_up_axis = get_stage_up(stage)
            avoid_penetration = 10 * stage_up_axis

            pad = self._brush["object_padding"] if self._brush["object_padding_enabled"] else 0
            for index in range(objects_prepared):
                asset_url = self._brush["assets"][ids[index]]["path"]
                asset_radius, up_offset = self._getCircleBound(stage, asset_url, scales[index], pad)
                real_radius = max(asset_radius, radius)
                pos = transform.Transform(positions[index])
                origin = carb.Float3(pos[0], pos[1], pos[2])
                if self._brush["physics"]:
                    hit = self._physx_scene_query.sweep_sphere_closest(
                        asset_radius, origin, ray_dir, ray_dist + real_radius
                    )
                else:
                    hit = self._painter.raycast_closest(origin, ray_dir, ray_dist + real_radius)

                if hit["hit"]:
                    dist = hit.get("distance", 1.0)
                    if dist > 0:
                        hit_pos = Gf.Vec3d(hit["position"][0], hit["position"][1], hit["position"][2])
                        hit_normal = Gf.Vec3d(hit["normal"][0], hit["normal"][1], hit["normal"][2])
                        hit_ids.append(ids[index])

                        # random rotation (by hit position/normal)
                        if self._brush["conform_to_surface"]:
                            random_rot = Gf.Rotation(stage_up_axis, rotations[index]) * Gf.Rotation(
                                stage_up_axis, hit_normal
                            )
                        else:
                            random_rot = Gf.Rotation(stage_up_axis, rotations[index])

                        # slerp the stroke direction rotation with the random rotation
                        rotation = Gf.Rotation(
                            Gf.Slerp(align_value, random_rot.GetQuaternion(), stroke_rot.GetQuaternion())
                        )
                        hit_rotations.append(rotation)

                        hit_scales.append(scales[index])
                        hit_meshes.append(hit["collision"])
                        hit_normals.append(hit_normal)
                        if self._brush["physics"]:
                            hit_positions.append(hit_pos + avoid_penetration - hit_normal * up_offset)
                        else:
                            hit_positions.append(hit_pos)
                        objects_ready += 1

        if objects_ready == 0:
            carb.log_info("[PaintTool] Nothing to paint")
        else:
            carb.log_info(f"[PaintTool] {objects_ready}/{objects_prepared} ready to paint")
            self.set_data(hit_ids, hit_positions, hit_rotations, hit_scales, hit_normals, hit_meshes)

        # carb.log_info("[PaintTool] scatter painting ends")
        return objects_ready > 0

    def _pause_property_window(self, paused: bool) -> bool:
        was_paused = False
        try:
            import omni.kit.window.property

            property_window = omni.kit.window.property.get_window()
            was_paused = property_window.paused
            property_window.paused = paused
        except Exception as e:
            pass
        return was_paused