import omni.ext
from omni.usd.commands.usd_commands import DeletePrimsCommand
from omni.paint.system.core import register_brush, unregister_brush
from .brush import (
    ScatterBrush,
    INSTANCING,
    PointInstancePainter,
    get_paint_prims_path,
    get_ext_path,
    get_paint_asset_path,
)

from pxr import Gf, Sdf


class ScatterBrushPaintCommand(omni.kit.commands.Command):
    def __init__(self, instancing_type, paint_candicator, out_paint_asset_prims):
        self._type = instancing_type
        self._paint_candicator = paint_candicator
        self._stage = omni.usd.get_context().get_stage()

        # For point instancing
        self._asset_counts = {}
        self._points = {}

        # For instanced asset
        self._asset_objects = {}
        self._asset_prims = []
        self._out_paint_asset_prims = out_paint_asset_prims

        if self._type == INSTANCING.POINT:
            self._point_instancing_list()
        else:
            self._create_list()

    def do(self):
        if self._type == INSTANCING.POINT:
            for painter in self._points:
                data = self._points[painter]
                painter.add_points(data["positions"], data["scales"], data["orients"])

        elif self._type == INSTANCING.ASSET:
            self._add_list(True)
        else:
            self._add_list(False)

        self._out_paint_asset_prims.extend(self._asset_prims)

    def undo(self):
        if self._type == INSTANCING.POINT:
            self._remove_point_instancings()
        elif self._type == INSTANCING.ASSET:
            self._remove_list()

    def _point_instancing_list(self):
        for asset_url in self._paint_candicator:
            objects = self._paint_candicator[asset_url]
            painter = PointInstancePainter(self._stage, asset_url)
            positions = []
            scales = []
            orients = []
            while len(objects) > 0:
                object_paint = objects.pop()
                positions.append(object_paint["position"])
                scales.append(object_paint["scale"])
                orients.append(Gf.Quath(object_paint["rotation"].GetQuat()))
            self._points[painter] = {"positions": positions, "scales": scales, "orients": orients}

            self._asset_counts[painter] = len(positions)

    def _remove_point_instancings(self):
        for painter in self._asset_counts:
            count = self._asset_counts[painter]
            painter.delete_last(count)

    def _create_list(self):
        for asset_url in self._paint_candicator:
            objects = self._paint_candicator[asset_url]
            asset_path = get_paint_prims_path(self._stage, asset_url)
            self._asset_objects[asset_path] = []
            while len(objects) > 0:
                object_paint = objects.pop()

                self._asset_objects[asset_path].append(object_paint)

    def _add_list(self, instancing):
        layer = self._stage.GetEditTarget().GetLayer()
        for asset_path in self._asset_objects:
            if len(self._asset_objects[asset_path]) == 0:
                continue

            asset_url = self._asset_objects[asset_path][0]["asset"]
            origin_asset_path = get_paint_asset_path(self._stage, asset_url)

            parent_prim = self._stage.DefinePrim(asset_path, "Xform")
            for primspec in parent_prim.GetPrimStack():
                if primspec.layer == layer:
                    parent_primspec = primspec
                    break

            with Sdf.ChangeBlock():
                id = len(parent_prim.GetChildren())
                for object_paint in self._asset_objects[asset_path]:
                    # here should use local id due to new painted prim not sync to stage yet,
                    # but still need check if painted asset used the name already
                    asset_name_pre = f"{asset_path}/inst_{id}"
                    asset_prim_path = omni.usd.get_stage_next_free_path(self._stage, asset_name_pre, False)
                    asset_name = asset_prim_path.split("/")[-1]
                    id = int(asset_prim_path.split("_")[-1]) + 1
                    primspec = Sdf.PrimSpec(parent_primspec, asset_name, Sdf.SpecifierDef)
                    Sdf.CopySpec(layer, origin_asset_path, layer, asset_prim_path)
                    t_attr = Sdf.AttributeSpec(primspec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
                    t_attr.default = object_paint["position"]
                    o_attr = Sdf.AttributeSpec(primspec, "xformOp:orient", Sdf.ValueTypeNames.Quath)
                    o_attr.default = Gf.Quath(object_paint["rotation"].GetQuat())
                    s_attr = Sdf.AttributeSpec(primspec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
                    s_attr.default = Gf.Vec3d(object_paint["scale"])
                    p_attr = Sdf.AttributeSpec(primspec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    p_attr.default = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                    primspec.instanceable = instancing

                    self._asset_prims.append(asset_prim_path)

    def _remove_list(self):
        delete_cmd = DeletePrimsCommand(paths=self._asset_prims)
        delete_cmd.do()
        self._asset_prims = []


class ScatterBrushEraseCommand(omni.kit.commands.Command):
    def __init__(self, erasers):
        self._erasers = erasers
        self._stage = omni.usd.get_context().get_stage()

        self._erased_data = {}
        self._temp_layers = {}

    def do(self):
        for url in self._erasers:
            self._erased_data[url] = self._erasers[url].erase()
            if self._erased_data[url]["prim"]["aabbs"] is not None:
                self._temp_layers[url] = {}
                self._delete_prims(url, self._erased_data[url]["prim"]["aabbs"])

    def undo(self):
        for url in self._erasers:
            if url in self._erased_data:
                self._erasers[url].restore(self._erased_data[url])
                if self._erased_data[url]["prim"]["aabbs"] is not None:
                    self._restore_prims(url, self._erased_data[url]["prim"]["aabbs"])

    def _delete_prims(self, url, erased_aabbs):
        layer_stack = self._stage.GetLayerStack()
        for layer in layer_stack:
            temp_layer = Sdf.Layer.CreateAnonymous()
            edit = Sdf.BatchNamespaceEdit()
            for path in erased_aabbs:
                prim_spec = layer.GetPrimAtPath(path)
                if prim_spec is None:
                    continue

                parent_spec = prim_spec.realNameParent
                if parent_spec is not None:
                    Sdf.CreatePrimInLayer(temp_layer, path)
                    Sdf.CopySpec(layer, path, temp_layer, path)
                    edit.Add(path, Sdf.Path.emptyPath)
                    # self._stage.RemovePrim(path)
            if layer.Apply(edit):
                self._temp_layers[url][temp_layer] = layer

    def _restore_prims(self, url, erased_aabbs):
        for key, value in self._temp_layers[url].items():
            restore_to = value
            restore_from = key
            for path in erased_aabbs:
                if restore_from.GetPrimAtPath(path):
                    Sdf.CreatePrimInLayer(restore_to, path)
                    Sdf.CopySpec(restore_from, path, restore_to, path)


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class BrushExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        register_brush(
            ScatterBrush.get_type(), __package__, ScatterBrush.__name__, default_brush_folder=get_ext_path("brushes")
        )

    def on_shutdown(self):
        unregister_brush(ScatterBrush.get_type(), __package__, ScatterBrush.__name__)


def get_extension():
    return BrushExtension()
