import omni.ext
import omni.ui as ui
import omni.kit.ui
from lightspeed.trex.asset_replacements.core.shared.data_models import GetAvailableAssetsQueryModel, AssetType
from omni.services.client import ServicesClient
import asyncio
import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom, Sdf
from omni.physx import get_physx_scene_query_interface
import random
import math
import omni.kit.commands
from .commands import ScatterPaintCommand

class ScatterToolExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[omni.remix.scatter.tool] extension startup")
        self._window = ui.Window("Scatter Tool", width=300, height=500, dockPreference=ui.DockPreference.LEFT_BOTTOM)
        self._assets_model = ui.SimpleListModel()
        self._physx_scene_query = None

        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("Scatter Brush Settings", style={"font_size": 16, "font_weight": "bold"})

                with ui.CollapsableFrame("Brush Settings"):
                    with ui.VStack(spacing=5, height=0):
                        with ui.HStack():
                            ui.Label("Radius:", width=100)
                            self._radius_slider = ui.FloatSlider(min=0.1, max=10.0)
                        with ui.HStack():
                            ui.Label("Density:", width=100)
                            self._density_slider = ui.FloatSlider(min=1.0, max=100.0)
                        with ui.HStack():
                            ui.Label("Jitter:", width=100)
                            ui.FloatSlider(min=0.0, max=1.0)
                        with ui.HStack():
                            ui.Label("Align to Normal", width=100)
                            ui.CheckBox()
                        with ui.HStack():
                            ui.Label("Scale Jitter:", width=100)
                            ui.FloatSlider(min=0.1, max=2.0)
                        with ui.HStack():
                            ui.Label("Rotation Jitter:", width=100)
                            ui.FloatSlider(min=0.0, max=360.0)
                        with ui.HStack():
                            ui.Label("Spacing:", width=100)
                            ui.FloatSlider(min=0.1, max=5.0)

                ui.Label("Ingested Assets", style={"font_size": 16, "font_weight": "bold"})
                with ui.ScrollingFrame(height=200):
                    ui.TreeView(self._assets_model, root_visible=False, header_visible=False, selection_changed_fn=self._on_asset_selected)

                ui.Button("Toggle Brush", clicked_fn=self._toggle_brush)
                ui.Button("Refresh Assets", clicked_fn=lambda: asyncio.ensure_future(self._refresh_assets()))

        self._brush_enabled = False
        self._is_painting = False
        self._input_subscriber = None
        self._stroke_points = []
        self._active_instancer = None
        self._selected_asset = None

        # Add a menu item to the Window menu to show/hide the scatter tool window
        self._menu = omni.kit.ui.get_editor_menu().add_item(
            "Window/Scatter Tool", self._on_menu_click, toggle=True, value=True
        )
        asyncio.ensure_future(self._refresh_assets())
        omni.kit.commands.register(ScatterPaintCommand)

    def on_shutdown(self):
        print("[omni.remix.scatter.tool] extension shutdown")
        if self._menu:
            omni.kit.ui.get_editor_menu().remove_item(self._menu)
        if self._window:
            self._window.destroy()
            self._window = None
        self._set_brush_enabled(False)
        omni.kit.commands.deregister(ScatterPaintCommand)

    def _toggle_brush(self):
        self._set_brush_enabled(not self._brush_enabled)

    def _set_brush_enabled(self, enabled):
        self._brush_enabled = enabled
        if self._brush_enabled:
            appwindow = omni.kit.app.get_app_interface()
            input_system = appwindow.get_input_system()
            self._input_subscriber = input_system.subscribe_to_input_events(self._on_input_event)
            print("Scatter brush enabled")
        else:
            if self._input_subscriber:
                self._input_subscriber.unsubscribe()
                self._input_subscriber = None
            print("Scatter brush disabled")

    def _on_input_event(self, event):
        if not self._brush_enabled:
            return True

        if event.type == omni.kit.app.InputEventType.MOUSE_PRESS and event.mouse.button == 0:
            self._is_painting = True
            self._stroke_points = []
            self._do_paint(event.mouse.x, event.mouse.y)
        elif event.type == omni.kit.app.InputEventType.MOUSE_RELEASE and event.mouse.button == 0:
            self._is_painting = False
            self._commit_stroke()
        elif event.type == omni.kit.app.InputEventType.MOUSE_MOVE and self._is_painting:
            self._do_paint(event.mouse.x, event.mouse.y)
        return True

    def _do_paint(self, x, y):
        if self._physx_scene_query is None:
            self._physx_scene_query = get_physx_scene_query_interface()

        viewport_window = omni.kit.ui.getViewportWindow()
        if not viewport_window:
            return
        viewport_api = viewport_window.viewport_api

        ray_origin, ray_dir = viewport_api.compute_ray_from_mouse_loc(x, y)
        hit = self._physx_scene_query.raycast_closest(ray_origin, ray_dir, 10000.0)

        if hit["hit"]:
            if not self._active_instancer:
                self._create_instancer(hit["rigid_body"])

            hit_pos = Gf.Vec3d(hit["position"])
            hit_normal = Gf.Vec3d(hit["normal"])

            density = self._density_slider.model.get_value_as_float()
            radius = self._radius_slider.model.get_value_as_float()

            for i in range(int(density)):
                rand_u = random.uniform(0, radius)
                rand_theta = random.uniform(0, 2 * math.pi)

                if abs(hit_normal[0]) < abs(hit_normal[1]):
                    tangent0 = Gf.Vec3d(1, 0, 0) - hit_normal * hit_normal[0]
                else:
                    tangent0 = Gf.Vec3d(0, 1, 0) - hit_normal * hit_normal[1]
                tangent0.Normalize()
                tangent1 = Gf.Cross(hit_normal, tangent0)

                point_on_plane = rand_u * (tangent0 * math.cos(rand_theta) + tangent1 * math.sin(rand_theta))

                final_pos = hit_pos + point_on_plane
                self._stroke_points.append(final_pos)

    def _create_instancer(self, hit_prim_path):
        stage = omni.usd.get_context().get_stage()
        hit_prim = stage.GetPrimAtPath(hit_prim_path)
        if not hit_prim:
            return

        anchor_path = f"/{hit_prim.GetName()}_anchor"
        instancer_path = f"{anchor_path}/scatter_instancer"

        anchor_prim = stage.GetPrimAtPath(anchor_path)
        if not anchor_prim:
            anchor_prim = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()

        instancer_prim = stage.GetPrimAtPath(instancer_path)
        if not instancer_prim:
            instancer_prim = UsdGeom.PointInstancer.Define(stage, instancer_path).GetPrim()

        self._active_instancer = UsdGeom.PointInstancer(instancer_prim)

        if self._selected_asset:
            proto_path = Sdf.Path(self._selected_asset)
            self._active_instancer.GetPrototypesRel().AddTarget(proto_path)
            # Make sure the prototype is instanceable
            proto_prim = stage.GetPrimAtPath(self._selected_asset)
            if proto_prim:
                proto_prim.SetInstanceable(True)

    def _commit_stroke(self):
        if not self._active_instancer or not self._stroke_points:
            return

        omni.kit.commands.execute(
            "ScatterPaintCommand",
            instancer_path=self._active_instancer.GetPath().pathString,
            points=self._stroke_points,
        )

        self._stroke_points = []
        self._active_instancer = None

    def _on_asset_selected(self, model, item):
        self._selected_asset = model.get_item_value(item)
        print(f"Selected asset: {self._selected_asset}")

    def _on_menu_click(self, value):
        self._window.visible = value

    async def _refresh_assets(self):
        client = ServicesClient()
        query_model = GetAvailableAssetsQueryModel(asset_type=AssetType.MESH)
        try:
            response = await client.assets.get_available_assets(query_params=query_model)
            self._assets_model.clear()
            if response and response.file_paths:
                for asset_path in response.file_paths:
                    self._assets_model.append_child(None, ui.SimpleString(asset_path))
        except Exception as e:
            print(f"Error refreshing assets: {e}")
