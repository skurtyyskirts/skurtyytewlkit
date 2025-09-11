# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carb
import omni.ui as ui

from lightspeed.common import constants as _constants
import omni.client
from lightspeed.trex.utils.common.asset_utils import is_asset_ingested as _is_asset_ingested
from omni.flux.utils.widget.collapsable_frame import PropertyCollapsableFrameWithInfoPopup

from .model import get_model


class ScatterBrushPane:
    """Docked left panel containing Scatter Brush settings and an asset picker."""

    TITLE = "SCATTER BRUSH"

    def __init__(self):
        self._model = get_model()
        self._window: Optional[ui.Window] = None
        self._asset_container: Optional[ui.Widget] = None
        self._preset_combo_model: Optional[ui.SimpleIntModel] = None
        self._preset_names: List[str] = []
        self._asset_list: List[Dict[str, Any]] = []
        self._asset_combo_model: Optional[ui.SimpleIntModel] = None
        self._build_ui()

    def _build_ui(self):
        self._window = ui.Window(
            "Scatter Brush",
            width=340,
            height=600,
            dockPreference=ui.DockPreference.LEFT,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_COLLAPSE),
        )
        with self._window.frame:
            with ui.ZStack():
                ui.Rectangle(name="WorkspaceBackground")
                with ui.ScrollingFrame(name="PropertiesPaneSection", horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=ui.Pixel(8)):
                        ui.Spacer(height=ui.Pixel(8))
                        with ui.HStack():
                            ui.Spacer(width=ui.Pixel(8))
                            with ui.VStack(spacing=ui.Pixel(8)):
                                # Collapsible group header
                                coll = PropertyCollapsableFrameWithInfoPopup(
                                    self.TITLE,
                                    info_text=(
                                        "Paint-scatter pre-ingested assets onto surfaces.\n\n"
                                        "- Use the toolbar toggle or 'B' to activate brush mode.\n"
                                        "- Settings persist across sessions.\n"
                                    ),
                                    collapsed=False,
                                )
                                with coll:
                                    self._build_content()

    def _build_content(self):
        # Asset section
        with ui.VStack(spacing=ui.Pixel(6)):
            ui.Label("Asset", name="PropertiesWidgetLabel")
            self._build_asset_dropdown()

        ui.Spacer(height=ui.Pixel(8))
        ui.Line(name="PropertiesPaneSectionTitle")

        # Brush settings
        with ui.VGrid(column_count=2, row_spacing=ui.Pixel(6), column_spacing=ui.Pixel(8)):
            ui.Label("Radius", name="PropertiesWidgetLabel")
            r = ui.FloatDrag(min=0.01, max=1000.0, step=0.01, style_type_name_override="Field")
            r.model.set_value(self._model.data.brush_radius)
            r.model.add_value_changed_fn(lambda m: self._on_change(brush_radius=float(m.get_value_as_float())))

            ui.Label("Spacing", name="PropertiesWidgetLabel")
            sp = ui.FloatDrag(min=0.0, max=1000.0, step=0.01, style_type_name_override="Field")
            sp.model.set_value(self._model.data.spacing)
            sp.model.add_value_changed_fn(lambda m: self._on_change(spacing=float(m.get_value_as_float())))

            ui.Label("Density (pts/m²)", name="PropertiesWidgetLabel")
            den = ui.FloatDrag(min=0.0, max=10000.0, step=0.1, style_type_name_override="Field")
            den.model.set_value(self._model.data.density)
            den.model.add_value_changed_fn(lambda m: self._on_change(density=float(m.get_value_as_float())))

            ui.Label("Random Scale Min", name="PropertiesWidgetLabel")
            smin = ui.FloatDrag(min=0.01, max=100.0, step=0.01, style_type_name_override="Field")
            smin.model.set_value(self._model.data.random_scale_min)
            smin.model.add_value_changed_fn(lambda m: self._on_change(random_scale_min=float(m.get_value_as_float())))

            ui.Label("Random Scale Max", name="PropertiesWidgetLabel")
            smax = ui.FloatDrag(min=0.01, max=100.0, step=0.01, style_type_name_override="Field")
            smax.model.set_value(self._model.data.random_scale_max)
            smax.model.add_value_changed_fn(lambda m: self._on_change(random_scale_max=float(m.get_value_as_float())))

            ui.Label("Random Yaw", name="PropertiesWidgetLabel")
            ry = ui.CheckBox()
            ry.model.set_value(self._model.data.random_yaw)
            ry.model.add_value_changed_fn(lambda m: self._on_change(random_yaw=bool(m.get_value_as_bool())))

            ui.Label("Align to Normals", name="PropertiesWidgetLabel")
            aln = ui.CheckBox()
            aln.model.set_value(self._model.data.align_to_normals)
            aln.model.add_value_changed_fn(lambda m: self._on_change(align_to_normals=bool(m.get_value_as_bool())))

            ui.Label("Surface Angle Limit", name="PropertiesWidgetLabel")
            ang = ui.FloatDrag(min=0.0, max=90.0, step=0.5, style_type_name_override="Field")
            ang.model.set_value(self._model.data.max_surface_angle_deg)
            ang.model.add_value_changed_fn(
                lambda m: self._on_change(max_surface_angle_deg=float(m.get_value_as_float()))
            )

            ui.Label("Seed", name="PropertiesWidgetLabel")
            sd = ui.IntDrag(min=0, max=2**31 - 1, step=1, style_type_name_override="Field")
            sd.model.set_value(self._model.data.seed)
            sd.model.add_value_changed_fn(lambda m: self._on_change(seed=int(m.get_value_as_int())))

            ui.Label("Erase Mode", name="PropertiesWidgetLabel")
            er = ui.CheckBox()
            er.model.set_value(self._model.data.erase_mode)
            er.model.add_value_changed_fn(lambda m: self._on_change(erase_mode=bool(m.get_value_as_bool())))

        ui.Spacer(height=ui.Pixel(8))
        ui.Line(name="PropertiesPaneSectionTitle")

        # Category dropdown
        with ui.HStack(spacing=ui.Pixel(8)):
            ui.Label("Category", name="PropertiesWidgetLabel", width=ui.Percent(40))
            names = list(_constants.REMIX_CATEGORIES.keys())
            initial_index = max(0, names.index(self._model.data.category) if self._model.data.category in names else 0)
            combo_model = ui.SimpleIntModel(initial_index)
            def _on_combo_changed(m):
                idx = int(m.get_value_as_int())
                value = names[idx] if 0 <= idx < len(names) else ""
                self._on_change(category=value)
            combo = ui.ComboBox(combo_model, *names)
            combo_model.add_value_changed_fn(_on_combo_changed)

        ui.Spacer(height=ui.Pixel(8))
        ui.Line(name="PropertiesPaneSectionTitle")

        # Presets controls
        with ui.VStack(spacing=ui.Pixel(6)):
            ui.Label("Presets", name="PropertiesWidgetLabel")
            with ui.HStack(spacing=ui.Pixel(6)):
                # Preset dropdown
                self._preset_names = self._model.list_presets()
                initial_idx = 0 if self._preset_names else -1
                self._preset_combo_model = ui.SimpleIntModel(max(0, initial_idx))
                self._preset_combo = ui.ComboBox(self._preset_combo_model, *(self._preset_names or ["<none>"]))
                # Buttons
                def _save():
                    # Prompt text field inline for name
                    def _commit(name_field_model):
                        name = name_field_model.get_value_as_string()
                        if name:
                            self._model.save_preset(name)
                            self._refresh_preset_dropdown(select=name)
                    name_field = ui.StringField(height=ui.Pixel(20), style_type_name_override="Field")
                    name_field.model.add_end_edit_fn(_commit)
                ui.Button("Save", clicked_fn=_save)

                def _load():
                    idx = int(self._preset_combo_model.get_value_as_int())
                    name = self._preset_names[idx] if 0 <= idx < len(self._preset_names) else None
                    if name:
                        self._model.load_preset(name)
                ui.Button("Load", clicked_fn=_load)

                def _delete():
                    idx = int(self._preset_combo_model.get_value_as_int())
                    name = self._preset_names[idx] if 0 <= idx < len(self._preset_names) else None
                    if name:
                        self._model.delete_preset(name)
                        self._refresh_preset_dropdown()
                ui.Button("Delete", clicked_fn=_delete)

    def _refresh_preset_dropdown(self, select: Optional[str] = None):
        self._preset_names = self._model.list_presets()
        # Rebuild combobox items
        if self._preset_combo_model is None:
            return
        # Find selection index
        sel_idx = 0
        if select and select in self._preset_names:
            sel_idx = self._preset_names.index(select)
        self._preset_combo_model.set_value(sel_idx if self._preset_names else 0)
        # Replace ComboBox items by recreating it inside a transient frame
        # Simpler approach: rebuild the whole content would also work, but avoid flicker.
        # Find parent container from current layout and replace widget
        # For brevity, we do a simple rebuild of the panel.
        self._window.frame.clear()
        with self._window.frame:
            with ui.ZStack():
                ui.Rectangle(name="WorkspaceBackground")
                with ui.ScrollingFrame(name="PropertiesPaneSection", horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=ui.Pixel(8)):
                        ui.Spacer(height=ui.Pixel(8))
                        with ui.HStack():
                            ui.Spacer(width=ui.Pixel(8))
                            with ui.VStack(spacing=ui.Pixel(8)):
                                coll = PropertyCollapsableFrameWithInfoPopup(
                                    self.TITLE,
                                    info_text=(
                                        "Paint-scatter pre-ingested assets onto surfaces.\n\n"
                                        "- Use the toolbar toggle or 'B' to activate brush mode.\n"
                                        "- Settings persist across sessions.\n"
                                    ),
                                    collapsed=False,
                                )
                                with coll:
                                    self._build_content()

    def _build_asset_dropdown(self):
        # Source assets from toolkit list (tools/custom/paintable_assets.py)
        try:
            from tools.custom.paintable_assets import list_paintable_assets
            assets = list_paintable_assets()
        except Exception as e:
            carb.log_warn(f"ScatterBrushPane: failed to query paintable assets: {e}")
            assets = []
        self._asset_list = [
            {
                "name": a.display_name,
                "usd_path": a.usd_path,
                "thumb": a.thumbnail_path,
            }
            for a in assets
        ]
        # Fallback to ingested USD enumeration if toolkit returned nothing
        if not self._asset_list:
            self._asset_list = self._enumerate_ingested_usds()

        # Build a simple dropdown for selection by name
        names = [str(a.get("name") or a.get("usd_path") or "asset") for a in self._asset_list]
        # Try to select current model asset if present
        selected_index = 0
        if self._model.data.asset_usd_path:
            for i, a in enumerate(self._asset_list):
                if str(a.get("usd_path")) == str(self._model.data.asset_usd_path):
                    selected_index = i
                    break
        self._asset_combo_model = ui.SimpleIntModel(selected_index if names else 0)
        def _on_asset_changed(m):
            idx = int(m.get_value_as_int())
            if 0 <= idx < len(self._asset_list):
                usd = self._asset_list[idx].get("usd_path")
                if usd:
                    self._on_change(asset_usd_path=str(usd))
        ui.ComboBox(self._asset_combo_model, *(names or ["<no assets>"]))
        self._asset_combo_model.add_value_changed_fn(_on_asset_changed)

    def _enumerate_ingested_usds(self) -> List[Dict[str, Any]]:
        protos: List[Dict[str, Any]] = []
        try:
            import omni.usd
            ctx = omni.usd.get_context()
            stage_url = ctx.get_stage_url()
            if stage_url:
                root_url = omni.client.normalize_url(str(omni.client.Uri(stage_url).get_dirname()))
                ingested_url = omni.client.combine_urls(root_url, _constants.REMIX_INGESTED_ASSETS_FOLDER)

                def walk(url: str):
                    res, entries = omni.client.list(url)
                    if res != omni.client.Result.OK:
                        return
                    for e in entries:
                        child = omni.client.combine_urls(url, e.relative_path)
                        if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                            walk(child)
                        elif e.flags & omni.client.ItemFlags.READABLE_FILE:
                            lower = e.relative_path.lower()
                            if any(lower.endswith(ext) for ext in _constants.USD_EXTENSIONS):
                                path = omni.client.break_url(child).path
                                if path and _is_asset_ingested(path):
                                    protos.append({"name": e.relative_path, "usd_path": path})
                walk(ingested_url)
                protos.sort(key=lambda d: str(d.get("name") or d.get("usd_path")).lower())
        except Exception as e:
            carb.log_warn(f"ScatterBrushPane: fallback enumeration failed: {e}")
        return protos

    def _on_change(self, **kwargs):
        self._model.update(**kwargs)

    def destroy(self):
        if self._window:
            self._window.visible = False
            self._window = None
