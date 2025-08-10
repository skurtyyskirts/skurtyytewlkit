# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import carb
import omni.ui as ui

from lightspeed.common import constants as _constants
from omni.flux.utils.widget.collapsable_frame import PropertyCollapsableFrameWithInfoPopup

from .model import get_model


class ScatterBrushPane:
    """Docked left panel containing Scatter Brush settings and an asset picker."""

    TITLE = "SCATTER BRUSH"

    def __init__(self):
        self._model = get_model()
        self._window: Optional[ui.Window] = None
        self._asset_container: Optional[ui.Widget] = None
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
        # Asset picker
        with ui.VStack(spacing=ui.Pixel(6)):
            ui.Label("Asset", name="PropertiesWidgetLabel")
            self._asset_container = ui.VStack()
            with self._asset_container:
                self._rebuild_assets()

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

    def _rebuild_assets(self):
        def _resolve_thumb(index_dir: Path, record: Dict[str, Any]) -> Optional[str]:
            thumb = record.get("thumbnail_128") or record.get("thumbnail")
            if not thumb:
                return None
            p = Path(thumb)
            if not p.is_absolute():
                p = index_dir / p
            return str(p)

        protos = self._model.load_prototypes()
        index_path = self._model.get_prototypes_index_path()
        index_dir = index_path.parent if index_path else None

        self._asset_container.clear()
        with self._asset_container:
            if not protos:
                ui.Label("No pre-ingested assets found.")
                return
            with ui.VStack(spacing=ui.Pixel(6)):
                for rec in protos:
                    name = str(rec.get("name") or rec.get("uuid") or "asset")
                    usd_path = rec.get("usd_path") or rec.get("usd") or ""
                    if not usd_path:
                        continue
                    with ui.HStack(height=ui.Pixel(56)):
                        # Thumbnail
                        thumb_url = _resolve_thumb(index_dir, rec) if index_dir else None
                        if thumb_url and Path(thumb_url).exists():
                            ui.Image(thumb_url, width=ui.Pixel(56), height=ui.Pixel(56))
                        else:
                            ui.Rectangle(width=ui.Pixel(56), height=ui.Pixel(56))
                        with ui.VStack():
                            ui.Label(name)
                            ui.Label(str(usd_path), name="PropertiesWidgetLabel")
                        # Select button
                        def _on_pick(this_usd=usd_path):
                            # Persist relative to index dir as provided in index json
                            self._on_change(asset_usd_path=str(this_usd))
                        ui.Button("Select", clicked_fn=_on_pick)

    def _on_change(self, **kwargs):
        self._model.update(**kwargs)

    def destroy(self):
        if self._window:
            self._window.visible = False
            self._window = None