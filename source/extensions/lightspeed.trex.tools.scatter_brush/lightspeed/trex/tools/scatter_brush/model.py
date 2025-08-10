# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import carb
import carb.settings
from omni.flux.utils.common import Event, EventSubscription

_SETTINGS_ROOT = 'exts."lightspeed.trex.tools.scatter_brush"'


@dataclass
class ScatterBrushSettings:
    asset_usd_path: str = ""
    brush_radius: float = 0.25
    spacing: float = 0.5
    density: float = 8.0
    random_scale_min: float = 0.9
    random_scale_max: float = 1.1
    random_yaw: bool = True
    align_to_normals: bool = True
    max_surface_angle_deg: float = 45.0
    seed: int = 1337
    erase_mode: bool = False
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScatterBrushModel:
    """Typed settings model for Scatter Brush with persistence and events."""

    def __init__(self):
        self._settings = carb.settings.get_settings()
        self._on_changed = Event()
        self._on_toggle = Event()
        self._enabled: bool = bool(self._settings.get(f"{_SETTINGS_ROOT}.enabled"))
        self._data: ScatterBrushSettings = self._load()

    # Persistence
    def _load(self) -> ScatterBrushSettings:
        gs = self._settings
        return ScatterBrushSettings(
            asset_usd_path=str(gs.get(f"{_SETTINGS_ROOT}.asset_usd_path") or ""),
            brush_radius=float(gs.get(f"{_SETTINGS_ROOT}.brush_radius") or 0.25),
            spacing=float(gs.get(f"{_SETTINGS_ROOT}.spacing") or 0.5),
            density=float(gs.get(f"{_SETTINGS_ROOT}.density") or 8.0),
            random_scale_min=float(gs.get(f"{_SETTINGS_ROOT}.random_scale_min") or 0.9),
            random_scale_max=float(gs.get(f"{_SETTINGS_ROOT}.random_scale_max") or 1.1),
            random_yaw=bool(gs.get(f"{_SETTINGS_ROOT}.random_yaw") or True),
            align_to_normals=bool(gs.get(f"{_SETTINGS_ROOT}.align_to_normals") or True),
            max_surface_angle_deg=float(gs.get(f"{_SETTINGS_ROOT}.max_surface_angle_deg") or 45.0),
            seed=int(gs.get(f"{_SETTINGS_ROOT}.seed") or 1337),
            erase_mode=bool(gs.get(f"{_SETTINGS_ROOT}.erase_mode") or False),
            category=str(gs.get(f"{_SETTINGS_ROOT}.category") or ""),
        )

    def _save(self):
        gs = self._settings
        gs.set(f"{_SETTINGS_ROOT}.enabled", bool(self._enabled))
        gs.set(f"{_SETTINGS_ROOT}.asset_usd_path", self._data.asset_usd_path)
        gs.set(f"{_SETTINGS_ROOT}.brush_radius", float(self._data.brush_radius))
        gs.set(f"{_SETTINGS_ROOT}.spacing", float(self._data.spacing))
        gs.set(f"{_SETTINGS_ROOT}.density", float(self._data.density))
        gs.set(f"{_SETTINGS_ROOT}.random_scale_min", float(self._data.random_scale_min))
        gs.set(f"{_SETTINGS_ROOT}.random_scale_max", float(self._data.random_scale_max))
        gs.set(f"{_SETTINGS_ROOT}.random_yaw", bool(self._data.random_yaw))
        gs.set(f"{_SETTINGS_ROOT}.align_to_normals", bool(self._data.align_to_normals))
        gs.set(f"{_SETTINGS_ROOT}.max_surface_angle_deg", float(self._data.max_surface_angle_deg))
        gs.set(f"{_SETTINGS_ROOT}.seed", int(self._data.seed))
        gs.set(f"{_SETTINGS_ROOT}.erase_mode", bool(self._data.erase_mode))
        gs.set(f"{_SETTINGS_ROOT}.category", self._data.category)

    # Accessors
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def data(self) -> ScatterBrushSettings:
        return self._data

    # Mutators trigger save + event
    def set_enabled(self, value: bool):
        value = bool(value)
        if self._enabled != value:
            self._enabled = value
            self._save()
            self._on_toggle(value)
            self._on_changed(self._data)

    def update(self, **kwargs):
        changed = False
        for k, v in kwargs.items():
            if hasattr(self._data, k) and getattr(self._data, k) != v:
                setattr(self._data, k, v)
                changed = True
        if changed:
            self._save()
            self._on_changed(self._data)

    # Events
    def subscribe_changed(self, fn: Callable[[ScatterBrushSettings], None]) -> EventSubscription:
        return EventSubscription(self._on_changed, fn)

    def subscribe_toggle(self, fn: Callable[[bool], None]) -> EventSubscription:
        return EventSubscription(self._on_toggle, fn)

    # Prototypes index helpers
    def get_prototypes_index_path(self) -> Optional[Path]:
        raw = self._settings.get(f"{_SETTINGS_ROOT}.prototypes_index")
        if raw:
            p = Path(str(raw))
            if p.exists():
                return p
        # Best-effort defaults: try common locations if present
        candidates: List[Path] = []
        try:
            import os
            root = Path(os.getcwd())
            candidates.append(root / "extensions/omniscatter/data/prototypes/prototypes_index.json")
            candidates.append(root / "_build/release/extensions/omniscatter/data/prototypes/prototypes_index.json")
            candidates.append(root / "_build/debug/extensions/omniscatter/data/prototypes/prototypes_index.json")
        except Exception:
            pass
        for c in candidates:
            if c.exists():
                return c
        return None

    def load_prototypes(self) -> List[Dict[str, Any]]:
        path = self.get_prototypes_index_path()
        if not path:
            return []
        try:
            import json
            data = json.loads(path.read_text())
            protos = data.get("prototypes") or []
            # Normalize paths
            out: List[Dict[str, Any]] = []
            for p in protos:
                item = dict(p)
                # Keep absolute/relative path as-is; UI will resolve relative to index dir
                out.append(item)
            return out
        except Exception as e:
            carb.log_warn(f"Failed to load prototypes index: {e}")
            return []

# Global/shared instance access
_global_model: ScatterBrushModel | None = None

def get_model() -> ScatterBrushModel:
    global _global_model
    if _global_model is None:
        _global_model = ScatterBrushModel()
    return _global_model