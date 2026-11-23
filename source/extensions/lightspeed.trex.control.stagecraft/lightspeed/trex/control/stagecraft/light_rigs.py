"""
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from __future__ import annotations

__all__ = ["LightRigMenu"]

import functools
import string
from collections.abc import Callable, Iterable
from pathlib import Path

import carb
import carb.settings
import carb.tokens
import omni.usd
from lightspeed.trex.contexts.setup import Contexts as _TrexContexts
from omni.kit.viewport.menubar.lighting import actions as _lighting_actions
from omni.kit.viewport.menubar.lighting import menu_container as _lighting_menu_container
from omni.kit.viewport.menubar.lighting.utility import VisibilityEdit as _VisibilityEdit
from pxr import Sdf, Usd, UsdGeom, UsdLux

from .commands import get_lighting_mode_setting_key as _get_lighting_mode_setting_key
from .mod_skies import ModSkyRigs as _ModSkyRigs

_RIGS_SETTING = "/exts/omni.kit.viewport.menubar.lighting/rigs"
_EXTRA_DIRECTORIES_SETTING = "/exts/lightspeed.trex.control.stagecraft/lightRigs/extraDirectories"
_MOD_SKY_DIRECTORIES_SETTING = "/exts/lightspeed.trex.control.stagecraft/lightRigs/modSkyDirectories"
# The Lighting menu's actions and menu modules each import this function by name.
_RIG_LISTER_NAME = "_get_rig_names_and_paths"
_LIGHT_RIG_PRIM_PATH = Sdf.Path("/OmniKit_Viewport_LightRig")
_DOME_LIGHT_TYPES = frozenset({"DomeLight", "DomeLight_1"})

RigLister = Callable[[str], Iterable[tuple[str, Path]] | None]


def _list_rig_directory(directory: Path) -> list[tuple[str, Path]]:
    """Return the rig files under a directory, labelled and ordered the way the Lighting menu lists its own."""
    rig_paths = sorted(directory.rglob("*.usd*"), key=lambda rig_path: rig_path.stem.lower())
    return [(string.capwords(rig_path.stem.replace("_", " ")), rig_path) for rig_path in rig_paths]


def _is_light_kept_with_mod_sky(prim: Usd.Prim, prim_path: Sdf.Path) -> bool:
    """Return whether a stage light stays lit while a mod sky rig is applied."""
    # The game keeps the level's lights under its Remix API dome, and the runtime draws only one dome per frame, so
    # the capture's own sky dome stays hidden behind the mod sky.
    return (
        prim.HasAPI(UsdLux.LightAPI)
        and prim.GetTypeName() not in _DOME_LIGHT_TYPES
        and not prim_path.HasPrefix(_LIGHT_RIG_PRIM_PATH)
    )


def _refresh_lighting_menus() -> None:
    """Make every viewport Lighting menu rebuild its list of light rigs."""
    # The menus rebuild their rig items only when the rig directory setting changes, so the setting switches between
    # two spellings of the same directory.
    settings = carb.settings.get_settings()
    rigs_directory = settings.get(_RIGS_SETTING)
    if not isinstance(rigs_directory, str) or not rigs_directory:
        return
    settings.set(_RIGS_SETTING, rigs_directory[:-1] if rigs_directory.endswith("/") else f"{rigs_directory}/")


class LightRigMenu:
    """Add extra light rigs and the mod skies of the open StageCraft project to the viewport Lighting menu."""

    def __init__(self, context_name: str = _TrexContexts.STAGE_CRAFT.value):
        """Start listing the extra rigs and the mod skies in every viewport Lighting menu.

        Args:
            context_name: The USD context whose project supplies the mod skies.
        """
        settings = carb.settings.get_settings()
        tokens = carb.tokens.get_tokens_interface()
        self._context_name = context_name
        self._extra_directories = [
            Path(tokens.resolve(directory)) for directory in settings.get(_EXTRA_DIRECTORIES_SETTING) or []
        ]
        self._mod_sky_rigs = _ModSkyRigs(
            Path(tokens.resolve("${temp}")) / "lightspeed.trex.control.stagecraft" / "mod_sky_rigs",
            list(settings.get(_MOD_SKY_DIRECTORIES_SETTING) or []),
        )
        self._original_rig_listers: dict = {}
        self._lighting_mode_key = None
        self._lighting_mode_subscription = None

        usd_context = omni.usd.get_context(context_name)
        self._stage_event_subscription = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self.__on_stage_event, name="LightRigMenuStageEvent"
        )
        for module in (_lighting_actions, _lighting_menu_container):
            rig_lister = vars(module)[_RIG_LISTER_NAME]
            self._original_rig_listers[module] = rig_lister
            vars(module)[_RIG_LISTER_NAME] = functools.partial(self.list_light_rigs, rig_lister)
        if usd_context.get_stage():
            self.__watch_stage(refresh_menus=False)

    def list_light_rigs(self, rig_lister: RigLister, extension_id: str) -> list[tuple[str, Path]]:
        """Return the Lighting menu's own rigs, then the extra rigs, then the mod skies of the StageCraft project.

        A rig whose label is already listed is left out, so the menu's own rig of a label wins.

        Args:
            rig_lister: The Lighting menu's own rig listing function.
            extension_id: The Lighting menu extension whose rig directory setting the listing reads.

        Returns:
            The label and file of every rig, in menu order.
        """
        rigs = list(rig_lister(extension_id) or [])
        labels = {label for label, _ in rigs}
        extra_rigs = [
            rig for directory in self._extra_directories if directory.is_dir() for rig in _list_rig_directory(directory)
        ]
        self.__refresh_mod_skies()
        for label, rig_path in [*extra_rigs, *self._mod_sky_rigs.rigs]:
            if label not in labels:
                labels.add(label)
                rigs.append((label, rig_path))
        return rigs

    def destroy(self) -> None:
        """Restore the Lighting menu's own rig listing and stop following the StageCraft stage."""
        for module, rig_lister in self._original_rig_listers.items():
            vars(module)[_RIG_LISTER_NAME] = rig_lister
        self._original_rig_listers = {}
        if self._lighting_mode_subscription:
            carb.settings.get_settings().unsubscribe_to_change_events(self._lighting_mode_subscription)
        self._lighting_mode_subscription = None
        self._stage_event_subscription = None

    def __refresh_mod_skies(self) -> bool:
        stage = omni.usd.get_context(self._context_name).get_stage()
        project_path = stage.GetRootLayer().realPath if stage else ""
        up_axis = UsdGeom.GetStageUpAxis(stage) if stage else UsdGeom.Tokens.z
        try:
            return self._mod_sky_rigs.refresh(Path(project_path) if project_path else None, up_axis)
        except (OSError, RuntimeError) as exc:
            carb.log_error(f"Could not write the mod sky light rigs of {project_path}: {exc}")
            return False

    def __watch_stage(self, refresh_menus: bool) -> None:
        settings = carb.settings.get_settings()
        if self._lighting_mode_subscription:
            settings.unsubscribe_to_change_events(self._lighting_mode_subscription)
        # The Lighting menu stores the lighting mode under a key of the stage it applies to, so each opened stage
        # needs its own subscription.
        self._lighting_mode_key = _get_lighting_mode_setting_key(omni.usd.get_context(self._context_name))
        self._lighting_mode_subscription = settings.subscribe_to_node_change_events(
            self._lighting_mode_key, self.__on_lighting_mode_changed
        )
        if self.__refresh_mod_skies() and refresh_menus:
            _refresh_lighting_menus()
        # The Lighting menu can reapply a mod sky while the stage opens, before this subscription exists.
        self.__keep_stage_lights_for_mod_sky()

    def __keep_stage_lights_for_mod_sky(self) -> None:
        lighting_mode = carb.settings.get_settings().get(self._lighting_mode_key)
        if not lighting_mode or lighting_mode not in self._mod_sky_rigs.labels:
            return
        stage = omni.usd.get_context(self._context_name).get_stage()
        if not stage:
            return
        # Applying any rig hides every stage light in the session layer; showing a light removes that opinion again.
        _VisibilityEdit(stage, _is_light_kept_with_mod_sky, None, api_types_for_prune=["LightAPI"]).run()

    def __on_lighting_mode_changed(self, _item, event_type: carb.settings.ChangeEventType) -> None:
        if event_type in (carb.settings.ChangeEventType.CREATED, carb.settings.ChangeEventType.CHANGED):
            self.__keep_stage_lights_for_mod_sky()

    def __on_stage_event(self, event) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            self.__watch_stage(refresh_menus=True)
