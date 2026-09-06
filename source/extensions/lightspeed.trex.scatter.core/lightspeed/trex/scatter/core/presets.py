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

__all__ = ["PresetStore", "get_default_presets_directory"]

import json
from pathlib import Path

import carb
import carb.settings
import carb.tokens
from pydantic import ValidationError

from .constants import PRESETS_DIR_SETTING, PRESETS_SUBDIRECTORY
from .settings import ScatterBrushSettings

_INVALID_NAME_CHARACTERS = frozenset('<>:"/\\|?*')


def get_default_presets_directory() -> Path:
    """Return the directory that holds the brush preset files.

    The carb setting ``PRESETS_DIR_SETTING`` wins when it is a non-empty string; otherwise the presets live under the
    application documents folder resolved from the ``${app_documents}`` token.
    """
    override = carb.settings.get_settings().get(PRESETS_DIR_SETTING)
    if isinstance(override, str) and override.strip():
        return Path(override.strip())
    documents = carb.tokens.get_tokens_interface().resolve("${app_documents}")
    return Path(documents) / PRESETS_SUBDIRECTORY


def _validate_preset_name(name: str) -> str:
    """Return the trimmed preset name, raising ``ValueError`` when it cannot be used as a file stem."""
    if not isinstance(name, str):
        raise ValueError(f"Preset name must be a string, not {type(name).__name__}")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Preset name cannot be empty")
    if cleaned.endswith("."):
        raise ValueError(f"Preset name '{cleaned}' cannot end with a period")
    invalid_characters = sorted(
        {character for character in cleaned if character in _INVALID_NAME_CHARACTERS or ord(character) < 32}
    )
    if invalid_characters:
        joined = "".join(invalid_characters)
        raise ValueError(f"Preset name '{cleaned}' contains characters that are not allowed in file names: {joined}")
    return cleaned


class PresetStore:
    """Named brush presets stored as one JSON file per preset inside a directory.

    The directory is created lazily on the first save. Preset names are used verbatim as file stems, so names that
    are empty, end with a period, or contain path separators or other characters that are invalid in file names are
    rejected with ``ValueError``. The name recorded inside a file is always derived from the file name on load.
    """

    def __init__(self, directory: Path):
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        """Directory holding the preset files."""
        return self._directory

    def list_names(self) -> list[str]:
        """Return the sorted names of every preset file in the directory."""
        if not self._directory.is_dir():
            return []
        return sorted(path.stem for path in self._directory.glob("*.json") if path.is_file())

    def exists(self, name: str) -> bool:
        """Return whether a preset file with this name exists.

        Raises:
            ValueError: If the name is not a valid preset name.
        """
        return self._path_for(name).is_file()

    def load(self, name: str) -> ScatterBrushSettings:
        """Read a preset file and build validated settings whose ``preset_name`` is the file name.

        Raises:
            ValueError: If the name is invalid, the file is missing or unreadable, or its content is not valid JSON
                describing acceptable brush settings.
        """
        path = self._path_for(name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Cannot read preset '{name}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Preset '{name}' is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Preset '{name}' must contain a JSON object")
        data["preset_name"] = path.stem
        try:
            return ScatterBrushSettings.from_json_dict(data)
        except ValidationError as exc:
            raise ValueError(f"Preset '{name}' contains invalid values: {exc}") from exc

    def save(self, name: str, settings: ScatterBrushSettings) -> Path:
        """Write the settings to ``<name>.json`` with ``preset_name`` set to the name and return the file path.

        Raises:
            ValueError: If the name is not a valid preset name.
            OSError: If the file cannot be written.
        """
        path = self._path_for(name)
        data = settings.to_json_dict()
        data["preset_name"] = path.stem
        self._directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def rename(self, old: str, new: str) -> None:
        """Move the preset file from one name to another.

        Raises:
            ValueError: If either name is invalid, the source is missing, or a different preset already uses the new
                name.
        """
        source = self._path_for(old)
        destination = self._path_for(new)
        if not source.is_file():
            raise ValueError(f"Preset '{old}' does not exist")
        if destination.exists() and not destination.samefile(source):
            raise ValueError(f"Preset '{new}' already exists")
        source.replace(destination)

    def clone(self, src: str, dst: str) -> None:
        """Copy a preset under a new name.

        Raises:
            ValueError: If either name is invalid, the source cannot be loaded, or the destination already exists.
        """
        settings = self.load(src)
        if self._path_for(dst).exists():
            raise ValueError(f"Preset '{dst}' already exists")
        self.save(dst, settings)

    def delete(self, name: str) -> None:
        """Delete the preset file; a missing preset is not an error.

        Raises:
            ValueError: If the name is not a valid preset name.
        """
        self._path_for(name).unlink(missing_ok=True)

    def _path_for(self, name: str) -> Path:
        """Return the file path for a validated preset name."""
        return self._directory / f"{_validate_preset_name(name)}.json"
