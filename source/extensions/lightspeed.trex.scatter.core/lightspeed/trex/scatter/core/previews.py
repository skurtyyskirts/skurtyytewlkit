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

__all__ = ["write_contact_sheets", "write_texture_preview"]

import math
from collections.abc import Sequence
from pathlib import Path

import carb
from PIL import Image, ImageDraw, ImageFont

_LOG_PREFIX = "[lightspeed.trex.scatter.core]"
_TILE_SIZE = 192
_COLUMNS = 6
_TILES_PER_SHEET = 36
_LABEL_HEIGHT = 30
_BACKGROUND = (32, 32, 32)
_LABEL_COLOR = (255, 255, 255)


def write_texture_preview(texture_path: str, output_path: Path) -> bool:
    """Write a PNG thumbnail of a texture's color channels, fitted in a tile.

    The alpha channel is dropped: games often keep something other than opacity in it, which would blank out a color
    that is otherwise easy to recognize.

    Args:
        texture_path: Texture file to read, in any format Pillow decodes, including the BC and RGBA DDS of captures.
        output_path: PNG file to write.

    Returns:
        False (logged) when the texture cannot be read or decoded.
    """
    try:
        with Image.open(texture_path) as image:
            preview = image.convert("RGB")
    except (OSError, NotImplementedError, ValueError) as error:
        carb.log_warn(f"{_LOG_PREFIX} Cannot preview texture '{texture_path}': {error}")
        return False
    preview.thumbnail((_TILE_SIZE, _TILE_SIZE))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)
    return True


def write_contact_sheets(tiles: Sequence[tuple[int, Path | None]], directory: Path) -> list[Path]:
    """Lay previews out on numbered sheets, so many surfaces can be told apart from a few images.

    Args:
        tiles: ``(number, preview)`` per surface, in sheet order; a surface without a preview keeps its numbered cell.
        directory: Directory the ``contact_sheet_<n>.png`` files are written to.

    Returns:
        The sheets written, empty when there are no tiles.
    """
    font = ImageFont.load_default(size=20)
    cell_height = _TILE_SIZE + _LABEL_HEIGHT
    sheets = []
    for sheet_index, start in enumerate(range(0, len(tiles), _TILES_PER_SHEET)):
        chunk = tiles[start : start + _TILES_PER_SHEET]
        rows = math.ceil(len(chunk) / _COLUMNS)
        sheet = Image.new("RGB", (_COLUMNS * _TILE_SIZE, rows * cell_height), _BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for index, (number, preview) in enumerate(chunk):
            left = (index % _COLUMNS) * _TILE_SIZE
            top = (index // _COLUMNS) * cell_height
            if preview is not None:
                with Image.open(preview) as tile:
                    sheet.paste(
                        tile.convert("RGB"),
                        (left + (_TILE_SIZE - tile.width) // 2, top + (_TILE_SIZE - tile.height) // 2),
                    )
            draw.text((left + 6, top + _TILE_SIZE + 4), f"#{number}", fill=_LABEL_COLOR, font=font)
        path = directory / f"contact_sheet_{sheet_index + 1:02d}.png"
        directory.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        sheets.append(path)
    return sheets
