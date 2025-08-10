from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Union


class Path:
    def __init__(self, path: str):
        self.pathString = str(path)

    def AppendChild(self, child: str) -> "Path":
        base = self.pathString.rstrip("/")
        child = str(child).lstrip("/")
        return Path(f"{base}/{child}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.pathString

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Path('{self.pathString}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self.pathString == other.pathString
        return False


class ValueTypeNames:
    Bool = "Bool"
    String = "String"
    # The rest are unused by the active tests, defined for completeness
    Float = "Float"
    Float2 = "Float2"
    Float3 = "Float3"
    Token = "Token"
    Asset = "Asset"
    TexCoord2fArray = "TexCoord2fArray"
    Normal3f = "Normal3f"
    Color3f = "Color3f"


class Layer:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.subLayerPaths: list[str] = []
        self.customLayerData = {}
        self._authored_prim_paths: Set[str] = set()

    @staticmethod
    def CreateAnonymous(name: str) -> "Layer":
        return Layer(name)

    def GetPrimAtPath(self, path: Union[str, Path]) -> Optional[object]:
        p = path.pathString if isinstance(path, Path) else str(path)
        return object() if p in self._authored_prim_paths else None

    # Internal helper used by Usd.Stage to record authored specs
    def _record_prim_spec(self, path: Union[str, Path]) -> None:
        p = path.pathString if isinstance(path, Path) else str(path)
        self._authored_prim_paths.add(p)