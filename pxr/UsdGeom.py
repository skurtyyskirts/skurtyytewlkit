from __future__ import annotations

from typing import Union

from .Sdf import Path
from .Usd import Stage


class _Defined:
    def __init__(self, stage: Stage, path: Union[str, Path], schema_name: str):
        self._stage = stage
        self._path = path
        self._schema = schema_name

    def GetPrim(self):  # pragma: no cover - simple accessor
        p = self._path.pathString if isinstance(self._path, Path) else str(self._path)
        return self._stage._define_prim(p, self._schema)


def _define(stage: Stage, path: Union[str, Path], schema_name: str):
    stage._define_prim(path, schema_name)
    return _Defined(stage, path, schema_name)


class Xform:
    @staticmethod
    def Define(stage: Stage, path: Union[str, Path]):
        return _define(stage, path, "Xform")


class Scope:
    @staticmethod
    def Define(stage: Stage, path: Union[str, Path]):
        return _define(stage, path, "Scope")


class PointInstancer:
    __name__ = "PointInstancer"

    @staticmethod
    def Define(stage: Stage, path: Union[str, Path]):
        return _define(stage, path, "PointInstancer")


class Mesh:  # pragma: no cover - placeholder to satisfy imports
    pass