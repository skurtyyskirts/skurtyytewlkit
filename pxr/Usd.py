from __future__ import annotations

from typing import Dict, Optional, Union

from .Sdf import Layer, Path

# Expose a module-level flag so higher-level code can detect the stub
_FAKE_PXR = True


class Attribute:
    def __init__(self, name: str):
        self._name = name
        self._value = None

    def Set(self, value):
        self._value = value

    def Get(self):  # pragma: no cover - trivial
        return self._value


class Prim:
    def __init__(self, path: Union[str, Path], type_name: str, stage: "Stage"):
        self._path = path if isinstance(path, Path) else Path(str(path))
        self._type_name = type_name
        self._stage = stage
        self._attributes: Dict[str, Attribute] = {}

    def IsValid(self) -> bool:  # pragma: no cover - trivial
        return True

    def GetPath(self) -> Path:  # pragma: no cover - trivial
        return self._path

    def CreateAttribute(self, name, value_type, custom=True) -> Attribute:  # noqa: ARG002
        attr = self._attributes.get(name)
        if not attr:
            attr = Attribute(name)
            self._attributes[name] = attr
        return attr

    def GetAttribute(self, name: str) -> Optional[Attribute]:
        return self._attributes.get(name)

    def IsA(self, schema_cls) -> bool:
        schema_name = getattr(schema_cls, "__name__", None)
        return schema_name == self._type_name


class EditTarget:
    def __init__(self, layer: Layer):
        self._layer = layer

    def GetLayer(self) -> Layer:  # pragma: no cover - trivial
        return self._layer


class Stage:
    # Marker so higher-level code can treat this stub as non-USD if desired
    _FAKE_PXR = True

    def __init__(self):
        self._prims: Dict[str, Prim] = {}
        self._edit_target: Optional[EditTarget] = None
        self._root_layer: Layer = Layer.CreateAnonymous("root.usda")
        self._default_prim: Optional[Prim] = None

    @staticmethod
    def CreateInMemory() -> "Stage":
        return Stage()

    @staticmethod
    def CreateNew(_path: str) -> "Stage":  # pragma: no cover - unused in active tests
        return Stage()

    @staticmethod
    def Open(_path: str) -> Optional["Stage"]:  # pragma: no cover - unused with stub
        return None

    def GetRootLayer(self) -> Layer:
        return self._root_layer

    def SetEditTarget(self, edit_target: EditTarget) -> None:
        self._edit_target = edit_target

    def GetEditTarget(self) -> EditTarget:
        return self._edit_target  # type: ignore[return-value]

    def GetPrimAtPath(self, path: Union[str, Path]) -> Optional[Prim]:
        p = path.pathString if isinstance(path, Path) else str(path)
        return self._prims.get(p)

    def SetDefaultPrim(self, prim: Prim) -> None:  # pragma: no cover - trivial
        self._default_prim = prim

    def Save(self) -> None:  # pragma: no cover - no-op for stub
        pass

    # Internal helper used by UsdGeom.Define
    def _define_prim(self, path: Union[str, Path], type_name: str) -> Prim:
        p = path.pathString if isinstance(path, Path) else str(path)
        prim = self._prims.get(p)
        if not prim:
            prim = Prim(p, type_name, self)
            self._prims[p] = prim
            # Record authored spec into current edit layer if any
            if self._edit_target is not None:
                layer = self._edit_target.GetLayer()
                layer._record_prim_spec(p)
        return prim