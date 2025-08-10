import pytest

try:
    from pxr import Usd, UsdGeom, Gf, Sdf
    HAVE_USD = True
except Exception:  # pragma: no cover
    HAVE_USD = False


pytestmark = pytest.mark.skipif(not HAVE_USD, reason="USD Python APIs not available")


def test_ensure_and_append_and_remove(monkeypatch):
    # Create an in-memory stage
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    # Monkeypatch omni.usd.get_context().get_stage to return our stage
    class _Ctx:
        def get_stage(self):
            return stage

    class _OCtx:
        def __init__(self):
            self._ctx = _Ctx()

        def get_context(self):
            return self._ctx

    import types, importlib.util, pathlib
mod_path = pathlib.Path('source/extensions/lightspeed.trex.viewports.shared.widget/lightspeed/trex/viewports/shared/widget/tools/point_instancer_authoring.py')
spec = importlib.util.spec_from_file_location('pia', str(mod_path))
pia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pia)  # type: ignore

pia.omni = types.SimpleNamespace(usd=_OCtx(), kit=types.SimpleNamespace(undo=types.SimpleNamespace(group=lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: False))))

from importlib import import_module
ensure_point_instancer = pia.ensure_point_instancer
append_instances = pia.append_instances
remove_instances_in_radius = pia.remove_instances_in_radius
InstanceSpec = pia.InstanceSpec

    # Ensure PI under /World
    pi_prim = ensure_point_instancer("/World", "TestSet", None)
    assert pi_prim.IsValid() and pi_prim.IsA(UsdGeom.PointInstancer)

    pi = UsdGeom.PointInstancer(pi_prim)
    # Append 3 instances around origin
    instances = [
        InstanceSpec(position=Gf.Vec3f(0.0, 0.0, 0.0)),
        InstanceSpec(position=Gf.Vec3f(1.0, 0.0, 0.0)),
        InstanceSpec(position=Gf.Vec3f(3.0, 0.0, 0.0)),
    ]
    append_instances(pi_prim, instances)

    pos = list(pi.GetPositionsAttr().Get())
    assert len(pos) == 3

    # Remove within radius 1.5 around (0,0,0) removes first two
    removed = remove_instances_in_radius(pi_prim, Gf.Vec3f(0.0, 0.0, 0.0), 1.5)
    assert removed == 2
    pos2 = list(pi.GetPositionsAttr().Get())
    assert len(pos2) == 1 and abs(pos2[0][0] - 3.0) < 1e-6