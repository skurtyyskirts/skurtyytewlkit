import omni.kit.commands
from pxr import UsdGeom, Sdf, Gf

class ScatterPaintCommand(omni.kit.commands.Command):
    def __init__(self, instancer_path: str, points: list):
        self._instancer_path = instancer_path
        self._points = points
        self._old_positions = None
        self._old_indices = None

    def do(self):
        stage = omni.usd.get_context().get_stage()
        instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(self._instancer_path))
        if not instancer:
            return

        self._old_positions = instancer.GetPositionsAttr().Get() or []
        self._old_indices = instancer.GetProtoIndicesAttr().Get() or []

        all_positions = list(self._old_positions) + self._points
        instancer.GetPositionsAttr().Set(all_positions)

        num_new_points = len(self._points)
        all_indices = list(self._old_indices) + [0] * num_new_points
        instancer.GetProtoIndicesAttr().Set(all_indices)

    def undo(self):
        stage = omni.usd.get_context().get_stage()
        instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(self._instancer_path))
        if not instancer:
            return

        if self._old_positions is not None:
            instancer.GetPositionsAttr().Set(self._old_positions)
        if self._old_indices is not None:
            instancer.GetProtoIndicesAttr().Set(self._old_indices)
