# DEVLOG

- 2025-01-XX: Initial Scatter Brush scaffolding added
  - Registered new scene: `omni.kit.lss.viewport.tools.scatter_brush`.
  - Added toolbar button `scatter_brush` (toggle) and basic placement logic.
  - Placement uses HdRemix world position query via `PointMousePicker` and creates `Xform` children under a group parent.
  - Style icon uses `Button.Image::scatter_brush` mapped to `_get_icons("brush")`.
  - Integration points prepared for asset references and point instancer.

- Notes:
  - MVP only places `Xform` prims. Next steps add reference placement using `lightspeed.trex.asset_replacements.core.shared.Setup.add_new_reference` and normal alignment.
