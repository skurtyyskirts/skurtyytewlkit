# Scatter Brush — Known Limitations

These limitations reflect the current implementation status of the Scatter Brush in Trex.

- No brush settings UI yet
  - Radius, density, jitter, rotation alignment to surface normal, random scale/seed are not exposed in UI.
- Placement rate throttled
  - Placement occurs at most every `min_interval_ms` (default 120ms) while dragging.
- Default primitive type
  - Creates `Xform` prims (no direct reference/integration with `UsdGeomPointInstancer` yet).
- No surface normal alignment
  - Instances are translated only; no orientation alignment to surface normals.
- No erase/flood modes
  - Only additive placement is supported.
- No asset prototype selection UI
  - The sample demonstrates simple geometry USDs; a prototype picker/weights UI is not implemented.
- Parent group behavior
  - Items are placed under an auto-created `ScatterBrush` group under the default prim or `/World`.
- Hotkeys
  - Toolbar toggle is primary; dedicated hotkeys are not assigned for brush modes.
- Undo/redo
  - Placement operations are grouped per placement, but not per stroke.