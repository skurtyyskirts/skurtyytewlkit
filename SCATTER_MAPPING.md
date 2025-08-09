# USD Composer Scatter Brush → RTX Remix Toolkit Mapping

This table maps USD Composer/Omniverse Scatter Brush features to RTX Remix Toolkit implementation, noting any differences and compliance references.

## Legend
- PI = UsdGeomPointInstancer
- Ref links are provided at bottom

## Feature Mapping

- Feature: Point instancing backend
  - USD Composer: UsdGeomPointInstancer
  - RTX Remix Toolkit: UsdGeomPointInstancer (authoring in Trex)
  - Notes: Ensure `prototypes`, `protoIndices`, `positions`, optional `orientations`, `scales`, and `ids` compliance
  - Refs: OpenUSD PI, Omniverse Scatter Tool

- Feature: Prototype asset selection (multi-asset, weights)
  - USD Composer: Asset list with selection/weights via Paint/Scatter tool
  - RTX Remix Toolkit: Trex asset browser + weights (to implement in Scatter UI)
  - Notes: Store per-asset weight; translate to probability when generating `protoIndices`
  - Refs: Paint Tool doc (asset painting)

- Feature: Placement algorithm (surface-based)
  - USD Composer: Raycast to surfaces, place instances aligned to hit
  - RTX Remix Toolkit: Use Trex/Kit picking/raycast; compute position from brush stamp and surface normal
  - Notes: Brush spacing/padding to reduce overlap
  - Refs: Paint Tool

- Feature: Brush modes (Paint, Erase, Flood)
  - USD Composer: Provided by Paint Tool scatter
  - RTX Remix Toolkit: Implement authoring commands for add/remove (erase) and flood (batch fill)
  - Notes: Use `ids` and optionally `invisibleIds` for non-destructive erase
  - Refs: Paint Tool

- Feature: Randomization (seed, rotation, scale, jitter)
  - USD Composer: Controls for random rot/scale and jitter
  - RTX Remix Toolkit: UI sliders/fields for seed, rotation ranges, non-uniform scale ranges, translation jitter
  - Notes: Persist seed for determinism; write arrays once per stroke
  - Refs: Omniverse Scatter Tool

- Feature: Density / Count controls
  - USD Composer: Density and max count in region
  - RTX Remix Toolkit: Density per area via brush radius & spacing; or absolute count for flood
  - Notes: Deterministic sampling with seed
  - Refs: Scatter Tool

- Feature: Orientation alignment
  - USD Composer: Align to surface normal optional
  - RTX Remix Toolkit: Toggle; generate `orientations` from normal and a random around-normal twist
  - Notes: Quaternion authoring per instance when enabled
  - Refs: Paint Tool

- Feature: Vertical offset / padding
  - USD Composer: Controls available in Paint Tool
  - RTX Remix Toolkit: Offset along normal; padding used in collision checks before authoring
  - Notes: Avoid interpenetration
  - Refs: Paint Tool

- Feature: Undo/Redo integration
  - USD Composer: Uses Omniverse/Kit undo
  - RTX Remix Toolkit: Wrap stroke in a single undoable command; granular ids preserved
  - Notes: Critical for parity and user trust
  - Refs: Scatter Tool tutorial

- Feature: Performance and batching
  - USD Composer: Efficient updates to PI arrays
  - RTX Remix Toolkit: Batch authoring; sparse edits; avoid full re-write per stroke
  - Notes: Keep arrays contiguous; rebuild with minimal churn
  - Refs: OpenUSD PI notes

- Feature: Packaging for runtime
  - USD Composer: Keep PI for Omniverse runtime
  - RTX Remix Toolkit: Prefer keeping PI; add optional bake-to-xforms if Remix Runtime lacks PI support
  - Notes: Compliance gate on runtime capability
  - Refs: RTX Remix Runtime docs (TBD)

## Divergences and Alternatives
- If RTX Remix Runtime cannot render `UsdGeomPointInstancer`, provide a bake option that expands instances into referenced `Xform` prims (with transforms and material bindings inherited from prototypes), gated by a packaging toggle.
- If Fabric Scene Delegate instancing options affect visibility or performance, document recommended settings for Scatter workflows.

## References
- OpenUSD UsdGeomPointInstancer: `https://openusd.org/docs/api/class_usd_geom_point_instancer.html`
- Omniverse Build a Scatter Tool: `https://docs.omniverse.nvidia.com/workflows/latest/extensions/scatter_tool.html`
- Omniverse Paint Tool: `https://docs.omniverse.nvidia.com/extensions/latest/ext_paint-tool.html`
- RTX Remix Toolkit (this repo docs): see `docs/toolkitinterface/remix-toolkitinterface-viewport.md`
- RTX Remix Runtime repo: `https://github.com/NVIDIAGameWorks/rtx-remix`
- DXVK-Remix repo: `https://github.com/NVIDIAGameWorks/dxvk-remix`
- Toolkit repo: `https://github.com/NVIDIAGameWorks/toolkit-remix`
- Bridge repo: (TBD)
- ComfyUI RTX Remix repo: (TBD)