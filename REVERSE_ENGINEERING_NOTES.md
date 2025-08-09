# Reverse Engineering Notes — Scatter Brush (USD Composer → RTX Remix Toolkit)

These notes capture the inferred behavior, UI, and data model of the USD Composer Scatter Brush (and Omniverse Paint Tool scatter) and translate them to an RTX Remix Toolkit implementation plan grounded in OpenUSD.

## High-Level Behavior
- The Scatter Brush uses OpenUSD `UsdGeomPointInstancer` (PI) as the backing data container.
- The user selects one or more prototype assets; the brush paints instances onto target surfaces with configurable randomness and density.
- Erase and Flood operations modify the same PI arrays, preserving per-instance identity (`ids`).

## Data Model (OpenUSD)
- `UsdGeomPointInstancer` attributes in scope:
  - `prototypes` (relationship[] to root prims of prototype subgraphs)
  - `protoIndices` (int[]), one entry per instance
  - `positions` (vec3f[] or vec3d[])
  - `orientations` (quath[]), optional; identity if absent
  - `scales` (vec3f[]), optional; ones if absent
  - `ids` (int64[]), optional; recommended for stable selection/undo/erase
  - Optional kinematic attributes (`velocities`, etc.) typically unused for static scatter
- Invariants
  - Array lengths match across authored arrays per time sample
  - Prototypes indexed by `protoIndices[i]` refer to `prototypes` order

## UI States and Controls (Composer/Paint parity)
- Mode: Paint | Erase | Flood
- Brush radius
- Density / stamp spacing
- Padding (object-object spacing)
- Vertical offset (along surface normal)
- Align to surface normal (toggle)
- Randomization:
  - Seed
  - Rotation randomization: min/max per-axis or around-normal twist
  - Scale randomization: uniform or per-axis min/max
  - Position jitter within brush stamp
- Asset library panel:
  - Asset selection (multi)
  - Per-asset weight/probability sliders

## Event Flow (Stroke)
1. Pointer-down: begin stroke
   - Capture seed state and target PI prim path
   - Start undoable command scope
2. Brush move: for each stamp along stroke path
   - Raycast points on target surfaces within brush radius
   - For each sample:
     - Check padding/collision constraints
     - Sample asset prototype using weights
     - Compute position (hit point + offset), orientation (aligned to normal + random twist), scale (random range)
     - Append to local buffers for `positions`, `orientations` (if authored), `scales` (if authored), `protoIndices`, `ids`
3. Pointer-up: end stroke
   - Batch-author array updates into PI attributes
   - Close undo scope

Erase mode:
- Raycast/select instances within brush; either mark via `invisibleIds` or rebuild arrays without the erased ids while preserving others.

Flood mode:
- Compute region seeds based on selection/volume; generate consistent deterministic distribution with given density and seed; author arrays in single operation.

## Authoring Strategy
- Prefer batched writes: build arrays in memory per stroke; author once per stroke to minimize USD change processing.
- Preserve stable `ids`; assign monotonic ids for new instances.
- For erasing, avoid reindexing unrelated instances; either filter by `ids` or use visibility masks.

## Performance Considerations
- PI scales to very large instance counts; still, authoring cost can spike on full-array rewrites.
- Strategies:
  - Sparse updates when feasible (append-only for paint)
  - Deferred rebuild for erase with minimal copy
  - Avoid per-sample Sdf change notifications (wrap in command/overlay)

## RTX Remix Toolkit Integration
- UI: Implement with `omni.ui` following Trex style; add Scatter Brush tool/panel.
- Viewport: Honor Merge Instances and Use Fabric Scene Graph Instancing settings (see local viewport docs).
- Asset paths: Prototypes should be referenced assets within the Toolkit project layout for portability.
- Packaging: If RTX Remix Runtime supports PI, keep instancers; otherwise implement a Bake to Xforms option for runtime packaging.

## Edge Cases
- Mixed prototype sets changing over time — maintain mapping and only append/remove references when needed
- Unit scale consistency — respect stage metersPerUnit and prototype authored scale
- Surface backface hits — optionally reject/backface-cull

## References
- OpenUSD UsdGeomPointInstancer: `https://openusd.org/docs/api/class_usd_geom_point_instancer.html`
- Omniverse Build a Scatter Tool: `https://docs.omniverse.nvidia.com/workflows/latest/extensions/scatter_tool.html`
- Omniverse Paint Tool: `https://docs.omniverse.nvidia.com/extensions/latest/ext_paint-tool.html`
- Toolkit local docs: `docs/toolkitinterface/remix-toolkitinterface-viewport.md`, `docs/tutorials/tutorial-restapi.md`
- RTX Remix Runtime repo: `https://github.com/NVIDIAGameWorks/rtx-remix`
- DXVK-Remix repo: `https://github.com/NVIDIAGameWorks/dxvk-remix`

## Open Questions for Web Research Agent
- Q1: Confirm RTX Remix Runtime support/limitations for `UsdGeomPointInstancer` rendering and material bindings.
- Q2: Official parameters list and defaults for Omniverse Paint Tool scatter brush (names, ranges).
- Q3: Bridge pipeline handling of PI (import/export) and any constraints.
- Q4: Recommended practices for PI `ids` management at scale from OpenUSD docs or NVIDIA guidance.