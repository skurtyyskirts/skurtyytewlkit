Author: Omniscatter Research Agent (GPT-5)
Date: 2025-08-09
License: CC-BY-4.0 for this report. Linked code/docs retain their original licenses.

Executive summary

The official NVIDIA Scatter Tool sample (Apache-2.0) and Paint Tool documentation map well to RTX-Remix requirements. The key work for Remix is adapting from one-shot world-space “scatter N items” to continuous brush placement onto hit surfaces, writing to `UsdGeom.PointInstancer` under per-mesh anchors, and ensuring robust undo/redo and persistence in Remix layers. All required APIs appear available in Kit/Usd; no closed components are required for MVP.

Feature mapping (identified → RTX-Remix integration)

- PointInstancer setup (sample + tutorial)
  - Integration: Use `UsdGeom.PointInstancer` with attributes `positions`, `orientations` (quats), `scales`, `protoIndices`, and `prototypes` list referencing USD prototype files. Create one instancer per anchor under `/World/ScatterAnchors/<mesh>`.

- UI controls
  - From: Scatter sample UI frames, Paint Tool UX (brush size, spacing, randomization)
  - To: Omniscatter panel with ON/OFF, Brush radius, Density (instances/m²), Prototype thumbnail grid, Erase toggle.

- Undo/redo
  - From: Sample uses `omni.kit.commands` and batched operations
  - To: Wrap appends/erases in commands; one history step per flush batch (100–250 instances). Maintain old/new attribute arrays.

- Viewport events and raycast
  - From: Kit viewport API (docs) for event subscriptions & picking
  - To: Subscribe to active viewport mouse events; perform raycast to surface; place instances aligned to hit normal with jitter.

- Performance
  - From: PointInstancer vectorization; Replicator examples for density/collision ideas
  - To: Batch writes, reuse arrays, avoid per-instance Authoring. Target ≥1k instances @ ≥30 FPS; split instancers per surface/tiles as needed.

- Persistence & load
  - From: USD composition (layers/references) best practices
  - To: Author to a dedicated, editable layer for scatter data. Ensure prototypes are referenced by path; test save/open reload.

Compatibility notes with RTX-Remix

- Stage structure: Remix stages use specific roots; placing under `/World/ScatterAnchors` is non-destructive and compatible.
- Materials: Prototypes may be USDs referencing MDL or UsdPreviewSurface. Remix supports MDL paths; provide both in ingest tool for compatibility.
- GPU Instancing: PointInstancer is supported by Omniverse RTX renderers; no extra settings required.

Gaps/assumptions to verify

1) Active Remix build has `omni.kit.viewport` raycast utility available.
2) Undo stack depth acceptable with batched updates (confirm default limits).
3) Prototype USDs with MDL bound materials render correctly in Remix app.

Actionable changes required in Remix integration

- Add `extensions/omniscatter` with UI, toolbar, events, painting core.
- Add ingestion utilities to generate prototypes and thumbnails from Remix-ingested outputs.
- Register a Scatter layer and ensure it is saved with project.

Licensing compatibility

- `kit-extension-sample-scatter`: Apache-2.0 → compatible. Keep NOTICE/headers.
- Paint Tool docs: for reference only. Do not copy text or UI art.


