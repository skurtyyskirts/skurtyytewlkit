# Scatter Brush Compliance Checklist (RTX Remix Toolkit)

This document is the authoritative compliance checklist for the RTX Remix Toolkit Scatter Brush. It maps requirements to official NVIDIA/OpenUSD guidance and flags any gaps requiring further confirmation.

## Scope
- Authoring and editing of Scatter Brush data using OpenUSD `UsdGeomPointInstancer` in RTX Remix Toolkit (Trex)
- UI and interaction semantics equivalent to USD Composer/Omniverse Scatter/Paint tools
- Packaging/export considerations for Remix Runtime (DXVK-Remix/RTX-Remix)

## Authoritative References
- OpenUSD
  - UsdGeomPointInstancer Class Reference — openusd: `https://openusd.org/docs/api/class_usd_geom_point_instancer.html`
  - Instancing overview (prototypes, instanceability) — openusd: `https://openusd.org/docs/UsdGeom-Instancing.html`
- NVIDIA Omniverse
  - Build a Scatter Tool (Workflows) — Omniverse: `https://docs.omniverse.nvidia.com/workflows/latest/extensions/scatter_tool.html`
  - Paint Tool (Scatter brush behaviors) — Omniverse: `https://docs.omniverse.nvidia.com/extensions/latest/ext_paint-tool.html`
  - Omnigraph Scatter nodes (optional pipeline) — Omniverse: `https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/node-library/nodes/omni-flora-core/scatter-1.html`
- RTX Remix Toolkit (local docs in repo)
  - Viewport options — Merge Instances / Use Fabric Scene Graph Instancing: see `docs/toolkitinterface/remix-toolkitinterface-viewport.md`
  - REST API tutorial (patterns for authoring ops) — `docs/tutorials/tutorial-restapi.md`
- GitHub Repositories
  - @ToolkitGithubDocs — Toolkit (this repo): `https://github.com/NVIDIAGameWorks/toolkit-remix`
  - @RTX-RemixGithub — Runtime: `https://github.com/NVIDIAGameWorks/rtx-remix`
  - @DXVK-RemixGithub — DXVK Remix: `https://github.com/NVIDIAGameWorks/dxvk-remix`
  - @Bridge-RemixGithub — Remix Bridge: (TBD — fetch exact repo URL)
  - @ComfyUI-RTX-RemixGithub — (TBD — fetch exact repo URL)
  - @USDToolset — OpenUSD docs (see links above) and Omniverse USD references
  - @RestAPIScreen — local `docs/tutorials/tutorial-restapi.md`

## USD Schema Compliance (PointInstancer)
- [ ] Use `UsdGeomPointInstancer` for scattered instances
  - [ ] Author `prototypes` relationship targeting root prims (OpenUSD PI: Prototypes)
  - [ ] Author `protoIndices` array (int) mapping each instance to a prototype index
  - [ ] Author `positions` (GfVec3f/d) per instance
  - [ ] Optional `orientations` (GfQuath) per instance; default identity if omitted
  - [ ] Optional `scales` (GfVec3f) per instance; default ones if omitted
  - [ ] Optional `ids` (int64) stable per instance for undo/erase/edit workflows
  - [ ] Optional `velocities`, `accelerations`, `angularVelocities` if needed
- [ ] Array-length invariants
  - [ ] `protoIndices`, `positions`, and any authored `orientations`/`scales` arrays are equal length for a given time sample
  - [ ] All arrays are consistently time-sampled when animation is required; no mixed-sample shape mismatches
- [ ] Transform/composition correctness
  - [ ] Instancer xform prim defines parent space; per-instance transforms are applied relative to instancer
  - [ ] World transform = instancer xform composed with per-instance position/orientation/scale
  - [ ] Do not author 4x4 matrices per instance (use PI vectorized attributes as per OpenUSD guidance)
- [ ] Prototype authoring
  - [ ] Prototypes are referenced assets or subgraphs; ensure correctness when swapping assets (no reauthoring of per-instance arrays)
  - [ ] Prefer prototype roots marked `instanceable` where beneficial; do not over-instance where it harms editability
- [ ] Selection/identity
  - [ ] Maintain stable `ids` to support deletion/selection of individual instances without reindexing arrays
  - [ ] Use `invisibleIds` for soft delete when non-destructive erasing is required

## Omniverse Interaction & UI Compliance
- [ ] Brush interaction model
  - [ ] Paint, Erase, and Flood modes consistent with Omniverse Paint Tool scatter
  - [ ] Brush radius, density, stamp spacing, jitter/padding, vertical offset, surface normal alignment
  - [ ] Randomization controls: seed, rotation (min/max or full random), non-uniform scale ranges
  - [ ] Asset library selection and per-asset weighting
  - [ ] Undo/Redo integrated using authoring command stack
- [ ] Scene evaluation
  - [ ] Surface raycast for placement; align orientation to hit normal (optional)
  - [ ] Collision/padding checks to minimize interpenetration when padding > 0
- [ ] Performance
  - [ ] Batch edits (minimize array churn); author arrays via scoped command for stroke
  - [ ] Recompute only changed segments; avoid per-sample re-authoring of entire arrays when erasing small subsets

## RTX Remix Toolkit Integration Compliance
- [ ] UI integration
  - [ ] Add Scatter Brush panel/tool into Trex UI in a way consistent with Toolkit UI patterns (omni.ui + Trex style)
- [ ] Viewport settings interplay
  - [ ] Honor/benefit from Fabric Scene Delegate instancing settings (local docs — Merge Instances / Use Fabric Scene Graph Instancing)
- [ ] Asset paths & project layout
  - [ ] Prototypes reference assets within project or configured libraries; paths resolve in Toolkit and during packaging
- [ ] Packaging/export for Runtime
  - [ ] If RTX Remix Runtime supports PointInstancer directly — preserve PI in packaged USD
  - [ ] If not supported — provide packaging option to Bake Instancer to explicit referenced prims/Xforms (compliance note below)

## DXVK-Remix / RTX-Remix Runtime Considerations
- [ ] Rendering parity
  - [ ] Validate whether runtime supports PointInstancer; if unsupported, enable baking
  - [ ] Ensure LOD/material bindings on prototypes are respected by instances in runtime
- [ ] Performance
  - [ ] Avoid pathological growth of instance count without LOD options
- [ ] Determinism
  - [ ] Author randomization with stored seed(s) for reproducible results across sessions

## Data Integrity & Undo Compliance
- [ ] Use a single undoable command scope per brush stroke
- [ ] Assign unique `ids` on create; preserve `ids` on edits
- [ ] For erase, prefer `invisibleIds` or sparse rebuild preserving order and ids where feasible

## Security & Licensing
- [ ] Ensure scattered assets comply with NVIDIA/Project licensing (see `artifacts/license_matrix.md`)

## Explicit Local Documentation Citations
- Viewport instancing toggles (Merge/Use Fabric Scene Graph Instancing)
  - See `docs/toolkitinterface/remix-toolkitinterface-viewport.md` rows describing options 9 and 15 (Merge Instances / Use Fabric Scene Graph Instancing)
- Authoring preference: Set "Instanceable" When Creating Reference
  - See `docs/toolkitinterface/remix-toolkitinterface-viewport.md` authoring option 17
- REST API tutorial patterns (command structure, authoring flows)
  - See `docs/tutorials/tutorial-restapi.md`

## Compliance Outcomes
- [ ] Authoring produces valid OpenUSD PointInstancer per schema
- [ ] Toolkit UI provides parity with Composer/Paint Tool scatter behaviors
- [ ] Packaging path validated for runtime support (preserve or bake)
- [ ] Performance guidelines met for interactive strokes and large instance counts

## Identified Gaps / Actions for Web Research Agent
- G1: Confirm RTX-Remix Runtime support status for `UsdGeomPointInstancer` and any constraints — fetch official statement from @RTX-RemixGithub docs/issues
- G2: Identify official @Bridge-RemixGithub repository URL and any guidance for PI support during import/export
- G3: Identify official @ComfyUI-RTX-RemixGithub repository URL and whether it exposes scatter automation
- G4: Omniverse Paint Tool scatter exact parameter names and defaults — capture authoritative list with links/anchors from Omniverse docs
- G5: Provide examples/best-practices from OpenUSD for large PI authoring (sparse updates, ids management)

Once gaps are resolved, update this checklist with the definitive references and mark items as Verified.