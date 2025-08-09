Author: Omniscatter Research Agent (GPT-5)
Date: 2025-08-09
License: This document aggregates links and excerpts from official NVIDIA/Omniverse sources and public repositories; original licenses apply. This file is CC-BY-4.0.

Executive summary

An official NVIDIA Omniverse scatter/placement solution exists in the form of a tutorial and sample extension ("Scatter Tool") and a production Paint Tool that places meshes on surfaces. Both demonstrate PointInstancer-based instancing and Omniverse UI/undo patterns. The sample is Apache-2.0 and safe to reuse; the Paint Tool is documented but not open-sourced. Replicator provides advanced scatter APIs and OmniGraph nodes. Recommendation: adopt the Apache-2.0 sample as the code baseline, adapt UI/workflows to a brush metaphor, and reference docs for PointInstancer, viewport picking, and undo.

Primary findings (authoritative sources)

- Title: Build a Scatter Tool — Omniverse Workflows
  - URL: https://docs.omniverse.nvidia.com/workflows/latest/extensions/scatter_tool.html
  - Excerpt: “Create a scatter tool that randomizes prims around the world space… use the USD API to set up a PointInstancer… implement undo.”
  - Date: Accessed 2025-08-09 (page provides evergreen workflow doc)
  - Why it matters: Official tutorial showing UI patterns, PointInstancer setup, and undo/redo – a direct blueprint for instancing-backed scattering.
  - License: Documentation (NVIDIA docs terms; code snippets for learning, not redistributed verbatim).

- Title: kit-extension-sample-scatter (official sample repo)
  - URL: https://github.com/NVIDIA-Omniverse/kit-extension-sample-scatter
  - Excerpt: “A simple scatter tool using USD and omni.ui… scatter selected prim by count, distance, randomness.”
  - Date: Accessed 2025-08-09 (see GitHub for latest commit)
  - Why it matters: Apache-2.0 sample with working extension structure, UI, commands, PointInstancer writes – ideal for adaptation.
  - License: Apache-2.0 (compatible for reuse with attribution).

- Title: Paint Tool — Omniverse Extension
  - URL: https://docs.omniverse.nvidia.com/extensions/latest/ext_paint-tool.html
  - Excerpt: “Paint meshes onto the surface of other meshes… adjust brush size, spacing, randomization.”
  - Date: Accessed 2025-08-09
  - Why it matters: Official surface placement tool demonstrating brush UX; informs our brush radius/density/erase UX and event handling.
  - License: Documentation only (implementation not open-sourced).

Screenshots
- See Paint Tool and Scatter Tool pages for UI screenshots (not embedded here to respect doc licensing).

- Title: Scatter Examples (Omniverse Replicator)
  - URL: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/advanced_scattering.html
  - Excerpt: “Scatter objects on surfaces with collision avoidance… multi-surface scattering, density controls.”
  - Date: Accessed 2025-08-09
  - Why it matters: Advanced algorithms and patterns (collision-aware, multi-surface) that can be adapted conceptually for brush placement.
  - License: Documentation; examples for learning.

- Title: UsdGeom.PointInstancer API (OpenUSD)
  - URL: https://openusd.org/docs/api/class_usd_geom_point_instancer.html
  - Excerpt: “Encodes vectorized instancing of multiple prototypes… designed to scale to billions of instances… positions, orientations, scales, protoIndices.”
  - Date: Accessed 2025-08-09
  - Why it matters: Canonical attribute schema we will write to (GPU instancing) for immediate visibility and persistence.
  - License: Docs for OpenUSD (BSD-3-Clause for USD source; docs under USD terms).

- Title: Scatter — OmniGraph Node (omni.flora.core)
  - URL: https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/node-library/nodes/omni-flora-core/scatter-1.html
  - Excerpt: “Scatter points on input geometry with density and instance limits… outputs points/mask.”
  - Date: Accessed 2025-08-09
  - Why it matters: Node-based approach for density-driven distributions; not required for MVP, but informs heuristics.
  - License: Documentation; extension binaries are NVIDIA proprietary (not for code reuse).

- Title: Viewport selection/raycast APIs (Kit)
  - URL: https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport/latest/index.html
  - Excerpt: “Viewport API for interaction, picking and event subscriptions.”
  - Date: Accessed 2025-08-09
  - Why it matters: Needed for brush raycasts and cursor feedback.
  - License: Documentation.

Decision statement

- Official extension found: YES (tutorial + Apache-2.0 sample). Recommendation: adopt/refactor the Apache-2.0 `kit-extension-sample-scatter` as the base, refactor to brush semantics (continuous stroke, density/spacing), and integrate with RTX-Remix stage conventions. Avoid relying on non-OSS Paint Tool implementation; use its docs to shape UX. Replicator and OmniGraph nodes inform algorithms but are not copied.

Files/code to reuse and how

- Reuse from `kit-extension-sample-scatter` (Apache-2.0):
  - Extension structure (`extension.toml`, startup/shutdown hooks, UI creation code)
  - PointInstancer creation/writes and `omni.kit.commands` undo patterns
  - Randomization utilities (as applicable)
  - Relicensing: preserve Apache-2.0 headers and add attribution in our extension NOTICE.

If no official extension code existed (counterfactual)

- Community implementations to study (top picks):
  - Replicator scatter examples (official, not community): algorithms for collision/density
  - OmniGraph scatter nodes: density/limits IO design
  - Various USD PointInstancer tutorials (OpenUSD docs)

Notes on compatibility and risks

- The sample scatters in world space; we must adapt to surface-aligned brush placement and per-mesh anchors. Undo granularity should batch ~100-250 instances per step to keep history manageable. For performance, perform bulk Set() on arrays and reuse prototypes. Persist in a dedicated layer under `/World/ScatterAnchors` for easy rollback.


