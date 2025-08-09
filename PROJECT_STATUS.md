# Scatter Brush Implementation - Project Status

- Project: RTX-Remix Toolkit - Scatter Brush Feature
- Overseer: AI Overseer Agent
- Last Updated: INITIAL

## Objectives
- Add a Scatter Brush tool to paint-scatter USD prims/assets directly in the viewport.
- Integrate UI (toolbar button) and scene factory for runtime behavior.
- Ensure compatibility with Remix picking (world position) and TREX contexts.
- Provide tests, documentation, and performance tuning.

## Current Status
- Core scaffolding added:
  - Scatter brush scene factory registered: `omni.kit.lss.viewport.tools.scatter_brush`.
  - Toolbar button `scatter_brush` created with toggle behavior.
  - Basic placement pipeline using world position → local placement under a group parent in the active stage.
- Pending work:
  - Brush UI Panel: radius, jitter, rotation align to surface normal, random scale, density.
  - Asset placement modes: reference USD asset, point instancer, or simple Xform.
  - Surface normal alignment and randomization.
  - Integration with `lightspeed.trex.asset_replacements.core.shared.Setup` to append references efficiently.
  - Hotkey definition and status indicator.
  - Tests and documentation updates.

## Work Breakdown
- Research/Web:
  - Validate latest Omniverse/USD APIs for point instancer, AddReference, and manipulators. [TODO]
- Development:
  - Implement brush settings panel and persistence. [TODO]
  - Implement asset reference placement using `Setup.add_new_reference`. [TODO]
  - Add surface normal query or compute method to align rotation. [TODO]
  - Add scatter density and jitter options. [TODO]
- Integration:
  - Ensure behavior across TREX contexts (StageCraft/TextureCraft/Ingest). [TODO]
  - Scene layer interoperability verified. [TODO]
- Performance:
  - Batch undo groups, throttle queries, consider instancing for large counts. [TODO]
- Testing:
  - Unit tests for placement, grouping, and commands. [TODO]
  - E2E tests simulating brush strokes. [TODO]
- Documentation:
  - User guide section and screenshots. [TODO]

## Risks
- World normal/transform retrieval with HdRemix may require additional API hooks. [Mitigation: Only position for MVP]
- Asset path resolution and reference layer handling requires care. [Mitigation: reuse `Setup` helpers]

## Next Actions
- Implement settings UI and asset selection.
- Integrate reference placement path via `Setup.add_new_reference`.
- Add tests and docs.