# Scatter Brush Project Status

- Feature: Scatter Brush for RTX-Remix Toolkit (TREX)
- Overseer: AI Agent (this workspace)
- State: Initialization
- Target App Contexts: `StageCraft` (primary), compatibility with shared viewport widget

## Current Milestone
- Establish code scaffolding and UI hook for Scatter Brush
- Add planning and dev logs
- Run build/lint to ensure repo stability

## Next Tasks
- Implement core scatter logic (place/instance selected asset along brush stroke)
- Add brush settings UI (radius, density, jitter, align to surface normal, random rotation/scale)
- Integrate with `AssetReplacementsCore` for prototype/instance handling
- Performance pass (batch edits, USD change blocks, minimal UI updates)
- Tests: unit and e2e smoke for tool creation and simple scatter
- Docs: user guide page and how-to

## Owners
- Developer: Scatter Brush Developer Agent
- UI/UX: UI/UX Designer Agent
- Integration: Integration Specialist Agent
- Research: Web Research Agent
- Perf: Performance Optimizer Agent
- QA: Testing Engineer Agent
- Docs: Documentation Writer Agent
- GitOps: Repo Manager Agent

## Risks/Notes
- PointInstancer support and instance/prototype mapping must match TREX conventions
- Ensure no regressions to existing tools/teleport