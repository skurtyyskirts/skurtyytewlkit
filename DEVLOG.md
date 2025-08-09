# DEV LOG

- Init: Created `PROJECT_STATUS.md` and planning. Identified viewport integration points:
  - Toolbar via `omni.kit.widget.toolbar` in `lightspeed.trex.viewports.shared.widget`
  - Scene registration via `RegisterScene` and `ViewportSceneLayer`
  - Tool pattern reference: `tools/teleport.py`
- Plan: Add minimal Scatter Brush tool with button and scene registration; implement later full scatter.