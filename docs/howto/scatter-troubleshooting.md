# Scatter Brush — Troubleshooting

- Scatter Brush button not visible
  - Ensure you are in a Trex viewport UI. Check `docs/toolkitinterface/remix-toolkitinterface-viewport.md`.
  - Verify the extension providing the toolbar is enabled in your app build.

- Clicking/dragging doesn’t place anything
  - Make sure the Scatter Brush toggle is on (button appears active).
  - Ensure you are over geometry that can be hit by picking; try the sample `hello_scatter.usda` ground.
  - Check for errors in the logs (see `docs/toolkitinterface/remix-toolkitinterface-layouttab.md` → Show Logs).

- Items appear but not where expected
  - Camera ray pick may be missing the intended surface; try orbiting and clicking again.
  - Placement occurs under a `ScatterBrush` parent; verify local transforms by expanding that group in the Stage.

- Undo/redo behaves oddly
  - Placement is currently grouped per placement, not per continuous stroke. Undo multiple times to remove multiple items.

- Performance stutters while dragging
  - Placement is throttled to one every ~120ms by default; reduce drag speed or click to place discretely.

- I want to place references or point instancer instances
  - Not yet implemented. Track progress in internal docs (`PROJECT_STATUS.md`, `SCATTER_MAPPING.md`).

- Need more viewport help
  - See `docs/toolkitinterface/remix-toolkitinterface-viewport.md` for controls and settings.