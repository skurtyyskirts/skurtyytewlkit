# RTX Remix Toolkit — Scatter Brush (Docs & Demos)

This repository includes user-facing documentation and a tiny sample project showcasing the new Scatter Brush tool in RTX Remix Toolkit (Trex).

- Left panel, toolbar toggle, and settings screenshots
- “Hello Scatter” sample USD stage with two ingested prototypes
- Step-by-step tutorial to try the tool end-to-end
- Known limitations and troubleshooting

## Quick Links

- Getting started with Toolkit UI: `docs/toolkitinterface/remix-toolkitinterface-viewport.md`
- “Hello Scatter” tutorial: `docs/tutorials/tutorial-hello-scatter.md`
- Known limitations: `docs/howto/scatter-known-limitations.md`
- Troubleshooting: `docs/howto/scatter-troubleshooting.md`

## UI Overview

### Where the Left Panel Lives

The left panel hosts Layers, Bookmarks, Selection History, and Properties panes used throughout Trex.

![Stage Manager / Left Panels](docs/data/images/remix-toolkitinterface-stagemanager.png)

See also: `docs/toolkitinterface/remix-toolkitinterface-layouttab.md`.

### Toolbar Toggle (Scatter Brush)

The Scatter Brush is toggled from the viewport toolbar. When active, holding LMB places items at the mouse hit position.

![Viewport Toolbar Reference](docs/data/images/remix-viewport-001.png)

- Button identifier: `scatter_brush`
- Tooltip: “Scatter Brush (paint to place objects under the mouse)”

### Settings

Use viewport settings and preferences to adjust UI visibility and other options.

![Reset and Preferences](docs/data/images/remix-viewport-017.png)
![Viewport UI Settings](docs/data/images/remix-viewport-007.png)

## Scatter Brush at a Glance

- Toolbar: Toggle the scatter brush via the toolbar button labeled `Scatter Brush` (identifier: `scatter_brush`).
- Behavior: When active, press and hold Left Mouse Button in the viewport to place items at the cursor hit position at a throttled rate.
- Default output: Creates an `Xform` child under a parent group named `ScatterBrush` beneath the stage default prim or `/World`.

## Sample: Hello Scatter

Located at `source/data/Kit/hello_scatter`.

- `hello_scatter.usda`: a tiny stage with a camera, light, ground plane, and a parent `ScatterBrush` group created on demand.
- Two simple prototype assets (USD) used for ingestion demonstration live in `extensions/omniscatter/data/prototypes/hello_scatter/`.

Follow the tutorial: `docs/tutorials/tutorial-hello-scatter.md`.

## Building and Running

- See `artifacts/how_to_run.md` or use the `build.sh` / `build.bat` in the repo root.
- Once running, open the `hello_scatter.usda` from the Project or via File → Open.

## Contributing

Please see `docs_dev/CONTRIBUTING.md` and `docs/contributing/contributing-overview.md`.
