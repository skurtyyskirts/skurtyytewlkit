# Tutorial: Hello Scatter

This short tutorial walks you through using the Scatter Brush in Trex with a tiny sample stage and two simple prototypes.

## Prerequisites

- Built and launched the Toolkit (see `artifacts/how_to_run.md`)
- Open the sample scene: `source/data/Kit/hello_scatter/hello_scatter.usda`

## 1. Locate the Toolbar and Toggle the Scatter Brush

- In the viewport toolbar, click the `Scatter Brush` button (identifier: `scatter_brush`) to toggle it on.
- Reference: `docs/data/images/remix-viewport-001.png`

Behavior when active:
- Hold Left Mouse Button (LMB) over geometry to place items at the mouse position.
- Items are created under the `ScatterBrush` group parent beneath the default prim or `/World`.

## 2. Optional: Adjust Viewport Settings

- Open Viewport Settings and preferences as needed.
- References: `docs/data/images/remix-viewport-017.png`, `docs/data/images/remix-viewport-007.png`

## 3. Prepare Prototypes (Ingest)

This sample includes simple prototypes:
- `extensions/omniscatter/data/prototypes/hello_scatter/proto_box.usda`
- `extensions/omniscatter/data/prototypes/hello_scatter/proto_sphere.usda`

If you want to generate thumbnails or ingest more assets, see:
- `tools/custom/prototype_ingest.py`
- `tools/custom/thumbnail_generator.py`

## 4. Scatter in the Scene

- With `Scatter Brush` active, hover on the ground and LMB-drag lightly; items appear at a limited rate.
- The default created prim type is `Xform` under the `ScatterBrush` parent group.

## 5. Save and Inspect

- Save your stage. In the Stage hierarchy, expand `World/ScatterBrush` to see created children, each transformed locally to the placement position.

## Where to Next

- For viewport controls and keyboard shortcuts, see `docs/toolkitinterface/remix-toolkitinterface-viewport.md`.
- For known limitations and troubleshooting, see:
  - `docs/howto/scatter-known-limitations.md`
  - `docs/howto/scatter-troubleshooting.md`