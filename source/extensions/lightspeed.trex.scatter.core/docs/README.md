# lightspeed.trex.scatter.core

Core logic for the RTX Remix asset scatter brush. It turns a picked point on a captured mesh into undoable
placements of ingested USD assets, with Paint-Tool-style density, falloff, spacing, padding, randomization and
presets. It has no UI.

## Responsibilities

- Own the brush settings model (`ScatterBrushSettings`, `ScatterAssetEntry`) and persist it in carb settings.
  The model rejects a low/high pair that is out of order; the controller keeps a pair ordered by carrying the
  end the caller did not set, so a drag field can move one end past the other
- Store named presets as JSON files (`PresetStore`)
- Resolve a surface hit to a Remix scatter target: the prototype `/RootNode/meshes/mesh_<HASH>` that will parent the
  placements, the instance that was hit, and the capture mesh used for geometry queries (`targets.py`)
- Cache captured mesh geometry as numpy arrays and answer closest-point, normal, ray and area-sampling queries
  (`MeshSurfaceCache`)
- Provide the surface picker seam: `HdRemixSurfacePicker` (GPU picking through `lightspeed.hydra.remix.core`) and
  `CpuRaySurfacePicker` (camera-ray fallback used when HdRemix is unavailable, e.g. headless tests)
- Generate stamps and floods (`placement.py`, `sampling.py`): seeded disk sampling, falloff and strength acceptance,
  padding against existing placements, weighted asset choice, parent-space transform composition with
  conform-to-surface, align-to-stroke, per-axis rotation and scale ranges, and per-asset up-axis correction
- Resolve the ingested texture set of a palette asset and author the inputs its material leaves empty as overrides under each placement (`materials.py`)
- Author and remove placements with pure `Sdf` edits and wrap each stroke in one undoable command
  (`ScatterStrokeCommand`, `ScatterFloodCommand`)
- Run the stroke state machine (`StrokeSession`) and expose the shared brush state and events through
  `ScatterBrushController`

## Non-Responsibilities

- Does not create any UI and does not import `omni.ui` or `omni.ui.scene`. The Scatter window lives in
  `lightspeed.trex.scatter.widget`; the viewport brush tool (gestures, cursor, toolbar button, hotkeys) lives in
  `lightspeed.trex.viewports.shared.widget`
- Does not ingest assets or show ingestion dialogs; callers gate assets with
  `lightspeed.trex.utils.widget.asset_validation` before adding them to the brush
- Does not create or bind materials, and never overwrites a texture an asset or the user already authored; an asset without a material is painted as it is and reported through a status message
- Does not author `UsdGeom.PointInstancer` prims or physics settling; every placement is an `Xform` with a reference so
  the Selection Panel, Stage Manager and packaging treat it as an ordinary replacement reference

## Architecture

Placements are authored under the captured **prototype**, never under an instance, because the Remix runtime replaces
geometry by draw-call hash and the Toolkit redirects all authoring to prototypes. One container `Xform`
(`scatter_<preset slug>`, marked `IsRemixScatter`) is created per prototype and brush; each placement is a child
`s_<uuid>` `Xform` carrying `IsRemixRef`, `IsRemixScatter`, `remixScatterAsset`, a reference to the ingested asset and
explicit `translate` / `rotateXYZ` / `scale` ops in the local space of the instance that was hit. The captured
`mesh` child is never modified, so the original geometry keeps rendering underneath the placements, and because the
container carries no `IsRemixRef` the Selection Panel does not enumerate thousands of reference rows.

Anything authored under a prototype appears under every instance of that hash. `ScatterTarget.instance_count` exposes
this so UIs can warn, and `TargetMode.ANCHOR` lets the user pin a single prototype whose canonical instance transform
is used for all placements.

HdRemix picking returns a prim path and a world position but no normal and allows one request in flight, so the brush
issues one pick per stamp and derives every other sample, the surface normal and the projection from the cached mesh
triangles on the CPU.

Texture assignment reuses Remix's own texture-set logic rather than repeating it: `determine_ideal_types` for the
type of each candidate file and `TEXTURE_TYPE_INPUT_MAP` for the AperturePBR input it belongs on. A set is resolved
once per palette asset (`resolve_asset_texture_set`), cached on the entry and rebuilt when the asset's ingestion hash
changes; a stroke turns the palette into one payload (`build_materials_payload`) before its first stamp. The inputs
are authored as `over` specs beneath the placement prim itself, so they belong to the placement: `remove_placements`
and the stroke snapshots take them along without knowing they exist, and the material that resolves is still the
asset's own. Only inputs the asset leaves empty are ever written, and `inputs:encoding` is written only alongside a
normal map the brush authors itself, because the value an asset declares without a normal map describes nothing.

A stroke writes placements directly into the edit layer while the mouse moves (so the viewport updates live) and
commits a single `ScatterStrokeCommand` on release; the first `do()` of that command is a no-op, `undo()` removes the
placements and redo re-authors them. Erase strokes snapshot the removed specs with `Sdf.CopySpec` so undo restores
them. No `omni.kit.undo` group is held open across frames.

### Key Classes

- `ScatterBrushSettings` / `ScatterAssetEntry` — validated settings model with JSON and carb persistence
- `PresetStore` — named preset files (list, load, save, rename, clone, delete)
- `MeshSurfaceCache` / `MeshGeometry` — numpy triangle cache with closest-point, raycast and area sampling
- `SurfacePicker`, `HdRemixSurfacePicker`, `CpuRaySurfacePicker`, `create_surface_picker` — picker seam
- `ScatterTarget`, `resolve_target` — prototype, instance and capture-mesh resolution for a hit
- `PaddingIndex`, `compose_parent_space_transform`, `generate_stamp`, `generate_flood`, `erase_candidates`
- `ScatterStrokeCommand`, `ScatterFloodCommand` — undoable mutations
- `StrokeSession` — spacing accumulator, per-target records, single commit
- `ScatterBrushController` — singleton brush state (mode, settings, assets, presets, anchor) with events
- `resolve_asset_texture_set`, `build_materials_payload` — texture-set resolution and the authoring payload

## Settings

| Setting path | Default | Description |
|---|---|---|
| `exts.lightspeed.trex.scatter.core.forceCpuPicker` | `false` | Force the CPU ray picker even when HdRemix is available (tests) |
| `exts.lightspeed.trex.scatter.core.presetsDirectory` | `""` | Override the presets directory (defaults to `${app_documents}/rtx-remix/scatter_presets`) |
| `persistent.exts.lightspeed.trex.scatter.core.brushSettings` | – | JSON of the last used brush settings |
| `persistent.exts.lightspeed.trex.scatter.core.assets` | – | JSON list of the brush asset entries |

## Known limitations

- Placements replicate onto every instance of the target hash (Remix topology); use the anchor target mode for a
  single-instance target
- Instances with non-uniform scale pass that scale on to placements
- No physics settling and no PointInstancer output
- Erase removes only placements authored in the current edit target layer; placements that live in another layer are
  left untouched and reported once per stroke in the log
- A single mouse move that would need more than 256 stamps (for example a grazing pick across the sky) restarts the
  stroke segment at the new sample instead of stamping the whole jump
- Texture assignment happens when a placement is authored, so it does not reach placements that were already painted;
  erase and repaint them, or assign the textures through the Texture Replacements surface, which lists scatter shaders
- Auto-detection claims a texture whose file name contains the material's or the asset's own name followed by the
  name of a texture type. The name may sit anywhere, so an asset pack's own prefix is fine
  (`Quaternius_Leaves_diff` belongs to a material called `Leaves`), and a descriptor may span two words (`nor_gl`).
  A material whose name carries words the texture does not (`Leaves_NormalTree` against `Quaternius_Leaves_diff`) is
  left unassigned rather than matched on a hunch; the per-asset override resolves those cases
