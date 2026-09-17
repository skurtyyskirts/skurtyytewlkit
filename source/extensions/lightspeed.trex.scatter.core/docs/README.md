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
- Keep placement size and vertical offset in world space: the parent instance's scale is divided out, each
  asset's `metersPerUnit` and measured natural size convert its own units into stage units, and the Absolute Size
  scale mode turns a target height into the scale a placement authors (`assets.py`, `sampling.py`)
- Resolve the ingested texture set of a palette asset and author the inputs its material leaves empty as overrides under each placement (`materials.py`)
- Author and remove placements with pure `Sdf` edits and wrap each stroke in one undoable command
  (`ScatterStrokeCommand`, `ScatterFloodCommand`)
- Run the stroke state machine (`StrokeSession`) and expose the shared brush state and events through
  `ScatterBrushController`
- Re-apply placement, scale and seed settings, and the palette's up axis and measurements, to scatter that already
  exists (`reapply.py`): resolve the selection into the placements an edit may write to, recompose them under the new
  settings, and commit one `ScatterReapplyCommand` per edit
- Move that same selection by a world-space delta and settle it back onto the surface (`move.py`), which is what the
  viewport's scatter move gizmo drags

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

### Re-applying settings to placements that already exist

A placement bakes the settings it was painted with into its three transform ops, so it records what it would need to
be composed again: `remixScatterOrigin` (the parent-space surface point, normal and, when the stamp had one, the
stroke heading), `remixScatterKey` (its own random key) and `remixScatterTransformKey` (a checksum of the transform
the brush gave it). The origin is held in parent space rather than world space so it survives the instance moving.

Giving every placement its own random key is what makes this possible at all: `compose_parent_space_transform` used
to draw from the generator of the whole stamp, which several placements share, so no placement's draws could be
replayed on their own. `_build_records` now draws a key per placement from the stamp generator and composes that
placement from `placement_rng(key, settings.seed)`, so painting and re-applying run the same function on the same
inputs and a re-apply at unchanged settings is a no-op. The brush seed is mixed in, which is what makes changing the
seed, or rerolling it, draw a new arrangement for placements that already exist.

`resolve_scope` decides what an edit may touch, and the narrowest selection wins: selected placements or a selected
scatter container scope the edit to exactly those placements, and anything else inside the capture topology scopes it
to the container of the current preset on that prototype. That last rule is what keeps two patches painted with two
presets on one mesh independent. Instance-side paths are rewritten to the prototype side first, because placements are
authored under the prototype and selected through instance proxies.

`ReapplyDriver` follows the controller's `settings_changed`, ignores every change outside `REAPPLIABLE_FIELDS`,
authors the recomposed transforms straight into the edit layer with notices deferred, and commits one
`ScatterReapplyCommand` when the caller says the edit is finished, so a whole slider drag is a single undo step. Above
`LIVE_EDIT_SOFT_LIMIT` placements it stops following every step of a drag and updates when the edit ends instead; like
the prim budget this changes how an edit is paced, never whether it is allowed.

### Moving a patch

`move_scope` transforms the transform a placement already carries rather than recomposing it, and **never reads the
brush settings**. Recomposing would make a drag apply whatever size and rotation ranges are in the Scatter window at
the time, so moving a patch could silently resize it; a move moves.

The gizmo's arrows and its rings are separate gestures, so a drag is either a slide or a turn, and a move treats the
two differently.

A slide carries each placement's recorded surface through the drag and projects it back onto the target's capture
mesh, so a patch slides along the ground instead of floating, and applies the rotation that takes the old normal to
the new one, which is what tilts it onto a slope. That rotation is applied only when `conform_to_surface` is on, so a
patch the brush was told to stand straight up is not tipped over by the ground it is slid across. How far a placement
looks for the surface is the size of the mesh it belongs to, so lifting a patch settles it back down and sliding it
past the edge stops it at the edge, while geometry it was not painted on is never searched.

A turn aims each placement where it stands. Its position and the surface it recorded are left exactly as they are and
only its rotation changes, which is why the rings behave like a trackball on the patch rather than a carousel. The
rotation is applied about the recorded surface point, so a blade pivots where it meets the ground. Carrying the
placements around the gizmo instead — the obvious reading of a rigid-body turn — swings the far side of a patch up
into the air on any ring but the one about the surface normal, and re-planting them afterwards then drops them
somewhere else on the mesh, so a turn became a move nobody asked for.

The rings send a world rotation, and everything a placement holds is in the space of the instance that parents it, so
`move_scope` carries the rotation into that space before applying it. A capture's instances stand at whatever angle the
game left them at; applied as if an instance stood square to the stage, a ring aims the patch about some other axis
than the one being dragged, which on a tilted instance leaves the ring about the stage up axis doing nothing at all
while the other two tip the patch flat.

A recomposition later reproduces the position and, through conform-to-surface, the settling; it cannot reproduce the
turn, so placements record it. `remixScatterTurn` holds the whole turn as X, Y, Z degrees in parent space and
`compose_parent_space_transform` applies it last, after conform, which is what the gizmo applied it to. The older
`remixScatterYaw` keeps its own meaning and its own place in the composition — added to the yaw that is sampled,
before conform — so scatter painted before turns were recorded composes to exactly what it always did; a move no
longer adds to it. With `conform_to_surface` off the settling is not reproduced either, because there is no conform
term to carry it.

`ScopeDragSession` holds the lifecycle of one drag: it takes the patch as it stands, measures every update from there
rather than from the last one, and commits the whole drag as a single `ScatterReapplyCommand`. The viewport gizmo and
the Scatter window's Turn fields are two ways of doing the same thing and share it. The turn a patch has been given
since it was selected lives on the controller (`selection_turn`, `set_selection_turn`,
`subscribe_selection_turn_changed`), kept with the scope it was given to so that a different selection reads as
untouched; that is what lets the gizmo and the window show and drive the same turn while each resolves its own scope.

A placement whose transform no longer matches its checksum was moved by hand and is left alone. One with no recorded
origin was painted before the brush recorded surfaces: it is projected back onto its capture mesh, which also removes
whatever vertical offset it was painted with, and reported, because its original random draws cannot be recovered and
its rotation and scale are drawn afresh.

### Key Classes

- `ScatterBrushSettings` / `ScatterAssetEntry` — validated settings model with JSON and carb persistence
- `PresetStore` — named preset files (list, load, save, rename, clone, delete)
- `MeshSurfaceCache` / `MeshGeometry` — numpy triangle cache with closest-point, raycast and area sampling
- `SurfacePicker`, `HdRemixSurfacePicker`, `CpuRaySurfacePicker`, `create_surface_picker` — picker seam
- `ScatterTarget`, `resolve_target` — prototype, instance and capture-mesh resolution for a hit
- `PaddingIndex`, `compose_parent_space_transform`, `generate_stamp`, `generate_flood`, `erase_candidates`
- `parent_uniform_scale`, `stage_meters_per_unit`, `read_asset_meters_per_unit`, `read_asset_natural_extent` —
  the unit and size facts that make a placement's scale mean a world size
- `ScatterStrokeCommand`, `ScatterFloodCommand`, `ScatterReapplyCommand` — undoable mutations
- `scope_pivot`, `move_scope`, `MoveOutcome` — where a move gizmo sits on a patch, what a drag of it does, and what
  it left behind
- `ReapplyScope`, `resolve_scope`, `reapply_transforms`, `ReapplyDriver`, `ReapplyOutcome` — what a settings change
  may write to, the recomposed transforms, the live edit that authors them, and what it left alone
- `ParentSpaceOrigin`, `to_parent_space_origin`, `placement_rng`, `transform_key` — the surface, the randomness and
  the ownership checksum a placement records so it can be composed again
- `StrokeSession` — spacing accumulator, per-target records, single commit; `rendered_prims` reports what the
  stroke costs the rendered stage, which is the placements it authored multiplied by each target's instance count
- `PrimBudgetStatus`, `prim_budget_status`, `ALERT_BUDGET_MULTIPLE` — classify a stroke's rendered-prim total
  against the `prim_budget` setting so a UI can report an expensive stroke
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
- An instance with non-uniform scale still stretches its placements along its own axes: only the average
  magnitude is compensated (the cube root of the transform's determinant), so sizes are right on average but a
  placement is squashed the way its instance is
- Absolute Size needs the asset's bounding box; an asset that cannot be opened is scattered at its own size and
  says so in the palette tooltip
- The palette's up axis is read from the asset once, when it is added, and can be overridden by hand; it is never
  re-derived afterwards. A wrong value scatters the asset on its side and skews Absolute Size, so a disagreement
  with the file is reported at the start of a stroke. Correcting it, or an asset's measurements landing, re-applies
  to the selected placements live, the way a brush setting does; adding, removing, re-weighting or re-texturing a
  palette entry does not, because none of those change a placement that is already painted
- `prim_budget` only classifies a stroke for reporting. Nothing in this extension caps, trims or stops a paint
  stroke: `flood_max_instances` bounds a flood, but a stroke's length is bounded only by `padding`, which stops
  governing it at all when set to 0
- No physics settling and no PointInstancer output
- Erase removes only placements authored in the current edit target layer; placements that live in another layer are
  left untouched and reported once per stroke in the log
- A single mouse move that would need more than 256 stamps (for example a grazing pick across the sky) restarts the
  stroke segment at the new sample instead of stamping the whole jump
- Texture assignment happens when a placement is authored, so it does not reach placements that were already painted;
  erase and repaint them, or assign the textures through the Texture Replacements surface, which lists scatter shaders
- A move never re-derives a patch: it will not resize, re-seed or re-rotate it from the current settings. The one
  setting it reads is `conform_to_surface`, which decides only whether a slide may turn a placement onto the surface
  it lands on
- A turn never moves a placement, so the rotate rings cannot be used to swing a patch around its middle; painting it
  where it belongs, or sliding it with the arrows, is what places a patch
- A turn the gizmo applies survives further moves but not a recomposition: only the yaw about the stage up axis is
  recorded, so a settings change puts a tilt back onto the surface. The palette's up axis, not the gizmo, is where an
  asset that is permanently mis-oriented is corrected
- Only placement, scale and seed settings re-apply to placements that already exist. The brush and density settings
  (radius, density, strength, spacing, padding) decide where stamps land, which depends on the mouse path a stroke
  followed; that path is not recorded, so changing them affects the next stroke only
- A re-apply writes the three transform ops and nothing else: it never re-points a placement at another asset and
  never rewrites its texture overrides
- A placement painted before the brush recorded surfaces is recognized by projecting it back onto its capture mesh,
  so one sitting farther from the mesh than twice the larger of the current radius and vertical offset is skipped and
  counted rather than moved to somewhere it does not belong
- Auto-detection claims a texture whose file name contains the material's or the asset's own name followed by the
  name of a texture type. The name may sit anywhere, so an asset pack's own prefix is fine
  (`Quaternius_Leaves_diff` belongs to a material called `Leaves`), and a descriptor may span two words (`nor_gl`).
  A material whose name carries words the texture does not (`Leaves_NormalTree` against `Quaternius_Leaves_diff`) is
  left unassigned rather than matched on a hunch; the per-asset override resolves those cases
