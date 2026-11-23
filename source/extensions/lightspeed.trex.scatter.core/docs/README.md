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
  (`MeshSurfaceCache`), keeping every query to the triangles near it and its memory bounded whatever the brush radius
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
- Keep the stroke just painted as the preview (`ScatterBrushController.begin_preview`): until anything else happens,
  live edits change that stroke and belong to it, so it can be tuned by eye and taken away with one undo while the
  settings it was tuned to stay set
- Merge painted placements into one `UsdGeom.PointInstancer` per patch and split them back (`instancing.py`), and
  keep a viewport stand-in of every instancer in the session layer (`preview.py`), because HdRemix does not draw a
  point instancer's instances while the runtime that plays the mod does
- Create, retune and remove scatter groups for the REST API and its MCP tools (`groups.py`, `group_models.py`), and
  list the captured surfaces a group can fill with PNG previews of their textures (`previews.py`)

## Non-Responsibilities

- Does not create any UI and does not import `omni.ui` or `omni.ui.scene`. The Scatter window lives in
  `lightspeed.trex.scatter.widget`; the viewport brush tool (gestures, cursor, toolbar button, hotkeys) lives in
  `lightspeed.trex.viewports.shared.widget`
- Does not ingest assets or show ingestion dialogs; callers gate assets with
  `lightspeed.trex.utils.widget.asset_validation` before adding them to the brush
- Does not create or bind materials, and never overwrites a texture an asset or the user already authored; an asset without a material is painted as it is and reported through a status message
- Does not do physics settling, and does not merge placements into a single baked mesh: instancing keeps one copy of
  each asset's geometry and stays splittable, where a baked mesh would copy the geometry once per blade and could not
  be edited again

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
commits a single `ScatterStrokeCommand` on release; the first `do()` of that command is a no-op, `undo()` snapshots
the placements as they stand and removes them, and redo restores that snapshot. Erase strokes snapshot the removed
specs with `Sdf.CopySpec` so undo restores them. No `omni.kit.undo` group is held open across frames.

A stamp spreads its candidates over the plane of the surface under the cursor and snaps each one onto the closest
surface within the brush radius. Near a wall or a ledge some candidates land on the side of it, and conformed to that
side they stand sideways. `max_surface_angle` skips every candidate, of a stamp or a flood, whose surface is steeper
than that from the stage up axis. It is measured from the up axis rather than from the surface under the cursor
because the cursor itself often lands on the bevel or slope beside a floor; measured from there, the blades on that
slope all counted as in range. Painting on a wall or a ceiling needs the default of 180.

Capture meshes cannot be trusted to say which side of them is up. Many games wind their triangles the other way round,
and on a Tomb Raider Anniversary capture every normal the brush recorded pointed down into the ground, so conformed
assets hung upside down under the floor and every floor failed the surface angle limit. `generate_stamp` therefore
turns the normal under the cursor to face `view_point` (the camera, handed in through `StrokeSession`) and every
candidate's normal to face the same side; without a view point, and always for a flood, normals are turned toward the
stage up axis. `conform_to_surface` is off by default, so an asset stands straight up whatever the slope, and without
it `compose_parent_space_transform` stands the asset on world up as the parent instance sees it rather than on the
parent's own up axis, which a rotated capture instance would otherwise tilt.

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
presets on one mesh independent, and it only holds while `apply_to` is `ALL`: with `SELECTED`, the default, the
selected mesh is where the next stroke goes, and treating it as a scope would make every setting changed between two
strokes rewrite everything already painted on it. Instance-side paths are rewritten to the prototype side first,
because placements are authored under the prototype and selected through instance proxies.

`ReapplyDriver` follows the controller's `settings_changed`, ignores every change outside `REAPPLIABLE_FIELDS`,
authors the recomposed transforms straight into the edit layer with notices deferred, and commits one
`ScatterReapplyCommand` when the caller says the edit is finished, so a whole slider drag is a single undo step. Above
`LIVE_EDIT_SOFT_LIMIT` placements it stops following every step of a drag and updates when the edit ends instead; like
the prim budget this changes how an edit is paced, never whether it is allowed. There is one driver per USD context
(`get_reapply_driver`), shared by the Scatter window and the viewport gizmo, so a change is applied once whichever of
them made it and the gizmo can edit the preview while the window is closed; a caller that drives an edit through
several settings changes, like a ring drag, holds it open with `hold` and `release`.

An edit reads its placements once, when it opens (`prepare_reapply`), and every step of the drag only composes and
writes them (`PreparedReapply.compose`): nothing a re-apply reads from a placement changes while a setting is dragged,
and reading a placement back from its layer costs more than composing it. A placement's random draws are taken from its
generator once per seed and replayed, with the same arithmetic numpy uses, so a replay is exact.

### The stroke preview

When the viewport brush commits a paint stroke it calls `begin_preview`, and the stroke on top of the undo stack
becomes the preview. Until something else is executed, undone or redone, the selection changes or the stage closes,
`preview_scope` is what the live-edit driver and the gizmo work on, ahead of the selection. An edit of the preview adds
no undo step of its own (`folds_into_preview`): it stays in the layer as part of the stroke, the stroke's `undo()`
snapshots it along with everything else, and a redo brings the stroke back tuned. Brush settings were never on the
undo stack, so one undo removes the stroke and leaves every setting it was tuned to in place for the next one.

What carries an orientation into later strokes is the base rotation (`base_rotation_x/y/z`), a brush setting applied
right after the up-axis correction and before the random tilt and yaw, so it turns the asset in its own frame: every
placement is stood up or leaned the same way whatever yaw it then draws. A turn of painted scatter cannot do that,
because it is applied after the yaw and lands differently on every placement; a turn also stays with the scatter it was
given to. On the preview the gizmo's rings therefore set the base rotation instead of turning the patch.

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

`ScopeDragSession` holds the lifecycle of one drag: it reads the patch once as it stands (`prepare_move`), measures
every update from there rather than from the last one, writes each step once, and commits the whole drag as a single
`ScatterReapplyCommand`, or as part of the stroke when the patch is the preview. A slide looks for the surface in one
batched query whose reach is the distance the slide carried the patch: each placement stood on the mesh before the
slide, so the nearest surface is never farther than that, and the bounded search finds exactly the point a search of
the whole mesh would. Above `LIVE_EDIT_SOFT_LIMIT` placements a drag moves the patch once, when it ends. The viewport gizmo and
the Scatter window's Turn fields are two ways of doing the same thing and share it. The turn a patch has been given
since it was selected lives on the controller (`selection_turn`, `set_selection_turn`,
`subscribe_selection_turn_changed`), kept with the scope it was given to so that a different selection reads as
untouched; that is what lets the gizmo and the window show and drive the same turn while each resolves its own scope.

A placement whose transform no longer matches its checksum was moved by hand and is left alone. One with no recorded
origin was painted before the brush recorded surfaces: it is projected back onto its capture mesh, which also removes
whatever vertical offset it was painted with, and reported, because its original random draws cannot be recovered and
its rotation and scale are drawn afresh.

### Instanced output

A loose placement is a prim of its own, so the runtime turns every blade into a replacement it processes and submits
every frame. The runtime's USD mod loader has a fast path for `UsdGeom.PointInstancer`
(`enableReplacementInstancerMeshRendering`, on by default): each prototype mesh becomes one replacement carrying a
list of transforms, its geometry is uploaded once and its instances are culled by distance on the GPU
(`rtx.pointInstancer.cullingRadius`, 5000 units by default, with an optional `fadeStartRadius`). `instancing.py` puts
painted scatter into that form.

Each patch, a `scatter_<slug>` container, gets a sibling `instanced_<slug>` `PointInstancer` marked `IsRemixScatter`,
with one prototype per distinct asset and texture-override set under its `Prototypes` scope (`p_<signature hash>`, an
`Xform` referencing the asset with an empty `xformOpOrder`). The instancer sits beside the container rather than inside
it, so a container still holds nothing but loose placements and keeps its one-edit removal and restore. It authors the
five arrays the runtime reads — `positions`, half-precision `orientations`, `scales`, `protoIndices` and `ids` — and,
beside them, every value a placement recorded (`remixScatter:translates`, `rotations`, `origins`, `keys`,
`transformKeys`, `yaws`, `turns`), so a split gives each placement back exactly, name included: an entry's id is the
twelve hex digits of the placement's name. The prototype's empty op order matters twice: a placement's own ops
replaced the ones the asset authors on its root, and the runtime ignores a prototype root's transform altogether, so
any other choice would make the toolkit and the game disagree.

`InstancingEdit` is the one reversible unit: `instance` puts a block of entries into an instancer, creating it and the
prototypes it needs from its snapshot, and removes the loose prims; `uninstance` does the opposite. Entries are found
by id, so an edit is undone correctly whatever else the instancer gained since, as long as later edits were undone
first, which the undo stack guarantees. A PLACE `ScatterStrokeCommand` with `instance_others` merges every other loose
placement of its layer when it is first done, as part of its own undo step: the stroke before it has just stopped being
the preview. A flood merges itself as well. `ScatterInstanceCommand` (Instance All) and `ScatterSplitCommand` cover the
rest, and an erase stroke removes entries through `InstancedEraser`, whose edits the erase command undoes and repeats.
A placement carrying anything an entry could not hold, such as a visibility or a texture set on it by hand, is left
loose; so is anything authored by another layer, and a new instancer never takes a name another layer already uses,
because its arrays would replace that layer's wholesale.

HdRemix in the toolkit draws each prototype of a point instancer once, where it stands, and none of its instances:
painted grass would turn into a few giant blades. `InstancerViewportPreview`, started by the extension for the StageCraft
context, therefore keeps a stand-in of every scatter instancer in the stage's **session layer**: an `over` that hides the
instancer, and an `<instancer>_viewport` container holding one `e_<id>` prim per entry that internally references the
entry's prototype with the entry's three ops. The session layer is never saved or packaged, so the mod holds only the
instancer while the toolkit shows exactly what the game will draw, at the toolkit's usual cost of one prim per
placement. It follows the brush through `subscribe_instancer_changes` rather than through every attribute change on the
stage, and only writes the entries that appeared, left or changed since its last sync. An entry that leaves is
deactivated rather than deleted, and activated again if its id comes back: Sdf removes a child by rewriting its
parent's list of children, so deleting the stand-ins of a large erase one by one cost the size of the patch for each.

### Keeping the brush cheap

The brush runs on every hover pick and every mouse move of a stroke, so each part of that path is kept from scaling
with the size of the capture or the strength of the settings:

- **Closest-point search.** A stamp's candidates sit on the tangent plane under the cursor and snap to the nearest
  surface within the brush radius. Searching the whole radius for every candidate paired each one with every triangle
  in a radius-sized box: 500 candidates at a radius of 300 over an 80,000-triangle mesh took 8.4 s and 6 GB. The search
  now starts at a reach about the size of one triangle and widens only for the candidates it has not settled; a
  candidate whose nearest triangle lies within the reach is settled exactly, because any closer triangle would have
  had it inside its bounds grown by that reach. Triangles are sorted along a Z-order curve into blocks of 64
  (`MeshGeometry.search_index`, built on the first query), each candidate tests only the blocks its own box overlaps,
  pairs whose bounds lie beyond the nearest corner found are dropped before the exact test, and pairs are reduced a
  slice at a time. The same case takes 24 ms and under 1 MB, and the result is the one an exhaustive search gives,
  ties going to the lowest triangle index. Ray casts test the same blocks first.
- **Stroke pacing.** One stamp can cost tens of milliseconds at heavy settings, and one mouse move can ask for up to
  `_MAX_STAMPS_PER_UPDATE` of them. `StrokeSession` queues the samples it is given and stops stamping once
  `_STAMP_TIME_BUDGET` (8 ms) of a frame is spent, carrying on from its last stamp at the next update or `resume`,
  which the viewport tool calls every frame while `pending`. The brush then trails the cursor instead of freezing the
  application, and the stamps are the ones an unhurried walk makes. `end` stamps what the stroke is still behind on
  for at most `_END_DRAIN_BUDGET` (1 s) before it commits; a stroke still behind after that leaves the rest of its
  path unpainted and reports it through `cut_short`, where it used to freeze the application at mouse-up for as long
  as the backlog took. The stamp centres of a segment are projected in one batched query
  (`MeshSurfaceCache.closest_samples`).
- **Stroke limit.** One stroke, and one flood, stops adding placements at `stroke_prim_limit(prim_budget)` rendered
  prims, five times the budget and the point where the counter turns red. A placement renders once per instance of
  its prototype, so the limit is counted in rendered prims: `StrokeSession` trims the stamp that would pass it, sets
  `limit_reached` and ignores the samples that follow. A mesh with more instances than the stroke has room left for
  is a different case, because the next mesh the stroke reaches may still fit: the stroke skips it, notes it in
  `blocked_prototypes` and carries on, and `can_place_on` lets the viewport grey the cursor over it.
  `flood_targets` draws `_FLOOD_OVERSAMPLE` samples for each
  placement a target still has room for, keeps what fits and reports it through `FloodOutcome.limited`. Without it a
  long drag with `padding` at 0, or a short one over
  a mesh with hundreds of instances, could author more than the game or the Toolkit can draw.
- **Padding across instances.** Padding is measured between the surface points placements were painted on, not
  between where they stand, so a vertical offset cannot be used to stack placements on one spot. A placement authored
  under a prototype stands on every instance of it, so a stroke that crosses from one instance to another of the same
  mesh seeds the padding index for each instance it reaches and keeps it current with what it has painted through
  the others.
- **Removing placements.** Sdf removes a child by searching and rewriting its parent's list of children, so deleting
  5,000 placements out of a container of 10,000 one by one took 0.85 s, and 10,000 out of 20,000 took 3.5 s.
  `remove_placements` deletes a container that loses every child in one edit, and rebuilds one that keeps a few by
  copying those out and back (`Sdf.CopySpec` with a filter on `primChildren`), about 50 µs per child kept, whenever
  that is the cheaper of the two. `author_placements` asks the layer for a path instead of searching a container's
  children for a name, which was the same linear search once per placement authored.
- **Erasing instanced scatter.** An instancer is rewritten whole however few entries leave it. `InstancedEraser`
  queues what each stamp erases and writes every instancer once per frame of the stroke (`flush`), rather than once
  per stamp.
- **Mesh cache.** `MeshSurfaceCache` keeps the world bounds of every mesh it has built after the geometry itself is
  evicted, and answers a ray that misses them without the triangles. Painting on a selection casts one ray at every
  selected mesh on every hover, and with more meshes selected than the cache holds that rebuilt each of them every
  time. A change notice drops the bounds with the geometry, and an invalidation looks each known mesh's ancestors up
  in the set of changed paths instead of comparing every mesh with every one of the thousands of paths a stroke
  changes.
- **Target resolution.** `resolve_target` runs on every hover pick and used to walk every child of
  `/RootNode/instances` twice to count and order a prototype's instances, which on a capture of 10,000 instances cost
  a whole frame each time. The instances of every hash are read once per stage and dropped only by a resync that can
  add, remove or rename one; painting never does. A hit on the capture mesh itself is used as it is rather than looked
  for again by bounding box, which also stops a neighbouring mesh whose bounds contain the point from being picked.
- **Padding.** `PaddingIndex` cells are as large as the padding, so a check visits the 27 cells around a candidate
  however fine the padding, and it compares plain floats, which at a few points per cell costs a fraction of a numpy
  call per cell.
- **CPU picker.** The fallback picker culls meshes by the world bounds the stage reports before building any triangles,
  and skips painted placements with everything beneath them, which a field of grass would otherwise make it walk on
  every pick.

### Scatter groups

A scatter group is a named flood that an agent asks for in words ("a heavy amount of grass on the grass, longer,
closer together") through `lightspeed.trex.service.core`, whose routes become `remix_*` MCP tools. `ScatterGroupsCore`
holds no state of its own:

- **Surfaces.** `get_surfaces` groups the captured prototypes under `/RootNode/meshes` by the material they render
  with, which is the replacement's material where a mod replaced the capture mesh, and reports each material's albedo
  texture. Capture textures are named by hash, so it also writes a PNG preview per texture and numbered contact sheets
  (`previews.py`, Pillow, alpha dropped because games keep other data there) for the agent to look at; decoding runs
  off the main loop.
- **Settings.** `group_settings` starts from the default brush settings, never from the user's current ones, and
  applies the amount (`light`, `medium`, `heavy`: a density per 50-unit disk and a cap, heavy stopping at the
  10,000-placement stroke budget) and then the group's tuning. The preset name `group <name>` gives the group its own
  `scatter_group_<name>` containers and instancers, so a group never touches scatter painted by hand.
- **Flood.** `flood_targets` is the loop `ScatterBrushController.flood` runs, taking settings and assets as arguments,
  so a group floods the same way the brush does. Targets, assets (`is_asset_ingested`, measured and textured as the
  palette does) and settings are all checked before the first edit, so a refused request changes nothing. A group
  stops at the same rendered-prim limit a flood from the window does, at the default budget, and records
  `prim_limited` when it did, so an agent reading a count below the cap is told a heavier amount would add nothing
  rather than left to conclude the surface is full.
- **Record.** Each group is recorded as JSON in the `remixScatterGroups` custom layer data of the layer it authors,
  with asset paths relative to that layer, so it can be retuned in a later session and travels with the mod.
- **Edits.** Create, update and remove are one undo group each. Update removes the group's specs with
  `ScatterRemoveBrushCommand` and floods again with the same seed unless asked to reseed, so changing only the size
  keeps the layout. `brush_spec_paths` finds a group's instancers by the container they record rather than by name,
  because a merge takes `instanced_<slug>_1` while the viewport's hidden stand-in of a just-removed instancer still
  holds `instanced_<slug>` on the stage. `ScatterSetGroupRecordCommand` writes or clears the record.

### Key Classes

- `ScatterBrushSettings` / `ScatterAssetEntry` — validated settings model with JSON and carb persistence;
  `from_json_dict` brings settings and presets saved before `SETTINGS_VERSION` 2 onto the defaults that version
  introduced (Apply To Selected, Conform off) once, because persisted values otherwise outlive a change of default
- `PresetStore` — named preset files (list, load, save, rename, clone, delete); `stored_name` says which stored
  preset a save would overwrite, which on a file system that ignores case can be one spelled differently
- `MeshSurfaceCache` / `MeshGeometry` — numpy triangle cache with closest-point (single, batched as arrays, or batched
  as samples through `closest_samples`), raycast and area sampling
- `SurfacePicker`, `HdRemixSurfacePicker`, `CpuRaySurfacePicker`, `create_surface_picker` — picker seam
- `ScatterTarget`, `resolve_target` — prototype, instance and capture-mesh resolution for a hit
- `selected_capture_meshes` — the instance-side capture meshes a selection names, which the brush casts its ray at
  when it paints on the selection
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
  stroke costs the rendered stage, which is the placements it authored multiplied by each target's instance count,
  `pending` / `resume` let a stroke that fell behind its frame budget catch up, and `prim_limit`, `limit_reached`,
  `blocked_prototypes`, `can_place_on`, `cut_short` and `failed` say where the stroke stops, what it skipped and why
  it ended short of its path
- `reset_instance_index` — release the per-stage index of instances that target resolution reads from (extension
  shutdown)
- `PrimBudgetStatus`, `prim_budget_status`, `ALERT_BUDGET_MULTIPLE`, `stroke_prim_limit` — classify a stroke's
  rendered-prim total against the `prim_budget` setting so a UI can report an expensive stroke, and give the total at
  which a stroke or a flood stops
- `FloodOutcome`, `flood_sample_count` — what a flood placed, what it renders and whether its limit stopped it, and
  how many samples a target would take before any cap
- `prune_inert_overs` — remove the empty `over` ancestors authoring left behind once the scatter under them is gone
- `ScatterBrushController` — singleton brush state (mode, settings, assets, presets, anchor) with events; also
  owns the stroke preview (`begin_preview`, `preview_scope`, `folds_into_preview`, `end_preview`,
  `subscribe_preview_changed`), and `reset_settings` puts every brush setting back to its default while keeping the
  palette and the presets
- `get_reapply_driver`, `prepare_reapply`, `PreparedReapply`, `prepare_move`, `PreparedMove` — the shared live-edit
  driver of a context, and the read-once state an edit or a drag works from
- `PlacementPool`, `placement_pool` — the placements of a prototype read once and taken out by distance, which is how
  an erase stroke finds what is under the brush at every stamp without reading the whole prototype again
- `resolve_asset_texture_set`, `build_materials_payload` — texture-set resolution and the authoring payload
- `merge_loose_placements`, `split_instancers`, `InstancingEdit`, `InstanceBlock`, `read_block` — moving placements
  into and out of a patch's `PointInstancer` as reversible edits, and the columnar entries they move
- `InstancedEraser`, `InstancedPool`, `instanced_pool`, `instanced_points` — erase and padding over instanced entries
- `ScatterInstanceCommand`, `ScatterSplitCommand`, `ScatterBrushController.instance_all`,
  `ScatterBrushController.split_selection`, `instancers_of_selection` — Instance All and Split Selection
- `InstancerViewportPreview`, `sync_instancer_preview`, `subscribe_instancer_changes` — the toolkit viewport's
  session-layer stand-in of instanced scatter
- `ScatterGroupsCore`, `group_settings`, `flood_targets`, `brush_spec_paths`, `ScatterRemoveBrushCommand`,
  `ScatterSetGroupRecordCommand` — scatter groups for the REST API; `ScatterGroup*Model`, `ScatterSurface*Model`
  and `ScatterAmount` are the request and response models its routes use

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
- One stroke or one flood stops at five times `prim_budget` rendered prims. The limit is per stroke, not per scene:
  nothing stops a second stroke from adding as much again, and the scene total is only reported. A mesh with more
  instances than the limit cannot be painted on at all until `prim_budget` is raised, because one placement on it
  already renders more than a stroke may add
- A live edit, a gizmo drag and a re-apply compose a placement against the first instance of its prototype, not the
  instance it was painted through. Instances of one mesh that the game placed at different rotations or scales
  therefore show an edited patch slightly differently from how it was painted on any but the first; recording the
  instance a placement was painted through would be needed to fix it
- Undoing a gizmo drag of a placement painted before surfaces were recorded leaves the surface the drag recovered on
  it, so it is no longer reported as a legacy placement afterwards
- On a Y-up stage the second tilt range leans a placement about Z, the other horizontal axis there. Scatter painted
  on a Y-up stage with a non-zero range before that was so is recomposed about Z by its first live edit, so it
  changes once; Remix captures are Z-up, where nothing changed
- A drag that spans more prototypes than the mesh cache holds (8) rebuilds their geometry as it goes from one to the
  next, and painting on a selection builds the geometry of each selected mesh the first time the cursor crosses its
  bounds, which on very large meshes is a visible hitch
- No physics settling
- Instanced scatter cannot be live-edited or moved with the gizmo; Split Selection turns it back into placements first.
  Erase and padding work on it as they do on placements
- The toolkit viewport draws instanced scatter through its session-layer stand-in, one prim per placement, so the
  toolkit itself is no faster after instancing; the saving is in the game. The stand-in follows the brush, undo and a
  re-opened stage; an instancer edited by hand outside the brush is re-shown only on the next resync or `refresh`
- In the game, instanced scatter farther than `rtx.pointInstancer.cullingRadius` (5000 units by default) from the
  camera is culled, with a hard edge unless `rtx.pointInstancer.fadeStartRadius` is set; those are runtime options in
  the game's `rtx.conf`, not brush settings
- Orientations are stored at the half precision the runtime reads, so an instanced placement can be turned by up to
  about a twentieth of a degree from its loose transform; the exact rotation is kept for Split
- Erase removes only placements authored in the current edit target layer; placements that live in another layer are
  left untouched and reported once per stroke in the log
- A single mouse move that would need more than 256 stamps (for example a grazing pick across the sky) restarts the
  stroke segment at the new sample instead of stamping the whole jump
- A stroke whose stamps cost more than a frame allows trails the cursor and says so; releasing the mouse finishes the
  stamps it is still behind on for up to a second, and a stroke dragged farther ahead of the brush than that loses
  the end of its path, which the brush reports. Holding the button until the brush has caught up paints all of it
- An erase stroke, or a stroke with padding, reads every placement already on a prototype the first time it reaches
  that prototype, through the composed stage; on a prototype holding tens of thousands of placements that first stamp
  takes a noticeable moment
- Texture assignment happens when a placement is authored, so it does not reach placements that were already painted;
  erase and repaint them, or assign the textures through the Texture Replacements surface, which lists scatter shaders
- A move never re-derives a patch: it will not resize, re-seed or re-rotate it from the current settings. The one
  setting it reads is `conform_to_surface`, which decides only whether a slide may turn a placement onto the surface
  it lands on
- A turn never moves a placement, so the rotate rings cannot be used to swing a patch around its middle; painting it
  where it belongs, or sliding it with the arrows, is what places a patch
- A turn stays with the scatter it was given to and never reaches the next stroke. The base rotation, or the
  palette's up axis, is where an asset that comes out lying down is corrected for good
- `max_surface_angle` decides where stamps land, so like the brush settings it affects the next stroke only
- The preview ends as soon as anything else reaches the undo stack, so a stroke tuned, then followed by some other
  edit, can no longer be taken away together with its tuning in one undo; each later edit of it is an undo step of its
  own
- Removing a few thousand placements from a container that keeps many more is still paid child by child: the rebuild
  only wins when few children stay, so removing 1,000 placements out of 20,000 loose ones in one edit costs roughly
  half a second at the rate measured per child. Instancing
  makes that rare, because once earlier strokes are merged the stroke just painted is usually all its container holds
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
