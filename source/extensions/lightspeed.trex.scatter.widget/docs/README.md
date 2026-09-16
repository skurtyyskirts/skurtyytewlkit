# lightspeed.trex.scatter.widget

Dockable **Scatter** window for the RTX Remix asset scatter brush. It edits the shared brush state owned by
`lightspeed.trex.scatter.core` and shows the assets that will be scattered.

## Responsibilities

- Create the `Scatter` workspace window (`ScatterWindow`, a `WorkspaceWindowBase`) and register it with
  `omni.ui.Workspace` so the StageCraft layout can dock it next to the Editor and the `Window` menu can show it
- Build the settings pane (`ScatterPane`): Paint / Erase mode buttons, Flood, and collapsible BRUSH, PLACEMENT (with
  a nested ADVANCED ROTATION frame), SCALE, RANDOM, TARGET, ASSETS and PRESETS sections whose fields are bound both
  ways to `ScatterBrushController`; the SCALE section swaps its rows with the Scale Mode combo, showing a height
  range for Absolute Size and the scale-factor rows for Multiplier
- Manage the scatterable asset list (`ScatterAssetListWidget`): add through the replacement asset file picker, add
  from the current selection, browse the project's ingested models, remove, enable/disable, weight and up-axis per row;
  every added asset passes through `accept_asset_if_valid_for_replacement` so non-ingested assets show the usual
  ingestion dialog, and an asset saved with a USD crate version this build cannot read is refused with a status message
- Report what a flood and a stroke will cost: the TARGET section shows the flood cap multiplied by the anchor's
  instance count, reads `per instance` in Hit Surface mode where the targets are only known once the flood runs, and
  carries the Prim Budget that colours the viewport's live stroke counter
- Own the live edit of scatter that is already painted (`ReapplyDriver`, from the core): the Edit Selection Live
  toggle at the top of PLACEMENT, the readout of what a settings change would update, and telling the driver when a
  control has finished being edited so a whole drag is one undo step
- Preset management UI (`PresetsWidget`): choose, save, save as, rename, clone and delete brush presets
- Surface controller status messages, the "placements replicate onto N instances" warning and the flood estimate
  (`scatter_flood_estimate`: the flood cap multiplied by the anchor's instance count, so a flood onto a replicated
  prototype shows the number of prims it will really add)

## Non-Responsibilities

- Does not pick surfaces, draw the brush cursor or handle viewport gestures; that is the scatter brush tool in
  `lightspeed.trex.viewports.shared.widget`
- Does not contain scatter math, USD authoring or undo logic; all of that lives in `lightspeed.trex.scatter.core`
- Does not ingest assets; it only validates and routes to the ingestion workflow through the shared dialogs

## Textures

The ASSETS section carries an **Assign Textures** toggle and a read-only **Maps** column per row. The column names
only what the brush will paint onto that asset: a badge per map it will author (`A N R`), `asset` when the asset
supplies its own textures and nothing needs adding, and `none` when there is nothing to assign at all. Maps a
texture set simply does not contain are listed in the row's tooltip rather than counted in the cell, because a set
without a normal map is ordinary rather than a fault. The button beside the cell picks a texture set by hand when
auto-detection finds nothing or picks the wrong one, and turns into **auto** to go back to detection.

## Architecture

`TrexScatterWindowExtension` instantiates `ScatterWindow` for the StageCraft USD context (`""`) and registers its
show function with `omni.ui.Workspace`. The window lazily builds a `ScatterPane`, which subscribes to the controller's
`settings_changed`, `assets_changed`, `mode_changed` and `status_message` events and pushes user edits back through
`controller.update_settings(...)`, `controller.add_asset(...)` and friends. Numeric fields are `FloatBoundedDrag` /
`IntBoundedDrag` widgets over a `DragValueModel`, clamped to the bounds `field_bounds` reads from the pydantic
constraints of `ScatterBrushSettings` and `ScatterAssetEntry`, so the UI can never offer a value the model rejects.
Every control carries an `identifier` (`scatter_<name>`) so end-to-end tests can drive the pane.

The pane owns the `ReapplyDriver` that applies placement, scale and seed changes to the scatter the user has
selected, and hands it the `MeshSurfaceCache` it already keeps for Flood. Knowing when an edit is finished is the
pane's job, because only it can tell a slider being dragged from a checkbox being clicked: a bounded drag field
commits on its `end_edit`, a checkbox or combo box commits on its value change, and a change that reached the
controller from somewhere else entirely, such as a preset being applied or the seed being rerolled, commits as soon
as it arrives. Brush and density settings stay editable while an edit is armed rather than being greyed out, because
the radius still has to be adjustable for the next stroke; the readout says that they apply to new strokes only.

The instance count behind the replication warning and the flood estimate walks `/RootNode/instances`, so the pane
caches it per `(target_mode, anchor_prototype_path)`; palette changes never recount. The "Browse ingested" combo box
re-lists the project directory on the frame after the click, never inside the click, and only replaces its entries
when the file names changed.

### Key Classes

- `ScatterWindow` — `WorkspaceWindowBase` subclass titled `WindowNames.SCATTER`
- `ScatterPane` — the `WorkspaceWidget` content with all sections
- `ScatterAssetModel` / `ScatterAssetDelegate` / `ScatterAssetListWidget` — the asset list; each `ScatterAssetItem`
  row owns the weight drag field built for it and releases it when the row is rebuilt or destroyed
- `PresetsWidget` — preset combo box and buttons
- `ChoicesComboModel` — refreshable `ui.ComboBox` model shared by the up-axis, browse and preset combo boxes
- `DragValueModel` / `field_bounds` — numeric model that lets the bounded drag widgets install their hard clamp, and
  the lookup of that clamp from the pydantic field constraints

## Usage

The window is loaded by `lightspeed.trex.control.stagecraft` and docked by
`lightspeed.trex.app.resources/data/layouts/stagecraft_default_layout.json`. Toggle it from `Window > Scatter`.

## Live editing what is already painted

**Edit Selection Live**, at the top of the PLACEMENT section and on by default, decides whether a settings change also
updates the scatter that is selected. The readout beside it says what is in scope before anything is changed
(`428 placements - mesh_A64083D9 / grass`), and afterwards adds what the edit left alone and why: placements moved by
hand, placements painted before the brush recorded surfaces, and placements authored outside the edit target layer.

The narrowest selection wins. Selecting painted placements or a scatter container edits exactly those; selecting a
mesh or an instance edits what the current preset painted on it, which is how two patches painted with two presets on
one mesh stay independent. With nothing selected, a settings change only affects the next stroke.
