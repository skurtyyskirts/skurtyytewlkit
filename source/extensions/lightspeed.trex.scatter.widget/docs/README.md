# lightspeed.trex.scatter.widget

Dockable **Scatter** window for the RTX Remix asset scatter brush. It edits the shared brush state owned by
`lightspeed.trex.scatter.core` and shows the assets that will be scattered.

## Responsibilities

- Create the `Scatter` workspace window (`ScatterWindow`, a `WorkspaceWindowBase`) and register it with
  `omni.ui.Workspace` so the StageCraft layout can dock it next to the Editor and the `Window` menu can show it
- Build the settings pane (`ScatterPane`): Paint / Erase mode buttons, Flood, Reset to Defaults, and collapsible
  BRUSH, PLACEMENT (with a nested ADVANCED ROTATION frame), SCALE, RANDOM, TARGET, OUTPUT, ASSETS and PRESETS sections
  whose fields are bound both ways to `ScatterBrushController`; the SCALE section swaps its rows with the Scale Mode
  combo, showing a height range for Absolute Size and the scale-factor rows for Multiplier
- Show and drive how painted scatter is stored (OUTPUT): the Instance Strokes toggle, Instance All and Split Selection,
  which call `ScatterBrushController.instance_all` and `split_selection`, and a count of the placements on the stage and
  how many of them are instanced, recounted when a stroke commits or the selection or preview changes
- Say which way each palette asset was modelled (`AssetUpAxisWidget`, at the top of PLACEMENT): the axis its file
  authors beside the Y-up/Z-up choice the brush stands it up by, flagged when the two disagree
- Manage the scatterable asset list (`ScatterAssetListWidget`): add through the replacement asset file picker, add
  from the current selection, browse the project's ingested models, remove, enable/disable, weight and up-axis per row;
  every added asset passes through `accept_asset_if_valid_for_replacement` so non-ingested assets show the usual
  ingestion dialog, and an asset saved with a USD crate version this build cannot read is refused with a status message
- Report what a flood and a stroke will cost: the TARGET section shows the flood cap multiplied by the anchor's
  instance count, held to the limit a flood stops at, reads `per instance` in Hit Surface mode where the targets are
  only known once the flood runs, and carries the Prim Budget that colours the viewport's live stroke counter and, at
  five times its value, stops a stroke or a flood. A flood asks for confirmation when its cap is above the default or
  when what it will render is above the Prim Budget, counting the anchor's instances or, without an anchor, those of
  the meshes selected when Flood is clicked
- Keep what it shows true to the stage: undo and redo recount the OUTPUT totals and reset the Turn fields, opening
  or closing a project clears the status line and the counts of the one before, and a control that does nothing with
  the settings as they stand, such as the scale range rows while Random Scale is off, is greyed out
- Own the live edit of scatter that is already painted (`ReapplyDriver`, from the core): the Edit Selection Live
  toggle at the top of PLACEMENT, the readout of what a settings change would update, and telling the driver when a
  control has finished being edited so a whole drag is one undo step
- Preset management UI (`PresetsWidget`): choose, save, save as, rename, clone and delete brush presets; Save As
  asks before it replaces another preset that already has the name, which on a file system that ignores case
  includes one spelled with different capitals, and keeps the spelling the stored preset has
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
as it arrives. The controller notifies the pane and the driver in no particular order, so before the pane finishes an
edit it hands the driver the change itself (`ReapplyDriver.follow_settings`); finishing first, whenever the pane
happened to be notified ahead of the driver, left the edit open with no undo step until something else closed it.
Brush and density settings stay
editable while an edit is armed rather than being greyed out, because the radius still has to be adjustable for the
next stroke; the readout says that they apply to new strokes only.

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
- `AssetUpAxisWidget` — one row per palette asset with the up axis its file authors and the one the brush uses; it
  reads each file once per path, because a weight drag changes the palette at every step
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

The narrowest selection wins. Selecting painted placements or a scatter container edits exactly those. With
**Apply To** set to All, selecting a mesh or an instance edits what the current preset painted on it, which is how two
patches painted with two presets on one mesh stay independent. With Selected, the default, a selected mesh is only
where the next stroke goes, so settings changed between strokes never rewrite the field already painted on it. With
nothing selected, a settings change only affects the next stroke.

Right after a stroke the readout says `Last stroke: ...`: until anything else happens, a settings change updates the
stroke just painted, and those changes are part of the stroke rather than undo steps of their own. That is the loop
for tuning a brush by eye: paint a test stroke, adjust **Base Rotation**, size or any other placement setting while
watching it, then undo once. The stroke disappears with everything done to it, the settings stay as they were left,
and the next stroke is painted with them. Selecting anything ends it, and edits then go to the selection as usual.

**Base Rotation X**, **Y** and **Z** turn every placement about its own root, before its random rotation, about the
stage axes. They are the setting for an asset that comes out lying down or leaning: they apply to every stroke painted
from now on and are saved with the preset. On the stroke just painted the viewport's rotate rings set them too.

**Max Surface Angle**, in the BRUSH section, is the steepest surface, in degrees from the stage up axis, the brush
still paints on. A stamp snaps each candidate onto the closest surface, so near a wall or a ledge some land on its side
and stand sideways; 45 suits grass, and the default of 180 keeps every candidate, which painting on a wall needs.

**Turn X**, **Turn Y** and **Turn Z**, below it, show the degrees that selection has been turned by since it was
selected. The viewport's rotate rings write them as they are dragged, and typing or dragging them turns the selection
the same way; either is one undo step. They are greyed out while nothing scattered is selected and read zero again for
a new selection, because a turn is measured from where the patch stood when it was picked rather than from any angle
the placements happen to carry - each already has its own random yaw.

They are not brush settings, which is why they are not bound to one: the Rotation ranges and Base Rotation decide
what the next stroke paints, while a turn edits scatter that is already painted. They are greyed out on the stroke
just painted for that reason; that stroke is tuned through the brush settings, so later strokes follow.

## Instanced output

With **Instance Strokes** on, the default, painted grass is stored as one `PointInstancer` per patch, which the game
draws from one copy of each asset instead of one replacement per blade. The stroke just painted stays a set of
prims, so it can be tuned live and undone in one step; painting the next stroke, or flooding, merges everything else
into the instancers, and undoing that stroke gives the earlier one back as prims. The OUTPUT section counts how many
placements are instanced and how many are still loose.

- **Instance All** merges every loose placement of the edit layer now: the last stroke of a session, scatter painted
  before instancing existed, or a patch that was split
- **Split Selection** turns the selected instanced scatter, or every instancer on a selected mesh, back into
  placements and selects them, so the brush settings and the gizmo can edit them; the next stroke merges them again

Selecting instanced scatter makes the live-edit readout say so instead of offering to edit it. The toolkit viewport
shows instanced scatter through a stand-in kept only in the session layer, because HdRemix does not draw a
`PointInstancer`'s instances; the viewport therefore looks and costs the same as before, and the saving is in the
game.

## Painting on the ground

The brush paints on the selected mesh by default: select the ground, turn the brush on and paint. The viewport casts
its ray at the selected mesh itself, so foliage cards, props and earlier scatter standing in front of the ground are
looked through, and with nothing selected a stroke is refused with a status message. Set **Apply To** to All to paint
whatever is under the cursor instead.

Every asset stands straight up by default; **Conform to Surface** tilts it to follow the slope instead. Which way is
up for an asset is the **Asset Up Axis** at the top of PLACEMENT: each palette asset gets a row saying the axis its
file authors (`File: Z-up`, or `File: not set` for a file that says nothing) beside the Y-up/Z-up choice the brush
stands it up by. A choice that disagrees with the file scatters the asset lying on its side, so that row is flagged and
the status line says what to set it to. The row and the asset list's Up column edit the same palette field.

**Reset to Defaults**, beside Flood, puts every brush setting back to its default after a confirmation; settings
changes are not undo steps. The asset list, the saved presets, the name of the current preset and the Edit Selection
Live toggle are kept, so a reset never starts rewriting the selected scatter. Right after a stroke, that stroke follows
the reset the way it follows a preset being loaded, and one undo still removes it. Settings are persisted between
sessions and in presets; ones saved before painting on the selection became the default switch to Apply To Selected
and Conform off once when they are read back, and keep everything else.
