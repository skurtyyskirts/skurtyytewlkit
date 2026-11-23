# lightspeed.trex.control.stagecraft

Controls the StageCraft USD context, workspace lifecycle, and StageCraft-owned navigation.

## Responsibilities

- Create and manage the StageCraft USD stage and its application event subscriptions.
- Validate project-open requests, present repair guidance, and transition valid or repaired projects to the workspace.
- Prefill the existing Project Wizard capture picker for project-open requests that require capture selection.
- Register the always-enabled Ingestion sidebar action separately from project-dependent StageCraft navigation.
- Open IngestCraft after on-demand activation completes, coalescing repeated requests and reporting activation timeouts.
- List extra light rig directories and the mod skies of the open StageCraft project in the viewport Lighting menu.

## Non-Responsibilities

- Does not create the IngestCraft stage or UI; the IngestCraft control and widget extensions own them.
- Does not implement extension activation; it calls the loader in `lightspeed.trex.utils.widget`.
- Does not own Home card visibility, validity, or missing-path guards; `lightspeed.trex.home.widget` owns them.
- Does not set the Lighting menu's own rig directory; the viewport bundles set it, and both use the lookdev HDRI rigs.
- Does not write mod skies into the mod: the game's skydome proxy creates the in-game dome from the same manifest.

## Architecture

`TrexStageCraftControlExtension` owns a `Setup` instance. `Setup` manages the StageCraft stage, event subscriptions,
and sidebar actions. The Modding and Ingestion actions use separate subscriptions so project state only controls
Modding. Ingestion awaits the shared IngestCraft loader before applying the IngestCraft layout. Project-open events are
fully validated once here before the unsaved-project prompt; the accepted callback opens the captured path without
revalidating it. Capture-selection requests open the prefilled Project Wizard instead, allowing the wizard core to save
the selected capture before StageCraft opens the project. `Setup` retains its deferred stage, layout, and IngestCraft tasks until they complete and cancels any
remaining work during teardown.

### Light rigs and mod skies

The extension also owns a `LightRigMenu`. It wraps the Lighting menu's rig listing in both modules that import it,
so the menu lists its own rig directory, then the `lightRigs.extraDirectories` setting's rigs (Kit's studio rigs),
then one "Mod Sky" rig per sky of the StageCraft project. `mod_skies.py` reads each `lightRigs.modSkyDirectories`
folder (default `assets/skydomes`, relative to the project file): the levels of its `skydomes.ini`, read with the
skydome proxy's defaults and limits, then every other `.dds` in the folder. It writes one rig per sky under
`${temp}/lightspeed.trex.control.stagecraft/mod_sky_rigs`. Each rig holds a lat-long `DomeLight` with intensity
`Radiance`, color `Tint` and `rotateZ` `Rotation`, the light-to-world transform the proxy passes to the Remix API, and
a `DistantLight` for the dome's own sun. Every rig is written in the stage's up axis, so the Lighting menu adds no
up-axis correction. Rigs are rewritten when the folder's files or the stage up axis change, and reloaded in place.

Applying any rig hides every stage light in the session layer. When the StageCraft stage's lighting mode becomes a
Mod Sky rig, `LightRigMenu` shows its non-dome stage lights again, because the game keeps the level's lights under its
dome. The capture's `DomeLight` stays hidden, because the runtime draws only one dome per frame. The Lighting menu
rebuilds its items only when its `rigs` setting changes, so a stage open that changes the mod skies switches that
setting between two spellings of the same directory.
