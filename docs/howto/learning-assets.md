# Setting Up Asset Replacements

Replacing the original game's visuals with modern, higher-quality assets is a key part of using RTX Remix to remaster a
game. Older games often have simpler, lower-detail models. RTX Remix allows modders to upgrade these visuals by
swapping out the original assets with more detailed and modern ones. This process is central to creating a visually
enhanced experience.

```{warning}
**All assets must be ingested into the project directory before they can be used in a mod.**

Refer to the [Ingesting Assets](learning-ingestion.md) section for information on asset ingestion.
```

## Ensuring Hash Stability

Asset replacement requires a stable hash for the asset. This stability ensures the RTX Remix Runtime can accurately
identify the asset and its corresponding replacements.

A hash, a unique identifier derived from the asset's data, serves as a reference for the RTX Remix Runtime to recognize
and manage the asset during gameplay.

Older games frequently exhibit unstable hashes in world geometry in part due to culling mechanisms.

To verify hash stability:

1. **In-Game Debugging:** Within the game, press "Alt+X", navigate to the "Debug" tab under "Rendering", and enable
   "Debug View".
2. **Geometry Hash Verification:** Switch to "Geometry Hash" in the debug view. If the game world displays color
   variations, it indicates hash instability, potentially necessitating workarounds or preventing replacement.

If the hashes appear stable, proceed with asset replacement. If signs of instability are present, consider using
anchor assets as an alternative way to replace assets.

### Using Anchor Assets

In scenarios involving unstable hashes, direct asset replacement may encounter difficulties. To address this, users can
employ "Anchor Assets" as stable, non-culled stand-in assets within the game level. These anchor assets serve as
reliable reference points for the placement of replacement assets, ensuring accurate positioning even when the original
game geometry exhibits hash instability.

The recommended workflow is as follows:

1. **Identify a Suitable Anchor Asset:** Select an asset within the game that is known to have a stable hash and is not
   subject to culling. Ideally, this asset should be a unique prop or element that can be easily identified and
   manipulated.
2. **Remove the Original Asset:** Remove the original asset that exhibits unstable hash behavior, as detailed in the
   [Managing Asset References](#managing-asset-references) section.
3. **Append the Anchor Asset:** Add a reference to the selected anchor asset into the scene, also described in the
   [Managing Asset References](#managing-asset-references) section.
4. **Transform the Anchor Asset:** Position, rotate, and scale the appended anchor asset to the precise location and
   orientation of the original asset that was removed, as detailed in the
   [Adjusting Replaced Assets](#adjusting-replaced-assets) section.
   This step ensures that the replacement asset will occupy the correct space within the game world.

By utilizing anchor assets and following this workflow, users can effectively overcome challenges posed by unstable
hashes and achieve accurate asset placement within their RTX Remix projects.

## Managing Asset References

The RTX Remix Toolkit supports asset removal, replacement, appending, and duplication. The process remains consistent
across these operations, with usage determined by the desired outcome.

```{warning}
Asset removal, addition, or movement in the RTX Remix Toolkit only alters rendering. Game-side events, such as
collisions, remain based on the original asset.
```

1. **Asset Removal:** Remove an asset from the scene without replacement by clicking the "Delete" button in the
   Selection Panel.

   ![Delete Assets](../data/images/remix-assets-delete.png)

2. **Asset Replacement:** Select an asset in the scene and choose a replacement asset. Modify the reference in the
   Object Properties panel by clicking the "Browse" button. **This is the most common asset modification.**

   ![Replace Assets](../data/images/remix-assets-replace.png)

3. **Asset Appending:** Select an asset in the scene and append a new asset. Select the reference item in the Selection
   Panel, click "Add new reference...", and choose the desired asset.

   ![Append Assets](../data/images/remix-assets-append.png)

4. **Asset Duplication:** Select an asset in the scene and create a duplicate instance. Select the reference item in the
   Selection Panel and click the "Duplicate" button.

   ![Duplicate Assets](../data/images/remix-assets-duplicate.png)

5. **Reset Asset Reference:** To revert the asset reference to its original state, click the **Restore all Properties** <img src="../../source/extensions/lightspeed.trex.app.resources/data/icons/restore.svg" class="svg-icon" style="height: 1em; vertical-align: text-bottom;"> icon located on the topmost item within the [Selection Panel](learning-toolkit.md#selection-panel).

   ```{warning}
   The "Restore all properties" function will reset all properties of the selected asset to their original values. This
   includes the object's transform, any added references, and other modifications.
   ```

## Adjusting Replaced Assets

After replacement, appending, or duplication, adjust asset position, rotation, and scale using the "Transform"
properties in the "Object Properties" panel.

![Adjust Position](../data/images/remix-assets-transforms.png)

```{tip}
Apply transforms to the "Xforms" prim, available on all ingested assets. Captured assets should have transforms applied
to the "mesh" prim.
```

## Scattering Assets

The scatter brush paints copies of ingested assets onto captured meshes directly in the viewport, the way a foliage
or debris painter works in other DCC tools. Drag across a mesh and it stamps grass, rocks, glass shards, or any other
ingested prop with randomized rotation and scale, instead of placing one reference at a time through the Selection
Panel.

### Enabling the Brush

Open the brush from any of these:

* The **Scatter** tab, docked next to the Editor tab by default.
* The scatter brush button in the viewport toolbar, directly under the Teleport button.
* `Ctrl+B`, which toggles paint mode from anywhere in the viewport.

```{tip}
The toolbar button, `Ctrl+B`, and the Paint button in the Scatter window all flip the same switch, so any of them
turns painting on or off.
```

### Building the Asset Palette

The **ASSETS** section of the Scatter window holds the palette the brush chooses from:

* **Add** opens the asset file picker to choose a USD model.
* **Add Selection** adds the asset referenced by the currently selected prim.
* The browse dropdown lists every model already ingested into the project.

Only ingested, in-project assets can join the palette. Adding anything else brings up the usual ingestion dialog
first; refer to [Ingesting Assets](learning-ingestion.md). Each row in the palette can be enabled or disabled, given a
relative selection **Weight**, and assigned the **Up Axis** it was authored with (Y or Z), which the brush uses to
keep the asset upright on the surface it paints onto.

### Painting

With at least one enabled asset in the palette and paint mode on:

* **Drag** across a captured mesh to stamp copies of the palette assets; a new stamp is placed every time the cursor
  travels the **Stamp Spacing** distance.
* **Shift+drag** erases placements under the brush instead of adding them.
* Hold **B** and scroll the mouse wheel to resize the brush without opening the panel.
* `Ctrl+Z` undoes a whole stroke at once, not stamp by stamp.

The brush cursor turns grey over meshes the current settings will not paint on, for example when **Apply To** is
restricted to the selection and the cursor is over something else.

### Brush Settings

The Scatter window groups the brush parameters into collapsible sections:

* **BRUSH** — the shape of one stamp: **Radius** (brush size in stage units), **Falloff** (the acceptance curve from
  the center to the edge: Constant, Linear, Smooth, Sphere, or Gaussian), **Density** (candidates drawn per stamp),
  **Strength** (the share of those candidates that is kept), **Stamp Spacing** (distance between stamps along a
  stroke), and **Padding** (minimum distance kept between placements).
* **PLACEMENT** — how each placement sits on the surface: **Vertical Offset** along the surface normal, **Conform to
  Surface** (aligns the asset's up axis with the surface normal), **Align to Stroke** (turns the asset to face the
  stroke direction), and random rotation ranges (yaw at the top level, with tilt around the other two axes under an
  **ADVANCED ROTATION** sub-section).
* **SCALE** — random scale per placement, with an enable toggle, a uniform **Min**/**Max** range, **Bias** (skews the
  distribution toward the minimum at -1 or the maximum at 1), **Weight** (sharpens the distribution above 1 or
  flattens it below 1), and, once **Uniform** is turned off, a separate Min/Max range per axis.
* **RANDOM** — the stroke's random **Seed**, with **Reroll** drawing a new one on demand, and **Randomize**, which
  draws a fresh seed for every stroke instead of reusing the fixed one.
* **TARGET** — which captured meshes receive placements: **Apply To** (All or Selected), **Target Mode** (Hit
  Surface authors under whatever mesh is under the cursor, Anchor always authors under one pinned prototype, set with
  the **Use Selection** button), **Erase Scope** (All Scattered or Brush Assets, controlling what Shift+drag or Erase
  mode removes), and **Flood Cap**, the placement limit for the **Flood** button, which fills the anchor or the
  selected captured meshes with placements in a single undoable step; requesting more than the default cap asks for
  confirmation first.

### Presets

The **PRESETS** section stores named brush configurations as files in the user's documents folder. **Save** writes
the current settings to the active preset, **Save As** and **Clone** prompt for a new name, **Rename** renames the
stored preset, and **Delete** removes the preset file after confirmation.

```{warning}
Placements are authored under the captured mesh's **prototype**, not the instance painted on, so they appear on
every instance that shares that mesh's hash. The Scatter window warns when the current target has more than one
instance. Use **Target Mode: Anchor** together with **Use Selection** to pin one specific mesh instead.
```

```{warning}
Like every other asset replacement, scattering only changes what is rendered; it does not add or move game-side
collision. The captured mesh itself is never modified, so it keeps rendering underneath the scattered assets.
```

Each placement is authored as its own reference prim (`s_<id>`), grouped under a `scatter_<preset>` container prim
beneath the target's `mesh_<HASH>` prototype in the mod. Placements and their container both carry the
`IsRemixScatter` attribute, which is how the brush tells its own placements apart from references added by hand,
for example when erasing. Placements are ordinary reference prims otherwise, so the Selection Panel and packaging
treat them like any other asset replacement.

## Handling Animated Assets

Animated asset replacement varies depending on the game's skeleton animation type.

### Skeleton Animation Types

**GPU-Based Skeleton Animation:** Replace existing 3D assets with new assets sharing the same skeleton. New assets adopt
animations from the original asset's bone transformations.

**Non-GPU-Based Skeleton Animation:** Replacement occurs on the engine side. Re-capture animations post-replacement,
then assign PBR textures in Remix. See the [Setting Up Material Replacements](learning-materials.md) section for more
information on texture replacements.

### Skeletons in Remix

Skinned replacements require expertise.

**Skeleton Data in USD Capture:** Replace 3D assets with assets using the same skeleton.

Considerations:

1. **Bone Indices and Weights:** The runtime reads bone indices and weights per vertex from replacement assets.
2. **Skeleton Changes:** Modeling tools may alter skeletons during import/export, disrupting mapping. Remapping to
   original vertices is supported, but requires manual specification.
3. **Limited Skeleton Information:** The GPU receives information from the bind pose to the current pose, complicating
   bind pose or hierarchy reconstruction.
4. **Differing Joint Counts:** Game skeletons often have fewer joints than replacements. Joint remapping is required.

### Remapping Skeletons

The RTX Remix Toolkit attempts automatic remapping upon adding a replacement skinned mesh with a detected USD skeleton.
Automatic remapping occurs when joints are named identically or are in the same order.

Alternatively, add a `skel:remix_joints` attribute to the bound mesh to specify joint mapping:
`uniform token[] skel:remix_joints = ["root", "root/joint1", "root/joint2", "root/joint3" ...]`.

### Remapping Skeleton Tool

The RTX Remix Toolkit provides a tool for manual joint remapping.

1. Open the [Stage Manager](../toolkitinterface/remix-toolkitinterface-layouttab.md#stage-manager) and navigate to the "
   Skeletons" tab.

   ![Skeleton Remapping](../data/images/remix-skeleton-interaction-tab.png)

2. Locate the bound replacement mesh.

3. Click "Remap Joint Indices" to open the remapping tool.

   ![Skeleton Remapping](../data/images/remix-skeleton-remapper.png)

4. Select a captured skeleton joint to drive each replacement asset joint. Use "Auto Remap Joints" for name/order-based
   mapping, "Reset" to start from scratch, and "Clear" to undo changes.

5. Click "Apply" to re-author joint influences on the replacement mesh, matching the captured joint index.

***
<sub> Need to leave feedback about the RTX Remix Documentation?  [Click here](https://github.com/NVIDIAGameWorks/rtx-remix/issues/new?assignees=nvdamien&labels=documentation%2Cfeedback%2Ctriage&projects=&template=documentation_feedback.yml&title=%5BDocumentation+feedback%5D%3A+) </sub>
