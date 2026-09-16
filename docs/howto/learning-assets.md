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

### Textures

Painted assets come out with their textures already assigned, so there is no second pass through the material
properties panel. With **Assign Textures** on, the brush looks for the asset's ingested texture set and fills in the
maps its material leaves empty — albedo, normal (with the right encoding), roughness, height, metallic and emissive —
using the same detection as the **Assign Texture Sets** button. Textures the asset already carries are never
replaced, and neither is one you assigned by hand.

The **Maps** column shows what will happen before you paint:

| Cell | Meaning |
|---|---|
| `asset` | The asset supplies its own textures; the brush adds nothing. |
| `A N R` | The brush will assign these maps: **A**lbedo, **N**ormal, **R**oughness, **H**eight, **M**etallic, **E**missive. Maps the set does not contain are named in the tooltip, not counted here — a texture set without a normal map is ordinary, not a fault. |
| `none` | Nothing to assign, or no matching texture set was found. The tooltip says which. |

A texture is matched when its file name contains the material's or the asset's own name followed by the name of a
texture type. The name does not have to come first, so an asset pack's prefix is no obstacle:
`Quaternius_Leaves_diff.a.rtex.dds` is claimed by a material called `Leaves`. What has to follow is the texture type,
which is why `grass_medium_02_diff_4k.a.rtex.dds` is left for `grass_medium_02` and never taken by a material merely
called `grass` — the word after `grass` there is `medium`, not a texture type.

That rule deliberately errs toward assigning nothing rather than assigning another asset's textures, so a material
whose name carries extra words (`Leaves_NormalTree`) may find nothing even when a sensible human guess exists. When
detection finds nothing, or picks the wrong set, use the button in the **Maps** cell to choose one texture of the
right set yourself; the rest are matched to it by name. The button then reads **auto**, which puts the row back on
automatic detection.

```{tip}
An asset whose material has no textures at all, and no texture set on disk to find, needs its textures ingested
before the brush can assign them. Name them after the asset's material so they are picked up automatically.
```

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
* **PLACEMENT** — how each placement sits on the surface: **Edit Selection Live** (see
  [Tuning What You Already Painted](#tuning-what-you-already-painted)), **Vertical Offset** along the surface normal,
  **Conform to Surface** (aligns the asset's up axis with the surface normal), **Align to Stroke** (turns the asset to
  face the stroke direction), and random rotation ranges (yaw at the top level, with tilt around the other two axes
  under an **ADVANCED ROTATION** sub-section).
* **SCALE** — how large each placement is. **Scale Mode** decides what the range means. **Absolute Size** (the
  default) reads the range as a height in stage units, measured along the asset's up axis, so setting grass to 30
  gives grass 30 units tall no matter how large the asset file happens to be; the brush measures each asset when it
  joins the palette, and its tooltip reports that measurement. **Multiplier** reads the range as a factor of the
  asset's own size, and adds the **Uniform** toggle with a separate Min/Max range per axis once it is turned off.
  **Random Scale** varies the size within the range; with it off, every placement uses the minimum. **Bias** skews
  the distribution toward the minimum at -1 or the maximum at 1, and **Weight** sharpens it above 1 or flattens it
  below 1. Dragging a **Min** above its **Max**, or a **Max** below its **Min**, carries the other end along, so
  either end can be moved anywhere without first making room for it. The same applies to the rotation ranges.

  Sizes are world-space: a placement comes out the same size whether the captured mesh under it was scaled by the
  game or not, and **Vertical Offset** is likewise a distance in stage units. An asset the brush cannot open is
  scattered at its own size and says so in its palette tooltip.

  Both the height and the upright orientation depend on the **Up** column in the ASSETS list, which is the axis
  the *asset file* was authored along — not the up axis of the scene. It is read from the file when the asset is
  added; setting it to match the stage instead tells the brush no correction is needed, and the asset is scattered
  lying on its side. The brush reports a disagreement between the column and the file when a stroke starts.
* **RANDOM** — the stroke's random **Seed**, with **Reroll** drawing a new one on demand, and **Randomize**, which
  draws a fresh seed for every stroke instead of reusing the fixed one.
* **TARGET** — which captured meshes receive placements: **Apply To** (All or Selected), **Target Mode** (Hit
  Surface authors under whatever mesh is under the cursor, Anchor always authors under one pinned prototype, set with
  the **Use Selection** button), **Erase Scope** (All Scattered or Brush Assets, controlling what Shift+drag or Erase
  mode removes), and **Flood Cap**, the placement limit for the **Flood** button, which fills the anchor or the
  selected captured meshes with placements in a single undoable step; requesting more than the default cap asks for
  confirmation first. **Prim Budget** sets when the brush's live stroke counter reports a stroke as expensive.

### Tuning What You Already Painted

A stroke bakes the settings it was painted with into each placement, so changing a setting afterwards would normally
only affect your next stroke. With **Edit Selection Live** on — it is on by default — a settings change also updates
the scatter you have selected, so you can paint a patch of grass first and find its size, rotation and offset
afterwards, watching the viewport while you drag the slider.

Select what you want to change, then drag. What you selected decides what moves:

| What you select | What a settings change updates |
|---|---|
| Painted placements | Exactly those placements. |
| A `scatter_<preset>` container | Every placement in that container. |
| A mesh or an instance | What the preset currently loaded in the window painted on it. |
| Nothing | Nothing already painted; the setting applies to your next stroke only. |

The readout beside the toggle always names what is in scope before you touch anything, for example
`428 placements - mesh_A64083D9 / grass`.

Because a plain mesh selection resolves through the preset name, two patches on the same mesh stay independent as
long as you painted them under different presets. Paint glass under a `Glass` preset and grass under a `Grass`
preset, and selecting that mesh with `Grass` loaded moves only the grass. If both were painted under the same
preset, select one patch's container or its placements directly instead.

A whole slider drag is a single undo step, so one `Ctrl+Z` puts the patch back the way it was.

```{note}
Only the **PLACEMENT**, **SCALE** and **RANDOM** settings re-apply. **Radius**, **Density**, **Strength**, **Stamp
Spacing** and **Padding** decide *where* stamps land, which depends on the path your mouse followed, and that path is
not recorded — those always apply to your next stroke only. Changing **Seed**, or pressing **Reroll**, draws a fresh
random arrangement for the placements you have selected, which is the quickest way to try a different look without
repainting.
```

Two kinds of placement are deliberately left alone, and the readout says how many:

* **Placements you moved by hand.** A placement remembers the transform the brush gave it, so one you have since
  nudged, turned or rescaled yourself keeps your change instead of being overwritten.
* **Placements painted before this feature existed.** They never recorded the surface they were placed on, so the
  brush projects them back onto their mesh to recover it. They follow every setting, but their original random
  rotation and scale cannot be recovered and are drawn afresh, so expect that patch to re-roll the first time you
  edit it.

Placements authored in a layer other than your current edit target are also skipped, the same way erasing skips them;
set that layer as the edit target to change them.

### Moving a Painted Patch

Selecting painted scatter also puts the usual transform gizmo — the red, green and blue arrows, and the rotate rings
— on the middle of that patch. Drag the arrows to slide the whole patch, or the rings to turn it. Both are a single
undo step, however far you drag.

The patch **stays on the surface**. Slide it across uneven ground and every placement re-settles onto whatever is
beneath it, tilting to match the slope, so grass never ends up floating above a hill or buried in it. Drag it
upwards and it simply settles back down; drag it past the edge of the mesh and it stops at the edge. It only ever
looks for the mesh it was painted on, so a patch can never jump onto some other piece of geometry.

The same rules decide what moves as everywhere else: select placements or a `scatter_<preset>` container to move
exactly those, or select a mesh or instance to move what the current preset painted on it. Nothing outside the
selection moves.

```{note}
The gizmo follows the viewport's own Move and Rotate tools, so the keys and toolbar buttons you already use to switch
between them switch this too. It is hidden while Paint or Erase is armed, because otherwise it would swallow the
brush stroke — turn painting off to move a patch. There is no scale handle: **SCALE ▸ Size** resizes a selected patch
live, in stage units, which is more precise than dragging a corner.
```

A move never re-derives the patch. It moves what is there, so dragging can never resize or re-randomize it even if
you have changed the brush settings since painting — and a placement you nudged by hand travels along with the rest
and stays yours, so a later settings change still leaves it alone.

### Keeping a scattered scene fast in game

A placement is authored under a captured mesh's prototype, so it is drawn once for **every instance of that
prototype's hash**. Painting 200 rocks onto a floor tile that appears 50 times in the level puts 10,000 rocks in
front of the player. That multiplier, not the number of strokes, is what decides the cost of a scattered scene.

While a paint stroke is running, the brush prints a running total next to the cursor. It reads `1,200 prims` on a
prototype with a single instance, and `1,200 placed  ->  60,000 prims` on one that replicates, so the multiplier is
visible while there is still time to undo. The text turns amber once the stroke passes **Prim Budget** and red at
five times it. It only reports: no stroke is ever cut short, and the counter is not shown while erasing.

Three settings decide how much a stroke can author:

* **Padding** is the real limit. It is the minimum distance between placements and it holds for the whole stroke, so
  it caps the total no matter how high **Density** goes. At the default of 10, a 1000-unit drag with a 50-unit radius
  cannot exceed about 1,250 placements. **Setting Padding to 0 removes the limit entirely** and lets one long drag
  author an unbounded number of placements.
* **Density** and **Stamp Spacing** decide how quickly the stroke approaches that cap. The defaults (density 8,
  spacing 25, Smooth falloff) author about 96 placements over a 1000-unit drag, which stays inside a 10,000-prim
  budget until the target has more than about 100 instances.
* **Falloff** thins every stamp: Smooth and Linear keep roughly 30% of the candidates drawn, Sphere about 67%, and
  Constant all of them. Switching from Smooth to Constant roughly triples a stroke.

If a stroke needs to stay inside 10,000 rendered prims, raise **Padding** as the instance count of the target rises:

| Instances of the target | Placements that fit the budget | Minimum Padding |
|---|---|---|
| 1 | 10,000 | 4 |
| 5 | 2,000 | 8 |
| 20 | 500 | 16 |
| 50 | 200 | 25 |
| 200 | 50 | 50 |

Prim count is not the whole cost. A scattered asset's triangle count and its material both multiply the same way, so
300 copies of a 50,000-triangle rock is 15 million triangles however modest the prim count looks. Prefer low-polygon
assets for anything scattered densely, and profile the result in game rather than trusting the counter alone.

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
