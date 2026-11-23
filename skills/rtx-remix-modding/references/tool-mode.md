# Driving Remix with the direct tools

Recipes for using the Remix MCP tools. Follow the prerequisites and save/readback requirements in [SKILL.md](../SKILL.md).

## `add_asset_reference`

Add an asset reference (USD/MDL) onto an existing prim.

Attach an asset reference to "<prim_path>".

1. `remix_append_prim_reference_file_path` prim_path="<prim_path>", asset_file_path="<asset_path>". (Skip prim-existence checks — the append surfaces errors.)
2. `remix_get_prim_reference_file_paths` prim_path="<prim_path>"; confirm the new reference is present.

## `replace_model_asset`

Swap the currently-selected 3D model in the viewport with an ingested asset.

Swap the selected model with an ingested asset.

1. `remix_get_available_ingested_assets` asset_type="models".
2. `remix_get_prim_paths` selection=true, prim_types=["models"].
3. Resolve the requested asset from step 1's list and the intended model from step 2. Use an exact path when provided; otherwise require an unambiguous match. If either is missing, report that and ask the user to ingest or select it. If multiple candidates remain, show them and ask the user to choose instead of taking the first entry.
4. `remix_replace_prim_reference_file_path` prim_path=<resolved model path>, asset_file_path=<resolved asset path>.

## `scatter_assets`

Scatter ingested assets over the surfaces the user describes, such as grass over every grass texture.

1. `remix_get_scatter_surfaces`. Capture textures are named by hash, so open each image in `contact_sheet_paths`; a tile's number is its surface's `number`.
2. `remix_get_available_ingested_assets` asset_type="models". If the user asked for assets the project lacks, download the source models yourself (Poly Haven and ambientCG publish CC0 ones), ingest each with `remix_ingest_model_asset`, and list again until they appear.
3. Reply with the surfaces and assets you picked and wait for the user's confirmation.
4. `remix_set_edit_target_layer` to the mod layer, then `remix_create_scatter_group` name, material_paths, assets, amount ("light", "medium" or "heavy").
5. `remix_save_layer`, then `remix_get_scatter_groups`; report `placement_count`, and when `capped` is true say a heavier amount would add more. When `prim_limited` is true the group stopped at the limit on what one scatter may add to the rendered stage, which counts each placement once per instance of its mesh, so a heavier amount adds nothing; say that instead.

## `retune_scatter`

Change a group with `remix_update_scatter_group`, sending only what changes; the layout stays unless reseed=true.

- More or less: amount, or tuning.density.
- Longer or shorter: tuning.size_min and tuning.size_max, a height in stage units; scale both from the current values.
- Closer together or further apart: lower or raise tuning.padding, and raise or lower tuning.density.
- More or fewer types: add_assets (ingested first, as above) or remove_asset_paths.

Then `remix_save_layer`. To take a group's scatter back out, `remix_remove_scatter_group` and save.
