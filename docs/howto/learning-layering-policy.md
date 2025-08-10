### Layering Policy: Asset Replacements, Remix Categories/Flags, and Persistence

- Authoring Target
  - All Toolkit-driven authoring for asset replacements must occur on the active mod layer (`LayerType.replacement`).
  - The capture layer is read-only: never author overrides or new prims in the capture layer.

- Edit Target Behavior
  - Before any authoring, the Toolkit sets the stage edit target to the replacement (mod) layer.
  - If a replacement layer is missing, users must import or create one before authoring.

- Reference Authoring
  - References appended/replaced are written relative to the current mod layer, ensuring relocatable paths.
  - When replacing references, author deletions/replacements against the mod layer to avoid mutating capture content.

- Remix Categories/Flags
  - Users can optionally tag created prims with Remix categories (e.g., `remix_category:world_ui`).
  - Tags are authored as USD attributes on the same prim(s) created by the reference operation, on the mod layer.
  - Category names are validated by prefix (`remix_category:`) and use `Sdf.ValueTypeNames.Bool` values.

- Persistence and Reload
  - After authoring operations, the mod layer is saved. Reloading the stage preserves authored references and Remix tags.
  - Save/reload cycles must reproduce the same composed results as long as the mod layer remains in the layer stack.

- Rationale
  - Keeping all authored data in the mod layer preserves capture fidelity and ensures predictable layering semantics.
  - Authoring categories/flags alongside replacements guarantees consistent runtime behavior (RTX Remix semantics).