# lightspeed.trex.viewports.shared.widget

Provides the shared RTX Remix viewport UI, including its viewport layers, tools, statistics, and adjacent Properties
pane.

If the active viewport camera is the capture game camera, camera-mutating gestures and frame/focus actions first copy
that view to `/OmniverseKit_Persp` and switch the viewport to perspective. Plain viewport focus keeps looking through
the capture game camera, which remains read-only source data. If the copy to perspective cannot be completed, the
camera-mutating action is canceled and a warning explains that the capture game camera was kept read-only.

## Responsibilities

- Build and manage shared RTX Remix viewport instances.
- Consume context-bound frame requests and route them to the matching viewport instance.
- Host the viewport/Properties pane splitter and keep the Properties pane at least 240 pixels wide.
- Coordinate viewport layers, tools, statistics, activation, and renderer startup behavior.
- Host the viewport toolbar tools: Teleport and the Scatter Brush (toolbar toggle, `Ctrl+B`, `B`+wheel radius
  sizing, paint/erase strokes with a surface-following cursor, and a live counter of the prims the stroke is adding
  to the rendered stage). The tool resumes a stroke that fell behind its frame budget on every frame, so a stroke
  heavier than a frame trails the cursor and catches up, and the counter reads "(catching up)" while it does. When a
  stroke reaches its prim limit the counter reads "(limit reached)" and the cursor turns grey, and every stroke ends
  with a status line saying what it painted or erased, or why it painted less than the path that was drawn.
- Let viewport tools claim the mouse wheel before the camera zooms through the mouse-wheel interceptor hook on
  `ViewportEventDelegate`.
- Retry supported viewport initialization once and report terminal startup failures through an Exit-only dialog.
- Protect capture game-camera source data while still allowing mutating camera workflows through perspective copies.

## Non-Responsibilities

- Defining the property editors displayed inside the Properties pane.
- Owning USD property models or property-widget delegates.
- Scatter placement math, USD authoring, undo commands, settings, and presets; `lightspeed.trex.scatter.core`
  owns those, and `lightspeed.trex.scatter.widget` owns the Scatter window.

## Architecture

- `SetupUI` builds the viewport, Properties pane, and splitter and coordinates their lifecycle.
- `TrexViewportSharedExtension` routes frame requests by USD context before delegating camera-safe framing to `SetupUI`.
- `ViewportLayers` owns the registered viewport-layer instances and their ordering.
- The `scene`, `stats`, and `tools` packages provide the viewport overlays and interactions hosted by `SetupUI`.
- `ViewportSceneLayer` owns the single `sc.SceneView` and only instantiates the scene factories listed in
  `INCLUDED_SCENE_LAYERS`; the scatter brush factory id comes from `lightspeed.trex.scatter.core.constants`.
- `tools/teleport.py` and `tools/scatter_brush.py` each provide a toolbar `WidgetGroup` singleton plus a
  per-viewport scene tool created through a registered scene factory.
- `events.ViewportEventDelegate` routes wheel, key, and drop events of a scene view;
  `events.register_mouse_wheel_interceptor` lets a tool consume wheel events before flight-speed and zoom
  handling.

## Scatter brush picking

The brush asks the renderer what is under the cursor, which on a captured ground is as often a foliage card, a prop or
earlier scatter as the ground itself. While the brush's Apply To is Selected, its default, `tools/scatter_brush.py`
casts the camera ray at the selected capture meshes on the CPU instead (`selected_capture_meshes` and
`MeshSurfaceCache.raycast` from the scatter core), cancels any renderer pick still in flight so it cannot move the
cursor back onto what stands in front, and falls back to the renderer pick only to show a grey cursor off the
selection. A stroke started with nothing selected is refused with a status message. The cursor's normal is turned
toward the camera, and every stroke is handed the camera position so the core paints the side of the surface the user
is looking at, whichever way the capture wound its triangles.

## Scatter brush stroke feedback

The core stops a stroke at five times the brush's Prim Budget in rendered prims, skips a mesh with more instances
than the stroke has room left for, and stops catching up a second after the mouse is released. Each leaves part of
the drawn path unpainted, which would read as a fault of the brush if nothing said so, so `tools/scatter_brush.py`
reports them: the status line of the Scatter window says what the stroke did paint, then names the reason and the
setting that changes it, and a notification says it the first time for each reason while the brush stays on. The
cursor is grey and the counter says why wherever a stamp would place nothing. A stroke that ends normally posts what
it painted (placements and the prims they render) or erased, which also takes an earlier stroke's message off the
status line, and one stopped by an error says so and keeps what it painted. The summary goes through
`ScatterBrushController.post_stroke_summary`, which keeps an up-axis warning the stroke started with in front of it. The cursor follows Shift as it is pressed
and released with the mouse at rest, so it shows red before an erase stroke starts rather than after the first move.

## Scatter move gizmo

`tools/scatter_gizmo.py` puts the standard `omni.kit.manipulator.transform` gizmo — the translate arrows and rotate
rings — on the middle of the painted scatter the user has selected, and turns a drag of it into one undoable move of
that whole patch through `lightspeed.trex.scatter.core.move_scope`.

It resolves its own scope with `resolve_scope`, the same rules the Scatter window's live editing uses, so it works
whether or not that window is open. The gizmo is model-driven rather than selection-driven, which is why it needs
nothing from `filter_transformable_prims`: that filter only passes paths with an `inst_` or `light_` segment, and
scatter placements are authored and worked with on the prototype side.

The gizmo hides itself while the brush is in paint or erase mode, because a handle sitting on the patch would take
the drag the brush is waiting for. It follows `/app/transform/operation`, so the viewport toolbar and the keys that
switch the prim gizmo between move and rotate switch this one too, and it offers no scale: resizing a patch is what
the Size range in the Scatter window does.

Every step of a drag is measured from the state captured when the gesture began, rather than from the step before
it, so a long drag cannot accumulate the error of every frame along the way.

The brush's Conform to Surface setting is passed into every move, because it decides whether a slide may turn the
patch onto the ground it lands on.

The rotate rings aim the patch rather than carrying it around: the gesture sends its rotation on its own, with no
pivot, and `move_scope` turns every placement where it stands. The handle still sits on the middle of the patch,
because that is what the rings are drawn around, but nothing about that point moves a placement. Sandwiching the
rotation between translations to and from the handle — a rigid-body turn — swung the far side of a patch up into the
air on any ring but the one about the surface normal, and re-planting it afterwards dropped it elsewhere on the
mesh.
