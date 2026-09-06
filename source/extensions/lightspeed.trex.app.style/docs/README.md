# lightspeed.trex.app.style

Global `omni.ui` style of the RTX Remix Toolkit: the shared palette and the named widget styles every Trex extension
draws with.

## Responsibilities

- Fill the default `ui.Style` dictionary (`style.default`) at import time with the Toolkit palette and the named
  styles of its widgets, including the viewport toolbar button images such as `Button.Image::teleport` and
  `Button.Image::scatter_brush` with their `:hovered` and `:checked` states
- Restyle the Kit viewport menubar and toolbar shades (`update_viewport_menu_style`) and the popup message dialog
  (`override_dialog_get_style`) so stock Kit widgets match the Toolkit
- Apply the ImGui-level overrides the `omni.ui` style cannot express (window shadow colour)

## Non-Responsibilities

- Does not ship icons, fonts or images; those live in `lightspeed.trex.app.resources` and are resolved through
  `omni.flux.utils.widget.resources` (`get_icons`, `get_fonts`, `get_image`)
- Does not create widgets or windows; extensions reference the style names defined here when they build their UI
- Does not own styles that a single widget uses; those stay next to the widget that draws them

## Architecture

The extension is one module, `trex_style.py`, evaluated once when the extension loads; its `[core] order = -100`
makes that happen before any Trex UI is built. The module defines the palette as `0xAABBGGRR` constants, then updates
`ui.Style.get_instance().default` in place, so every window created afterwards picks the styles up without importing
anything. Style keys follow the `omni.ui` selector syntax: `<Type>::<name>` for a named widget and `:hovered`,
`:checked` or `:disabled` suffixes for its states. A toolbar toggle therefore needs the base key plus a `:checked`
entry, which is what `Button.Image::scatter_brush` and `Button.Image::scatter_brush:checked` provide for the scatter
brush `WidgetGroup` of `lightspeed.trex.viewports.shared.widget`.

`update_viewport_menu_style` writes the `cl.viewport_menubar_*` and `cl.toolbar_button_*` shades that Kit's viewport
menubar and toolbar read; it runs at import and can be called again after the menus are created.
`override_dialog_get_style` is installed over `omni.kit.window.popup_dialog.message_dialog.get_style` (or over its
`UI_STYLES` table on Kit versions without that function) so the message dialog uses the Toolkit button colours.

### Key Members

- `style` — the `ui.Style` singleton whose `default` dictionary carries the Toolkit styles
- `update_viewport_menu_style` — viewport menubar and toolbar shade overrides
- `override_dialog_get_style` — message dialog style override

## Usage

```python
from lightspeed.trex.app.style import style

checked_style = style.default["Button.Image::scatter_brush:checked"]
```
