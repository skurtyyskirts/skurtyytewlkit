# lightspeed.trex.hotkeys

Defines the application-wide hotkeys of the RTX Remix Toolkit and lets extensions subscribe to them as events,
optionally filtered by Trex context.

## Responsibilities

- Declare the supported key combinations as `TrexHotkeyEvent` members and register each one once with
  `omni.kit.hotkeys.core` (through an `omni.kit.actions.core` action) when the extension starts
- Own the process-wide `HotkeyManager` (`get_global_hotkey_manager`) that fans a hotkey press out to every subscriber
- Route a press to the subscribers registered for the current Trex context first and to context-free subscribers
  otherwise, and let each subscription add its own `enable_fn` filter (for example "only in the active viewport")

## Non-Responsibilities

- Does not implement any hotkey action: undo, save, frame, teleport, scatter brush toggling and every other effect
  live in the extension that subscribes (for example `lightspeed.trex.control.stagecraft` handles `CTRL_Z`, and
  `lightspeed.trex.viewports.shared.widget` handles `CTRL_T` and `CTRL_B`)
- Does not define Trex contexts; it reads the current one from `lightspeed.trex.contexts`
- Does not offer per-extension or user-remappable hotkeys; extensions that need those register them directly with
  `omni.kit.hotkeys.core`

## Architecture

`TrexHotkeysExtension.on_startup` creates the global `HotkeyManager` and calls `register_global_hotkeys`, which
defines one `AppHotkey` per `TrexHotkeyEvent`. `AppHotkey` wraps an `omni.kit.hotkeys.core.Hotkey` together with its
`omni.kit.actions` action and registers or deregisters both whenever `omni.kit.hotkeys.core` is enabled or disabled,
so the hotkeys survive that extension being reloaded.

One `Hotkey` per key combination is registered globally rather than one per consumer: `omni.kit.hotkeys.core` supports
a single hotkey context, so context filtering is done here at subscription time instead. When a key combination fires,
the manager looks up the `Event` registered for `(hotkey_event, current context)`, falls back to
`(hotkey_event, None)`, and calls the first non-empty one. A subscription is an `EventSubscription` that unsubscribes
when it is garbage-collected, so subscribers keep it alive for as long as they want the hotkey.

Supported `TrexHotkeyEvent` members: `CTRL_Z` (undo), `CTRL_Y` (redo), `CTRL_S` (save), `CTRL_SHIFT_S` (save as),
`F` (frame selection), `ESC` (clear selection), `CTRL_T` (teleport selection) and `CTRL_B` (toggle the scatter
brush).

### Key Classes

- `TrexHotkeyEvent` — enum of the supported key combinations (`omni.kit.hotkeys.core.KeyCombination` values)
- `HotkeyManager` — defines the hotkey events and holds the per-context `Event` objects subscribers attach to
- `AppHotkey` — registration helper binding one key combination to one `omni.kit.actions` action
- `TrexHotkeysExtension` — creates and destroys the global manager

## Usage

```python
from lightspeed.trex.hotkeys import TrexHotkeyEvent, get_global_hotkey_manager

self._subscription = get_global_hotkey_manager().subscribe_hotkey_event(
    TrexHotkeyEvent.CTRL_B, self._toggle_brush, enable_fn=self._is_viewport_active
)
```
