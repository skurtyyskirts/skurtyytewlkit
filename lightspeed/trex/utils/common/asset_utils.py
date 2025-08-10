from __future__ import annotations


def is_layer_from_capture(layer_identifier: str | None) -> bool:
    ident = (layer_identifier or "").lower()
    return "capture" in ident