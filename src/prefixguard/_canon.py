"""Canonical serialization — same logical content must produce the same bytes."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["canon"]


def canon(obj: Any) -> str:
    """Deterministic string form of a prompt component.

    Dict key order and unicode escaping never count as divergence: providers
    see the serialized request, but *your* serializer should be deterministic —
    we compare the logical content, sorted and normalized, so only real edits
    show up as byte drift.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(
            obj, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), default=str,
        )
    except (TypeError, ValueError):
        return str(obj)
