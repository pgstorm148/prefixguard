"""Token counting: exact when tiktoken is installed, honest heuristic otherwise."""

from __future__ import annotations

__all__ = ["count", "counter_name"]

_encoder = None  # None = not probed yet, False = unavailable
_name = "heuristic:chars/4"


def _load():
    global _encoder, _name
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
            _name = "tiktoken:cl100k_base"
        except Exception:
            _encoder = False
    return _encoder


def count(text: str) -> int:
    if not text:
        return 0
    enc = _load()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def counter_name() -> str:
    _load()
    return _name
