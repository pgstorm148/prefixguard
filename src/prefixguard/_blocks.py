"""Decompose a request into ordered, hashable blocks: system -> tools -> messages.

The block order mirrors provider serialization order, which is the order the
cache prefix is matched in. Divergence at block N invalidates everything from
N onward.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from ._canon import canon
from ._tokens import count

__all__ = ["Block", "decompose"]


@dataclass
class Block:
    kind: str  # "system" | "tool" | "message"
    id: str  # "system", "tools[0]", "messages[3]"
    label: str  # tool name, message role, or "system"
    raw: Any
    text: str
    hash: str
    tokens: int

    @property
    def preview(self) -> str:
        return self.text[:80] + ("..." if len(self.text) > 80 else "")


def _make(kind: str, block_id: str, label: str, raw: Any) -> Block:
    text = canon(raw)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return Block(kind, block_id, label, raw, text, digest, count(text))


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        if "name" in tool:
            return str(tool["name"])
        fn = tool.get("function")
        if isinstance(fn, dict) and "name" in fn:  # OpenAI function-tool shape
            return str(fn["name"])
    name = getattr(tool, "name", None)
    return str(name) if name else "?"


def decompose(
    messages: Sequence[Any] | None = None,
    system: Any = None,
    tools: Sequence[Any] | None = None,
) -> list[Block]:
    blocks: list[Block] = []
    if system is not None:
        blocks.append(_make("system", "system", "system", system))
    for i, tool in enumerate(tools or []):
        blocks.append(_make("tool", f"tools[{i}]", _tool_name(tool), tool))
    for i, msg in enumerate(messages or []):
        if isinstance(msg, dict):
            role = str(msg.get("role", "?"))
        else:
            role = str(getattr(msg, "role", "?"))
        blocks.append(_make("message", f"messages[{i}]", role, msg))
    return blocks
