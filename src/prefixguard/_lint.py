"""Static lint — catch cache-breakers *before* you ship, with no session needed."""

from __future__ import annotations

import re
from typing import Any, Sequence

from ._blocks import decompose
from ._canon import canon  # noqa: F401
from .report import Cause, Finding

__all__ = ["lint"]

_VOLATILE = [
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", "ISO timestamp"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "date"),
    (r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "UUID"),
    (r"\b1[6-9]\d{8}\b", "unix epoch seconds"),
    (r"(?i)\b(today|current time|right now) is\b", "natural-language clock"),
    (r"(?i)\bsession[_ -]?id\b\s*[:=]", "session id"),
    (r"(?i)\brequest[_ -]?id\b\s*[:=]", "request id"),
]


def lint(
    system: Any = None,
    tools: Sequence[Any] | None = None,
    messages: Sequence[Any] | None = None,
) -> list[Finding]:
    """Inspect a prompt for content that will break the cache on the *next* turn."""
    findings: list[Finding] = []
    blocks = decompose(messages=messages, system=system, tools=tools)

    # Volatile content only matters in the stable prefix: system + tools, plus
    # every message except the last (the last one is new anyway).
    msg_blocks = [b for b in blocks if b.kind == "message"]
    stable = [b for b in blocks if b.kind != "message"] + msg_blocks[:-1]

    for b in stable:
        text = canon(b.raw)
        for pattern, name in _VOLATILE:
            m = re.search(pattern, text)
            if m:
                findings.append(
                    Finding(
                        Cause.VOLATILE_CONTENT,
                        b.id,
                        f"Found {name} ({m.group(0)!r}) at char {m.start()} of "
                        f"{b.id}. If this changes per turn, you will miss the "
                        f"cache on every single call and re-prefill "
                        f"~{sum(x.tokens for x in blocks[blocks.index(b):]):,} tokens.",
                        after=m.group(0),
                    )
                )
                break

    # Tool ordering: not sorted is a latent reorder bug.
    names = [b.label for b in blocks if b.kind == "tool"]
    if names and names != sorted(names):
        findings.append(
            Finding(
                Cause.TOOL_REORDER,
                "tools",
                "Tool list is not in a deterministic (sorted) order. If it is "
                "built from a dict, set, or plugin registry, the order can shift "
                "between turns and silently invalidate everything after it.",
                after=names,
            )
        )

    # Deliberately NOT linted: unsorted dict keys. Idiomatic schemas are almost
    # never alphabetical, so flagging them fires on every tool and trains people
    # to ignore the linter. Real byte drift is caught at diff time as
    # TOOL_SCHEMA_DRIFT, where there is actual evidence for it.
    return findings
