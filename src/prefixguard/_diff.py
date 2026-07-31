"""Find *where* the prefix diverged and *why*."""

from __future__ import annotations

from ._blocks import Block
from .report import Cause, Finding

__all__ = ["first_divergence", "classify"]


def first_divergence(prev: list[Block], curr: list[Block]) -> int | None:
    """Index of the first block whose bytes differ. None if prefix intact.

    A pure append (curr extends prev) is intact — that is the happy path.
    """
    for i in range(min(len(prev), len(curr))):
        if prev[i].hash != curr[i].hash:
            return i
    if len(curr) < len(prev):
        # Blocks disappeared. The prefix is shorter than what was cached; the
        # provider will still match up to len(curr), so this is only a break
        # if it also implies rewritten content downstream. Treat as a drop.
        return len(curr)
    return None


def _by_kind(blocks: list[Block], kind: str) -> list[Block]:
    return [b for b in blocks if b.kind == kind]


def classify(prev: list[Block], curr: list[Block], idx: int) -> list[Finding]:
    findings: list[Finding] = []

    p = prev[idx] if idx < len(prev) else None
    c = curr[idx] if idx < len(curr) else None
    kind = (c or p).kind if (c or p) else "message"

    if kind == "system":
        findings.append(
            Finding(
                Cause.SYSTEM_MUTATED,
                "system",
                "System prompt bytes changed between turns. Every downstream "
                "token must be re-prefilled.",
                before=p.preview if p else None,
                after=c.preview if c else None,
            )
        )
        return findings

    if kind == "tool":
        pt, ct = _by_kind(prev, "tool"), _by_kind(curr, "tool")
        pn = [b.label for b in pt]
        cn = [b.label for b in ct]
        if set(pn) == set(cn) and pn != cn:
            findings.append(
                Finding(
                    Cause.TOOL_REORDER,
                    (c or p).id,
                    f"Same {len(cn)} tools, different order. "
                    "Ordering is not stable across turns.",
                    before=pn,
                    after=cn,
                )
            )
        elif set(pn) != set(cn):
            added = [n for n in cn if n not in pn]
            removed = [n for n in pn if n not in cn]
            findings.append(
                Finding(
                    Cause.TOOL_SET_CHANGED,
                    (c or p).id,
                    f"Tool set changed. added={added or '-'} removed={removed or '-'}",
                    before=pn,
                    after=cn,
                )
            )
        else:
            findings.append(
                Finding(
                    Cause.TOOL_SCHEMA_DRIFT,
                    (c or p).id,
                    f"Tool {(c or p).label!r} has the same name but different "
                    "serialized bytes. Usually key ordering or a re-generated schema.",
                    before=p.preview if p else None,
                    after=c.preview if c else None,
                )
            )
        return findings

    # message-level
    pm, cm = _by_kind(prev, "message"), _by_kind(curr, "message")
    if len(cm) < len(pm):
        shrink = len(pm) - len(cm)
        cause = Cause.COMPACTION if shrink >= 4 else Cause.HISTORY_DROP
        findings.append(
            Finding(
                cause,
                (c or p).id if (c or p) else "messages",
                f"History shrank by {shrink} message(s) — "
                f"{len(pm)} -> {len(cm)}. This is not append-only.",
                before=f"{len(pm)} messages",
                after=f"{len(cm)} messages",
            )
        )
    else:
        findings.append(
            Finding(
                Cause.HISTORY_REWRITE,
                (c or p).id,
                f"Message at {(c or p).id} was modified in place. "
                "Prior turns must never change.",
                before=p.preview if p else None,
                after=c.preview if c else None,
            )
        )
    return findings
