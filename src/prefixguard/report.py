"""Findings and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Cause", "Finding", "TurnReport", "CachePrefixBroken"]


class Cause:
    SYSTEM_MUTATED = "SYSTEM_MUTATED"
    TOOL_REORDER = "TOOL_REORDER"
    TOOL_SET_CHANGED = "TOOL_SET_CHANGED"
    TOOL_SCHEMA_DRIFT = "TOOL_SCHEMA_DRIFT"
    HISTORY_REWRITE = "HISTORY_REWRITE"
    HISTORY_DROP = "HISTORY_DROP"
    COMPACTION = "COMPACTION"
    VOLATILE_CONTENT = "VOLATILE_CONTENT"
    NON_DETERMINISTIC_SERIALIZATION = "NON_DETERMINISTIC_SERIALIZATION"


_FIXES = {
    Cause.SYSTEM_MUTATED: "Freeze the system prompt for the session. Move anything "
    "per-turn (time, user id, retrieved context) into the latest user message.",
    Cause.TOOL_REORDER: "Sort tools by name before every call. Set ordering is not "
    "stable across dict/set iteration or dynamic registration.",
    Cause.TOOL_SET_CHANGED: "Keep the tool list fixed for the session; gate "
    "availability in your handler, not in the schema list.",
    Cause.TOOL_SCHEMA_DRIFT: "Serialize schemas with json.dumps(..., sort_keys=True) "
    "and cache the serialized string.",
    Cause.HISTORY_REWRITE: "History must be append-only. Do not edit, re-annotate, or "
    "re-serialize prior turns.",
    Cause.HISTORY_DROP: "Dropping from the middle/front rewrites the prefix. Compact "
    "to a summary block at a cache breakpoint instead of sliding a window.",
    Cause.COMPACTION: "Expected on compaction. Place the cache breakpoint on the "
    "summary block so the next turns re-warm against it.",
    Cause.VOLATILE_CONTENT: "Remove per-turn volatile values from the stable prefix.",
    Cause.NON_DETERMINISTIC_SERIALIZATION: "Same logical content, different bytes. "
    "Canonicalize serialization at the boundary.",
}


@dataclass
class Finding:
    cause: str
    block_id: str
    detail: str
    before: Any = None
    after: Any = None

    @property
    def fix(self) -> str:
        return _FIXES.get(self.cause, "")


@dataclass
class TurnReport:
    session: str
    turn: int
    broken: bool
    stable_prefix_tokens: int = 0
    invalidated_tokens: int = 0
    total_prefix_tokens: int = 0
    divergence_block: str | None = None
    divergence_token_offset: int | None = None
    findings: list[Finding] = field(default_factory=list)
    counter: str = ""

    @property
    def hit_ratio(self) -> float:
        if not self.total_prefix_tokens:
            return 0.0
        return self.stable_prefix_tokens / self.total_prefix_tokens

    def explain(self, color: bool = True) -> str:
        g, r, y, d, x = (
            ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")
            if color
            else ("", "", "", "", "")
        )
        if not self.broken:
            if self.turn == 1:
                head = (
                    f"{g}OK{x} turn 1 — baseline captured, "
                    f"~{self.total_prefix_tokens:,} tokens in prefix"
                )
            else:
                head = (
                    f"{g}OK{x} turn {self.turn} — prefix intact, "
                    f"~{self.stable_prefix_tokens:,} tokens reused "
                    f"({self.hit_ratio:.0%} of prefix)"
                )
            if not self.findings:
                return head
            out = [head]
            for f_ in self.findings:
                out.append(f"\n  {y}WARN {f_.cause}{x} @ {f_.block_id}")
                out.append(f"    {f_.detail}")
                if f_.fix:
                    out.append(f"    {g}fix:{x} {f_.fix}")
            return "\n".join(out)

        lines = [
            f"{r}CACHE PREFIX BROKEN{x} — session {self.session!r}, turn {self.turn}",
            f"  divergence at {y}{self.divergence_block}{x}"
            f" (~token {self.divergence_token_offset:,})",
            f"  reused    ~{self.stable_prefix_tokens:,} tokens"
            f"  ({self.hit_ratio:.0%})",
            f"  re-prefill ~{self.invalidated_tokens:,} tokens"
            f"  ({1 - self.hit_ratio:.0%})",
        ]
        for f_ in self.findings:
            lines.append(f"\n  {r}{f_.cause}{x} @ {f_.block_id}")
            lines.append(f"    {f_.detail}")
            if f_.before is not None or f_.after is not None:
                lines.append(f"    {d}before:{x} {_trunc(f_.before)}")
                lines.append(f"    {d}after :{x} {_trunc(f_.after)}")
            if f_.fix:
                lines.append(f"    {g}fix:{x} {f_.fix}")
        lines.append(f"\n  {d}token counts are estimates ({self.counter}){x}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.explain(color=False)


def _trunc(v: Any, n: int = 120) -> str:
    s = str(v)
    return s[:n] + ("..." if len(s) > n else "")


class CachePrefixBroken(AssertionError):
    """Raised in strict mode when the cacheable prefix is invalidated."""

    def __init__(self, report: TurnReport):
        self.report = report
        super().__init__(report.explain(color=False))
