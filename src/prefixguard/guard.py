"""The stateful guard: observe each turn, get told what broke."""

from __future__ import annotations

from typing import Any, Sequence

from ._blocks import Block, decompose
from ._diff import classify, first_divergence
from ._lint import lint
from ._tokens import counter_name
from .report import CachePrefixBroken, Finding, TurnReport

__all__ = ["PrefixGuard"]


class PrefixGuard:
    """Track the cacheable prefix of a conversation across turns.

    >>> guard = PrefixGuard(strict=False)
    >>> r = guard.observe(messages=[{"role": "user", "content": "hi"}])
    >>> r.broken
    False
    """

    def __init__(
        self,
        strict: bool = False,
        min_prefix_tokens: int = 0,
        static_lint: bool = True,
    ):
        self.strict = strict
        self.min_prefix_tokens = min_prefix_tokens
        self.static_lint = static_lint
        self._sessions: dict[str, list[Block]] = {}
        self._turns: dict[str, int] = {}
        self._history: list[TurnReport] = []

    # -- main entry point -------------------------------------------------

    def observe(
        self,
        messages: Sequence[Any] | None = None,
        system: Any = None,
        tools: Sequence[Any] | None = None,
        session: str = "default",
    ) -> TurnReport:
        blocks = decompose(messages=messages, system=system, tools=tools)
        turn = self._turns.get(session, 0) + 1
        self._turns[session] = turn
        total = sum(b.tokens for b in blocks)

        prev = self._sessions.get(session)
        self._sessions[session] = blocks

        findings: list[Finding] = []
        if self.static_lint and turn == 1:
            findings += lint(system=system, tools=tools, messages=messages)

        if prev is None:
            report = TurnReport(
                session=session,
                turn=turn,
                broken=False,
                stable_prefix_tokens=0,
                invalidated_tokens=0,
                total_prefix_tokens=total,
                findings=findings,
                counter=counter_name(),
            )
            self._history.append(report)
            return report

        idx = first_divergence(prev, blocks)
        if idx is None:
            stable = sum(b.tokens for b in blocks[: len(prev)])
            report = TurnReport(
                session=session,
                turn=turn,
                broken=False,
                stable_prefix_tokens=stable,
                invalidated_tokens=total - stable,
                total_prefix_tokens=total,
                findings=findings,
                counter=counter_name(),
            )
            self._history.append(report)
            return report

        stable = sum(b.tokens for b in blocks[:idx])
        lost = sum(b.tokens for b in prev[idx:])
        findings += classify(prev, blocks, idx)

        report = TurnReport(
            session=session,
            turn=turn,
            broken=True,
            stable_prefix_tokens=stable,
            invalidated_tokens=lost,
            total_prefix_tokens=total,
            divergence_block=(blocks[idx].id if idx < len(blocks) else prev[idx].id),
            divergence_token_offset=stable,
            findings=findings,
            counter=counter_name(),
        )
        self._history.append(report)

        if self.strict and lost >= self.min_prefix_tokens:
            raise CachePrefixBroken(report)
        return report

    # -- aggregate --------------------------------------------------------

    @property
    def history(self) -> list[TurnReport]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """Session-wide numbers. This is the line that goes in your dashboard."""
        turns = [r for r in self._history if r.turn > 1]
        reused = sum(r.stable_prefix_tokens for r in turns)
        served = sum(r.total_prefix_tokens for r in turns)
        broken = [r for r in turns if r.broken]
        causes: dict[str, int] = {}
        for r in broken:
            for f in r.findings:
                causes[f.cause] = causes.get(f.cause, 0) + 1
        return {
            "turns": len(turns) + (1 if self._history else 0),
            "breaks": len(broken),
            "prefix_hit_ratio": (reused / served) if served else 0.0,
            "tokens_reused": reused,
            "tokens_reprefilled": sum(r.invalidated_tokens for r in broken),
            "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
            "counter": counter_name(),
        }

    def reset(self, session: str | None = None) -> None:
        if session is None:
            self._sessions.clear()
            self._turns.clear()
            self._history.clear()
        else:
            self._sessions.pop(session, None)
            self._turns.pop(session, None)
