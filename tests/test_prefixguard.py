import pytest
from prefixguard import PrefixGuard, lint, Cause, CachePrefixBroken

SYS = "You are a helpful agent."
TOOLS = [{"name": "alpha", "input_schema": {}}, {"name": "beta", "input_schema": {}}]
M1 = [{"role": "user", "content": "hello " * 500}]


def _g(**kw):
    return PrefixGuard(**kw)


def test_append_only_is_intact():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    m2 = M1 + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "go"}]
    r = g.observe(m2, system=SYS, tools=TOOLS, session="s")
    assert not r.broken
    assert r.hit_ratio > 0.9


def test_system_mutation_detected():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    r = g.observe(M1, system=SYS + " Now: 10:04.", tools=TOOLS, session="s")
    assert r.broken
    assert r.findings[0].cause == Cause.SYSTEM_MUTATED
    assert r.divergence_block == "system"


def test_tool_reorder_detected():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    r = g.observe(M1, system=SYS, tools=list(reversed(TOOLS)), session="s")
    assert r.broken
    assert any(f.cause == Cause.TOOL_REORDER for f in r.findings)


def test_tool_schema_drift_detected():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    drifted = [dict(TOOLS[0], description="new"), TOOLS[1]]
    r = g.observe(M1, system=SYS, tools=drifted, session="s")
    assert any(f.cause == Cause.TOOL_SCHEMA_DRIFT for f in r.findings)


def test_key_order_does_not_false_positive():
    g = _g()
    g.observe(M1, system=SYS, tools=[{"name": "a", "z": 1, "b": 2}], session="s")
    r = g.observe(M1, system=SYS, tools=[{"b": 2, "name": "a", "z": 1}], session="s")
    assert not r.broken


def test_sliding_window_trim_detected():
    g = _g()
    long = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    g.observe(long, system=SYS, tools=TOOLS, session="s")
    r = g.observe(long[5:], system=SYS, tools=TOOLS, session="s")
    assert r.broken


def test_compaction_flagged_separately():
    g = _g()
    long = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    g.observe(long, system=SYS, tools=TOOLS, session="s")
    r = g.observe([{"role": "user", "content": "summary"}], system=SYS, tools=TOOLS, session="s")
    assert any(f.cause == Cause.COMPACTION for f in r.findings)


def test_strict_mode_raises():
    g = _g(strict=True)
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    with pytest.raises(CachePrefixBroken):
        g.observe(M1, system="different", tools=TOOLS, session="s")


def test_sessions_are_isolated():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="a")
    r = g.observe(M1, system="other", tools=TOOLS, session="b")
    assert not r.broken


def test_lint_finds_timestamp():
    f = lint(system="Agent. Time is 2026-07-31T09:00:00Z", tools=TOOLS)
    assert any(x.cause == Cause.VOLATILE_CONTENT for x in f)


def test_lint_finds_unsorted_tools():
    f = lint(system=SYS, tools=[{"name": "zeta"}, {"name": "alpha"}])
    assert any(x.cause == Cause.TOOL_REORDER for x in f)


def test_lint_clean_prompt_is_quiet():
    assert lint(system=SYS, tools=TOOLS) == []


def test_summary_shape():
    g = _g()
    g.observe(M1, system=SYS, tools=TOOLS, session="s")
    g.observe(M1, system="x", tools=TOOLS, session="s")
    s = g.summary()
    assert s["breaks"] == 1 and 0.0 <= s["prefix_hit_ratio"] <= 1.0
