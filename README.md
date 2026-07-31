# prefixguard

**Your prompt cache is probably broken. You have no error to tell you.**

```bash
pip install prefixguard
```

Prompt caching (Anthropic, OpenAI, Bedrock, vLLM, SGLang, llama.cpp) requires an
**exact byte-for-byte prefix match**. Change one token early in the prompt and
every cached token after it is thrown away and re-prefilled.

Then you write perfectly ordinary agent code:

```python
messages = trim_to_window(messages)          # drops from the middle
system   = f"You are an agent. Today is {date.today()}."   # changes daily
tools    = [t.schema for t in registry.values()]           # order not guaranteed
```

…and you silently re-prefill your entire context on every turn. No exception.
No warning. Just a 5–10x bill and a TTFT cliff.

`prefixguard` is the missing assertion.

---

## Usage

```python
from prefixguard import PrefixGuard

guard = PrefixGuard()

# call it right before you hit the API, every turn
report = guard.observe(messages, system=system, tools=tools, session=conv_id)

if report.broken:
    print(report.explain())
```

```
CACHE PREFIX BROKEN — session 's1', turn 2
  divergence at tools[1] (~token 27)
  reused    ~27 tokens  (1%)
  re-prefill ~2,030 tokens  (99%)

  TOOL_REORDER @ tools[1]
    Same 3 tools, different order. Ordering is not stable across turns.
    before: ['search', 'memory', 'feishu_doc_read']
    after : ['search', 'feishu_doc_read', 'memory']
    fix: Sort tools by name before every call. Set ordering is not stable
         across dict/set iteration or dynamic registration.
```

## Fail the build instead

```python
guard = PrefixGuard(strict=True)   # raises CachePrefixBroken
```

Drop it in a pytest fixture and cache regressions never reach production.

## Catch it before you ever run a turn

```python
from prefixguard import lint

for issue in lint(system=SYSTEM_PROMPT, tools=TOOLS):
    print(issue.cause, "-", issue.detail)
```

```
VOLATILE_CONTENT - Found ISO timestamp ('2026-07-31T09:00') at char 33 of
system. If this changes per turn, you will miss the cache on every single
call and re-prefill ~865 tokens.
```

## Session numbers

```python
guard.summary()
# {'turns': 24, 'breaks': 9, 'prefix_hit_ratio': 0.31,
#  'tokens_reused': 412_882, 'tokens_reprefilled': 903_114,
#  'causes': {'HISTORY_DROP': 6, 'SYSTEM_MUTATED': 3}}
```

## What it detects

| Cause | What happened |
|---|---|
| `SYSTEM_MUTATED` | System prompt bytes changed mid-session |
| `TOOL_REORDER` | Same tools, different order |
| `TOOL_SET_CHANGED` | A tool was added or removed |
| `TOOL_SCHEMA_DRIFT` | Same tool name, different serialized schema |
| `HISTORY_REWRITE` | A prior message was edited in place |
| `HISTORY_DROP` | Sliding-window trim rewrote the prefix |
| `COMPACTION` | History replaced by a summary (expected, but priced) |
| `VOLATILE_CONTENT` | Timestamp / UUID / session id in the stable prefix |
| `NON_DETERMINISTIC_SERIALIZATION` | Identical content, different bytes |

## Design notes

- **Zero dependencies.** `pip install prefixguard[tokens]` adds `tiktoken` for
  exact counts; otherwise a character heuristic is used and every report says so.
- **Provider-agnostic.** It reads the request you were going to send. It does not
  wrap your client, proxy your traffic, or need an API key.
- **Canonical comparison.** Dict key order and unicode escaping are normalised
  before hashing, so it reports real divergence — not JSON noise.
- **Append-only is the happy path.** Extending `messages` is never a break.

## License

MIT
