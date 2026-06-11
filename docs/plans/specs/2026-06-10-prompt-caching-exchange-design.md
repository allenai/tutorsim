# Prompt Caching on Exchange Calls -- Design

*Status: spec / 2026-06-10*

## Goal

Cut Phase 1 input-token cost (the dominant chunk of benchmark spend, ~85% of
total) by caching the static head of every tutor / student turn within a
scenario. The head -- system prompt + scenario transcript prefix -- is
byte-identical across all rounds of one scenario, so it hits cache on round
2 onward. Each scenario gets its own cache because the prefix differs.

Expected savings, observed in the dyn_smoke baseline (~30k tokens of prefix
per round, 50 rounds): ~$20 saved per scenario at Anthropic Opus 4.8 pricing,
~$200 per 10-scenario varied smoke. Roughly 85% reduction in Phase 1 input
cost.

## Scope

In scope:
- Caching on **Anthropic** exchange calls (tutor + student turns) -- explicit
  `cache_control: {"type": "ephemeral"}` on the head text block.
- Caching on **OpenAI** exchange calls -- automatic prefix caching kicks in
  when a request shares >= 1024 tokens of byte-identical prefix with a recent
  call. No SDK config; we just keep the head deterministic.
- A new `ModelClient.generate(... cacheable_prefix=...)` API used by the
  exchange loop.

Out of scope:
- Real **Gemini** context-caching lifecycle (`CachedContent.create/refresh/
  delete`). Architecture supports it via the same `cacheable_prefix` kwarg,
  but the Gemini implementation is a no-op concatenation for now with
  `TODO(gemini-cache)` annotation. We don't run Gemini in benchmark today.
- Caching trait-generator, annotator, or labeller calls. Each of those has a
  unique prompt per scenario; no shared prefix to hit.
- Extended (1-hour) cache TTL. Use 5-min ephemeral; revisit if batch-mode hit
  rates are poor.

## Architecture

### New `ModelClient.generate` kwarg

`annotator/core/client.py`:

```python
def generate(
    self,
    prompt: str,
    *,
    cacheable_prefix: str | None = None,
    json_mode: bool = False,
    max_tokens: int = 4096,
    timeout: int = 600,
    ...
) -> ModelResponse:
    """When cacheable_prefix is set, the model sees `cacheable_prefix + prompt`
    with the prefix marked cacheable per provider. When None, behavior is
    unchanged (the existing string-prompt path).
    """
```

Per-provider behavior when `cacheable_prefix` is set:

**Anthropic** (`_generate_anthropic`):
```python
messages = [{"role": "user", "content": [
    {"type": "text", "text": cacheable_prefix,
     "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": prompt},
]}]
```
Extract cache usage from response: `cache_creation_input_tokens` and
`cache_read_input_tokens` populate the existing `usage` dict.

**OpenAI** (`_generate_openai`):
Concatenate the two: send a single user message with `cacheable_prefix +
prompt` as content. OpenAI's automatic prefix cache hits when the prefix is
byte-identical to a recent call and >= 1024 tokens. Extract
`usage.prompt_tokens_details.cached_tokens` into `usage["cached_tokens"]` so
we can observe hit rate.

**Gemini** (`_generate_gemini`):
Concatenate the two and send as the single prompt. No cache lifecycle yet.
Mark with `# TODO(gemini-cache): wire CachedContent here`.

When `cacheable_prefix is None`, no change to any of the three providers.

### `_build_role_prompt` returns a tuple

`benchmark/core/exchange.py`:

```python
def _build_role_prompt(
    role: str,
    transcript_prefix: str,
    extra: str,
    student_context: str,
    prompt_version: str = "v1",
    student_mode: str | None = None,
    scenario=None,
    trait_client=None,
    trait_model: str | None = None,
) -> tuple[str, str]:
    """Return (cacheable_head, tail).

    cacheable_head = system_prompt (with substitutions) + standard scaffolding
                     + transcript_prefix
    tail            = extra + "\\n\\n" + role_instruction

    Both are concatenated client-side to form the full prompt; the head is
    marked cacheable per provider.
    """
```

Notably, the trait-persona substitution into the student system prompt still
happens on the head side (it's invariant for the scenario), so trait mode
also benefits from caching.

### Exchange-loop change

Both `run_exchange` and `run_exchanges_batch` already maintain a
`running_transcript` string. The refactor is to track an additional `extra`
string (the per-round growing suffix) alongside it. The two are equivalent
to `scenario.transcript_prefix + extra`; we keep `running_transcript` for
existing in-prompt logging / display use, and use `extra` for the cache split.

Two call shapes change:
```python
head, tail = _build_role_prompt("TUTOR", scenario.transcript_prefix, extra, ...)
response = tutor_client.generate(prompt=tail, cacheable_prefix=head, ...)
```
Same shape for STUDENT. The cache key is identical across all
tutor+student calls within one scenario (the head doesn't change), so every
call after the first should hit cache.

### Cache TTL strategy

5-minute ephemeral cache. In sync mode rounds fire seconds apart -- well
within the TTL. In batch mode, Anthropic processes batch entries in parallel;
multiple scenarios submitted together with shared `cacheable_prefix` will
benefit if processed within the TTL window. We log hit rate and revisit if
batch hit rates are poor.

## Usage tracking

Extend `_add_usage` to handle three new keys (`cache_creation_input_tokens`,
`cache_read_input_tokens`, `cached_tokens`). These are zero in current
runs; with caching they populate naturally. The viewer / scoring code
doesn't need changes -- usage is for cost reporting only.

Add a one-line per-scenario log:
```
[<scenario_id>] cache hit rate: X cached / Y input tokens (~Z%)
```

## Tests

In `tests/test_client_caching.py` (new):

- `generate(cacheable_prefix=X, prompt=Y)` on Anthropic produces a request
  whose `messages[0].content` is two text blocks; first has
  `cache_control: {"type":"ephemeral"}`, second does not. Verified by
  patching `self._client.messages.create` and inspecting the kwargs.
- `generate(cacheable_prefix=None, prompt=Y)` on Anthropic produces the
  legacy single-string content (no behavior change).
- `generate(cacheable_prefix=X, prompt=Y)` on OpenAI sends a single user
  message whose content is the concatenation `X + Y`.
- OpenAI cache usage: when API response includes
  `prompt_tokens_details.cached_tokens=N`, the returned `usage` dict
  contains `cached_tokens=N`.
- Anthropic cache usage: when response includes
  `cache_creation_input_tokens=N` and `cache_read_input_tokens=M`, those
  appear in the returned `usage` dict.

In `tests/test_benchmark_exchange_dynamic.py` (extend):

- `_build_role_prompt` returns `(head, tail)` where `head` is the same
  string across multiple round invocations on the same scenario.
- `run_exchange` invokes `tutor_client.generate` with the same
  `cacheable_prefix` value on every tutor turn within one scenario.

## Out-of-scope follow-ups

- Pre-batched trait gen as a Phase 0.5 step (separate spec; only matters at
  scale).
- Caching the annotator pass (each annotator call has its own scenario-
  specific transcript; no shared prefix benefit).
- Gemini real lifecycle (separate spec when we actually run Gemini in
  benchmark).
- Extended (1h) cache for batch mode if 5-min hit rate is poor.

## Risk / open questions

- **Whitespace drift breaking OpenAI auto-cache:** OpenAI requires byte-
  identical prefix for cache hits. Make sure the cacheable_head we send is
  deterministic per scenario (no `datetime.now()` strings, no trailing
  whitespace variation). The test that asserts head invariance across
  rounds catches regressions.
- **Anthropic 1024-token minimum:** prefixes below the minimum aren't cached
  by Anthropic (the SDK does cache_control marking unconditionally but the
  API rejects sub-minimum cache attempts gracefully and just doesn't cache).
  No code change needed; small prefixes pay full price.
- **Cache eviction across runs:** caches are per-organization with 5-min TTL.
  A re-run of the same smoke after >5 min cold-starts the cache for round 1
  and hits on round 2+ -- not "free" but still 85%+ savings.
- **Refactor blast radius:** `_build_role_prompt` changes its return type.
  Every caller in `benchmark/core/exchange.py` and the two tests that import
  it have to be updated. No callers outside the benchmark package, so the
  refactor is contained.
