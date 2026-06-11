# Prompt Caching on Exchange Calls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cacheable_prefix` kwarg to `ModelClient.generate` so the exchange loop can mark the per-scenario static head (system prompt + transcript prefix) as cacheable. Anthropic uses explicit `cache_control`; OpenAI relies on automatic prefix caching; Gemini is a no-op concat stub. Cuts Phase 1 input cost ~85% on multi-round exchanges.

**Architecture:** `ModelClient.generate(prompt, *, cacheable_prefix=None, ...)`. When set, the effective prompt is `cacheable_prefix + prompt`, with the prefix marked cacheable per provider. `_build_role_prompt` in `benchmark/core/exchange.py` is refactored to return a `(head, tail)` tuple; both `run_exchange` and `run_exchanges_batch` track `extra` (the growing suffix) separately from the static `scenario.transcript_prefix` and pass them to the client.

**Tech Stack:** Python 3.11, pytest, existing Anthropic / OpenAI / Gemini SDKs already in `annotator/core/client.py`.

**Spec:** [`docs/plans/specs/2026-06-10-prompt-caching-exchange-design.md`](specs/2026-06-10-prompt-caching-exchange-design.md)

---

## File Map

- **Modify:** `annotator/core/client.py` — add `cacheable_prefix` kwarg to `generate`; per-provider plumbing in `_generate_anthropic`, `_generate_openai`, `_generate_gemini`; extract cache-usage keys into the returned `usage` dict.
- **Modify:** `benchmark/core/exchange.py` — refactor `_build_role_prompt` to return `(head, tail)`; track an `extra` string in `run_exchange` and `run_exchanges_batch`; call `client.generate(prompt=tail, cacheable_prefix=head, ...)` from both tutor and student turn paths.
- **Create:** `tests/test_client_caching.py` — unit tests for the new client-side caching plumbing across all three providers.
- **Modify:** `tests/test_benchmark_exchange_dynamic.py` — extend existing tests where `_build_role_prompt`'s return type change ripples; add a cache-key-stability test.

---

## Task 1: Add `cacheable_prefix` to `ModelClient.generate` + per-provider plumbing (TDD)

**Files:**
- Modify: `annotator/core/client.py`
- Create: `tests/test_client_caching.py`

### Steps

- [ ] **Step 1: Write the failing tests**

Create `tests/test_client_caching.py`:

```python
"""Tests for ModelClient.generate cacheable_prefix kwarg + per-provider plumbing."""
from unittest.mock import MagicMock, patch

import pytest

from annotator.core.client import ModelClient


def _make_anthropic_client():
    client = ModelClient.__new__(ModelClient)
    client.model = "claude-opus-4-8"
    client.provider = "anthropic"
    client._client = MagicMock()
    return client


def _make_openai_client():
    client = ModelClient.__new__(ModelClient)
    client.model = "gpt-5.5"
    client.provider = "openai"
    client._client = MagicMock()
    return client


def _anthropic_response(cache_creation=0, cache_read=0):
    resp = MagicMock()
    text_block = MagicMock(); text_block.type = "text"; text_block.text = "ok"
    resp.content = [text_block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 5
    resp.usage.cache_creation_input_tokens = cache_creation
    resp.usage.cache_read_input_tokens = cache_read
    return resp


def _openai_response(cached=0):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "ok"
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 105
    resp.usage.prompt_tokens_details = MagicMock()
    resp.usage.prompt_tokens_details.cached_tokens = cached
    return resp


def test_anthropic_cacheable_prefix_marks_head_block():
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response()

    client.generate("the tail", json_mode=False, max_tokens=64,
                    cacheable_prefix="the cacheable head")

    kwargs = client._client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "the cacheable head",
                          "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "the tail"}


def test_anthropic_cacheable_prefix_none_uses_single_string_content():
    """Back-compat: when cacheable_prefix is None, behavior is unchanged."""
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response()

    client.generate("just the tail", json_mode=False, max_tokens=64)

    kwargs = client._client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    # Legacy path: content is a plain string.
    assert content == "just the tail"


def test_anthropic_usage_captures_cache_creation_and_read():
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response(
        cache_creation=80, cache_read=20,
    )

    resp = client.generate("tail", json_mode=False, max_tokens=64,
                           cacheable_prefix="head"*200)

    assert resp.usage["cache_creation_input_tokens"] == 80
    assert resp.usage["cache_read_input_tokens"] == 20


def test_openai_cacheable_prefix_concatenates_into_single_user_message():
    client = _make_openai_client()
    client._client.chat.completions.create.return_value = _openai_response()

    client.generate("the tail", json_mode=False, max_tokens=64,
                    cacheable_prefix="the head")

    kwargs = client._client.chat.completions.create.call_args.kwargs
    msgs = kwargs["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    # OpenAI gets the concatenated text; auto-cache handles the rest.
    assert msgs[0]["content"] == "the head" + "the tail"


def test_openai_usage_captures_cached_tokens():
    client = _make_openai_client()
    client._client.chat.completions.create.return_value = _openai_response(cached=42)

    resp = client.generate("tail", json_mode=False, max_tokens=64,
                           cacheable_prefix="head")

    assert resp.usage["cached_tokens"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client_caching.py -v`
Expected: `TypeError: generate() got an unexpected keyword argument 'cacheable_prefix'`.

- [ ] **Step 3: Extend `ModelClient.generate` to accept `cacheable_prefix`**

In `annotator/core/client.py`, update the `generate` method signature to add `cacheable_prefix` as a keyword-only argument:

```python
    def generate(self, prompt: str,
                 images: list[str] | None = None,
                 json_mode: bool = True,
                 max_tokens: int = 0, timeout: int = 120,
                 thinking: bool = False,
                 thinking_budget: int = 0,
                 reasoning_effort: str = "",
                 enable_cache: bool = False,
                 *,
                 cacheable_prefix: str | None = None) -> ModelResponse:
```

Then update each per-provider dispatch call to forward `cacheable_prefix`. The three `if self.provider == ...` branches inside `generate` become:

```python
                if self.provider == "gemini":
                    return self._generate_gemini(prompt, json_mode, max_tokens, timeout,
                                                 thinking, thinking_budget, images,
                                                 cacheable_prefix=cacheable_prefix)
                elif self.provider == "openai":
                    return self._generate_openai(prompt, json_mode, max_tokens, timeout,
                                                  thinking, thinking_budget,
                                                  reasoning_effort=reasoning_effort,
                                                  images=images,
                                                  cacheable_prefix=cacheable_prefix)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, json_mode, max_tokens, timeout,
                                                     thinking, thinking_budget,
                                                     images=images,
                                                     enable_cache=enable_cache,
                                                     cacheable_prefix=cacheable_prefix)
```

- [ ] **Step 4: Implement `cacheable_prefix` in `_generate_anthropic`**

Update `_generate_anthropic`'s signature to accept `cacheable_prefix`:

```python
    def _generate_anthropic(self, prompt, json_mode, max_tokens, timeout,
                            thinking=False, thinking_budget=0,
                            images=None, enable_cache=False,
                            cacheable_prefix: str | None = None):
```

Replace the existing `if images: ... else: content = prompt` block with logic that builds a structured-content list when caching is requested, otherwise preserves the legacy string-content path:

```python
        if images:
            image_blocks = _build_image_blocks_anthropic(
                images, use_url=_should_use_presigned_url(), enable_cache=enable_cache,
            )
            content = _interleave_text_and_images(
                prompt, image_blocks, lambda s: {"type": "text", "text": s},
            )
            if cacheable_prefix is not None:
                # Prepend the cacheable head as its own text block in front of the interleaved content.
                content = [
                    {"type": "text", "text": cacheable_prefix,
                     "cache_control": {"type": "ephemeral"}},
                ] + content
        elif cacheable_prefix is not None:
            content = [
                {"type": "text", "text": cacheable_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt
```

Update the usage extraction at the bottom of `_generate_anthropic` to capture cache fields when present:

```python
        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
            "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
```

- [ ] **Step 5: Implement `cacheable_prefix` in `_generate_openai`**

Update `_generate_openai`'s signature to accept `cacheable_prefix`:

```python
    def _generate_openai(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0,
                         reasoning_effort: str = "", images=None,
                         cacheable_prefix: str | None = None):
```

Replace the existing `if images: ... else: content = prompt` block with:

```python
        if images:
            image_blocks = _build_image_blocks_openai(
                images, use_url=_should_use_presigned_url(),
            )
            # Prepend the cacheable head (if any) so auto-cache sees the same prefix on repeat calls.
            head_text = cacheable_prefix or ""
            content = _interleave_text_and_images(
                head_text + prompt, image_blocks, lambda s: {"type": "text", "text": s},
            )
        else:
            content = (cacheable_prefix or "") + prompt
```

Update the usage extraction at the bottom of `_generate_openai`:

```python
        cached = 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        usage = {
            "input_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
            "cached_tokens": cached,
        }
        return ModelResponse(text=text, usage=usage)
```

- [ ] **Step 6: Implement `cacheable_prefix` stub in `_generate_gemini`**

Update `_generate_gemini`'s signature to accept `cacheable_prefix`:

```python
    def _generate_gemini(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0, images=None,
                         cacheable_prefix: str | None = None):
```

Replace the existing `if images: ... else: contents = prompt` block with:

```python
        # TODO(gemini-cache): wire CachedContent.create/refresh/delete here.
        # For now we concatenate the cacheable head into the prompt so the
        # behavior is semantically correct even though we don't get a cache hit.
        effective_prompt = (cacheable_prefix or "") + prompt
        if images:
            image_blocks = _build_image_blocks_gemini(images)
            parts = _interleave_text_and_images(
                effective_prompt, image_blocks, lambda s: {"text": s},
            )
            contents = [{"role": "user", "parts": parts}]
        else:
            contents = effective_prompt
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_client_caching.py -v`
Expected: 5 passed.

Full suite:
Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add annotator/core/client.py tests/test_client_caching.py
git commit -m "client: add cacheable_prefix kwarg + per-provider plumbing"
```

---

## Task 2: Refactor `_build_role_prompt` to return `(head, tail)` (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py`
- Modify: `tests/test_benchmark_exchange_dynamic.py`

### Background

`_build_role_prompt` currently takes one string `transcript_so_far` and returns a single string. The refactor: take `transcript_prefix` (immutable per-scenario) and `extra` (growing per-round) separately, and return a `(cacheable_head, tail)` tuple. The head is the part that's safe to cache (unchanged across all rounds of one scenario); the tail is the dynamic part.

Both `run_exchange` and `run_exchanges_batch` need to track `extra` alongside the existing `running_transcript`. The simplest refactor: keep `running_transcript` for logging / `_append_turns` compatibility, but additionally maintain `extra` per scenario as the post-prefix portion.

### Steps

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_exchange_dynamic.py` at the END of the file:

```python
# ---------------------------------------------------------------------------
# _build_role_prompt cache-tuple tests
# ---------------------------------------------------------------------------

def test_build_role_prompt_returns_head_tail_tuple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    from benchmark.core.exchange import _build_role_prompt
    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello",
        extra="\nTurn 3. TUTOR: ok",
        student_context="Grade 5",
        prompt_version="v5",
    )
    assert isinstance(head, str) and isinstance(tail, str)
    assert "Grade 5" in head             # system + context is in the head
    assert "Turn 1." in head             # prefix is in the head
    assert "Turn 3." in tail             # extra is in the tail
    assert "Turn 3." not in head


def test_build_role_prompt_head_invariant_across_extras(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    from benchmark.core.exchange import _build_role_prompt
    head1, _ = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx", prompt_version="v5",
    )
    head2, _ = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok",
        student_context="ctx", prompt_version="v5",
    )
    # Head must be byte-identical -- this is what enables cache hits.
    assert head1 == head2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_build_role_prompt_returns_head_tail_tuple -v`
Expected: failure -- `_build_role_prompt` doesn't accept `transcript_prefix=` / `extra=` and doesn't return a tuple.

- [ ] **Step 3: Refactor `_build_role_prompt`**

Replace the existing `_build_role_prompt` in `benchmark/core/exchange.py`:

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
    """Build (cacheable_head, tail) for either tutor or student.

    head = system_prompt (with substitutions) + "Here is the conversation so far:\\n" + transcript_prefix
    tail = extra + "\\n\\n" + role_instruction

    The head is byte-identical across all rounds of one scenario, so it hits
    the prompt cache (Anthropic explicit / OpenAI automatic) on round 2+.

    When student_mode == "trait", scenario / trait_client / trait_model must
    be provided; the persona is resolved (cached per (conv_id, cut_turn)) and
    substituted into the {trait_persona} placeholder in the system prompt --
    which is in the head, so trait mode benefits from caching too.
    """
    if role == "TUTOR":
        system_prompt = _load_prompt(prompt_version, "tutor_system.txt")
        role_instruction = "Respond as the TUTOR. Give only your response, no labels or prefixes."
    else:
        if student_mode:
            student_file = f"students/{student_mode}.txt"
        else:
            student_file = "student_system.txt"
        system_prompt = _load_prompt(prompt_version, student_file)
        role_instruction = "Respond as the STUDENT. Give only your response, no labels or prefixes."

    system_prompt = system_prompt.replace("{student_context}", student_context)

    if role == "STUDENT" and student_mode == "trait":
        if scenario is None or trait_client is None or trait_model is None:
            raise ValueError(
                "_build_role_prompt: student_mode='trait' requires scenario, "
                "trait_client, and trait_model"
            )
        from benchmark.core.traits import get_or_generate_trait
        persona = get_or_generate_trait(
            scenario, prompt_version, trait_client, trait_model,
        )
        system_prompt = system_prompt.replace("{trait_persona}", persona)

    head = f"{system_prompt}\n\nHere is the conversation so far:\n\n{transcript_prefix}"
    tail = f"{extra}\n\n{role_instruction}"
    return head, tail
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_build_role_prompt_returns_head_tail_tuple tests/test_benchmark_exchange_dynamic.py::test_build_role_prompt_head_invariant_across_extras -v`
Expected: 2 passed.

Note: existing `_build_role_prompt` callers in `run_exchange` / `run_exchanges_batch` still expect a single string return value. Task 3 fixes those call sites. For now, the full suite will fail on the existing tests -- that's expected and intentional. Do NOT try to make them pass yet.

Run only the helper-test file to confirm the new tests pass:
Run: `pytest tests/test_benchmark_exchange_dynamic.py -v -k "head_tail or head_invariant or parse_tutor_tokens or check_end_token"`
Expected: all green for tests that don't touch run_exchange's call site.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: _build_role_prompt returns (cacheable_head, tail) tuple"
```

(Yes, this commit intentionally leaves `run_exchange` / `run_exchanges_batch` broken because they still call `_build_role_prompt` the old way. Task 3 fixes them. Each task should leave tests green in isolation, but for this two-task refactor it's cleaner to land them together if you prefer -- combine Tasks 2 and 3 into one commit if so. The plan keeps them separate for review clarity.)

---

## Task 3: Wire `cacheable_prefix` through the exchange loops

**Files:**
- Modify: `benchmark/core/exchange.py` (both `run_exchange` and `run_exchanges_batch`)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (extend existing tests so they pass with the new tuple return shape; add a cache-key-stability test)

### Steps

- [ ] **Step 1: Write the cache-key-stability test**

Append to `tests/test_benchmark_exchange_dynamic.py`:

```python
def test_run_exchange_sends_same_cacheable_prefix_each_round(tmp_path, monkeypatch):
    """Every tutor call within one scenario should pass the same cacheable_prefix.
    This is what makes the prompt cache hit on round 2+."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    seen_prefixes = []
    def _tutor_generate(prompt, **kwargs):
        seen_prefixes.append(kwargs.get("cacheable_prefix"))
        resp = MagicMock()
        resp.text = "next" if len(seen_prefixes) < 3 else "wrap [NEXT_PROBLEM]"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    tutor = MagicMock(); tutor.model = "stub"; tutor.generate = _tutor_generate

    student = _stub_client(["one", "two"])

    from benchmark.core.exchange import run_exchange
    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
    )
    assert ex.completed is True
    assert len(seen_prefixes) >= 3
    # All non-None cacheable_prefix values must be identical.
    non_none = [p for p in seen_prefixes if p is not None]
    assert len(non_none) == len(seen_prefixes), "every call should send a cacheable_prefix"
    assert len(set(non_none)) == 1, f"prefix changed across rounds: {set(non_none)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_run_exchange_sends_same_cacheable_prefix_each_round -v`
Expected: failure -- `run_exchange` doesn't pass `cacheable_prefix` to the client yet.

- [ ] **Step 3: Update `run_exchange` to use the tuple return and pass `cacheable_prefix`**

In `benchmark/core/exchange.py`, replace the existing `run_exchange` body so it tracks an `extra` string and passes `cacheable_prefix=head, prompt=tail` to each `client.generate` call. The full replacement:

```python
def run_exchange(
    scenario: Scenario,
    tutor_client: ModelClient,
    student_client: ModelClient,
    max_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    prompt_version: str,
    images: list[str] | None = None,
    student_mode: str | None = None,
    trait_client: ModelClient | None = None,
    trait_model: str | None = None,
) -> Exchange:
    """Sync mode multi-turn exchange.

    Both [END] and [NEXT_PROBLEM] terminate; recorded on Exchange.ended_via.
    Each tutor/student call passes scenario.transcript_prefix as cacheable_prefix
    so the static head hits the prompt cache on round 2+.
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    extra = ""
    next_turn_num = scenario.cut_turn + 1
    ended_via = ""

    while len(exchange.generated_turns) < max_turns:
        # --- Tutor turn ---
        head, tail = _build_role_prompt(
            "TUTOR", scenario.transcript_prefix, extra, scenario.student_context,
            prompt_version,
        )
        response = tutor_client.generate(
            tail, json_mode=False, max_tokens=tutor_max_tokens,
            images=images, cacheable_prefix=head,
        )
        _add_usage(exchange.tutor_usage, response.usage)

        text, ended, next_problem = _parse_tutor_tokens(response.text)
        messages = _split_messages(text)
        if not messages and not (ended or next_problem):
            messages = ["..."]
        if messages:
            extra, next_turn_num = _append_turns_to_extra(
                exchange, messages, "TUTOR", extra, next_turn_num,
            )

        if ended:
            ended_via = "END"
            break
        if next_problem:
            ended_via = "NEXT_PROBLEM"
            break
        if len(exchange.generated_turns) >= max_turns:
            ended_via = "MAX_TURNS"
            break

        # --- Student turn ---
        head, tail = _build_role_prompt(
            "STUDENT", scenario.transcript_prefix, extra, scenario.student_context,
            prompt_version, student_mode=student_mode,
            scenario=scenario, trait_client=trait_client, trait_model=trait_model,
        )
        response = student_client.generate(
            tail, json_mode=False, max_tokens=student_max_tokens,
            images=images, cacheable_prefix=head,
        )
        _add_usage(exchange.student_usage, response.usage)

        messages = _split_messages(response.text) or ["..."]
        extra, next_turn_num = _append_turns_to_extra(
            exchange, messages, "STUDENT", extra, next_turn_num,
        )

    if not ended_via:
        ended_via = "MAX_TURNS"

    exchange.completed = True
    exchange.ended_via = ended_via
    return exchange
```

This requires a new helper `_append_turns_to_extra` that mirrors the existing `_append_turns` but appends to `extra` instead of `running_transcript`. Add it just below `_append_turns`:

```python
def _append_turns_to_extra(
    exchange: Exchange,
    messages: list[str],
    role: str,
    extra: str,
    next_turn_num: int,
) -> tuple[str, int]:
    """Append messages as turns and grow the `extra` suffix.

    Used by the cache-aware exchange loop. The transcript prefix stays
    fixed at scenario.transcript_prefix; this only mutates the per-round
    growing portion.
    """
    for msg in messages:
        turn = {"turn_number": next_turn_num, "role": role, "text": msg}
        exchange.generated_turns.append(turn)
        extra += f"\nTurn {next_turn_num}. {role}: {msg}"
        next_turn_num += 1
    return extra, next_turn_num
```

- [ ] **Step 4: Update `run_exchanges_batch` the same way**

In `benchmark/core/exchange.py`, refactor `run_exchanges_batch` to track an `extra` dict (one string per scenario) and pass `cacheable_prefix=head, prompt=tail` on each batch entry. The full replacement:

```python
def run_exchanges_batch(
    scenarios: list[Scenario],
    tutor_client: ModelClient,
    student_client: ModelClient,
    max_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    poll_interval: int,
    save_callback: callable = None,
    prompt_version: str = "v1",
    images_by_scenario: dict[str, list[str]] | None = None,
    student_mode: str | None = None,
    trait_client: ModelClient | None = None,
    trait_model: str | None = None,
) -> dict[str, Exchange]:
    """Batch mode multi-turn exchanges.

    Per-scenario state tracks `extra` (growing suffix) separate from the
    static scenario.transcript_prefix; the head is passed as cacheable_prefix
    on every per-scenario batch entry.
    """
    exchanges = {}
    extras: dict[str, str] = {}
    next_turns = {}
    ended_via: dict[str, str] = {}

    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        extras[scenario.scenario_id] = ""
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1

    scenario_map = {s.scenario_id: s for s in scenarios}
    active_ids = list(scenario_map.keys())

    for round_num in range(math.ceil(max_turns / 2)):
        if not active_ids:
            break

        # --- Tutor batch ---
        logger.info("Round %d - tutor batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "TUTOR", scenario.transcript_prefix, extras[sid], scenario.student_context,
                prompt_version,
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            tutor_entries.append(
                build_batch_entry(sid, tail, json_mode=False, max_tokens=tutor_max_tokens,
                                  images=scenario_images, cacheable_prefix=head)
            )

        tutor_raw = run_batch(
            tutor_client, tutor_entries, json_mode=False,
            display_name=f"tutor_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = tutor_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("tutor failed for %s", sid[:50])
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.tutor_usage, result["usage"])

            text, ended, next_problem = _parse_tutor_tokens(result["text"])
            messages = _split_messages(text)
            if not messages and not (ended or next_problem):
                messages = ["..."]
            if messages:
                extras[sid], next_turns[sid] = _append_turns_to_extra(
                    exchange, messages, "TUTOR", extras[sid], next_turns[sid],
                )

            if ended:
                ended_via[sid] = "END"
                ended_this_round.append(sid)
                continue
            if next_problem:
                ended_via[sid] = "NEXT_PROBLEM"
                ended_this_round.append(sid)
                continue
            if len(exchange.generated_turns) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        # --- Student batch ---
        if not active_ids:
            if save_callback:
                for sid in scenario_map:
                    save_callback(sid, exchanges[sid])
            continue

        logger.info("Round %d - student batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        student_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "STUDENT", scenario.transcript_prefix, extras[sid], scenario.student_context,
                prompt_version, student_mode=student_mode,
                scenario=scenario, trait_client=trait_client, trait_model=trait_model,
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            student_entries.append(
                build_batch_entry(sid, tail, json_mode=False, max_tokens=student_max_tokens,
                                  images=scenario_images, cacheable_prefix=head)
            )

        student_raw = run_batch(
            student_client, student_entries, json_mode=False,
            display_name=f"student_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = student_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("student failed for %s", sid[:50])
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.student_usage, result["usage"])

            messages = _split_messages(result["text"]) or ["..."]
            extras[sid], next_turns[sid] = _append_turns_to_extra(
                exchange, messages, "STUDENT", extras[sid], next_turns[sid],
            )

            if len(exchange.generated_turns) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        if save_callback:
            for sid in scenario_map:
                save_callback(sid, exchanges[sid])

    for sid in scenario_map:
        exchanges[sid].completed = True
        exchanges[sid].ended_via = ended_via.get(sid, "MAX_TURNS")

    logger.info("Exchanges complete: %d scenarios", len(scenario_map))
    return exchanges
```

- [ ] **Step 5: Extend `build_batch_entry` to accept `cacheable_prefix`**

Look up `build_batch_entry` in `annotator/core/client.py`. Update its signature to accept a keyword-only `cacheable_prefix=None` and store it on the returned entry dict so `run_batch` (which submits the batch) can forward it when constructing per-entry requests.

Run: `grep -n "def build_batch_entry\|def run_batch\b" annotator/core/client.py`
to find the exact line numbers. The shape of the change: `build_batch_entry` accepts `cacheable_prefix` and tucks it inside the entry's request payload (the Anthropic batch API supports `cache_control` in the same way as the messages API). The batch-runner path inside `run_batch` constructs the message content the same way `_generate_anthropic` does -- when `cacheable_prefix` is present, build a structured-content message with the head block marked cacheable.

If touching `build_batch_entry` / `run_batch` is too invasive for this task, leave `cacheable_prefix` silently ignored in batch mode by deleting the `cacheable_prefix=head` arg from the `build_batch_entry` calls in the batch path above. Sync mode still benefits. Document this in the commit message.

Recommended path: implement batch caching now if `build_batch_entry` is small and clearly handles per-provider request shapes; defer if it's a tangle.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

If any existing tests in `test_benchmark_exchange_dynamic.py` still reference the old single-string `_build_role_prompt` signature or pass `transcript_so_far=`, update them to use the new `transcript_prefix=` + `extra=""` shape.

- [ ] **Step 7: Commit**

```bash
git add benchmark/core/exchange.py annotator/core/client.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: pass cacheable_prefix on every exchange call"
```

---

## Task 4: End-to-end smoke + cache-hit observation

**Files:** none modified -- verification only.

### Steps

- [ ] **Step 1: Run a 2-scenario sync smoke and observe cache stats**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version cache_smoke_2026_06_10 --scenario-mode human --max-scenarios 2 --mode sync
```

Watch the logs for the per-scenario exchange usage. Look at saved exchange files:

```bash
PYTHONIOENCODING=utf-8 python -c "
from annotator.core.storage import load_benchmark_result, list_benchmark_result_files
for f in list_benchmark_result_files('cache_smoke_2026_06_10', 'exchanges', 'anthropic'):
    ex = load_benchmark_result('cache_smoke_2026_06_10', 'exchanges', 'anthropic', f)
    t = ex.get('tutor_usage', {}) or {}
    s = ex.get('student_usage', {}) or {}
    print(f[-30:],
          'tutor in/out/cache_read/cache_create:', t.get('input_tokens',0), '/', t.get('output_tokens',0),
          '/', t.get('cache_read_input_tokens',0), '/', t.get('cache_creation_input_tokens',0),
          '| student in/out/cache_read/cache_create:', s.get('input_tokens',0), '/', s.get('output_tokens',0),
          '/', s.get('cache_read_input_tokens',0), '/', s.get('cache_creation_input_tokens',0))
"
```

Expected: `cache_creation_input_tokens` is non-zero on the first round of each scenario (the head was just written to cache); `cache_read_input_tokens` grows on subsequent rounds (cache hits). If both are zero, caching isn't engaging -- inspect the request shape sent to Anthropic.

- [ ] **Step 2: Estimate savings**

```bash
PYTHONIOENCODING=utf-8 python -c "
from annotator.core.storage import load_benchmark_result, list_benchmark_result_files
read = create = full = 0
for f in list_benchmark_result_files('cache_smoke_2026_06_10', 'exchanges', 'anthropic'):
    ex = load_benchmark_result('cache_smoke_2026_06_10', 'exchanges', 'anthropic', f)
    for u in [ex.get('tutor_usage', {}) or {}, ex.get('student_usage', {}) or {}]:
        read += u.get('cache_read_input_tokens', 0)
        create += u.get('cache_creation_input_tokens', 0)
        full += u.get('input_tokens', 0)
hit_pct = 100.0 * read / max(1, full)
print(f'cache read: {read:,} | created: {create:,} | total input: {full:,} | hit rate: {hit_pct:.1f}%')
"
```

Expected: hit rate >50% on a multi-round scenario.

- [ ] **Step 3: No commit on this task**

This is verification only. If cache stats look healthy, the feature is shipped.

---

## Self-Review

**Spec coverage:**
- `cacheable_prefix` kwarg on `ModelClient.generate` -- Task 1.
- Anthropic explicit `cache_control` on head block -- Task 1.
- OpenAI auto-cache via byte-identical concatenation -- Task 1.
- Gemini stub with TODO -- Task 1.
- Usage extraction (`cache_creation_input_tokens`, `cache_read_input_tokens`, `cached_tokens`) -- Task 1.
- `_build_role_prompt` returns `(head, tail)` -- Task 2.
- Exchange loops track `extra` and pass `cacheable_prefix=head` -- Task 3.
- Cache key stability test -- Task 3.
- Hit-rate observation -- Task 4.

All spec items mapped to tasks.

**Placeholder scan:** no TBDs in code blocks; all step bodies are concrete. Step 5 of Task 3 leaves the user a choice of whether to also wire batch-mode caching; both branches are explicit (do it now, or skip with `cacheable_prefix=head` deleted from batch calls).

**Type/name consistency:**
- `cacheable_prefix` parameter name matches across Tasks 1, 2, 3, 4.
- `_append_turns_to_extra` (new helper) defined in Task 3 Step 3 and used in both `run_exchange` and `run_exchanges_batch` in Task 3 Steps 3 + 4.
- `head, tail` tuple return shape from `_build_role_prompt` consistent across Tasks 2 and 3.
- Cache usage keys (`cache_creation_input_tokens`, `cache_read_input_tokens`, `cached_tokens`) used consistently in Task 1 implementation and Task 4 verification.
