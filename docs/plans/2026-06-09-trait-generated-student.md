# Trait-Generated Student Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `trait` student mode where a generator LLM reads only the transcript prefix and writes a per-scenario student persona; the synthetic student then embodies that persona. Cached across runs. Oracle-safe.

**Architecture:** New module `benchmark/core/traits.py` with `get_or_generate_trait(scenario, ...)`. New prompts `prompts/benchmark/v5/trait_generator.txt` and `prompts/benchmark/v5/students/trait.txt` (with `{trait_persona}` placeholder). `_build_role_prompt` gains optional `scenario`, `trait_client`, `trait_model` args used only when `student_mode == "trait"`.

**Tech Stack:** Python 3.11, pytest, existing `annotator.core.client.ModelClient`, existing storage backend for cache files.

**Spec:** [`docs/plans/specs/2026-06-09-trait-generated-student-design.md`](specs/2026-06-09-trait-generated-student-design.md)

---

## File Map

- **Create:** `prompts/benchmark/v5/trait_generator.txt` — generator system prompt.
- **Create:** `prompts/benchmark/v5/students/trait.txt` — trait-mode student prompt with `{trait_persona}` placeholder.
- **Create:** `benchmark/core/traits.py` — `get_or_generate_trait` + cache helpers.
- **Modify:** `benchmark/core/exchange.py` — extend `_build_role_prompt` signature; thread `scenario` + trait client/model through the student turn in both `run_exchange` and `run_exchanges_batch`.
- **Modify:** `benchmark/run.py` — build `trait_client` from the student profile when `student_mode == "trait"` and pass it through.
- **Create:** `tests/test_benchmark_traits.py` — unit tests for cache hit/miss, oracle-leak guard, integration with `_build_role_prompt`.

---

## Task 1: Add trait prompts to v5

**Files:**
- Create: `prompts/benchmark/v5/trait_generator.txt`
- Create: `prompts/benchmark/v5/students/trait.txt`

### Steps

- [ ] **Step 1: Write `trait_generator.txt`**

Create `prompts/benchmark/v5/trait_generator.txt` with this content:

```
You are an expert at observing K-12 students and characterizing their learning traits.

## Context
You have the following metadata about the student:
{student_context}

## Task
Read the conversation prefix below. Write a 2-3 paragraph student persona description that will be used to simulate the same student continuing the session.

Cover, where there is evidence in the transcript:
- Math skill level and conceptual gaps.
- Common mistakes / misconceptions you observe.
- Affect and emotional state (confident, frustrated, engaged, distracted, etc.).
- Attention patterns (focused, drifting, easily redirected, etc.).
- Learning style (asks questions, explains in own words, prefers examples, etc.).

Stay grounded in what's visible in the transcript. If the transcript is short or doesn't show a trait, say so rather than inventing.

## Conversation prefix
{transcript_prefix}

## Output
Write only the persona description, 2-3 paragraphs. No headings, no preamble.
```

- [ ] **Step 2: Write `students/trait.txt`**

Create `prompts/benchmark/v5/students/trait.txt` with this content:

```
You are role-playing as a specific K-12 student in a continuing tutoring session. Embody the student described below closely -- match how they reason, how they ask questions, the kinds of mistakes they make, their affect, and their attention.

## Student metadata
{student_context}

## Student persona
{trait_persona}

## Task
Continue the conversation naturally as this student. Respond to the tutor's most recent message. Stay in character. Use the same length, tone, and conceptual level the persona suggests.

To break up your response across multiple messages, use [NEW_MESSAGE] on its own line.
```

- [ ] **Step 3: Verify the files are present**

Run: `ls prompts/benchmark/v5/ prompts/benchmark/v5/students/`
Expected: `trait_generator.txt` next to `tutor_system.txt`; `trait.txt` alongside the other four student-mode files.

- [ ] **Step 4: Commit**

```bash
git add prompts/benchmark/v5/trait_generator.txt prompts/benchmark/v5/students/trait.txt
git commit -m "prompts: add v5 trait_generator + trait student prompts"
```

---

## Task 2: Add `traits.py` module (TDD)

**Files:**
- Create: `benchmark/core/traits.py`
- Create: `tests/test_benchmark_traits.py`

### Background

`get_or_generate_trait(scenario, prompt_version, model_client, model_name) -> str` returns the cached persona for `(scenario.conv_id, scenario.cut_turn)`, or generates one by calling `model_client.generate(prompt, ...)` with `prompts/benchmark/{prompt_version}/trait_generator.txt`. Cache lives at `results/benchmark/_trait_cache/<conv_id>__<cut_turn>.json` and is accessed via the storage backend (so it works in both local and s3 modes).

The cache file shape:

```json
{
  "conv_id": "...",
  "cut_turn": 22,
  "persona": "<persona text>",
  "generator_model": "claude-opus-4-8",
  "prompt_version": "v5",
  "prefix_length_chars": 12345,
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
  "generated_at": "2026-06-09T12:34:56"
}
```

The cache key includes both `conv_id` and `cut_turn` so different cuts of the same conversation get distinct personas.

### Steps

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_traits.py`:

```python
"""Tests for benchmark.core.traits: trait generator + per-scenario cache."""
from unittest.mock import MagicMock, patch

import pytest

from benchmark.core.traits import (
    get_or_generate_trait, _trait_cache_filename,
)
from benchmark.core.scenarios import Scenario


def _stub_response(text="A focused 5th grader who confuses long division steps."):
    resp = MagicMock()
    resp.text = text
    resp.usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    return resp


def _scenario(conv_id="conv1", cut_turn=10, prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello"):
    return Scenario(
        scenario_id=f"{conv_id}__hum_x_y",
        conv_id=conv_id,
        cut_turn=cut_turn,
        transcript_prefix=prefix,
        student_context="Grade 5, math",
        last_student_message="hello",
        mode="human",
        detection={"turn_start": 5, "turn_end": 12,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def test_trait_cache_filename_uses_conv_id_and_cut_turn():
    fname = _trait_cache_filename(_scenario(conv_id="abc123", cut_turn=42))
    assert "abc123" in fname
    assert "42" in fname
    assert fname.endswith(".json")


def test_cache_miss_invokes_client_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    # Clear any cached backend so STORAGE_ROOT takes effect.
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    monkeypatch.setattr(
        "benchmark.core.traits._load_prompt",
        lambda version, fname: "GEN {student_context} | {transcript_prefix}",
    )

    client = MagicMock()
    client.generate.return_value = _stub_response("persona-A")

    s = _scenario(conv_id="conv-aa", cut_turn=7)
    persona = get_or_generate_trait(s, prompt_version="v5",
                                    model_client=client, model_name="m1")
    assert persona == "persona-A"
    assert client.generate.called

    # File written to the trait cache dir.
    cache_dir = tmp_path / "results" / "benchmark" / "_trait_cache"
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    import json
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["persona"] == "persona-A"
    assert saved["conv_id"] == "conv-aa"
    assert saved["cut_turn"] == 7
    assert saved["generator_model"] == "m1"
    assert saved["prompt_version"] == "v5"


def test_cache_hit_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    monkeypatch.setattr(
        "benchmark.core.traits._load_prompt",
        lambda version, fname: "GEN {student_context} | {transcript_prefix}",
    )

    s = _scenario(conv_id="conv-bb", cut_turn=3)

    # Prime the cache by calling once with one client.
    primer = MagicMock()
    primer.generate.return_value = _stub_response("primed-persona")
    persona1 = get_or_generate_trait(s, prompt_version="v5",
                                     model_client=primer, model_name="m1")
    assert persona1 == "primed-persona"

    # Second client: must NOT be called.
    second = MagicMock()
    second.generate.side_effect = AssertionError("client should not be called on cache hit")
    persona2 = get_or_generate_trait(s, prompt_version="v5",
                                     model_client=second, model_name="m1")
    assert persona2 == "primed-persona"
    assert not second.generate.called


def test_generator_prompt_contains_only_prefix_no_post_cut_text(tmp_path, monkeypatch):
    """Oracle-leak guard: generator must only see transcript_prefix."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    monkeypatch.setattr(
        "benchmark.core.traits._load_prompt",
        lambda version, fname: "PREFIX-BLOCK:\n{transcript_prefix}\nEND-PREFIX",
    )

    captured = {}
    def _record_generate(prompt, **_kw):
        captured["prompt"] = prompt
        return _stub_response("ok")

    client = MagicMock()
    client.generate = _record_generate

    s = _scenario(conv_id="conv-cc", cut_turn=5,
                  prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok")
    # Sanity: prefix does not include the secret post-cut sentinel.
    SECRET = "POST_CUT_SECRET_TURN_42_TEXT"
    assert SECRET not in s.transcript_prefix

    get_or_generate_trait(s, prompt_version="v5",
                          model_client=client, model_name="m1")

    assert "Turn 1." in captured["prompt"]
    assert "Turn 2." in captured["prompt"]
    assert "Turn 3." in captured["prompt"]
    # The post-cut secret must not appear, since trait gen only reads
    # scenario.transcript_prefix (never the full conversation object).
    assert SECRET not in captured["prompt"]


def test_persona_caches_per_cut_turn(tmp_path, monkeypatch):
    """Same conv, different cuts -> different cache entries."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    monkeypatch.setattr(
        "benchmark.core.traits._load_prompt",
        lambda version, fname: "GEN {transcript_prefix}",
    )

    client = MagicMock()
    client.generate.side_effect = [_stub_response("p-cut3"), _stub_response("p-cut9")]

    s_a = _scenario(conv_id="conv-dd", cut_turn=3)
    s_b = _scenario(conv_id="conv-dd", cut_turn=9)
    p1 = get_or_generate_trait(s_a, prompt_version="v5",
                               model_client=client, model_name="m1")
    p2 = get_or_generate_trait(s_b, prompt_version="v5",
                               model_client=client, model_name="m1")
    assert p1 == "p-cut3"
    assert p2 == "p-cut9"
    cache_dir = tmp_path / "results" / "benchmark" / "_trait_cache"
    assert len(list(cache_dir.glob("*.json"))) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_traits.py -v`
Expected: `ImportError: cannot import name 'get_or_generate_trait'`.

- [ ] **Step 3: Implement `benchmark/core/traits.py`**

Create `benchmark/core/traits.py`:

```python
"""Per-scenario trait generator + cache for trait-mode synthetic students.

The generator reads ONLY the scenario's transcript_prefix and student_context.
It never touches the full conversation object -- this is what keeps trait mode
from being an oracle on the real student's post-cut turns.
"""
import datetime
import logging

from annotator.core.storage import (
    _get_backend, get_benchmark_result_path,
)
from benchmark.core.scenarios import Scenario
from benchmark.core.exchange import _load_prompt

logger = logging.getLogger(__name__)


_TRAIT_CACHE_DIR_NAME = "_trait_cache"


def _trait_cache_filename(scenario: Scenario) -> str:
    """Cache file relpath: <conv_id>__<cut_turn>.json"""
    safe_conv = scenario.conv_id.replace("/", "_")
    return f"{safe_conv}__{scenario.cut_turn}.json"


def _trait_cache_relpath(scenario: Scenario) -> str:
    """Storage-backend relative path for the cache file."""
    return f"results/benchmark/{_TRAIT_CACHE_DIR_NAME}/{_trait_cache_filename(scenario)}"


def _load_cached_persona(scenario: Scenario) -> str | None:
    be = _get_backend()
    data = be.read_json(_trait_cache_relpath(scenario))
    if data and isinstance(data, dict) and "persona" in data:
        return data["persona"]
    return None


def _save_cached_persona(scenario: Scenario, persona: str, *,
                         generator_model: str, prompt_version: str,
                         usage: dict) -> None:
    payload = {
        "conv_id": scenario.conv_id,
        "cut_turn": scenario.cut_turn,
        "persona": persona,
        "generator_model": generator_model,
        "prompt_version": prompt_version,
        "prefix_length_chars": len(scenario.transcript_prefix),
        "usage": usage,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    be = _get_backend()
    be.write_json(_trait_cache_relpath(scenario), payload)


def get_or_generate_trait(
    scenario: Scenario,
    prompt_version: str,
    model_client,
    model_name: str,
) -> str:
    """Return cached persona for (conv_id, cut_turn), else generate via the LLM.

    The generator sees only scenario.transcript_prefix and scenario.student_context.
    It NEVER reads the full conversation object -- oracle-safe by construction.
    """
    cached = _load_cached_persona(scenario)
    if cached is not None:
        logger.info("trait cache hit: %s cut=%d", scenario.conv_id[:24], scenario.cut_turn)
        return cached

    template = _load_prompt(prompt_version, "trait_generator.txt")
    prompt = (
        template
        .replace("{student_context}", scenario.student_context or "")
        .replace("{transcript_prefix}", scenario.transcript_prefix or "")
    )

    response = model_client.generate(prompt, json_mode=False, max_tokens=1024)
    persona = (response.text or "").strip()
    usage = response.usage or {}

    _save_cached_persona(
        scenario, persona,
        generator_model=model_name,
        prompt_version=prompt_version,
        usage=usage,
    )
    logger.info("trait generated: %s cut=%d (%d chars)",
                scenario.conv_id[:24], scenario.cut_turn, len(persona))
    return persona
```

The `get_benchmark_result_path` import is unused — leave it out if unused; the backend `read_json` / `write_json` accept a relpath directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_traits.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/traits.py tests/test_benchmark_traits.py
git commit -m "benchmark: add trait generator + per-scenario cache"
```

---

## Task 3: Wire `student_mode == "trait"` into `_build_role_prompt`

**Files:**
- Modify: `benchmark/core/exchange.py` (extend `_build_role_prompt` signature; trait persona substitution)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (add an integration test)

### Steps

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark_exchange_dynamic.py` (after the existing tests):

```python
def test_build_role_prompt_trait_mode_substitutes_persona(tmp_path, monkeypatch):
    """When student_mode='trait', _build_role_prompt resolves a persona via
    get_or_generate_trait and substitutes {trait_persona}."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: (
            "STUDENT-TRAIT-PROMPT context={student_context} persona={trait_persona}"
            if "trait" in fname else "OTHER {student_context}"
        ),
    )
    monkeypatch.setattr(
        "benchmark.core.traits._load_prompt",
        lambda version, fname: "GEN {transcript_prefix}",
    )

    # Patch the trait client so we don't actually hit an API.
    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "A determined 5th grader who skips multiplication facts."
    fake_response.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    fake_client.generate.return_value = fake_response

    from benchmark.core.exchange import _build_role_prompt
    scenario = _make_scenario()
    out = _build_role_prompt(
        "STUDENT",
        transcript_so_far=scenario.transcript_prefix,
        student_context=scenario.student_context,
        prompt_version="v5",
        student_mode="trait",
        scenario=scenario,
        trait_client=fake_client,
        trait_model="m1",
    )

    assert "A determined 5th grader" in out
    assert "{trait_persona}" not in out
    assert scenario.student_context in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_build_role_prompt_trait_mode_substitutes_persona -v`
Expected: TypeError on `scenario=` / `trait_client=` / `trait_model=` kwargs.

- [ ] **Step 3: Extend `_build_role_prompt`**

In `benchmark/core/exchange.py`, replace the existing `_build_role_prompt` function with:

```python
def _build_role_prompt(
    role: str, transcript_so_far: str, student_context: str,
    prompt_version: str = "v1",
    student_mode: str | None = None,
    scenario=None,
    trait_client=None,
    trait_model: str | None = None,
) -> str:
    """Build a prompt for either tutor or student.

    When role == "STUDENT" and student_mode is set, loads
    students/{student_mode}.txt under the prompt version. Otherwise falls
    back to the legacy single-file student_system.txt so older versions
    (v1) keep working without a students/ subfolder.

    When student_mode == "trait", `scenario`, `trait_client`, and `trait_model`
    must be provided. The persona is resolved via traits.get_or_generate_trait
    (cached per (conv_id, cut_turn)) and substituted into the {trait_persona}
    placeholder.
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

    return f"""{system_prompt}

Here is the conversation so far:

{transcript_so_far}

{role_instruction}"""
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: all pass, including the new trait test.

Run: `pytest tests/test_benchmark_traits.py -v`
Expected: still passes.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: _build_role_prompt substitutes {trait_persona} for trait mode"
```

---

## Task 4: Thread trait client through `run_exchange` and `run_exchanges_batch`

**Files:**
- Modify: `benchmark/core/exchange.py` (both functions accept + pass `trait_client`, `trait_model`)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (one round-trip test through `run_exchange`)

### Steps

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark_exchange_dynamic.py`:

```python
def test_run_exchange_with_trait_mode(tmp_path, monkeypatch):
    """run_exchange with student_mode='trait' should resolve a persona via
    trait_client and produce student turns that include the persona text in
    the student prompt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    def _stub_loader(version, fname):
        if "trait_generator" in fname:
            return "GEN {transcript_prefix}"
        if "students/trait" in fname:
            return "STUDENT context={student_context} persona={trait_persona}"
        if fname == "tutor_system.txt":
            return "TUTOR {student_context}"
        return "OTHER"

    monkeypatch.setattr("benchmark.core.exchange._load_prompt", _stub_loader)
    monkeypatch.setattr("benchmark.core.traits._load_prompt", _stub_loader)

    from unittest.mock import MagicMock
    fake_response = MagicMock()
    fake_response.text = "calm but distracted 5th grader"
    fake_response.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    trait_client = MagicMock()
    trait_client.generate.return_value = fake_response

    # The student client records the prompts it sees so we can assert the
    # persona was substituted.
    student_prompts = []
    def _student_generate(prompt, **_kw):
        student_prompts.append(prompt)
        resp = MagicMock()
        resp.text = "I'll try the next step"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    student_client = MagicMock()
    student_client.model = "stub-student"
    student_client.generate = _student_generate

    tutor = _stub_client(["Try this problem.", "Great, [NEXT_PROBLEM]"])

    from benchmark.core.exchange import run_exchange
    scenario = _make_scenario()
    ex = run_exchange(
        scenario=scenario,
        tutor_client=tutor,
        student_client=student_client,
        max_turns=10,
        tutor_max_tokens=128,
        student_max_tokens=128,
        prompt_version="v5",
        student_mode="trait",
        trait_client=trait_client,
        trait_model="stub-trait",
    )

    assert ex.completed is True
    assert trait_client.generate.called
    # The student's system prompt for at least one turn should include the
    # generated persona text.
    assert any("calm but distracted" in p for p in student_prompts), (
        f"persona not found in any student prompt; saw {len(student_prompts)} prompt(s)"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_run_exchange_with_trait_mode -v`
Expected: failure -- `run_exchange` doesn't accept `trait_client=` / `trait_model=`.

- [ ] **Step 3: Extend `run_exchange` and `run_exchanges_batch` signatures**

In `benchmark/core/exchange.py`, update `run_exchange`'s signature and student-turn call:

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
```

Inside the student turn (where `_build_role_prompt("STUDENT", ...)` is called), add the new kwargs:

```python
        prompt = _build_role_prompt(
            "STUDENT", running_transcript, scenario.student_context,
            prompt_version, student_mode=student_mode,
            scenario=scenario, trait_client=trait_client, trait_model=trait_model,
        )
```

Do the same change in `run_exchanges_batch`:

Signature:

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
```

Student-batch loop (where `_build_role_prompt("STUDENT", ...)` is called):

```python
            prompt = _build_role_prompt(
                "STUDENT", transcripts_buf[sid], scenario.student_context,
                prompt_version, student_mode=student_mode,
                scenario=scenario, trait_client=trait_client, trait_model=trait_model,
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: all pass (including the new trait round-trip test).

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: thread trait_client/trait_model through run_exchange{s_batch}"
```

---

## Task 5: Wire `trait_client` in `benchmark/run.py`

**Files:**
- Modify: `benchmark/run.py` (build `trait_client` from student profile when `student_mode == "trait"` and pass through)

### Steps

- [ ] **Step 1: Inspect current student-client construction**

Run: `grep -n "student_client\|student_mode\|run_exchange" benchmark/run.py | head -20`
Expected: see where `student_client = ModelClient(student_cfg["model"])` is built and the two `run_exchange*` call sites.

- [ ] **Step 2: Build trait client + pass through**

In `benchmark/run.py`, find the block where the student profile is loaded and the `student_client` is constructed. Just after `student_client = ModelClient(student_cfg["model"])`, add:

```python
    trait_client = None
    trait_model = None
    if student_mode == "trait":
        trait_client = student_client       # reuse student profile per spec
        trait_model = student_cfg["model"]
```

Then in both `run_exchanges_batch(...)` and `run_exchange(...)` call sites, add the two kwargs at the end of the call:

```python
                    student_mode=student_mode,
                    trait_client=trait_client,
                    trait_model=trait_model,
                )
```

(Add to both places — the batch branch around line 213 and the sync branch around line 228.)

- [ ] **Step 3: Verify import path / no syntax errors**

Run: `python -c "import benchmark.run"`
Expected: clean import, no error.

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py
git commit -m "benchmark: wire trait_client through run.py when student_mode='trait'"
```

---

## Task 6: End-to-end smoke (manual, no commit)

**Files:** none.

### Steps

- [ ] **Step 1: Run a tiny smoke**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version trait_smoke_2026_06_09 --scenario-mode human --max-scenarios 2 --mode sync
```

Override the student mode by editing `config.yaml` temporarily to set
`benchmark.student.mode: trait` before running, OR add `--student-mode trait`
if the CLI supports it (check `benchmark/run.py` — if it doesn't, edit config).

Expected logs:
- "trait generated: <conv_id_prefix> cut=<N> (<chars> chars)" for each scenario on first encounter.
- Subsequent student turns within the same scenario should NOT re-log generation (in-process reuse is fine; cache-file reuse only matters across runs).

- [ ] **Step 2: Inspect the cache files**

```bash
ls results/benchmark/_trait_cache/
PYTHONIOENCODING=utf-8 python -c "
import json, os
for f in os.listdir('results/benchmark/_trait_cache'):
    d = json.load(open('results/benchmark/_trait_cache/' + f, encoding='utf-8'))
    print(f, '|', d.get('persona','')[:100].encode('ascii','replace').decode())
"
```
Expected: one JSON file per scenario; persona text is a short coherent paragraph.

- [ ] **Step 3: Re-run the same smoke**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version trait_smoke_2026_06_09_b --scenario-mode human --max-scenarios 2 --mode sync
```
(Same student_mode setting.) Expected: logs say "trait cache hit" for both scenarios; no new trait API calls.

- [ ] **Step 4: Reset config**

If you edited `config.yaml` to switch student.mode to `trait`, restore it to `imitate_example`:

```yaml
  student:
    profile: anthropic
    mode: imitate_example
```

(No commit on this task. The smoke is just verification.)

---

## Self-Review

**Spec coverage:**
- New mode name `trait` — Task 1 (prompt) + Task 2 (module reads `students/trait.txt` via `_build_role_prompt`).
- New prompts `trait_generator.txt` + `students/trait.txt` with `{trait_persona}` placeholder — Task 1.
- New module `traits.py` with `get_or_generate_trait` — Task 2.
- Per-scenario persistent cache at `results/benchmark/_trait_cache/<conv_id>__<cut_turn>.json` — Task 2 (filename helper + storage backend write).
- Oracle-leak guard — Task 2 (test + implementation only reads `scenario.transcript_prefix` and `scenario.student_context`).
- `_build_role_prompt` integration — Task 3.
- Reuse student profile's model — Task 5 (`trait_client = student_client`).
- Pass-through in `run_exchange` and `run_exchanges_batch` — Task 4.
- Tests for cache hit / cache miss / oracle leak / per-cut-turn / round-trip integration — Tasks 2, 3, 4.
- Smoke verification — Task 6.

All spec items covered.

**Placeholder scan:** no TBDs or hand-waves; all code blocks are concrete.

**Type/name consistency:**
- `get_or_generate_trait`, `_trait_cache_filename`, `_trait_cache_relpath`, `_load_cached_persona`, `_save_cached_persona` consistent across Task 2 + Task 3.
- `trait_client` and `trait_model` parameter names match across Tasks 3, 4, 5.
- `{trait_persona}` placeholder in `students/trait.txt` matches the `.replace("{trait_persona}", persona)` call in `_build_role_prompt`.
- Cache path uses `results/benchmark/_trait_cache/` consistently in Task 2 implementation and Task 6 inspection.
