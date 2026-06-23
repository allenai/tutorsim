# Oracle Tutor Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `tutor.mode: oracle` benchmark mode. When set, the AI tutor sees the post-cut real human turns as a reference and is instructed to mimic the real tutor as closely as possible. Functions as a ceiling cell for the tutor side.

**Architecture:** New prompt `prompts/benchmark/v5/tutors/oracle.txt` with `{reference_transcript}` placeholder. `_build_role_prompt` gains `tutor_mode` + `reference_transcript` kwargs (parallel to student trait wiring). `run_exchange` / `run_exchanges_batch` gain `tutor_mode` + `transcripts` kwargs and compute the reference once per scenario. `benchmark/run.py` reads `config["tutor"]["mode"]` and threads everything through. Reference lives in the cacheable head, so prompt caching still hits.

**Tech Stack:** Python 3.11, pytest, existing benchmark exchange + caching infrastructure.

**Spec:** [`docs/plans/specs/2026-06-10-oracle-tutor-mode-design.md`](specs/2026-06-10-oracle-tutor-mode-design.md)

---

## File Map

- **Create:** `prompts/benchmark/v5/tutors/oracle.txt` — oracle tutor prompt.
- **Modify:** `benchmark/core/exchange.py` — add `_build_reference_transcript` helper; extend `_build_role_prompt` with `tutor_mode` + `reference_transcript` kwargs; thread through `run_exchange` and `run_exchanges_batch`.
- **Modify:** `benchmark/run.py` — read `config["tutor"]["mode"]`; load transcripts unconditionally when oracle; pass through to both call sites; raise clear error if oracle + transcripts missing.
- **Modify:** `config.yaml` — add `benchmark.tutor.mode: null` default.
- **Create:** `tests/test_benchmark_oracle_tutor.py` — unit + integration tests for the new mode.

---

## Task 1: Add v5 oracle tutor prompt

**Files:**
- Create: `prompts/benchmark/v5/tutors/oracle.txt`

### Steps

- [ ] **Step 1: Create the prompt directory + file**

```bash
mkdir -p prompts/benchmark/v5/tutors
```

Create `prompts/benchmark/v5/tutors/oracle.txt` with EXACTLY this content (preserve formatting, end with newline):

```
You are an online tutor in a live tutoring session with a K-12 student.

## Goal
The real conversation continued past this point in the actual session. Your task is to continue the conversation as the tutor, matching the real tutor's style, strategy, length, register, and pedagogical moves as closely as possible. You don't need to copy them verbatim, but stay faithful to their approach.

## Context
You have the following metadata about the student:
{student_context}

## What the real tutor did from this point on
{reference_transcript}

## Sending multiple messages
To break up your response over multiple messages, use [NEW_MESSAGE].
This will separate your response into multiple lines / messages to the student.

## Moving on
When you judge the student ready to move on to the next problem,
return [NEXT_PROBLEM]. Our system will display the next problem.
You do not have to generate the next problem yourself,
just return [NEXT_PROBLEM].
```

- [ ] **Step 2: Verify file is present**

Run: `ls prompts/benchmark/v5/tutors/`
Expected: `oracle.txt` listed.

- [ ] **Step 3: Commit**

```bash
git add prompts/benchmark/v5/tutors/oracle.txt
git commit -m "prompts: add v5 oracle tutor prompt"
```

---

## Task 2: Add `_build_reference_transcript` helper + `tutor_mode` wiring in `_build_role_prompt` (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py`
- Create: `tests/test_benchmark_oracle_tutor.py`

### Background

`_build_reference_transcript(conversation, cut_turn)` returns a string with the post-cut real human turns formatted exactly like the rest of the transcript (`Turn N. ROLE: text`), joined with newlines. Returns empty string if no post-cut turns exist.

`_build_role_prompt` gains:
- `tutor_mode: str | None = None`
- `reference_transcript: str | None = None`

When `role == "TUTOR"` and `tutor_mode` is set, load `tutors/{tutor_mode}.txt` instead of `tutor_system.txt`, and substitute `{reference_transcript}` (raise `ValueError` if it's None).

Reference goes in the head (cacheable). Tail is unchanged.

### Steps

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_oracle_tutor.py`:

```python
"""Tests for oracle tutor mode."""
from unittest.mock import MagicMock

import pytest

from benchmark.core.exchange import (
    _build_role_prompt, _build_reference_transcript, run_exchange,
)
from benchmark.core.scenarios import Scenario


def _conv_with_turns(num_turns=10):
    return {
        "conversation_id": "conv1",
        "turns": [
            {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
             "text": f"real-turn-{n}"} for n in range(1, num_turns + 1)
        ],
    }


def _scenario(conv_id="conv1", cut_turn=4):
    return Scenario(
        scenario_id="s1",
        conv_id=conv_id,
        cut_turn=cut_turn,
        transcript_prefix="Turn 1. TUTOR: real-turn-1\nTurn 2. STUDENT: real-turn-2\nTurn 3. TUTOR: real-turn-3\nTurn 4. STUDENT: real-turn-4",
        student_context="Grade 5",
        last_student_message="real-turn-4",
        mode="human",
        detection={"turn_start": 2, "turn_end": 5,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def _stub_client(replies):
    client = MagicMock()
    client.model = "stub"
    it = iter(replies)
    def _gen(*args, **kwargs):
        resp = MagicMock()
        resp.text = next(it)
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    client.generate = _gen
    return client


def test_build_reference_transcript_returns_post_cut_turns_only():
    conv = _conv_with_turns(num_turns=8)
    ref = _build_reference_transcript(conv, cut_turn=4)
    # Pre-cut turns must NOT appear:
    assert "Turn 1." not in ref
    assert "Turn 4." not in ref
    # Post-cut turns must appear in order:
    assert "Turn 5." in ref
    assert "Turn 6." in ref
    assert "Turn 7." in ref
    assert "Turn 8." in ref
    # Format matches the prefix format ("Turn N. ROLE: text"):
    assert "Turn 5. TUTOR: real-turn-5" in ref


def test_build_reference_transcript_empty_when_cut_is_last_turn():
    conv = _conv_with_turns(num_turns=4)
    ref = _build_reference_transcript(conv, cut_turn=4)
    assert ref == ""


def test_build_role_prompt_oracle_mode_substitutes_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: (
            "ORACLE ctx={student_context} ref={reference_transcript}"
            if "tutors/oracle" in fname
            else "DEFAULT-TUTOR {student_context}"
        ),
    )
    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx5",
        prompt_version="v5",
        tutor_mode="oracle",
        reference_transcript="Turn 2. STUDENT: please help",
    )
    assert "ORACLE ctx=ctx5" in head
    assert "ref=Turn 2. STUDENT: please help" in head
    # Reference must live in the head (cacheable), not the tail:
    assert "Turn 2. STUDENT: please help" not in tail


def test_build_role_prompt_oracle_without_reference_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "ORACLE {student_context} {reference_transcript}",
    )
    with pytest.raises(ValueError, match="reference_transcript"):
        _build_role_prompt(
            "TUTOR",
            transcript_prefix="Turn 1. TUTOR: hi",
            extra="",
            student_context="ctx",
            prompt_version="v5",
            tutor_mode="oracle",
            reference_transcript=None,
        )


def test_build_role_prompt_tutor_mode_unset_uses_default_prompt(tmp_path, monkeypatch):
    """Back-compat: tutor_mode=None loads the legacy tutor_system.txt."""
    monkeypatch.chdir(tmp_path)
    loaded = []
    def _loader(version, fname):
        loaded.append(fname)
        return "DEFAULT {student_context}"
    monkeypatch.setattr("benchmark.core.exchange._load_prompt", _loader)

    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx",
        prompt_version="v5",
        # tutor_mode left unset
    )
    assert any(f == "tutor_system.txt" for f in loaded)
    assert not any("tutors/" in f for f in loaded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_oracle_tutor.py -v`
Expected: ImportError on `_build_reference_transcript`, plus failures on the `tutor_mode=` / `reference_transcript=` kwargs.

- [ ] **Step 3: Add `_build_reference_transcript` helper**

In `benchmark/core/exchange.py`, near the existing transcript-formatting helpers (`_format_prefix` lives in `scenarios.py`, so add this new helper next to `_append_turns_to_extra` in `exchange.py`), add:

```python
def _build_reference_transcript(conversation: dict, cut_turn: int) -> str:
    """Format the post-cut real human turns from a full conversation.

    Returns a newline-joined string of `Turn N. ROLE: text` lines for every
    turn whose turn_number > cut_turn. Empty string if no post-cut turns.

    Used by oracle tutor mode -- the reference shown to the AI so it can
    mimic the real tutor's continuation.
    """
    lines = []
    for turn in conversation.get("turns", []):
        n = turn.get("turn_number")
        if n is None or n <= cut_turn:
            continue
        role = turn.get("role", "")
        text = turn.get("text", "")
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)
```

- [ ] **Step 4: Extend `_build_role_prompt` to handle `tutor_mode`**

Replace the existing `_build_role_prompt` in `benchmark/core/exchange.py` with:

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
    tutor_mode: str | None = None,
    reference_transcript: str | None = None,
) -> tuple[str, str]:
    """Build (cacheable_head, tail) for either tutor or student.

    head = system_prompt (with substitutions) + "Here is the conversation so far:\\n" + transcript_prefix
    tail = extra + "\\n\\n" + role_instruction

    The head is byte-identical across all rounds of one scenario, so it hits
    the prompt cache (Anthropic explicit / OpenAI automatic) on round 2+.

    Tutor mode: when tutor_mode is set and role=="TUTOR", loads
    tutors/{tutor_mode}.txt instead of tutor_system.txt and substitutes
    {reference_transcript} (which must be supplied).

    Student trait mode: see existing docstring.
    """
    if role == "TUTOR":
        if tutor_mode:
            system_prompt = _load_prompt(prompt_version, f"tutors/{tutor_mode}.txt")
        else:
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

    if role == "TUTOR" and tutor_mode:
        if reference_transcript is None:
            raise ValueError(
                f"_build_role_prompt: tutor_mode={tutor_mode!r} requires "
                "reference_transcript"
            )
        system_prompt = system_prompt.replace("{reference_transcript}", reference_transcript)

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

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_benchmark_oracle_tutor.py -v`
Expected: 5 passed (or whatever the test count is — all green).

Full suite:
Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_oracle_tutor.py
git commit -m "benchmark: _build_role_prompt supports tutor_mode + reference_transcript"
```

---

## Task 3: Thread `tutor_mode` + `transcripts` through `run_exchange` and `run_exchanges_batch` (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py`
- Modify: `tests/test_benchmark_oracle_tutor.py`

### Steps

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_benchmark_oracle_tutor.py`:

```python
def test_run_exchange_oracle_mode_passes_reference_in_head(tmp_path, monkeypatch):
    """run_exchange with tutor_mode='oracle' must put the post-cut reference
    in the cacheable_prefix passed to the tutor client."""
    monkeypatch.chdir(tmp_path)

    def _loader(version, fname):
        if "tutors/oracle" in fname:
            return "ORACLE {student_context} REF={reference_transcript}"
        return "DEFAULT {student_context}"
    monkeypatch.setattr("benchmark.core.exchange._load_prompt", _loader)

    seen_prefixes = []
    def _tutor_generate(prompt, **kwargs):
        seen_prefixes.append(kwargs.get("cacheable_prefix"))
        resp = MagicMock()
        resp.text = "Wrap up. [NEXT_PROBLEM]"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    tutor = MagicMock(); tutor.model = "stub"; tutor.generate = _tutor_generate
    student = _stub_client([])

    scenario = _scenario(conv_id="conv1", cut_turn=4)
    transcripts = {"conv1": _conv_with_turns(num_turns=8)}

    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
        tutor_mode="oracle",
        transcripts=transcripts,
    )
    assert ex.completed is True
    # cacheable_prefix from round 1 must contain the post-cut real turns.
    assert any("Turn 5. TUTOR: real-turn-5" in (p or "") for p in seen_prefixes)
    assert any("Turn 8. STUDENT: real-turn-8" in (p or "") for p in seen_prefixes)


def test_run_exchange_oracle_without_transcripts_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "X {student_context} {reference_transcript}",
    )
    tutor = _stub_client(["Hi"])
    student = _stub_client([])

    with pytest.raises(ValueError, match="oracle"):
        run_exchange(
            scenario=_scenario(), tutor_client=tutor, student_client=student,
            max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
            prompt_version="v5",
            tutor_mode="oracle",
            transcripts=None,
        )


def test_run_exchange_tutor_mode_none_ignores_transcripts(tmp_path, monkeypatch):
    """Legacy back-compat: when tutor_mode is None, no reference is loaded
    even if transcripts is passed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "DEFAULT {student_context}",
    )
    tutor = _stub_client(["Done. [NEXT_PROBLEM]"])
    student = _stub_client([])

    ex = run_exchange(
        scenario=_scenario(), tutor_client=tutor, student_client=student,
        max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
        # tutor_mode unset
        transcripts={"conv1": _conv_with_turns(num_turns=8)},
    )
    assert ex.completed is True
    assert ex.ended_via == "NEXT_PROBLEM"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_oracle_tutor.py -v -k "oracle_mode or oracle_without or tutor_mode_none"`
Expected: failures — `run_exchange` doesn't accept `tutor_mode=` / `transcripts=` yet.

- [ ] **Step 3: Extend `run_exchange` signature and body**

Replace the existing `run_exchange` in `benchmark/core/exchange.py` with:

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
    tutor_mode: str | None = None,
    transcripts: dict[str, dict] | None = None,
) -> Exchange:
    """Sync mode multi-turn exchange.

    Both [END] and [NEXT_PROBLEM] terminate; recorded on Exchange.ended_via.
    Each tutor/student call passes scenario.transcript_prefix's head as
    cacheable_prefix so the static head hits the prompt cache on round 2+.

    When tutor_mode is set, transcripts must include scenario.conv_id; the
    post-cut reference is computed once and substituted into the tutor prompt.
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    extra = ""
    next_turn_num = scenario.cut_turn + 1
    ended_via = ""

    # Compute reference once per scenario when oracle (or any tutor_mode) is on.
    reference_transcript = None
    if tutor_mode:
        if not transcripts:
            raise ValueError(
                f"run_exchange: tutor_mode={tutor_mode!r} requires transcripts"
            )
        conv = transcripts.get(scenario.conv_id)
        if conv is None:
            raise ValueError(
                f"run_exchange: tutor_mode={tutor_mode!r} but no transcript "
                f"loaded for conv_id={scenario.conv_id!r}"
            )
        reference_transcript = _build_reference_transcript(conv, scenario.cut_turn)

    while len(exchange.generated_turns) < max_turns:
        # --- Tutor turn ---
        head, tail = _build_role_prompt(
            "TUTOR", scenario.transcript_prefix, extra, scenario.student_context,
            prompt_version,
            tutor_mode=tutor_mode,
            reference_transcript=reference_transcript,
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

- [ ] **Step 4: Extend `run_exchanges_batch` signature and body**

Replace the existing `run_exchanges_batch` in `benchmark/core/exchange.py` with:

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
    tutor_mode: str | None = None,
    transcripts: dict[str, dict] | None = None,
) -> dict[str, Exchange]:
    """Batch mode multi-turn exchanges.

    Per-scenario state tracks `extra` (growing suffix) separate from the
    static scenario.transcript_prefix; the head is passed as cacheable_prefix
    on every per-scenario batch entry.

    When tutor_mode is set, transcripts must include every scenario's conv_id;
    the post-cut reference is computed once per scenario and reused across
    rounds.
    """
    exchanges = {}
    extras: dict[str, str] = {}
    next_turns = {}
    ended_via: dict[str, str] = {}
    refs: dict[str, str] = {}

    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        extras[scenario.scenario_id] = ""
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1
        if tutor_mode:
            if not transcripts:
                raise ValueError(
                    f"run_exchanges_batch: tutor_mode={tutor_mode!r} requires transcripts"
                )
            conv = transcripts.get(scenario.conv_id)
            if conv is None:
                raise ValueError(
                    f"run_exchanges_batch: tutor_mode={tutor_mode!r} but no transcript "
                    f"loaded for conv_id={scenario.conv_id!r}"
                )
            refs[scenario.scenario_id] = _build_reference_transcript(conv, scenario.cut_turn)

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
                tutor_mode=tutor_mode,
                reference_transcript=refs.get(sid),
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

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_benchmark_oracle_tutor.py -v`
Expected: all pass.

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_oracle_tutor.py
git commit -m "benchmark: thread tutor_mode + transcripts through exchange loops"
```

---

## Task 4: Wire `tutor.mode` in `config.yaml` and `benchmark/run.py`

**Files:**
- Modify: `config.yaml`
- Modify: `benchmark/run.py`

### Steps

- [ ] **Step 1: Add the `tutor.mode` default in `config.yaml`**

Find the `benchmark:` block, locate the `student:` sub-block (currently with `profile` and `mode` keys). Just below `student:`, add a sibling `tutor:` block:

```yaml
  tutor:
    mode: null         # null = default v5 tutor; "oracle" = mimic real tutor (sees post-cut)
```

Place it immediately after the `student:` block, before `detect:`.

- [ ] **Step 2: Verify the config parses**

Run:
```bash
python -c "import yaml; c = yaml.safe_load(open('config.yaml')); print(c['benchmark'].get('tutor'))"
```
Expected: `{'mode': None}`.

- [ ] **Step 3: Read `tutor.mode` in `benchmark/run.py`**

In `benchmark/run.py`, find where `student_mode = config["student"].get("mode")` is read (around line 72 / 141 -- there are two reads currently). Just below the SECOND read (the one near the `student_client` construction, before the `tutor_profiles` loop), add:

```python
    tutor_mode = config.get("tutor", {}).get("mode")
```

- [ ] **Step 4: Load transcripts when tutor_mode is set**

Currently `transcripts_for_screenshots` is loaded only when `with_screenshots` is true. We need to also load when `tutor_mode` is set. Find the block (around line 95-98):

```python
    transcripts_for_screenshots = None
    if with_screenshots:
        from annotator.core.storage import load_all_transcripts
        transcripts_for_screenshots = load_all_transcripts()
```

Replace with:

```python
    transcripts_for_screenshots = None
    if with_screenshots or config.get("tutor", {}).get("mode"):
        from annotator.core.storage import load_all_transcripts
        transcripts_for_screenshots = load_all_transcripts()
```

Reuse the same dict for both purposes -- transcripts are read-only, so sharing is safe.

- [ ] **Step 5: Pass `tutor_mode` + `transcripts` to both call sites**

Find the two `run_exchange*` call sites in `benchmark/run.py`.

In the batch path (`run_exchanges_batch(...)`), add the two kwargs at the end:

```python
                    trait_client=trait_client,
                    trait_model=trait_model,
                    tutor_mode=tutor_mode,
                    transcripts=transcripts_for_screenshots,
                )
```

In the sync path (`run_exchange(...)`), add the two kwargs at the end:

```python
                            trait_client=trait_client,
                            trait_model=trait_model,
                            tutor_mode=tutor_mode,
                            transcripts=transcripts_for_screenshots,
                        )
```

- [ ] **Step 6: Verify import + run tests**

Run: `python -c "import benchmark.run"`
Expected: clean import.

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add config.yaml benchmark/run.py
git commit -m "benchmark: wire tutor.mode + transcripts through run.py"
```

---

## Task 5: End-to-end smoke (manual, no commit)

**Files:** none modified — verification only.

### Steps

- [ ] **Step 1: Pick a small smoke shape**

We'll run a 2-scenario sync smoke with `tutor.mode: oracle` against the existing `varied_smoke.py`-style flow. Since `varied_smoke.py` doesn't yet support `--tutor-mode`, either:

- **(a)** Add a `--tutor-mode` CLI flag to `scripts/varied_smoke.py` (~10 lines: parse arg, build reference per scenario, pass to `run_exchange`). Quickest if you want a one-off; this script is already throwaway.
- **(b)** Temporarily flip `config.yaml`'s `tutor.mode` to `oracle`, run `python -m benchmark --version oracle_smoke_2026_06_10 --scenario-mode human --max-scenarios 2 --mode sync`, then revert.

Go with (b) for the verification smoke since the production pipeline is what we care about.

- [ ] **Step 2: Run the smoke**

Temporarily edit `config.yaml` to set `tutor.mode: oracle`, then:

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version oracle_smoke_2026_06_10 --scenario-mode human --max-scenarios 2 --mode sync
```

Expected logs:
- 2 scenarios run end to end.
- Exchanges complete with `ended_via` of `END` or `NEXT_PROBLEM` (oracle should naturally hit the end of the real conversation and emit one of the tokens).

Restore `config.yaml`'s `tutor.mode: null` after.

- [ ] **Step 3: Inspect cacheable_prefix size on round 1**

```bash
PYTHONIOENCODING=utf-8 python -c "
from annotator.core.storage import load_benchmark_result, list_benchmark_result_files
for f in list_benchmark_result_files('oracle_smoke_2026_06_10', 'exchanges', 'anthropic'):
    ex = load_benchmark_result('oracle_smoke_2026_06_10', 'exchanges', 'anthropic', f)
    t = ex.get('tutor_usage', {}) or {}
    print(f[-30:],
          'turns:', len(ex.get('generated_turns', [])),
          'cache_create:', t.get('cache_creation_input_tokens', 0),
          'cache_read:', t.get('cache_read_input_tokens', 0))
"
```

Expected: `cache_creation_input_tokens` non-zero (the head is written to cache on round 1); `cache_read_input_tokens` non-zero (cache hit on round 2+). Cache creation will be larger than baseline since the reference is in the head.

- [ ] **Step 4: Update `docs/status.md`**

Prepend a short block at the top of `docs/status.md`:

```markdown
## Recently Shipped: Oracle Tutor Mode (2026-06-10)

Benchmark gains an opt-in `tutor.mode: oracle` that hands the AI tutor the
post-cut real human turns and instructs it to mimic the real tutor's style /
strategy. Functions as a tutor-side ceiling cell for the experimental matrix.

- New prompt `prompts/benchmark/v5/tutors/oracle.txt`.
- `_build_role_prompt` accepts `tutor_mode` + `reference_transcript`.
- `run_exchange` / `run_exchanges_batch` compute the reference per scenario
  from the loaded transcripts.
- Default `tutor.mode: null` keeps existing behavior unchanged.

Spec: [plans/specs/2026-06-10-oracle-tutor-mode-design.md](plans/specs/2026-06-10-oracle-tutor-mode-design.md)
Plan: [plans/2026-06-10-oracle-tutor-mode.md](plans/2026-06-10-oracle-tutor-mode.md)
```

Update the `*Last updated:*` line to `2026-06-10`.

- [ ] **Step 5: Commit**

```bash
git add docs/status.md
git commit -m "docs: status update for oracle tutor mode"
```

---

## Self-Review

**Spec coverage:**
- New prompt `prompts/benchmark/v5/tutors/oracle.txt` — Task 1.
- `_build_reference_transcript` helper — Task 2.
- `_build_role_prompt(tutor_mode=..., reference_transcript=...)` — Task 2.
- Missing-reference guard — Task 2 test + raise.
- `run_exchange` / `run_exchanges_batch` accept `tutor_mode` + `transcripts` — Task 3.
- Reference computed once per scenario — Task 3 (sync: before loop; batch: in init loop).
- Missing-transcript guard — Task 3 raise.
- Config field + default null — Task 4.
- `run.py` plumbing + transcripts load — Task 4.
- End-to-end smoke + status doc — Task 5.

All spec items mapped.

**Placeholder scan:** no TBDs in step bodies; all code blocks are concrete.

**Type/name consistency:**
- `tutor_mode`, `reference_transcript`, `_build_reference_transcript` consistent across Tasks 2–5.
- Cache key stability preserved (reference is in head, fixed per scenario).
- Existing trait-mode signature and existing exchange loop structure preserved.
