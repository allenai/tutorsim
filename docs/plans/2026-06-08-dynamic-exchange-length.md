# Dynamic Exchange Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the AI tutor end a benchmark scenario by emitting `[END]` in its reply, capped by `max_turns: 100` generated turns. Replace fixed `num_turns: 2`. Introduce prompt version `v3` (= v2 with the `[END]` instruction).

**Architecture:** New helper `_check_end_token(text) -> (cleaned, ended)` in `benchmark/core/exchange.py`. Sync loop becomes `while not ended and len(generated) < max_turns`. Batch loop iterates up to `ceil(max_turns / 2)` rounds, removes scenarios from `active_ids` when their tutor reply contains `[END]` or when generated-turn count hits the cap. Config swaps `num_turns: 2` -> `max_turns: 100`, `prompt_version: v2` -> `v3`. v1/v2 stay frozen.

**Tech Stack:** Python 3.11, pytest. Modifies `benchmark/core/exchange.py`, `benchmark/run.py`, `config.yaml`, and adds `prompts/benchmark/v3/`.

**Spec:** [`docs/plans/specs/2026-06-08-dynamic-exchange-length-design.md`](specs/2026-06-08-dynamic-exchange-length-design.md)

---

## File Map

- **Create:** `prompts/benchmark/v3/tutor_system.txt` (v2 verbatim + appended `[END]` paragraph)
- **Create:** `prompts/benchmark/v3/students/imitate_example.txt`, `simple.txt`, `expert.txt`, `paraphrase_with_example.txt` (verbatim copies of v2 student prompts)
- **Modify:** `benchmark/core/exchange.py` — add `_check_end_token`, rewrite `run_exchange` and `run_exchanges_batch` loops, rename `num_turns` parameter to `max_turns`
- **Modify:** `benchmark/run.py` — pass `max_turns` instead of `num_turns` at both call sites (~lines 217 and 234)
- **Modify:** `config.yaml` — `num_turns: 2` -> `max_turns: 100`; `prompt_version: v2` -> `v3`
- **Create:** `tests/test_benchmark_exchange_dynamic.py` — new test module for token helper + dynamic-loop behavior in both sync and batch modes

---

## Task 1: Create v3 prompt directory

**Files:**
- Create: `prompts/benchmark/v3/tutor_system.txt`
- Create: `prompts/benchmark/v3/students/imitate_example.txt`
- Create: `prompts/benchmark/v3/students/simple.txt`
- Create: `prompts/benchmark/v3/students/expert.txt`
- Create: `prompts/benchmark/v3/students/paraphrase_with_example.txt`

### Steps

- [ ] **Step 1: Verify v2 layout**

Run: `ls prompts/benchmark/v2/ && ls prompts/benchmark/v2/students/`
Expected: `tutor_system.txt students/` and `imitate_example.txt expert.txt paraphrase_with_example.txt simple.txt`.

- [ ] **Step 2: Copy v2 into v3 verbatim**

```bash
mkdir -p prompts/benchmark/v3/students
cp prompts/benchmark/v2/tutor_system.txt prompts/benchmark/v3/tutor_system.txt
cp prompts/benchmark/v2/students/*.txt prompts/benchmark/v3/students/
```

- [ ] **Step 3: Append the `[END]` instruction to v3 tutor prompt**

Open `prompts/benchmark/v3/tutor_system.txt` and append the following at the end of the file (preserving the existing content above it). Make sure there's a single blank line between the existing content and the new section, and the file ends with a newline.

```
Ending the scenario: when the moment has played out and there's nothing useful left to add (the student has reached an answer they can run with, the misconception is resolved, or the problem is finished), end your final message with `[END]` on its own line. Include a brief, natural wrap-up before the token — e.g. "Great work, let me know if you want to try another. [END]". Don't use `[END]` until the scenario genuinely feels resolved; it's fine for some moments to take several exchanges.
```

- [ ] **Step 4: Verify v3 differs from v2 only in `tutor_system.txt`**

Run: `diff -r prompts/benchmark/v2 prompts/benchmark/v3`
Expected output: a single difference flagged on `tutor_system.txt`. All student files must be byte-identical (no diff for them).

- [ ] **Step 5: Commit**

```bash
git add prompts/benchmark/v3
git commit -m "prompts: add benchmark v3 (= v2 + [END] tutor instruction)"
```

---

## Task 2: Add `_check_end_token` helper (TDD)

**Files:**
- Create: `tests/test_benchmark_exchange_dynamic.py`
- Modify: `benchmark/core/exchange.py` (add `END_TOKEN` constant and `_check_end_token` function)

### Steps

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_exchange_dynamic.py`:

```python
"""Tests for dynamic-length benchmark exchanges (tutor [END] + max_turns cap)."""

import pytest

from benchmark.core.exchange import _check_end_token, END_TOKEN


def test_check_end_token_trailing():
    text, ended = _check_end_token("Great job on that. [END]")
    assert ended is True
    assert text == "Great job on that."


def test_check_end_token_trailing_on_own_line():
    text, ended = _check_end_token("Great job on that.\n[END]")
    assert ended is True
    assert text == "Great job on that."


def test_check_end_token_absent():
    text, ended = _check_end_token("Keep going!")
    assert ended is False
    assert text == "Keep going!"


def test_check_end_token_mid_text():
    text, ended = _check_end_token("ok [END] bye")
    assert ended is True
    # Token stripped; surrounding text preserved (interior whitespace acceptable):
    assert "[END]" not in text
    assert "ok" in text and "bye" in text


def test_check_end_token_alone():
    text, ended = _check_end_token("[END]")
    assert ended is True
    assert text == ""


def test_check_end_token_constant_value():
    assert END_TOKEN == "[END]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: ImportError or AttributeError for `_check_end_token` / `END_TOKEN`.

- [ ] **Step 3: Implement the helper**

In `benchmark/core/exchange.py`, near the top (after `NEXT_DELIMITER`):

```python
END_TOKEN = "[END]"


def _check_end_token(text: str) -> tuple[str, bool]:
    """Strip [END] token from text and report whether it was present.

    The token usually appears at the end of the message but may appear
    anywhere; either way the scenario should end. Trailing whitespace
    after the strip is removed.
    """
    if END_TOKEN not in text:
        return text, False
    cleaned = text.replace(END_TOKEN, "").rstrip()
    return cleaned, True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: add _check_end_token helper for dynamic exchanges"
```

---

## Task 3: Make `run_exchange` (sync) dynamic (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py` (rewrite `run_exchange`; rename `num_turns` -> `max_turns`)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (add sync-mode tests)

### Steps

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_exchange_dynamic.py`:

```python
from dataclasses import dataclass
from unittest.mock import MagicMock

from benchmark.core.exchange import run_exchange
from benchmark.core.scenarios import Scenario


def _make_scenario():
    return Scenario(
        scenario_id="s1",
        conv_id="conv1",
        cut_turn=3,
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok",
        student_context="Grade 5, fractions",
        last_student_message="hello",
        mode="human",
        detection={"turn_start": 2, "turn_end": 3,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def _stub_client(replies):
    """ModelClient stub whose .generate(...) returns reply objects in order."""
    client = MagicMock()
    client.model = "stub-model"
    iterator = iter(replies)

    def _generate(*args, **kwargs):
        text = next(iterator)
        resp = MagicMock()
        resp.text = text
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp

    client.generate = _generate
    return client


def test_run_exchange_ends_on_first_tutor_end_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # avoid touching real prompt files
    # Patch prompt loader to return a fixed string, so we don't need v3 on disk in test
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    tutor = _stub_client(["Great work! [END]"])
    student = _stub_client([])  # never called

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    assert len(ex.generated_turns) == 1
    assert ex.generated_turns[0]["role"] == "TUTOR"
    assert ex.generated_turns[0]["text"] == "Great work!"


def test_run_exchange_runs_full_turn_then_ends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client([
        "What did you get for x?",          # round 1 tutor
        "Nice! That matches. [END]",        # round 2 tutor, ends
    ])
    student = _stub_client(["I got 5"])     # round 1 student only

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    roles = [t["role"] for t in ex.generated_turns]
    assert roles == ["TUTOR", "STUDENT", "TUTOR"]
    assert ex.generated_turns[-1]["text"] == "Nice! That matches."


def test_run_exchange_respects_max_turns_when_no_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    # Both clients always reply with single-message text and never emit [END].
    tutor = _stub_client([f"tutor msg {i}" for i in range(50)])
    student = _stub_client([f"student msg {i}" for i in range(50)])

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    assert len(ex.generated_turns) == 4
    # Alternates tutor/student starting with tutor:
    assert [t["role"] for t in ex.generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]


def test_run_exchange_end_token_alone_skips_empty_turn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client(["What did you get?", "[END]"])
    student = _stub_client(["I got 5"])

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    # Tutor-only [END] adds no turn for the empty wrap-up:
    roles = [t["role"] for t in ex.generated_turns]
    assert roles == ["TUTOR", "STUDENT"]
    assert ex.generated_turns[0]["text"] == "What did you get?"
    assert ex.generated_turns[1]["text"] == "I got 5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: 4 new failures — `run_exchange` doesn't accept `max_turns` yet (it still has `num_turns`).

- [ ] **Step 3: Rewrite `run_exchange` in `benchmark/core/exchange.py`**

Replace the existing `run_exchange` function (lines ~119-174) with:

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
) -> Exchange:
    """Run a multi-turn exchange for a single scenario (sync mode).

    Runs until the tutor emits END_TOKEN in its reply, or until the count of
    generated turns reaches `max_turns`. The student never ends a scenario.

    When images is provided, every tutor and student call receives them.
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    running_transcript = scenario.transcript_prefix
    next_turn_num = scenario.cut_turn + 1
    ended = False

    while not ended and len(exchange.generated_turns) < max_turns:
        # Tutor turn(s)
        prompt = _build_role_prompt("TUTOR", running_transcript, scenario.student_context, prompt_version)
        response = tutor_client.generate(
            prompt, json_mode=False, max_tokens=tutor_max_tokens,
            images=images,
        )
        _add_usage(exchange.tutor_usage, response.usage)

        text, ended = _check_end_token(response.text)
        messages = _split_messages(text)
        if not messages and not ended:
            messages = ["..."]
        if messages:
            running_transcript, next_turn_num = _append_turns(
                exchange, messages, "TUTOR", running_transcript, next_turn_num,
            )

        if ended or len(exchange.generated_turns) >= max_turns:
            break

        # Student turn(s)
        prompt = _build_role_prompt("STUDENT", running_transcript, scenario.student_context,
                                    prompt_version, student_mode=student_mode)
        response = student_client.generate(
            prompt, json_mode=False, max_tokens=student_max_tokens,
            images=images,
        )
        _add_usage(exchange.student_usage, response.usage)

        messages = _split_messages(response.text) or ["..."]
        running_transcript, next_turn_num = _append_turns(
            exchange, messages, "STUDENT", running_transcript, next_turn_num,
        )

    exchange.completed = True
    return exchange
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: 10 passed (6 helper + 4 sync).

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: dynamic run_exchange (tutor [END] + max_turns cap)"
```

---

## Task 4: Make `run_exchanges_batch` dynamic (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py` (rewrite `run_exchanges_batch`; rename `num_turns` -> `max_turns`)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (add batch-mode tests)

### Steps

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_exchange_dynamic.py`:

```python
from benchmark.core.exchange import run_exchanges_batch


def _scenario(sid):
    return Scenario(
        scenario_id=sid, conv_id=sid, cut_turn=3,
        transcript_prefix=f"Turn 1. TUTOR: a\nTurn 2. STUDENT: b\nTurn 3. TUTOR: c",
        student_context="ctx", last_student_message="b", mode="human",
        detection={"turn_start": 2, "turn_end": 3,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def test_run_exchanges_batch_ends_per_scenario(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    # Two scenarios: s1 ends on first tutor reply, s2 runs through all rounds (no [END]).
    # Batch driver should drop s1 from active_ids after round 1 and keep going for s2.

    # Capture the order of run_batch calls and their entries.
    calls = []

    def fake_run_batch(client, entries, json_mode, display_name, poll_interval):
        calls.append({"display_name": display_name,
                      "sids": [e["custom_id"] for e in entries]})
        results = {}
        for e in entries:
            sid = e["custom_id"]
            # tutor batches: alternate end-state per scenario
            if display_name.startswith("tutor"):
                if sid == "s1":
                    text = "wrap [END]"
                else:
                    text = "more please"
            else:  # student
                text = "ok"
            results[sid] = {"text": text,
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        return results

    def fake_build_batch_entry(custom_id, prompt, json_mode, max_tokens, images=None):
        return {"custom_id": custom_id}

    monkeypatch.setattr("benchmark.core.exchange.run_batch", fake_run_batch)
    monkeypatch.setattr("benchmark.core.exchange.build_batch_entry", fake_build_batch_entry)

    tutor = MagicMock(); tutor.model = "t"
    student = MagicMock(); student.model = "s"

    exchanges = run_exchanges_batch(
        scenarios=[_scenario("s1"), _scenario("s2")],
        tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=64, student_max_tokens=64,
        poll_interval=0, prompt_version="v3",
    )

    # s1 ended after round 1 tutor -> exactly 1 turn (TUTOR only), no student.
    assert [t["role"] for t in exchanges["s1"].generated_turns] == ["TUTOR"]
    assert exchanges["s1"].generated_turns[0]["text"] == "wrap"
    assert exchanges["s1"].completed is True

    # s2 ran until generated_turns hit max_turns=4: T, S, T, S
    assert [t["role"] for t in exchanges["s2"].generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]
    assert exchanges["s2"].completed is True

    # After s1 ends, subsequent batches must NOT include s1:
    later_calls = calls[1:]  # everything after the first tutor batch
    for c in later_calls:
        assert "s1" not in c["sids"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: 1 new failure — `run_exchanges_batch` doesn't accept `max_turns` and doesn't strip `[END]`.

- [ ] **Step 3: Rewrite `run_exchanges_batch`**

Replace the existing `run_exchanges_batch` function (lines ~181-315) with:

```python
import math


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
) -> dict[str, Exchange]:
    """Run multi-turn exchanges for all scenarios using batch API.

    Each round submits one batch of tutor prompts (for currently-active scenarios),
    waits, then (if any still active) submits one student batch. Scenarios are
    removed from active_ids when their tutor reply ends with END_TOKEN or when
    generated turn count hits max_turns.

    Args:
        save_callback: Optional function(scenario_id, exchange) called after
            each round to save progress incrementally.

    Returns: {scenario_id: Exchange}
    """
    exchanges = {}
    transcripts = {}
    next_turns = {}

    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        transcripts[scenario.scenario_id] = scenario.transcript_prefix
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1

    scenario_map = {s.scenario_id: s for s in scenarios}
    active_ids = list(scenario_map.keys())

    max_rounds = math.ceil(max_turns / 2)

    for round_num in range(max_rounds):
        if not active_ids:
            break

        # --- Tutor batch ---
        logger.info("Round %d - tutor batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            prompt = _build_role_prompt("TUTOR", transcripts[sid], scenario.student_context, prompt_version)
            scenario_images = (images_by_scenario or {}).get(sid)
            tutor_entries.append(
                build_batch_entry(sid, prompt, json_mode=False, max_tokens=tutor_max_tokens,
                                  images=scenario_images)
            )

        tutor_raw = run_batch(
            tutor_client, tutor_entries, json_mode=False,
            display_name=f"tutor_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        # Process tutor results
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

            text, ended = _check_end_token(result["text"])
            messages = _split_messages(text)
            if not messages and not ended:
                messages = ["..."]
            if messages:
                transcripts[sid], next_turns[sid] = _append_turns(
                    exchange, messages, "TUTOR", transcripts[sid], next_turns[sid],
                )

            if ended or len(exchange.generated_turns) >= max_turns:
                ended_this_round.append(sid)

        for sid in failed:
            active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        # --- Student batch (only if scenarios still active) ---
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
            prompt = _build_role_prompt("STUDENT", transcripts[sid], scenario.student_context,
                                        prompt_version, student_mode=student_mode)
            scenario_images = (images_by_scenario or {}).get(sid)
            student_entries.append(
                build_batch_entry(sid, prompt, json_mode=False, max_tokens=student_max_tokens,
                                  images=scenario_images)
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
            transcripts[sid], next_turns[sid] = _append_turns(
                exchange, messages, "STUDENT", transcripts[sid], next_turns[sid],
            )

            if len(exchange.generated_turns) >= max_turns:
                ended_this_round.append(sid)

        for sid in failed:
            active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        if save_callback:
            for sid in scenario_map:
                save_callback(sid, exchanges[sid])

    # Mark all scenarios as completed (we ran them to their natural end or cap).
    for sid in scenario_map:
        exchanges[sid].completed = True

    logger.info("Exchanges complete: %d scenarios", len(scenario_map))
    return exchanges
```

Make sure the `import math` is at module-level (top of file, with the other imports).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: 11 passed.

Also run the existing exchange-related tests to confirm no regression:
Run: `pytest tests/test_benchmark_student_modes.py tests/test_benchmark_resume.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: dynamic run_exchanges_batch with per-scenario end detection"
```

---

## Task 5: Wire `max_turns` + v3 default into config and run.py

**Files:**
- Modify: `config.yaml` (lines around 167-170)
- Modify: `benchmark/run.py` (two call sites at lines ~217 and ~234)

### Steps

- [ ] **Step 1: Inspect current `exchange:` block in config**

Run: `grep -n "exchange:" -A 6 config.yaml`
You should see something like:
```yaml
  exchange:
    num_turns: 2
    poll_interval: 60
    prompt_version: v2
```

- [ ] **Step 2: Update `config.yaml`**

Replace `num_turns: 2` with `max_turns: 100` and change `prompt_version: v2` to `prompt_version: v3`. Preserve `poll_interval` and any other adjacent keys. Final block should look like:

```yaml
  exchange:
    max_turns: 100               # was: num_turns: 2 — tutor decides end via [END]; this is a hard cap on generated turns
    poll_interval: 60
    prompt_version: v3           # v3 = v2 + [END] tutor instruction
```

- [ ] **Step 3: Update `benchmark/run.py` call sites**

Find both `num_turns=exchange_cfg["num_turns"],` lines (one in the batch branch ~line 217, one in the sync branch ~line 234) and change each to:

```python
                    max_turns=exchange_cfg["max_turns"],
```

- [ ] **Step 4: Verify the file parses**

Run:
```bash
python -c "import yaml; c = yaml.safe_load(open('config.yaml')); print('exchange:', c['benchmark']['exchange'])"
```
Expected output includes `'max_turns': 100, 'poll_interval': 60, 'prompt_version': 'v3'` (key order may vary).

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -q`
Expected: all green (pre-existing `test_eval_metrics.py` collection error from missing `krippendorff` is fine to ignore).

- [ ] **Step 6: Commit**

```bash
git add config.yaml benchmark/run.py
git commit -m "benchmark: switch to max_turns + prompt_version v3"
```

---

## Task 6: End-to-end smoke

**Files:** none modified — verification only.

### Steps

- [ ] **Step 1: Run a small benchmark in sync mode against human scenarios**

Run (Windows PowerShell or bash):
```bash
python -m benchmark --version dyn_smoke_2026_06_08 --scenario-mode human --max-scenarios 5 --mode sync
```

Watch the logs. Expected:
- Step 0 detection is skipped (human mode).
- 5 scenarios run through Phase 1 exchange. Each exchange's `generated_turns` count should vary (some shorter if tutor emitted `[END]`, some longer up to the 100-turn cap).
- Phase 2 annotation across 3 styles completes without errors.
- Phase 3 scores print.

If any scenario reports exactly `max_turns` turns or 0 turns, inspect that exchange's saved JSON to confirm the tutor either never emitted `[END]` (expected for long scenarios) or that the loop is honoring the cap.

- [ ] **Step 2: Inspect the saved exchanges to confirm dynamic behavior**

Run:
```bash
python -c "
import json
from annotator.core.storage import load_benchmark_result, list_benchmark_result_files
files = list_benchmark_result_files('dyn_smoke_2026_06_08', 'exchanges', 'anthropic')
for f in files:
    ex = load_benchmark_result('dyn_smoke_2026_06_08', 'exchanges', 'anthropic', f)
    last = ex['generated_turns'][-1]['text'][-60:] if ex['generated_turns'] else ''
    print(f, '| turns=', len(ex['generated_turns']), '| last=', repr(last))
"
```

Expected: at least one scenario should have fewer than 5 turns, indicating the tutor ended early via `[END]` (token is stripped from saved text). If every scenario has the same turn count, the tutor isn't emitting `[END]` and Task 1's prompt addition needs review.

- [ ] **Step 3: Report results, then stop**

This is a no-commit verification step. Capture the per-scenario turn counts and report. No code changes needed unless smoke surfaces a real problem.

---

## Self-Review

**Spec coverage:**
- New prompt version v3 — Task 1
- `_check_end_token` helper — Task 2
- Sync dynamic loop — Task 3
- Batch dynamic loop + dropout — Task 4
- Config + run.py wiring — Task 5
- End-to-end smoke — Task 6

All sections of the spec map to a task.

**Placeholder scan:** no TBDs, no "handle edge cases" hand-waves. All test code, helper bodies, and loop bodies are concrete.

**Type/name consistency:**
- `END_TOKEN`, `_check_end_token`, `max_turns`, `ended`, `active_ids`, `ended_this_round` used consistently across Tasks 2-4.
- `max_turns: 100` matches `exchange_cfg["max_turns"]` in Task 5.
- Prompt path `prompts/benchmark/v3/tutor_system.txt` in Task 1 matches `prompt_version: v3` in Task 5.
