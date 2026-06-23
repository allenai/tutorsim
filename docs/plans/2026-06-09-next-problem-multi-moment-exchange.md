# Multi-Problem Exchange with [NEXT_PROBLEM] Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the AI tutor walk multiple teachable moments from the same conversation by emitting `[NEXT_PROBLEM]` to advance and `[END]` to terminate. The system queues subsequent moments from `ground_truth_hybrid`, injects a SYSTEM divider + the real student turn at the popped moment's cut point, and continues the exchange.

**Architecture:** Additive change. `Scenario` gains `next_problem_queue`. Helpers `_parse_tutor_tokens` and `_inject_next_problem` extend the dynamic-exchange loop (sync + batch). New prompt version `v4`. `extract_human_scenarios` populates the queue by reusing the modal-cut + role-adjust logic for each subsequent in-conversation moment.

**Tech Stack:** Python 3.11, pytest. Touches `benchmark/core/scenarios.py`, `benchmark/core/exchange.py`, `benchmark/eval/view.py`, `benchmark/eval/view_replay.py`, `prompts/benchmark/v4/`, `config.yaml`, and tests.

**Spec:** [`docs/plans/specs/2026-06-09-next-problem-multi-moment-exchange-design.md`](specs/2026-06-09-next-problem-multi-moment-exchange-design.md)

---

## File Map

- **Create:** `prompts/benchmark/v4/tutor_system.txt` (v3 verbatim with the `[END]` paragraph replaced by a two-token instruction).
- **Create:** `prompts/benchmark/v4/students/{imitate_example,simple,expert,paraphrase_with_example}.txt` — byte-identical copies of v3 students.
- **Modify:** `benchmark/core/scenarios.py` — extend `Scenario` dataclass with `next_problem_queue`; refactor cluster logic so the modal-cut + role-adjust path is reusable; populate queue in `extract_human_scenarios`.
- **Modify:** `benchmark/core/exchange.py` — add `NEXT_PROBLEM_TOKEN` and `_parse_tutor_tokens`; add `_inject_next_problem`; rewrite the tutor-result handling in `run_exchange` and `run_exchanges_batch` to honour the two-token system; record `ended_via` and `problem_boundaries` on `Exchange`.
- **Modify:** `config.yaml` — default `benchmark.exchange.prompt_version: v4`.
- **Modify:** `benchmark/eval/view.py` and `benchmark/eval/view_replay.py` — render `role=SYSTEM` turns with a gray-divider style (cosmetic).
- **Modify:** `tests/test_benchmark_exchange_dynamic.py` — token parser tests + multi-problem loop tests.
- **Modify:** `tests/test_benchmark_human_scenarios.py` — queue population tests.

---

## Task 1: Create v4 prompt directory

**Files:**
- Create: `prompts/benchmark/v4/tutor_system.txt`
- Create: `prompts/benchmark/v4/students/imitate_example.txt`
- Create: `prompts/benchmark/v4/students/simple.txt`
- Create: `prompts/benchmark/v4/students/expert.txt`
- Create: `prompts/benchmark/v4/students/paraphrase_with_example.txt`

### Steps

- [ ] **Step 1: Copy v3 to v4 verbatim**

```bash
mkdir -p prompts/benchmark/v4/students
cp prompts/benchmark/v3/tutor_system.txt prompts/benchmark/v4/tutor_system.txt
cp prompts/benchmark/v3/students/*.txt prompts/benchmark/v4/students/
```

- [ ] **Step 2: Replace the `[END]` paragraph in v4 tutor prompt**

Open `prompts/benchmark/v4/tutor_system.txt`. Find the existing paragraph that begins `Ending the scenario:` and ends with the `[END]` example sentence. Replace that paragraph with:

```
Signaling problem transitions: this conversation may contain several distinct
problems / teachable moments. Use these two tokens at the very end of your
final message, each on its own line, to signal what should happen next:

- `[NEXT_PROBLEM]` — the current problem has played out (the student has reached
  an answer they can run with, the misconception is resolved, or the problem is
  finished) and you'd hand off to the next problem now. Include a brief, natural
  wrap-up before the token (e.g. "Great work, ready for the next one? [NEXT_PROBLEM]").
  Our system will introduce the next problem for the student.

- `[END]` — you're fully done with this student / conversation (no more problems
  to work through, time to wrap up entirely). Include a final goodbye before the
  token (e.g. "Have a great day! [END]").

If both `[END]` and `[NEXT_PROBLEM]` appear, the conversation ends. Don't use
either token until the current problem is genuinely resolved; some moments take
several exchanges.
```

Preserve all content before that paragraph (the tutor role / chat style / `[NEXT]` multi-message instructions) and ensure the file ends with a newline.

- [ ] **Step 3: Verify v4 differs from v3 only in `tutor_system.txt`**

Run: `diff -r prompts/benchmark/v3 prompts/benchmark/v4`
Expected: a single difference flagged on `tutor_system.txt`. Student files must be byte-identical.

- [ ] **Step 4: Commit**

```bash
git add prompts/benchmark/v4
git commit -m "prompts: add benchmark v4 with [NEXT_PROBLEM] + [END] tokens"
```

---

## Task 2: Add `_parse_tutor_tokens` helper (TDD)

**Files:**
- Modify: `benchmark/core/exchange.py` (add `NEXT_PROBLEM_TOKEN` constant and `_parse_tutor_tokens` near the existing `END_TOKEN` / `_check_end_token`)
- Modify: `tests/test_benchmark_exchange_dynamic.py` (add helper tests at the top of the file, after existing helper tests)

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/test_benchmark_exchange_dynamic.py`, add to the existing import line that pulls from `benchmark.core.exchange`:

```python
from benchmark.core.exchange import (
    _check_end_token, END_TOKEN,
    _parse_tutor_tokens, NEXT_PROBLEM_TOKEN,
)
```

(If those names are already imported individually, consolidate into the multi-line import above. The remaining `from benchmark.core.exchange import ...` lines stay unchanged.)

After the existing `_check_end_token` tests (and before the `run_exchange` tests), append:

```python
def test_parse_tutor_tokens_end_only():
    text, ended, next_p = _parse_tutor_tokens("Wrap up. [END]")
    assert ended is True and next_p is False
    assert text == "Wrap up."


def test_parse_tutor_tokens_next_problem_only():
    text, ended, next_p = _parse_tutor_tokens("Great work. [NEXT_PROBLEM]")
    assert ended is False and next_p is True
    assert text == "Great work."


def test_parse_tutor_tokens_both_end_wins():
    text, ended, next_p = _parse_tutor_tokens("done [NEXT_PROBLEM] [END]")
    assert ended is True
    assert next_p is False   # END takes precedence
    assert "[END]" not in text
    assert "[NEXT_PROBLEM]" not in text


def test_parse_tutor_tokens_neither():
    text, ended, next_p = _parse_tutor_tokens("Try this next.")
    assert ended is False and next_p is False
    assert text == "Try this next."


def test_parse_tutor_tokens_constant_values():
    assert END_TOKEN == "[END]"
    assert NEXT_PROBLEM_TOKEN == "[NEXT_PROBLEM]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v -k "parse_tutor_tokens"`
Expected: ImportError on `_parse_tutor_tokens` / `NEXT_PROBLEM_TOKEN`.

- [ ] **Step 3: Implement the helper**

In `benchmark/core/exchange.py`, immediately after the existing `END_TOKEN = "[END]"` line and the `_check_end_token` function, add:

```python
NEXT_PROBLEM_TOKEN = "[NEXT_PROBLEM]"


def _parse_tutor_tokens(text: str) -> tuple[str, bool, bool]:
    """Strip tutor control tokens and report which were present.

    Returns (cleaned_text, ended, next_problem).
    END takes precedence: if both tokens appear, ended=True, next_problem=False.
    """
    has_end = END_TOKEN in text
    has_next = NEXT_PROBLEM_TOKEN in text
    cleaned = text.replace(END_TOKEN, "").replace(NEXT_PROBLEM_TOKEN, "").rstrip()
    if has_end:
        return cleaned, True, False
    return cleaned, False, has_next
```

Keep `_check_end_token` around for any caller still importing it (no functional change there).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v -k "parse_tutor_tokens"`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: add _parse_tutor_tokens for two-token exchange control"
```

---

## Task 3: Populate `next_problem_queue` on `Scenario` (TDD)

**Files:**
- Modify: `benchmark/core/scenarios.py` (extend `Scenario` dataclass; refactor cluster-resolve into a helper; populate queue inside `extract_human_scenarios`)
- Modify: `tests/test_benchmark_human_scenarios.py` (add queue-population tests; update existing tests to assert the new field exists)

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/test_benchmark_human_scenarios.py`, append new tests at the END of the file (after the helper tests from prior tasks):

```python
# ---------------------------------------------------------------------------
# next_problem_queue tests
# ---------------------------------------------------------------------------

def test_queue_populated_with_subsequent_moments():
    scenarios = _run_extract()
    # Pick a scenario from Transcript A (which has multiple kept clusters).
    by_id = {s.scenario_id: s for s in scenarios}
    s_first = next(s for s in scenarios if s.scenario_id.endswith("__hum_3_5"))
    # Seed turn_start=3. Queue should contain all kept clusters with
    # turn_start > seed.turn_end (=5): (5,8) excluded (ts=5 not > te=5? actually
    # turn_start > seed.turn_end means strictly later, so 5,8 keeps), then (10,12)
    # and (16,18). Actually 5 > 5 is false, so (5,8) is NOT included since its
    # turn_start equals seed.turn_end. Keep both queue entries (10,12) and (16,18).
    assert isinstance(s_first.next_problem_queue, list)
    queue_ranges = [(q["turn_start"], q["turn_end"]) for q in s_first.next_problem_queue]
    assert (10, 12) in queue_ranges
    assert (16, 18) in queue_ranges
    # (5,8) overlaps with seed (turn_start=5 == seed.turn_end=5), so excluded.
    assert (5, 8) not in queue_ranges


def test_queue_entries_carry_adjusted_cut():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_3_5"))
    by_range = {(q["turn_start"], q["turn_end"]): q for q in s.next_problem_queue}
    # (10,12) was a singleton with cut=11 TUTOR -> adjusted to 10.
    assert by_range[(10, 12)]["cut_turn"] == 10
    # (16,18) modal=17 TUTOR -> adjusted to 16.
    assert by_range[(16, 18)]["cut_turn"] == 16


def test_queue_is_sorted_by_turn_start():
    scenarios = _run_extract()
    for s in scenarios:
        starts = [q["turn_start"] for q in s.next_problem_queue]
        assert starts == sorted(starts)


def test_queue_empty_when_seed_is_last_moment():
    scenarios = _run_extract()
    # Transcript B has only the single (8,9) cluster -> queue empty.
    s_b = next(s for s in scenarios if s.conv_id.startswith("bbbbbbbb"))
    assert s_b.next_problem_queue == []


def test_queue_skips_overlapping_or_earlier_moments():
    """A moment with turn_start <= seed.turn_end must be excluded from the queue."""
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_5_8"))
    # Seed turn_end=8. Anything with turn_start <= 8 must NOT appear.
    for q in s.next_problem_queue:
        assert q["turn_start"] > 8


def test_existing_scenario_fields_still_present():
    """Sanity: queue addition didn't break existing fields."""
    scenarios = _run_extract()
    s = scenarios[0]
    assert hasattr(s, "scenario_id")
    assert hasattr(s, "next_problem_queue")
    assert hasattr(s, "cut_turn")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_human_scenarios.py -v -k "queue"`
Expected: `AttributeError: 'Scenario' object has no attribute 'next_problem_queue'`.

- [ ] **Step 3: Extend the `Scenario` dataclass**

In `benchmark/core/scenarios.py`, near the existing `@dataclass class Scenario:` block, add the new field at the end of the field list:

```python
@dataclass
class Scenario:
    scenario_id: str
    conv_id: str
    cut_turn: int
    transcript_prefix: str
    student_context: str
    last_student_message: str
    mode: str                     # "detected" | "random" | "human"
    detection: "dict | None"
    next_problem_queue: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["transcript_prefix_length"] = len(self.transcript_prefix)
        del d["transcript_prefix"]
        return d
```

If `field` is not already imported from `dataclasses` at the top of the file (it currently is — line 7: `from dataclasses import dataclass, asdict`), update that import to:

```python
from dataclasses import dataclass, asdict, field
```

- [ ] **Step 4: Extract a per-cluster resolver**

Just below `_pick_representative_member`, add a helper that turns a moment cluster into the resolved (cut_turn, representative) data the queue / scenario both need:

```python
def _resolve_cluster(members: list[dict], conversation: dict,
                     turn_start: int, turn_end: int) -> "dict | None":
    """Return {"cut_turn", "chosen_cut_turn", "cut_votes", "cluster_size",
    "representative"} for a moment cluster, or None if it should be dropped.

    Applies vote filtering (cut in [ts, te]), modal selection, role adjustment,
    and representative pick. Identical logic to extract_human_scenarios' inner
    loop -- factored out so the next_problem_queue can reuse it.
    """
    votes: list[int] = []
    for m in members:
        cut = m.get("cut_turn")
        if not isinstance(cut, int):
            continue
        if cut < turn_start or cut > turn_end:
            continue
        votes.append(cut)
    chosen = _pick_modal_cut(votes)
    if chosen is None:
        return None
    adjusted = _role_adjust_cut(chosen, conversation)
    if adjusted is None:
        return None
    rep = _pick_representative_member(members, chosen)
    return {
        "cut_turn": adjusted,
        "chosen_cut_turn": chosen,
        "cut_votes": dict(Counter(votes)),
        "cluster_size": len(members),
        "representative": rep,
    }
```

- [ ] **Step 5: Refactor `extract_human_scenarios` to use the resolver and populate the queue**

Replace the body of `extract_human_scenarios` (keeping its signature `(transcripts: dict[str, dict]) -> list[Scenario]`) with:

```python
def extract_human_scenarios(transcripts: dict[str, dict]) -> list[Scenario]:
    """Build one scenario per moment cluster + a queue of subsequent moments.

    See the design doc for selection / role-adjust details. Each produced
    Scenario carries a next_problem_queue listing later moments from the
    same conversation (filtered the same way as the seed), so the exchange
    loop can advance the AI tutor through multiple problems.
    """
    uuid_to_conv = {_conv_id_to_uuid(cid): cid for cid in transcripts}

    # First pass: group all kept moments per conversation by (ts, te).
    clusters_by_conv: dict[str, dict[tuple, list[dict]]] = {}
    for gt in load_all_ground_truth_files():
        gt_uuid = gt.get("conversation_id")
        full_conv_id = uuid_to_conv.get(_conv_id_to_uuid(gt_uuid or ""))
        if not full_conv_id or full_conv_id in EXAMPLE_CONV_IDS:
            continue
        for m in gt.get("key_moments", []):
            if m.get("situation_label_agg") not in _SCAFF_AGGS:
                continue
            ts = m.get("turn_start")
            te = m.get("turn_end")
            if ts is None or te is None:
                continue
            clusters_by_conv.setdefault(full_conv_id, {}).setdefault((ts, te), []).append(m)

    scenarios: list[Scenario] = []
    for full_conv_id, clusters in clusters_by_conv.items():
        conversation = transcripts[full_conv_id]
        # Resolve every cluster once, in turn order.
        resolved: list[tuple] = []  # (ts, te, resolved_dict)
        for (ts, te) in sorted(clusters.keys()):
            r = _resolve_cluster(clusters[(ts, te)], conversation, ts, te)
            if r is not None:
                resolved.append((ts, te, r))

        # Each kept cluster becomes a scenario; its queue is all later clusters.
        for i, (ts, te, r) in enumerate(resolved):
            prefix = _format_prefix(conversation, r["cut_turn"])
            if not prefix:
                continue

            queue = [
                {"turn_start": qts, "turn_end": qte, "cut_turn": qr["cut_turn"]}
                for (qts, qte, qr) in resolved[i + 1:]
                if qts > te
            ]

            rep = r["representative"]
            detection = {
                "turn_start": ts,
                "turn_end": te,
                # situation_label_agg only set on annotation_type=="scaffolding"
                # records, so all selected moments are scaffolding.
                "annotation_type": "scaffolding",
                "situation": rep.get("situation", ""),
                "situation_label_agg": rep.get("situation_label_agg"),
                "moment_id": rep.get("moment_id"),
                "annotator_id": rep.get("annotator_id"),
                "chosen_cut_turn": r["chosen_cut_turn"],
                "cut_votes": r["cut_votes"],
                "cluster_size": r["cluster_size"],
            }

            scenarios.append(Scenario(
                scenario_id=f"{full_conv_id}__hum_{ts}_{te}",
                conv_id=full_conv_id,
                cut_turn=r["cut_turn"],
                transcript_prefix=prefix,
                student_context=_get_student_context(conversation),
                last_student_message=_last_student_msg(conversation, r["cut_turn"]),
                mode="human",
                detection=detection,
                next_problem_queue=queue,
            ))

    return scenarios
```

- [ ] **Step 6: Run all human-scenarios tests**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: all pass — both pre-existing tests AND the new queue tests.

- [ ] **Step 7: Commit**

```bash
git add benchmark/core/scenarios.py tests/test_benchmark_human_scenarios.py
git commit -m "benchmark: populate next_problem_queue on human Scenarios"
```

---

## Task 4: Multi-problem loop in `run_exchange` (sync, TDD)

**Files:**
- Modify: `benchmark/core/exchange.py` — add `_inject_next_problem`; rewrite tutor result handling in `run_exchange`; add `ended_via` and `problem_boundaries` to `Exchange`.
- Modify: `tests/test_benchmark_exchange_dynamic.py` — add multi-problem tests.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/test_benchmark_exchange_dynamic.py`, append at the end (after existing tests):

```python
# ---------------------------------------------------------------------------
# Multi-problem (sync) tests (Task 4)
# ---------------------------------------------------------------------------

def _make_scenario_with_queue(queue):
    """Same as _make_scenario but with a next_problem_queue."""
    s = _make_scenario()
    s.next_problem_queue = list(queue)
    return s


def _conv_with_known_text(num_turns=30):
    """A long conversation with predictable turn text per turn number."""
    return {
        "conversation_id": "conv1",
        "turns": [
            {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
             "text": f"original-text-turn-{n}"} for n in range(1, num_turns + 1)
        ],
    }


def test_next_problem_with_empty_queue_ends_via_exhausted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client(["Ready for next? [NEXT_PROBLEM]"])
    student = _stub_client([])  # never called

    scenario = _make_scenario_with_queue([])
    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
    )
    assert ex.completed is True
    assert ex.ended_via == "NEXT_PROBLEM_EXHAUSTED"
    assert ex.problem_boundaries == []
    # No SYSTEM injection because queue was empty.
    assert all(t["role"] in ("TUTOR", "STUDENT") for t in ex.generated_turns)


def test_next_problem_injects_system_and_student_turns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    conv = _conv_with_known_text()
    transcripts = {"conv1": conv}

    # Tutor: round 1 says NEXT_PROBLEM. Round 2 (after injection) says END.
    tutor = _stub_client([
        "Done with this one! [NEXT_PROBLEM]",
        "All wrapped up. [END]",
    ])
    student = _stub_client([])  # student turns come from the injected real text

    scenario = _make_scenario_with_queue([
        {"turn_start": 10, "turn_end": 12, "cut_turn": 10},
    ])

    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
        transcripts=transcripts,
    )

    assert ex.completed is True
    assert ex.ended_via == "END"
    # generated_turns: [TUTOR wrap1, SYSTEM divider, STUDENT injected, TUTOR wrap2]
    roles = [t["role"] for t in ex.generated_turns]
    assert roles == ["TUTOR", "SYSTEM", "STUDENT", "TUTOR"]
    assert ex.generated_turns[0]["text"] == "Done with this one!"
    assert "New problem" in ex.generated_turns[1]["text"]
    # Injected student text is the literal real-transcript text at cut_turn=10:
    assert ex.generated_turns[2]["text"] == "original-text-turn-10"
    assert ex.generated_turns[3]["text"] == "All wrapped up."
    assert ex.problem_boundaries == [
        {"after_turn": ex.generated_turns[0]["turn_number"],
         "source_moment": {"turn_start": 10, "turn_end": 12, "cut_turn": 10}},
    ]


def test_end_token_takes_precedence_no_injection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    conv = _conv_with_known_text()
    transcripts = {"conv1": conv}
    tutor = _stub_client(["Wrap up [NEXT_PROBLEM] [END]"])
    student = _stub_client([])

    scenario = _make_scenario_with_queue([
        {"turn_start": 10, "turn_end": 12, "cut_turn": 10},
    ])
    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
        transcripts=transcripts,
    )
    assert ex.completed is True
    assert ex.ended_via == "END"
    # No SYSTEM divider because END won.
    assert all(t["role"] in ("TUTOR", "STUDENT") for t in ex.generated_turns)
    assert ex.problem_boundaries == []


def test_max_turns_ends_via_max(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client([f"tutor {i}" for i in range(50)])
    student = _stub_client([f"student {i}" for i in range(50)])
    scenario = _make_scenario_with_queue([])
    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
    )
    assert ex.completed is True
    assert ex.ended_via == "MAX_TURNS"
    assert len(ex.generated_turns) == 4


def test_two_next_problems_then_exhausted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )
    conv = _conv_with_known_text()
    transcripts = {"conv1": conv}
    tutor = _stub_client([
        "p1 done [NEXT_PROBLEM]",
        "p2 done [NEXT_PROBLEM]",
        "p3 done [NEXT_PROBLEM]",  # queue empty here -> EXHAUSTED
    ])
    student = _stub_client([])
    scenario = _make_scenario_with_queue([
        {"turn_start": 10, "turn_end": 12, "cut_turn": 10},
        {"turn_start": 14, "turn_end": 16, "cut_turn": 14},
    ])
    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
        transcripts=transcripts,
    )
    assert ex.completed is True
    assert ex.ended_via == "NEXT_PROBLEM_EXHAUSTED"
    assert len(ex.problem_boundaries) == 2
    # SYSTEM dividers + injected student text appear twice:
    assert sum(1 for t in ex.generated_turns if t["role"] == "SYSTEM") == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v -k "next_problem or max_turns_ends_via_max or end_token_takes_precedence"`
Expected: failures — `run_exchange` doesn't accept `transcripts=`, doesn't honour `[NEXT_PROBLEM]`, doesn't set `ended_via` / `problem_boundaries`.

- [ ] **Step 3: Extend the `Exchange` dataclass**

In `benchmark/core/exchange.py`, update the `Exchange` dataclass to add the new fields with safe defaults:

```python
@dataclass
class Exchange:
    scenario_id: str
    tutor_model: str
    generated_turns: list[dict] = field(default_factory=list)
    tutor_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    student_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    completed: bool = False
    ended_via: str = ""                                  # "END" | "NEXT_PROBLEM_EXHAUSTED" | "MAX_TURNS" | ""
    problem_boundaries: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
```

(`field` is already imported with `dataclass`; verify the existing top-of-file import line.)

- [ ] **Step 4: Add the `_inject_next_problem` helper**

After the existing `_append_turns` helper, add:

```python
def _inject_next_problem(
    exchange: Exchange,
    running_transcript: str,
    next_turn_num: int,
    popped: dict,
    conversation: dict,
) -> tuple[str, int]:
    """Append a SYSTEM divider + the real student turn at popped['cut_turn'].

    Mutates exchange.generated_turns and exchange.problem_boundaries.
    Returns updated (running_transcript, next_turn_num).
    """
    ts = popped["turn_start"]
    te = popped["turn_end"]
    cut = popped["cut_turn"]

    # Boundary points at the LAST emitted turn before injection (the tutor's
    # wrap-up). If no turns generated yet, after_turn=0.
    after_turn = exchange.generated_turns[-1]["turn_number"] if exchange.generated_turns else 0
    exchange.problem_boundaries.append({
        "after_turn": after_turn,
        "source_moment": {"turn_start": ts, "turn_end": te, "cut_turn": cut},
    })

    divider_text = f"--- New problem (source turns {ts}-{te}, cut {cut}) ---"
    divider = {"turn_number": next_turn_num, "role": "SYSTEM", "text": divider_text}
    exchange.generated_turns.append(divider)
    running_transcript += f"\nTurn {next_turn_num}. SYSTEM: {divider_text}"
    next_turn_num += 1

    # Inject the literal real student turn at cut_turn from the original transcript.
    real_turn = next(
        (t for t in conversation.get("turns", []) if t["turn_number"] == cut),
        None,
    )
    student_text = real_turn["text"] if real_turn else "..."
    student = {"turn_number": next_turn_num, "role": "STUDENT", "text": student_text}
    exchange.generated_turns.append(student)
    running_transcript += f"\nTurn {next_turn_num}. STUDENT: {student_text}"
    next_turn_num += 1

    return running_transcript, next_turn_num
```

- [ ] **Step 5: Rewrite the tutor-result handling in `run_exchange`**

Replace the existing `run_exchange` function with:

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
    transcripts: dict[str, dict] | None = None,
) -> Exchange:
    """Run a multi-turn exchange for a single scenario (sync mode).

    Runs until the tutor emits END_TOKEN, the next_problem_queue is empty
    after a NEXT_PROBLEM_TOKEN, or generated turns reach max_turns. When
    NEXT_PROBLEM fires we pop the queue and inject a SYSTEM divider + the
    real student turn at the popped moment's cut_turn.

    `transcripts` is required when the scenario's next_problem_queue may
    advance (i.e. queue is non-empty). It defaults to None for back-compat
    with callers that don't need next_problem injection.
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    running_transcript = scenario.transcript_prefix
    next_turn_num = scenario.cut_turn + 1
    queue = list(scenario.next_problem_queue or [])

    conversation = None
    if transcripts is not None:
        conversation = transcripts.get(scenario.conv_id)

    ended_via = ""

    while len(exchange.generated_turns) < max_turns:
        # --- Tutor turn ---
        prompt = _build_role_prompt("TUTOR", running_transcript, scenario.student_context, prompt_version)
        response = tutor_client.generate(
            prompt, json_mode=False, max_tokens=tutor_max_tokens,
            images=images,
        )
        _add_usage(exchange.tutor_usage, response.usage)

        text, ended, next_problem = _parse_tutor_tokens(response.text)
        messages = _split_messages(text)
        if not messages and not (ended or next_problem):
            messages = ["..."]
        if messages:
            running_transcript, next_turn_num = _append_turns(
                exchange, messages, "TUTOR", running_transcript, next_turn_num,
            )

        if ended:
            ended_via = "END"
            break
        if next_problem:
            if not queue:
                ended_via = "NEXT_PROBLEM_EXHAUSTED"
                break
            popped = queue.pop(0)
            if conversation is None:
                # Can't honour the injection without a conversation to read from.
                ended_via = "NEXT_PROBLEM_EXHAUSTED"
                break
            running_transcript, next_turn_num = _inject_next_problem(
                exchange, running_transcript, next_turn_num, popped, conversation,
            )
            # Don't run a synthetic student turn this round -- the injected
            # STUDENT turn is the student's message.
            if len(exchange.generated_turns) >= max_turns:
                ended_via = "MAX_TURNS"
                break
            continue
        if len(exchange.generated_turns) >= max_turns:
            ended_via = "MAX_TURNS"
            break

        # --- Student turn ---
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

    if not ended_via:
        ended_via = "MAX_TURNS"

    exchange.completed = True
    exchange.ended_via = ended_via
    return exchange
```

- [ ] **Step 6: Run the new tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py -v`
Expected: all tests pass — Task 2 parser tests + the new Task 4 multi-problem tests + the prior `run_exchange` / batch tests (which don't pass `transcripts` and will exercise the back-compat None path).

- [ ] **Step 7: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: dynamic run_exchange honours [NEXT_PROBLEM] with queue injection"
```

---

## Task 5: Multi-problem loop in `run_exchanges_batch`

**Files:**
- Modify: `benchmark/core/exchange.py` — extend the batch loop with the same [NEXT_PROBLEM] handling per scenario.
- Modify: `tests/test_benchmark_exchange_dynamic.py` — one batch-mode multi-problem test.

### Steps

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark_exchange_dynamic.py`:

```python
def test_batch_next_problem_per_scenario(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.exchange._load_prompt",
        lambda version, fname: "SYS {student_context}",
    )

    # Two scenarios. s1 emits NEXT_PROBLEM then END. s2 hits max_turns.
    def fake_run_batch(client, entries, json_mode, display_name, poll_interval):
        results = {}
        for e in entries:
            sid = e["custom_id"]
            if display_name.startswith("tutor"):
                round_n = int(display_name.rsplit("_", 1)[-1])
                if sid == "s1" and round_n == 1:
                    text = "p1 done [NEXT_PROBLEM]"
                elif sid == "s1" and round_n == 2:
                    text = "all done [END]"
                else:
                    text = "more"
            else:
                text = "ok"
            results[sid] = {"text": text,
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        return results

    def fake_build_batch_entry(custom_id, prompt, json_mode, max_tokens, images=None):
        return {"custom_id": custom_id}

    monkeypatch.setattr("benchmark.core.exchange.run_batch", fake_run_batch)
    monkeypatch.setattr("benchmark.core.exchange.build_batch_entry", fake_build_batch_entry)

    tutor = MagicMock(); tutor.model = "t"
    student = MagicMock(); student.model = "st"

    s1 = _scenario("s1")
    s1.next_problem_queue = [{"turn_start": 10, "turn_end": 12, "cut_turn": 10}]
    s2 = _scenario("s2")
    s2.next_problem_queue = []

    transcripts = {"s1": _conv_with_known_text(), "s2": _conv_with_known_text()}

    exchanges = run_exchanges_batch(
        scenarios=[s1, s2],
        tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=64, student_max_tokens=64,
        poll_interval=0, prompt_version="v4",
        transcripts=transcripts,
    )

    # s1: round 1 NEXT_PROBLEM -> SYSTEM + STUDENT injected, then round 2 END.
    # Final generated_turns for s1: [TUTOR p1, SYSTEM, STUDENT injected, TUTOR end] -> 4 turns
    assert exchanges["s1"].ended_via == "END"
    assert [t["role"] for t in exchanges["s1"].generated_turns] == [
        "TUTOR", "SYSTEM", "STUDENT", "TUTOR",
    ]
    assert len(exchanges["s1"].problem_boundaries) == 1

    # s2: never emits NEXT_PROBLEM; runs until max_turns=4: T, S, T, S
    assert exchanges["s2"].ended_via == "MAX_TURNS"
    assert [t["role"] for t in exchanges["s2"].generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]
    assert exchanges["s2"].problem_boundaries == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_exchange_dynamic.py::test_batch_next_problem_per_scenario -v`
Expected: failure — `run_exchanges_batch` doesn't accept `transcripts=` and doesn't honour `[NEXT_PROBLEM]`.

- [ ] **Step 3: Extend `run_exchanges_batch`**

In `benchmark/core/exchange.py`, modify the function signature and tutor-result-handling block. Replace the existing function with:

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
    transcripts: dict[str, dict] | None = None,
) -> dict[str, Exchange]:
    """Run multi-turn exchanges for all scenarios using batch API.

    Per-scenario state tracks an independent `next_problem_queue`. On NEXT_PROBLEM
    from a tutor reply, the system injects a SYSTEM divider + the real STUDENT
    turn at the popped moment's cut_turn (from `transcripts`), then continues
    the loop. Scenarios are removed from active_ids when END is signalled, the
    queue is empty after NEXT_PROBLEM, generated turns reach max_turns, or a
    batch result is missing.

    Returns: {scenario_id: Exchange}
    """
    exchanges = {}
    transcripts_buf = {}
    next_turns = {}
    queues: dict[str, list[dict]] = {}
    ended_via: dict[str, str] = {}

    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        transcripts_buf[scenario.scenario_id] = scenario.transcript_prefix
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1
        queues[scenario.scenario_id] = list(scenario.next_problem_queue or [])

    scenario_map = {s.scenario_id: s for s in scenarios}
    active_ids = list(scenario_map.keys())
    needs_student: dict[str, bool] = {sid: True for sid in active_ids}

    max_rounds = math.ceil(max_turns / 2) + len(scenarios)  # +N for possible injections

    for round_num in range(max_rounds):
        if not active_ids:
            break

        # --- Tutor batch ---
        logger.info("Round %d - tutor batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            prompt = _build_role_prompt("TUTOR", transcripts_buf[sid], scenario.student_context, prompt_version)
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
                transcripts_buf[sid], next_turns[sid] = _append_turns(
                    exchange, messages, "TUTOR", transcripts_buf[sid], next_turns[sid],
                )

            if ended:
                ended_via[sid] = "END"
                ended_this_round.append(sid)
                continue
            if next_problem:
                if not queues[sid]:
                    ended_via[sid] = "NEXT_PROBLEM_EXHAUSTED"
                    ended_this_round.append(sid)
                    continue
                conv = (transcripts or {}).get(scenario_map[sid].conv_id)
                if conv is None:
                    ended_via[sid] = "NEXT_PROBLEM_EXHAUSTED"
                    ended_this_round.append(sid)
                    continue
                popped = queues[sid].pop(0)
                transcripts_buf[sid], next_turns[sid] = _inject_next_problem(
                    exchange, transcripts_buf[sid], next_turns[sid], popped, conv,
                )
                needs_student[sid] = False     # injected turn is the student's message
                if len(exchange.generated_turns) >= max_turns:
                    ended_via[sid] = "MAX_TURNS"
                    ended_this_round.append(sid)
                continue

            if len(exchange.generated_turns) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)
                continue

            needs_student[sid] = True

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        # --- Student batch (only scenarios that need one) ---
        student_active = [sid for sid in active_ids if needs_student.get(sid, True)]
        if not student_active:
            if save_callback:
                for sid in scenario_map:
                    save_callback(sid, exchanges[sid])
            continue

        logger.info("Round %d - student batch (%d scenarios)",
                    round_num + 1, len(student_active))
        student_entries = []
        for sid in student_active:
            scenario = scenario_map[sid]
            prompt = _build_role_prompt("STUDENT", transcripts_buf[sid], scenario.student_context,
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
        for sid in student_active:
            result = student_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("student failed for %s", sid[:50])
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.student_usage, result["usage"])

            messages = _split_messages(result["text"]) or ["..."]
            transcripts_buf[sid], next_turns[sid] = _append_turns(
                exchange, messages, "STUDENT", transcripts_buf[sid], next_turns[sid],
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

        # Reset needs_student for surviving scenarios so the next round runs a tutor turn.
        for sid in active_ids:
            needs_student[sid] = True

        if save_callback:
            for sid in scenario_map:
                save_callback(sid, exchanges[sid])

    for sid in scenario_map:
        exchanges[sid].completed = True
        exchanges[sid].ended_via = ended_via.get(sid, "MAX_TURNS")

    logger.info("Exchanges complete: %d scenarios", len(scenario_map))
    return exchanges
```

- [ ] **Step 4: Run all benchmark tests**

Run: `pytest tests/test_benchmark_exchange_dynamic.py tests/test_benchmark_screenshots.py tests/test_benchmark_resume.py tests/test_benchmark_student_modes.py -v`
Expected: all pass.

Then full suite:
Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/exchange.py tests/test_benchmark_exchange_dynamic.py
git commit -m "benchmark: batch run_exchanges_batch honours [NEXT_PROBLEM]"
```

---

## Task 6: Wire `transcripts` through `benchmark/run.py` and flip default to v4

**Files:**
- Modify: `benchmark/run.py` — pass `transcripts` to both `run_exchange` and `run_exchanges_batch` call sites.
- Modify: `config.yaml` — `prompt_version: v3 -> v4`.

### Steps

- [ ] **Step 1: Find the call sites**

Run: `grep -n "run_exchange\|run_exchanges_batch" benchmark/run.py`
Expected: see the two existing call sites (the batch path around line ~213 and the sync path around line ~228) and confirm where the transcripts dict is already loaded for screenshots.

- [ ] **Step 2: Ensure transcripts are loaded for both modes**

The run already loads `transcripts_for_screenshots` only when `with_screenshots=True`. The `[NEXT_PROBLEM]` injection needs the same dict unconditionally when scenarios have non-empty queues. Modify `benchmark/run.py` to load transcripts once early in `run_benchmark` (after the existing config-validation block) regardless of `with_screenshots`:

Search for `transcripts_for_screenshots = None` and update the loader to:

```python
    from annotator.core.storage import load_all_transcripts
    transcripts_for_screenshots = load_all_transcripts()
    transcripts_for_exchange = transcripts_for_screenshots
```

If `transcripts_for_screenshots` was loaded only conditionally before, lift it out so it's always available. Use the same dict for both purposes.

- [ ] **Step 3: Pass transcripts to both exchange call sites**

Find and update the batch path:

```python
                new_exchanges = run_exchanges_batch(
                    scenarios=missing,
                    tutor_client=tutor_client,
                    student_client=student_client,
                    max_turns=exchange_cfg["max_turns"],
                    tutor_max_tokens=tutor_cfg["max_tokens"],
                    student_max_tokens=student_cfg["max_tokens"],
                    poll_interval=exchange_cfg["poll_interval"],
                    save_callback=_save_exchange,
                    prompt_version=exchange_prompt_version,
                    images_by_scenario=images_by_scenario,
                    student_mode=student_mode,
                    transcripts=transcripts_for_exchange,
                )
```

And the sync path:

```python
                        exchange = run_exchange(
                            scenario=scenario,
                            tutor_client=tutor_client,
                            student_client=student_client,
                            max_turns=exchange_cfg["max_turns"],
                            tutor_max_tokens=tutor_cfg["max_tokens"],
                            student_max_tokens=student_cfg["max_tokens"],
                            prompt_version=exchange_prompt_version,
                            images=(images_by_scenario or {}).get(scenario.scenario_id),
                            student_mode=student_mode,
                            transcripts=transcripts_for_exchange,
                        )
```

- [ ] **Step 4: Flip default in `config.yaml`**

Edit `config.yaml`. Change the `benchmark.exchange.prompt_version` value from `v3` to `v4`. Update the inline comment too.

Final block:

```yaml
  exchange:
    max_turns: 100               # was: num_turns: 2 — tutor decides end via [END]/[NEXT_PROBLEM]; this is a hard cap on generated turns
    poll_interval: 60
    prompt_version: v4           # v4 = v3 + [NEXT_PROBLEM] token alongside [END]
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -q --ignore=tests/test_eval_metrics.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add benchmark/run.py config.yaml
git commit -m "benchmark: wire transcripts to exchanges + flip default to v4"
```

---

## Task 7: SYSTEM-role styling in viewers

**Files:**
- Modify: `benchmark/eval/view.py` — add a CSS rule and role class for `SYSTEM` turns.
- Modify: `benchmark/eval/view_replay.py` — same.

### Steps

- [ ] **Step 1: Add SYSTEM styling to `view.py`**

In `benchmark/eval/view.py`, find the CSS block that defines `.turn .role.tutor` / `.turn .role.student` and add a sibling rule. Also extend the JS that picks `roleClass` from the turn role.

CSS additions (near the existing `.turn .role.tutor` rule, ~line 215-225):

```
.turn .role.system {{ color: #6c757d; font-style: italic; }}
.turn.system {{ background: #f0f0f3; border-left: 3px solid #9e9e9e; color: #555; font-style: italic; }}
```

JS update inside `renderTranscript` -- change:

```javascript
    const roleClass = role === 'tutor' ? 'tutor' : 'student';
```

to:

```javascript
    const roleClass = role === 'tutor' ? 'tutor' : (role === 'system' ? 'system' : 'student');
```

And update the `.turn` className construction to include `system` when applicable:

```javascript
    let bgClass = isGen ? 'generated' : 'original';
    if (role === 'system') bgClass = 'system';
```

- [ ] **Step 2: Add SYSTEM styling to `view_replay.py`**

In `benchmark/eval/view_replay.py`, find the CSS block with `.turn .role.tutor` and `.turn.prefix` etc., add:

```
.turn .role.system {{ color: #6c757d; font-style: italic; }}
.turn.system {{ background: #f0f0f3; border-left-color: #9e9e9e; color: #555; font-style: italic; }}
```

In the JS function `renderTurns`, replace:

```javascript
    const roleClass = (t.role || '').toLowerCase() === 'tutor' ? 'tutor' : 'student';
```

with:

```javascript
    const role = (t.role || '').toLowerCase();
    const roleClass = role === 'tutor' ? 'tutor' : (role === 'system' ? 'system' : 'student');
```

(`t.kind` already drives the row background; add a guard so SYSTEM-roled turns get the dimmer background even when their kind is `ai_generated`):

```javascript
    let kindClass = t.kind;
    if (role === 'system') kindClass = 'system';
```

Then use `kindClass` in the row's `class="turn ..."`.

- [ ] **Step 3: Regenerate viewers against the prior smoke (no SYSTEM turns expected yet)**

Run:
```bash
PYTHONIOENCODING=utf-8 python -m benchmark.eval.view_replay --version dyn_smoke_v12_2026_06_09 --profile anthropic
```
Expected: HTML re-generates without errors. No SYSTEM turns in this old smoke; the styling change is just future-proofing.

- [ ] **Step 4: Commit**

```bash
git add benchmark/eval/view.py benchmark/eval/view_replay.py
git commit -m "viewer: style SYSTEM-role turns (gray divider) for multi-problem exchanges"
```

---

## Task 8: End-to-end smoke

**Files:** none modified -- verification only.

### Steps

- [ ] **Step 1: Run a small smoke under v4**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version next_problem_smoke_2026_06_09 --scenario-mode human --max-scenarios 5 --mode sync
```

Expected:
- Step 0 detection is skipped (human mode).
- 5 scenarios run. At least some scenarios should have non-empty `next_problem_queue` and emit one or more `[NEXT_PROBLEM]` tokens.
- Phase 1 logs show shorter exchanges than the old `dyn_smoke` (since the tutor now hands off rather than inventing problems).

If all 5 scenarios end via `MAX_TURNS` with zero `problem_boundaries`, the tutor isn't using `[NEXT_PROBLEM]` -- inspect a saved tutor reply for the prompt-version mismatch or token misspelling.

- [ ] **Step 2: Inspect saved exchanges**

```bash
PYTHONIOENCODING=utf-8 python -c "
from annotator.core.storage import load_benchmark_result, list_benchmark_result_files
files = list_benchmark_result_files('next_problem_smoke_2026_06_09', 'exchanges', 'anthropic')
for f in files:
    ex = load_benchmark_result('next_problem_smoke_2026_06_09', 'exchanges', 'anthropic', f)
    pb = ex.get('problem_boundaries') or []
    last = (ex['generated_turns'][-1]['text'][-60:].encode('ascii','replace').decode() if ex['generated_turns'] else '')
    print(f[:30], '| turns=', len(ex['generated_turns']), '| via=', ex.get('ended_via',''), '| problems=', len(pb), '| last=', repr(last))
"
```

Expected: a mix of `ended_via` values (`END`, `NEXT_PROBLEM_EXHAUSTED`, occasionally `MAX_TURNS`); some scenarios show `problems >= 1`.

- [ ] **Step 3: Generate the replay viewer**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark.eval.view_replay --version next_problem_smoke_2026_06_09 --profile anthropic
```

Expected: HTML at `results/benchmark/next_problem_smoke_2026_06_09/viewer_replay_anthropic.html`. SYSTEM divider turns visible in the replayed column at problem boundaries.

- [ ] **Step 4: Update `docs/status.md`**

Prepend a new "Recently Shipped" block at the top of `docs/status.md`:

```markdown
## Recently Shipped: Multi-Problem Exchanges (2026-06-09)

Benchmark tutor can now emit `[NEXT_PROBLEM]` to advance to the next teachable
moment in the same conversation (system serves up the real next-moment student
turn from the original transcript) and `[END]` to terminate the scenario.

- New `prompts/benchmark/v4/tutor_system.txt` with two-token instruction.
- `Scenario` gains `next_problem_queue` (subsequent moments in the same conv,
  modal-cut + role-adjusted).
- `Exchange` gains `problem_boundaries` and `ended_via` (`END` /
  `NEXT_PROBLEM_EXHAUSTED` / `MAX_TURNS`).
- Replay viewer renders SYSTEM divider turns at problem boundaries.

Spec: [plans/specs/2026-06-09-next-problem-multi-moment-exchange-design.md](plans/specs/2026-06-09-next-problem-multi-moment-exchange-design.md)
Plan: [plans/2026-06-09-next-problem-multi-moment-exchange.md](plans/2026-06-09-next-problem-multi-moment-exchange.md)
```

Update the `*Last updated:*` line to `2026-06-09`.

- [ ] **Step 5: Commit**

```bash
git add docs/status.md
git commit -m "docs: status update for multi-problem exchanges"
```

---

## Self-Review

**Spec coverage:**
- Tutor system prompt v4 with two tokens — Task 1.
- `_parse_tutor_tokens` helper with END precedence — Task 2.
- `next_problem_queue` populated from subsequent moments via modal-cut + role-adjust — Task 3.
- `_inject_next_problem` and dynamic sync loop — Task 4.
- Batch loop with per-scenario queue + injection — Task 5.
- `Exchange.problem_boundaries` + `Exchange.ended_via` — Tasks 4 & 5.
- `transcripts` wired into both call sites + v4 default — Task 6.
- SYSTEM-role styling — Task 7.
- End-to-end smoke + status doc — Task 8.

All spec sections map to a task.

**Placeholder scan:** no TBDs or hand-waves; all helper bodies, test bodies, and steps include concrete code.

**Type/name consistency:**
- `_parse_tutor_tokens`, `_inject_next_problem`, `NEXT_PROBLEM_TOKEN`, `END_TOKEN`, `next_problem_queue`, `problem_boundaries`, `ended_via`, `_resolve_cluster` used consistently across Tasks 2–7.
- `transcripts` parameter name matches in `run_exchange` / `run_exchanges_batch` / `benchmark.run.py`.
- `Scenario.next_problem_queue` reads from spec; queue entry shape `{turn_start, turn_end, cut_turn}` matches throughout.
- `Exchange.ended_via` values are exactly `"END"`, `"NEXT_PROBLEM_EXHAUSTED"`, `"MAX_TURNS"` everywhere.
