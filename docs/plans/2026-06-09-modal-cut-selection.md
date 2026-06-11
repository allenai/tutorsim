# Modal Cut Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-moment scenario emission with per-cluster modal-cut selection in `extract_human_scenarios`. Cluster moments by `(conv_id, turn_start, turn_end)`, pick the modal cut across teacher votes (smallest on tie), then role-adjust (cut-1 when cut lands on a TUTOR turn).

**Architecture:** All changes inside `benchmark/core/scenarios.py`. Three new private helpers (`_pick_modal_cut`, `_role_adjust_cut`, `_pick_representative_member`) added in TDD. `extract_human_scenarios` rewritten to cluster-first; `scenario_id` format changes from `__hum_{idx}` to `__hum_{ts}_{te}`. Existing tests updated for new IDs and behaviour; new cluster-specific tests added.

**Tech Stack:** Python 3.11, pytest. Single test file: `tests/test_benchmark_human_scenarios.py`.

**Spec:** [`docs/plans/specs/2026-06-09-modal-cut-selection-design.md`](specs/2026-06-09-modal-cut-selection-design.md)

---

## File Map

- **Modify:** `benchmark/core/scenarios.py` — add three helpers; rewrite `extract_human_scenarios` to cluster-and-pick.
- **Modify:** `tests/test_benchmark_human_scenarios.py` — replace fixture with one that exercises the new selection logic; update existing assertions for new `scenario_id` format; add new tests for modal pick / tie-break / role adjustment / vote filtering.

No other files touched. `benchmark/run.py`, `benchmark/core/annotator_bridge.py`, `config.yaml` are unchanged. The `Scenario` dataclass is unchanged.

---

## Task 1: Add helper functions (TDD)

**Files:**
- Modify: `benchmark/core/scenarios.py` (add three helpers near the existing `_SCAFF_AGGS` constant)
- Modify: `tests/test_benchmark_human_scenarios.py` (add a new test class / section for helpers)

### Steps

- [ ] **Step 1: Write failing tests for `_pick_modal_cut`**

Append to the **top** of `tests/test_benchmark_human_scenarios.py` (immediately after the existing imports), a new import line and a fresh test section. (Existing tests stay intact; we'll update them in Task 2.)

Add the import:

```python
from benchmark.core.scenarios import (
    _pick_modal_cut, _role_adjust_cut, _pick_representative_member,
)
```

Add these tests at the very end of the file:

```python
# ---------------------------------------------------------------------------
# Helper tests (Task 1)
# ---------------------------------------------------------------------------

def test_pick_modal_cut_single_winner():
    assert _pick_modal_cut([4, 4, 5]) == 4


def test_pick_modal_cut_tie_returns_smallest():
    assert _pick_modal_cut([4, 5]) == 4
    assert _pick_modal_cut([7, 3, 7, 3]) == 3


def test_pick_modal_cut_singleton():
    assert _pick_modal_cut([9]) == 9


def test_pick_modal_cut_empty_returns_none():
    assert _pick_modal_cut([]) is None


def _conv(turns_pattern):
    """Build a conversation dict with turn roles per turns_pattern[i] for turn_number i+1."""
    return {
        "turns": [
            {"turn_number": n + 1, "role": role, "text": f"t{n+1}"}
            for n, role in enumerate(turns_pattern)
        ],
    }


def test_role_adjust_cut_student_no_change():
    # Odd-numbered turns are TUTOR, even are STUDENT (matches the main fixture below).
    conv = _conv(["TUTOR", "STUDENT"] * 10)  # turns 1..20
    assert _role_adjust_cut(6, conv) == 6  # turn 6 is STUDENT


def test_role_adjust_cut_tutor_decrements():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    assert _role_adjust_cut(11, conv) == 10  # turn 11 is TUTOR -> cut-1


def test_role_adjust_cut_tutor_at_first_turn_returns_none():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    # cut=1 is TUTOR; adjustment would give 0, which is below the 1-turn minimum.
    assert _role_adjust_cut(1, conv) is None


def test_role_adjust_cut_missing_turn_returns_none():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    assert _role_adjust_cut(99, conv) is None


def test_pick_representative_member_prefers_modal_voter():
    members = [
        {"annotator_id": "z", "cut_turn": 9, "situation": "z-other"},
        {"annotator_id": "b", "cut_turn": 6, "situation": "b-modal"},
        {"annotator_id": "a", "cut_turn": 6, "situation": "a-modal"},
    ]
    rep = _pick_representative_member(members, chosen_cut=6)
    # Among members voting modal=6, smallest annotator_id wins.
    assert rep["annotator_id"] == "a"
    assert rep["situation"] == "a-modal"


def test_pick_representative_member_falls_back_when_no_modal_voter():
    # Defensive: if no member voted the chosen cut (shouldn't happen in practice),
    # return the smallest-annotator_id member overall.
    members = [
        {"annotator_id": "z", "cut_turn": 9, "situation": "z"},
        {"annotator_id": "a", "cut_turn": 8, "situation": "a"},
    ]
    rep = _pick_representative_member(members, chosen_cut=6)
    assert rep["annotator_id"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: `ImportError: cannot import name '_pick_modal_cut'` (and the two siblings) — none of the helpers exist yet.

- [ ] **Step 3: Implement the helpers**

In `benchmark/core/scenarios.py`, near the existing `_SCAFF_AGGS` constant (around the top of the helpers area), add:

```python
from collections import Counter


def _pick_modal_cut(votes: list[int]) -> int | None:
    """Return the most-voted cut. On tie, return the smallest.

    Returns None when votes is empty.
    """
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [c for c, n in counts.items() if n == max_count]
    return min(winners)


def _role_adjust_cut(cut_turn: int, conversation: dict) -> int | None:
    """Adjust cut_turn based on the role of the turn at cut_turn.

    - STUDENT: cut stays as-is (prefix includes the student turn).
    - TUTOR: cut_turn -= 1 (prefix excludes the human tutor turn; AI replaces it).
    - turn not found OR adjustment falls below 1: returns None (caller drops the cluster).
    """
    turns_by_n = {t["turn_number"]: t for t in conversation.get("turns", [])}
    turn = turns_by_n.get(cut_turn)
    if turn is None:
        return None
    if turn.get("role") == "TUTOR":
        adjusted = cut_turn - 1
        if adjusted < 1:
            return None
        return adjusted
    return cut_turn


def _pick_representative_member(members: list[dict], chosen_cut: int) -> dict:
    """Return the member to use for the scenario's detection payload.

    Preference: members whose own cut_turn equals the chosen (modal) cut,
    smallest annotator_id (lexicographic) on tie. Falls back to smallest
    annotator_id overall if none of the members voted the chosen cut.
    """
    matching = [m for m in members if m.get("cut_turn") == chosen_cut]
    pool = matching if matching else members
    return min(pool, key=lambda m: (m.get("annotator_id") or ""))
```

(If `from collections import Counter` is already imported at the top of the file, don't duplicate — move the import up there.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_human_scenarios.py -v -k 'pick_modal_cut or role_adjust_cut or pick_representative_member'`
Expected: 11 passed (4 modal + 4 role_adjust + 2 rep_member + the import line covered).

Then run the rest of the file to confirm existing tests still pass (the function hasn't been rewritten yet, so they should):
Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: helpers pass; existing tests still pass (they don't exercise the helpers).

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/scenarios.py tests/test_benchmark_human_scenarios.py
git commit -m "benchmark: add cluster/modal/role helpers for human scenarios"
```

---

## Task 2: Rewrite `extract_human_scenarios` to cluster-and-pick (TDD)

**Files:**
- Modify: `benchmark/core/scenarios.py` (rewrite `extract_human_scenarios`)
- Modify: `tests/test_benchmark_human_scenarios.py` (replace `_gt_files()` fixture and update existing assertions; add cluster-specific tests)

### Steps

- [ ] **Step 1: Replace the test fixture and add new behavior tests**

Edit `tests/test_benchmark_human_scenarios.py`. Replace the existing `_gt_files()` function (and adjust the existing tests below) with the new fixture and test bodies below. Keep `_make_transcripts()`, `_run_extract()`, and the helper-tests from Task 1 unchanged.

Replace `_gt_files` with:

```python
def _gt_files():
    """Hybrid GT shape that exercises clustering + modal cut + tie + role adjust.

    Transcript A turn roles: odd=TUTOR, even=STUDENT (1..20).
    Clusters and their expected scenario behavior:
      (5,8)   singleton, cut=6 STUDENT -> kept, cut_turn=6
      (10,12) singleton, cut=11 TUTOR  -> kept, cut_turn=10 (adjusted)
      (3,5)   two members vote 4 and 5 (tie) -> smallest=4 STUDENT -> cut_turn=4
      (16,18) three members vote 17, 17, 18 -> modal=17 TUTOR -> cut_turn=16 (adjusted)
      (13,15) mixed agg -> cluster dropped
      (17,19) rapport -> cluster dropped (situation_label_agg absent)
      (2,4)   votes 1 (< ts) and 5 (> te) -> all votes filtered -> cluster dropped
    Transcript B:
      (8,9)   rigor singleton, cut=8 STUDENT -> kept, cut_turn=8
    Transcript C: not in transcripts dict -> all moments dropped.
    """
    def m(ts, te, ann, cut, agg="scaffolding", ann_type="scaffolding",
          situation="S", moment_id=None):
        d = {
            "turn_start": ts, "turn_end": te,
            "annotation_type": ann_type, "annotator_id": ann,
            "situation": situation, "action": "A", "result": "R",
            "strategy_label": "effective",
        }
        if agg is not None:
            d["situation_label_agg"] = agg
        if cut is not None:
            d["cut_turn"] = cut
        if moment_id is not None:
            d["moment_id"] = moment_id
        return d

    return [
        {
            "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "num_turns": 20,
            "key_moments": [
                m(5, 8, "ann1", 6, situation="S_5_8", moment_id="m1"),
                m(10, 12, "ann1", 11, situation="S_10_12", moment_id="m2"),
                m(3, 5, "ann1", 4, situation="S_3_5_v4"),
                m(3, 5, "ann2", 5, situation="S_3_5_v5"),
                m(16, 18, "ann1", 17, situation="S_16_18_a"),
                m(16, 18, "ann2", 17, situation="S_16_18_b"),
                m(16, 18, "ann1", 18, situation="S_16_18_c"),
                m(13, 15, "ann1", 14, agg="mixed"),
                m(17, 19, "ann1", 18, ann_type="rapport", agg=None),
                m(2, 4, "ann1", 1),                      # cut < ts -> vote dropped
                m(2, 4, "ann2", 5),                      # cut > te -> vote dropped
            ],
        },
        {
            "conversation_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "num_turns": 20,
            "key_moments": [
                m(8, 9, "ann2", 8, agg="rigor", situation="S_8_9", moment_id="m5"),
            ],
        },
        {
            "conversation_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "num_turns": 20,
            "key_moments": [
                m(3, 5, "ann3", 4),
            ],
        },
    ]
```

Now replace the existing top-of-file tests with the following set. Find the block from `def test_filters_to_scaff_or_rigor_with_cut_turn():` through `def test_load_scenarios_human_mode_skips_detection_and_extracts():` (inclusive) and replace it with:

```python
def test_extracts_one_scenario_per_cluster():
    scenarios = _run_extract()
    # Transcript A keeps 4 clusters (5-8, 10-12, 3-5, 16-18); transcript B keeps 1
    # (8-9). Transcripts C / dropped clusters contribute 0.
    assert len(scenarios) == 5


def test_scenario_id_uses_turn_range():
    scenarios = _run_extract()
    ids = {s.scenario_id.rsplit("__", 1)[1] for s in scenarios}
    assert ids == {"hum_5_8", "hum_10_12", "hum_3_5", "hum_16_18", "hum_8_9"}


def test_singleton_student_cut_is_unchanged():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_5_8"))
    assert s.cut_turn == 6
    assert s.detection["chosen_cut_turn"] == 6
    assert s.detection["situation"] == "S_5_8"
    assert s.detection["moment_id"] == "m1"
    assert s.detection["annotator_id"] == "ann1"
    assert s.detection["situation_label_agg"] == "scaffolding"
    assert s.detection["cluster_size"] == 1
    assert s.detection["cut_votes"] == {6: 1}
    # Prefix includes turn 6:
    assert "Turn 6." in s.transcript_prefix
    assert "Turn 7." not in s.transcript_prefix


def test_singleton_tutor_cut_is_adjusted():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_10_12"))
    # cut_turn=11 is TUTOR (odd) -> adjusted to 10.
    assert s.cut_turn == 10
    assert s.detection["chosen_cut_turn"] == 11
    assert "Turn 10." in s.transcript_prefix
    assert "Turn 11." not in s.transcript_prefix


def test_tie_resolves_to_smallest_cut():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_3_5"))
    # votes were 4 and 5 -> tie -> smallest=4 (STUDENT, no adjustment)
    assert s.cut_turn == 4
    assert s.detection["chosen_cut_turn"] == 4
    assert s.detection["cut_votes"] == {4: 1, 5: 1}
    assert s.detection["cluster_size"] == 2


def test_same_annotator_dupes_inflate_votes_and_modal_wins():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_16_18"))
    # votes: ann1 -> 17, ann2 -> 17, ann1 -> 18.  modal=17 (TUTOR) -> adjusted to 16.
    assert s.cut_turn == 16
    assert s.detection["chosen_cut_turn"] == 17
    assert s.detection["cut_votes"] == {17: 2, 18: 1}
    assert s.detection["cluster_size"] == 3
    # Representative should be one of the modal voters (ann1 or ann2, smallest annotator_id):
    assert s.detection["annotator_id"] == "ann1"


def test_drops_cluster_when_all_votes_out_of_range():
    scenarios = _run_extract()
    assert not any(s.scenario_id.endswith("__hum_2_4") for s in scenarios)


def test_drops_mixed_and_rapport_clusters():
    scenarios = _run_extract()
    assert not any(s.scenario_id.endswith("__hum_13_15") for s in scenarios)
    assert not any(s.scenario_id.endswith("__hum_17_19") for s in scenarios)


def test_skips_when_transcript_missing():
    scenarios = _run_extract()
    assert not any(s.conv_id.startswith("cccccccc") for s in scenarios)


def test_scenario_id_stable_across_runs():
    a = _run_extract()
    b = _run_extract()
    assert sorted(s.scenario_id for s in a) == sorted(s.scenario_id for s in b)


def test_detection_shape_is_compatible_with_annotator_bridge():
    """Round-trip: a human scenario must work with build_synthetic_detections."""
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_5_8"))
    fake_exchange = Exchange(
        scenario_id=s.scenario_id,
        tutor_model="x",
        generated_turns=[
            {"turn_number": s.cut_turn + 1, "role": "TUTOR", "text": "ok"},
            {"turn_number": s.cut_turn + 2, "role": "STUDENT", "text": "yes"},
        ],
        tutor_usage={}, student_usage={}, completed=True,
    )
    detections = build_synthetic_detections(s, fake_exchange)
    assert isinstance(detections, dict)
    assert s.conv_id in detections
    conv_detections = detections[s.conv_id]["detections"]
    assert len(conv_detections) == 1
    det = conv_detections[0]
    assert det["annotation_type"] == "scaffolding"
    assert det["turn_start"] == s.detection["turn_start"]
    assert det["turn_end"] == s.cut_turn + 2


def test_load_scenarios_human_mode_skips_detection_and_extracts():
    transcripts = _make_transcripts()
    with patch("benchmark.core.scenarios.load_all_ground_truth_files",
               return_value=_gt_files()), \
         patch("benchmark.core.scenarios.load_transcripts",
               return_value=transcripts):
        scenarios = load_scenarios(
            {"mode": "human"},
            detections_by_conv=None,   # MUST NOT raise -- human mode skips detection
        )
    assert len(scenarios) == 5
    assert all(s.mode == "human" for s in scenarios)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: many tests fail in the non-helper section (the current `extract_human_scenarios` still emits 2 scenarios with old IDs and lacks the new `cut_votes` / `cluster_size` / `chosen_cut_turn` keys). Helpers from Task 1 still pass.

- [ ] **Step 3: Rewrite `extract_human_scenarios`**

Replace the existing `extract_human_scenarios` body in `benchmark/core/scenarios.py` with:

```python
def extract_human_scenarios(transcripts: dict[str, dict]) -> list[Scenario]:
    """Build one scenario per moment cluster, using the modal teacher cut.

    Algorithm:
      1. Filter to situation_label_agg in {scaffolding, rigor}.
      2. Cluster moments by (conv_id, turn_start, turn_end).
      3. Per cluster, collect cut votes; drop a vote if cut_turn is absent
         or outside [turn_start, turn_end].
      4. Drop clusters with zero valid votes.
      5. Pick modal cut (smallest on tie).
      6. Role-adjust the chosen cut against the transcript's turns (TUTOR -> cut-1,
         STUDENT -> unchanged). Drop the cluster if adjustment fails.
      7. Pick a representative cluster member for the detection payload.

    The produced Scenario.detection dict stays shape-compatible with
    annotator_bridge.build_synthetic_detections (keys: turn_start, turn_end,
    annotation_type, situation) and adds chosen_cut_turn / cut_votes /
    cluster_size for traceability.
    """
    uuid_to_conv = {_conv_id_to_uuid(cid): cid for cid in transcripts}

    # Build clusters: (full_conv_id, ts, te) -> list[moment]
    clusters: dict[tuple, list[dict]] = {}
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
            clusters.setdefault((full_conv_id, ts, te), []).append(m)

    scenarios: list[Scenario] = []
    for (full_conv_id, ts, te), members in clusters.items():
        votes: list[int] = []
        for m in members:
            cut = m.get("cut_turn")
            if not isinstance(cut, int):
                continue
            if cut < ts or cut > te:
                continue
            votes.append(cut)

        chosen = _pick_modal_cut(votes)
        if chosen is None:
            continue

        conversation = transcripts[full_conv_id]
        adjusted = _role_adjust_cut(chosen, conversation)
        if adjusted is None:
            continue

        prefix = _format_prefix(conversation, adjusted)
        if not prefix:
            continue

        rep = _pick_representative_member(members, chosen)
        vote_counts = dict(Counter(votes))

        detection = {
            "turn_start": ts,
            "turn_end": te,
            # situation_label_agg is only set on annotation_type=="scaffolding"
            # records, so all selected moments are scaffolding.
            "annotation_type": "scaffolding",
            "situation": rep.get("situation", ""),
            "situation_label_agg": rep.get("situation_label_agg"),
            "moment_id": rep.get("moment_id"),
            "annotator_id": rep.get("annotator_id"),
            "chosen_cut_turn": chosen,
            "cut_votes": vote_counts,
            "cluster_size": len(members),
        }

        scenarios.append(Scenario(
            scenario_id=f"{full_conv_id}__hum_{ts}_{te}",
            conv_id=full_conv_id,
            cut_turn=adjusted,
            transcript_prefix=prefix,
            student_context=_get_student_context(conversation),
            last_student_message=_last_student_msg(conversation, adjusted),
            mode="human",
            detection=detection,
        ))

    return scenarios
```

If `from collections import Counter` was added by Task 1, leave it; otherwise add it now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: all tests pass (helpers from Task 1 + the new cluster tests). Roughly 22 tests total in this file.

Then confirm nothing else regressed:
Run: `pytest tests/ -q`
Expected: all green (pre-existing `test_eval_metrics.py` `krippendorff` import error is unrelated and OK to ignore).

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/scenarios.py tests/test_benchmark_human_scenarios.py
git commit -m "benchmark: cluster human moments and pick modal teacher cut"
```

---

## Task 3: Smoke against real data

**Files:** none modified — verification only.

### Steps

- [ ] **Step 1: Recompute scenario counts against the real `ground_truth_hybrid`**

Run (Windows shell — escape inner quotes appropriately, or save as a script):
```bash
PYTHONIOENCODING=utf-8 python -c "
from benchmark.core.scenarios import extract_human_scenarios
from annotator.core.utils import load_transcripts
ss = extract_human_scenarios(load_transcripts())
print(f'scenarios={len(ss)}, convs={len({s.conv_id for s in ss})}')
# Distribution: how many came from clusters with multiple votes?
multi = [s for s in ss if s.detection.get('cluster_size', 1) > 1]
print(f'multi-vote clusters: {len(multi)} ({100*len(multi)/max(1,len(ss)):.1f}%)')
adjusted = [s for s in ss if s.detection.get('chosen_cut_turn') != s.cut_turn]
print(f'role-adjusted (chosen != cut_turn): {len(adjusted)} ({100*len(adjusted)/max(1,len(ss)):.1f}%)')
"
```

Expected: roughly `scenarios=~993, convs=~115` (numbers may drift as `ground_truth_hybrid` is resynced). Multi-vote clusters should be a substantial chunk (~727 by the data inspected during brainstorming). Some non-trivial fraction will be role-adjusted.

If the scenario count is wildly off (e.g. 0 or ~3000), stop and investigate — the cluster keying or vote filter may be wrong.

- [ ] **Step 2: Spot-check one tie and one role-adjusted scenario**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "
from benchmark.core.scenarios import extract_human_scenarios
from annotator.core.utils import load_transcripts
ss = extract_human_scenarios(load_transcripts())
# Sample a tied cluster
tied = next((s for s in ss if max(s.detection['cut_votes'].values()) == 1 and len(s.detection['cut_votes']) > 1), None)
print('tied sample:', tied.scenario_id, 'votes=', tied.detection['cut_votes'], 'cut_turn=', tied.cut_turn, 'chosen=', tied.detection['chosen_cut_turn']) if tied else print('no tied cluster found')
# Sample a role-adjusted scenario
adj = next((s for s in ss if s.detection['chosen_cut_turn'] != s.cut_turn), None)
print('role-adj sample:', adj.scenario_id, 'chosen=', adj.detection['chosen_cut_turn'], 'cut_turn=', adj.cut_turn) if adj else print('no adjusted scenarios found')
"
```

Expected: smallest vote wins on ties; `cut_turn == chosen - 1` on role-adjusted samples.

- [ ] **Step 3: Update `docs/status.md`**

Prepend a short block at the top of `docs/status.md` summarizing the change. Keep prior sections intact.

```markdown
## Recently Shipped: Modal Cut Selection (2026-06-09)

Benchmark human-mode scenario extraction now clusters moments by exact turn-range
and picks the modal teacher cut (smallest on tie) per cluster. Role-adjusts the
chosen cut so the AI tutor always speaks first (TUTOR cut -> cut - 1).

- New helpers in `benchmark/core/scenarios.py`: `_pick_modal_cut`, `_role_adjust_cut`,
  `_pick_representative_member`.
- `scenario_id` format: `{conv_id}__hum_{turn_start}_{turn_end}` (range-based, stable).
- `Scenario.detection` gains `chosen_cut_turn`, `cut_votes`, `cluster_size`.
- Scope: ~993 scenarios (down from 2,975 raw moments; ~3x dedup).

Spec: [plans/specs/2026-06-09-modal-cut-selection-design.md](plans/specs/2026-06-09-modal-cut-selection-design.md)
Plan: [plans/2026-06-09-modal-cut-selection.md](plans/2026-06-09-modal-cut-selection.md)
```

Update the `*Last updated:*` line at the top of the file to `2026-06-09`.

- [ ] **Step 4: Commit**

```bash
git add docs/status.md
git commit -m "docs: status update for modal cut selection"
```

---

## Self-Review

**Spec coverage:**
- Cluster by (conv_id, ts, te) — Task 2 (rewrite body).
- Filter by situation_label_agg — Task 2.
- Drop votes with absent/out-of-range cut_turn — Task 2 (vote-collection loop) + Task 2 test (`test_drops_cluster_when_all_votes_out_of_range`).
- Modal pick + tie-break to smallest — Task 1 (`_pick_modal_cut`) + Task 2 test (`test_tie_resolves_to_smallest_cut`).
- Same-annotator duplicates inflate votes — Task 2 test (`test_same_annotator_dupes_inflate_votes_and_modal_wins`).
- Role adjustment (STUDENT/TUTOR/missing/below-1) — Task 1 (`_role_adjust_cut`) + tests + Task 2 tests for singleton TUTOR/STUDENT cases.
- Representative member selection — Task 1 (`_pick_representative_member`) + tests + Task 2 test asserts `detection["annotator_id"]`.
- New `scenario_id` format — Task 2 test (`test_scenario_id_uses_turn_range`).
- Detection enrichment (`chosen_cut_turn`, `cut_votes`, `cluster_size`) — Task 2 tests.
- Smoke against real data — Task 3.

All spec items covered.

**Placeholder scan:** no TBDs, no hand-waves; all code blocks are concrete.

**Type/name consistency:**
- `_pick_modal_cut`, `_role_adjust_cut`, `_pick_representative_member` defined in Task 1 and called by name in Task 2 body.
- `cluster_size`, `cut_votes`, `chosen_cut_turn` keys used consistently across Task 2 implementation and tests.
- `scenario_id` format `__hum_{ts}_{te}` matches between implementation and tests.
- `_SCAFF_AGGS` is referenced (unchanged from prior).
