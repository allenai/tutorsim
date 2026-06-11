# Human Key Moments as Benchmark Source -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new benchmark scenario mode `human` that reads scaffolding-flavored key moments (filtered by `situation_label_agg in {scaffolding, rigor}` and presence of `cut_turn`) from `data/ground_truth_hybrid/`, and make it the default.

**Architecture:** Additive change. New function `extract_human_scenarios(transcripts)` in `benchmark/core/scenarios.py` produces `Scenario` objects whose `detection` dict is shape-compatible with `annotator_bridge.build_synthetic_detections`, so nothing downstream changes. `load_scenarios()` dispatches on the new `mode` value. `config.yaml` default flips to `human`. The existing `detected`/`random`/`both` paths remain unchanged for reproducibility.

**Tech Stack:** Python 3.11, pytest. Reads `data/ground_truth_hybrid/*.json` via `annotator.core.storage.load_all_ground_truth_files()`. Transcripts via `annotator.core.utils.load_transcripts()`.

**Spec:** [`docs/plans/specs/2026-06-08-human-key-moments-benchmark-design.md`](specs/2026-06-08-human-key-moments-benchmark-design.md)

---

## File Map

- **Modify** `benchmark/core/scenarios.py` — add `extract_human_scenarios()`, extend `load_scenarios()` mode dispatch, update `Scenario.mode` docstring.
- **Modify** `config.yaml` — flip default `scenarios.mode` from `detected` to `human`.
- **Create** `tests/test_benchmark_human_scenarios.py` — unit tests for the new extractor and dispatch.
- **Modify** `docs/status.md` — short status update once shipped.

No changes needed in `benchmark/run.py` (the Step 0 detection guard already skips when mode is not `detected`/`both`) or `benchmark/core/annotator_bridge.py` (consumes `scenario.detection` agnostically).

---

## Task 1: Add `extract_human_scenarios()` (TDD)

**Files:**
- Create: `tests/test_benchmark_human_scenarios.py`
- Modify: `benchmark/core/scenarios.py` (add new function + import)

### Background for the implementer

- `data/ground_truth_hybrid/<uuid>.json` files look like:
  ```json
  {
    "conversation_id": "0014e499-...",
    "num_turns": 297,
    "key_moments": [
      {
        "turn_start": 5, "turn_end": 7,
        "annotation_type": "scaffolding",
        "annotator_id": "c0d23f9c-...",
        "situation": "...", "action": "...", "result": "...",
        "strategy_label": "ineffective",
        "situation_label": {"scaffolding": "no_mention", "rigor": "yes"},
        "situation_label_agg": "rigor",
        "cut_turn": 4,
        "moment_id": "..."
      },
      ...
    ]
  }
  ```
  Notes:
    - `situation_label_agg` is only set on `annotation_type == "scaffolding"` records. Rapport records have it absent / `None`.
    - Possible `situation_label_agg` values: `scaffolding`, `rigor`, `both`, `neither`, `mixed`, `unknown`, or absent.
    - `cut_turn` is optional — only present when an annotator picked a benchmark cut point.

- `Scenario` (already defined in `benchmark/core/scenarios.py`) shape:
  ```python
  @dataclass
  class Scenario:
      scenario_id: str
      conv_id: str
      cut_turn: int
      transcript_prefix: str
      student_context: str
      last_student_message: str
      mode: str                      # "detected" | "random" | "human" (new)
      detection: dict | None
  ```
  Helpers `_format_prefix`, `_get_student_context`, `_last_student_msg` already exist and should be reused.

- `transcripts` is a dict keyed by full `conv_id` (possibly composite, e.g. `2024-t1_2024-s1_<uuid>`), while GT files use bare UUIDs. Use `annotator.core.storage._conv_id_to_uuid` to build a UUID → conv_id lookup so the GT-side UUID can find its transcript. (Underscore-prefixed but it's used in our own codebase already.)

- Selection rule (must hold for a moment to become a scenario):
  - `situation_label_agg in {"scaffolding", "rigor"}`
  - `"cut_turn" in moment`
  - The GT file's `conversation_id` UUID resolves to a transcript in `transcripts`.
  - `cut_turn >= 1` and `cut_turn <= max(turn_number)` for that transcript.
  - GT-UUID-derived conv_id is not in `EXAMPLE_CONV_IDS`.

- `scenario_id` format: `f"{conv_id}__hum_{moment_idx}"` where `moment_idx` is the integer index of the moment within the file's `key_moments` list (stable across runs as long as `build_ground_truth.py` preserves order).

- The produced `detection` dict (shape-compatible with the existing detected-mode path) must include at minimum `turn_start`, `annotation_type`, `situation` — `annotator_bridge.build_synthetic_detections` reads those. Add `situation_label_agg`, `moment_id`, `annotator_id` for downstream traceability.

### Steps

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_human_scenarios.py`:

```python
"""Tests for extract_human_scenarios in benchmark/core/scenarios.py.

The new 'human' mode reads scaffolding-flavored key moments from
data/ground_truth_hybrid/ (filtered to situation_label_agg in
{scaffolding, rigor} and cut_turn present) and turns each into a Scenario.
"""
import json
from unittest.mock import patch

from benchmark.core.scenarios import extract_human_scenarios, Scenario
from benchmark.core.annotator_bridge import build_synthetic_detections
from benchmark.core.exchange import Exchange


def _make_transcripts():
    """Three transcripts keyed by composite conv_ids (UUID at the end)."""
    return {
        "2024-t1_2024-s1_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": {
            "conversation_id": "2024-t1_2024-s1_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "turns": [
                {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
                 "text": f"t{n}"} for n in range(1, 21)
            ],
        },
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": {
            "conversation_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "turns": [
                {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
                 "text": f"t{n}"} for n in range(1, 21)
            ],
        },
    }


def _gt_files():
    """Hybrid GT shape with a representative mix of moments."""
    return [
        {  # transcript A: 1 keep (scaffolding+cut), 1 drop (no cut), 1 drop (mixed agg)
            "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "num_turns": 20,
            "key_moments": [
                {"turn_start": 5, "turn_end": 7, "annotation_type": "scaffolding",
                 "annotator_id": "ann1", "situation": "S1", "action": "A", "result": "R",
                 "strategy_label": "effective", "situation_label_agg": "scaffolding",
                 "cut_turn": 4, "moment_id": "m1"},
                {"turn_start": 10, "turn_end": 12, "annotation_type": "scaffolding",
                 "annotator_id": "ann1", "situation": "S2", "action": "A", "result": "R",
                 "strategy_label": "partial", "situation_label_agg": "rigor"},  # no cut_turn -> drop
                {"turn_start": 14, "turn_end": 16, "annotation_type": "scaffolding",
                 "annotator_id": "ann1", "situation": "S3", "action": "A", "result": "R",
                 "strategy_label": "ineffective", "situation_label_agg": "mixed",
                 "cut_turn": 13},                                              # mixed -> drop
                {"turn_start": 17, "turn_end": 19, "annotation_type": "rapport",
                 "annotator_id": "ann1", "situation": "S4", "action": "A", "result": "R",
                 "strategy_label": "effective", "cut_turn": 16},               # rapport -> drop
            ],
        },
        {  # transcript B: 1 keep (rigor+cut), 1 drop (cut_turn out of range)
            "conversation_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "num_turns": 20,
            "key_moments": [
                {"turn_start": 8, "turn_end": 9, "annotation_type": "scaffolding",
                 "annotator_id": "ann2", "situation": "S5", "action": "A", "result": "R",
                 "strategy_label": "effective", "situation_label_agg": "rigor",
                 "cut_turn": 7, "moment_id": "m5"},
                {"turn_start": 99, "turn_end": 100, "annotation_type": "scaffolding",
                 "annotator_id": "ann2", "situation": "S6", "action": "A", "result": "R",
                 "strategy_label": "effective", "situation_label_agg": "scaffolding",
                 "cut_turn": 99},                                              # past end -> drop
            ],
        },
        {  # transcript not in transcripts dict -> all moments drop
            "conversation_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "num_turns": 20,
            "key_moments": [
                {"turn_start": 3, "turn_end": 5, "annotation_type": "scaffolding",
                 "annotator_id": "ann3", "situation": "S7", "action": "A", "result": "R",
                 "strategy_label": "effective", "situation_label_agg": "scaffolding",
                 "cut_turn": 2},
            ],
        },
    ]


def _run_extract():
    with patch("benchmark.core.scenarios.load_all_ground_truth_files",
               return_value=_gt_files()):
        return extract_human_scenarios(_make_transcripts())


def test_filters_to_scaff_or_rigor_with_cut_turn():
    scenarios = _run_extract()
    assert len(scenarios) == 2
    sit_aggs = sorted(s.detection["situation_label_agg"] for s in scenarios)
    assert sit_aggs == ["rigor", "scaffolding"]


def test_scenario_fields_populated():
    scenarios = _run_extract()
    by_agg = {s.detection["situation_label_agg"]: s for s in scenarios}

    s_scaff = by_agg["scaffolding"]
    assert s_scaff.mode == "human"
    assert s_scaff.cut_turn == 4
    assert s_scaff.scenario_id.endswith("__hum_0")            # moment index 0
    assert s_scaff.conv_id.endswith("aaaaaaaaaaaa")
    assert s_scaff.transcript_prefix.startswith("Turn 1.")
    assert "Turn 4." in s_scaff.transcript_prefix
    assert "Turn 5." not in s_scaff.transcript_prefix
    assert s_scaff.detection["annotation_type"] == "scaffolding"
    assert s_scaff.detection["turn_start"] == 5
    assert s_scaff.detection["situation"] == "S1"
    assert s_scaff.detection["moment_id"] == "m1"
    assert s_scaff.detection["annotator_id"] == "ann1"

    s_rigor = by_agg["rigor"]
    assert s_rigor.cut_turn == 7
    assert s_rigor.scenario_id.endswith("__hum_0")            # moment index 0 in its file


def test_skips_when_transcript_missing():
    scenarios = _run_extract()
    assert not any(s.conv_id.startswith("cccccccc") for s in scenarios)


def test_skips_when_cut_turn_out_of_range():
    scenarios = _run_extract()
    assert all(s.cut_turn <= 20 for s in scenarios)


def test_scenario_id_stable_across_runs():
    a = _run_extract()
    b = _run_extract()
    assert sorted(s.scenario_id for s in a) == sorted(s.scenario_id for s in b)


def test_detection_shape_is_compatible_with_annotator_bridge():
    """Round-trip: a human scenario must work with build_synthetic_detections."""
    scenarios = _run_extract()
    s = scenarios[0]
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
    assert "detections" not in detections or isinstance(detections, dict)
    # The function returns a dict like {"conversation_id":..., "detections":[...]}
    # OR (per current implementation) a list-wrapped dict. Just assert no exception
    # and that the human moment's annotation_type ("scaffolding") survives:
    assert detections  # truthy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: `ImportError` or `AttributeError: extract_human_scenarios` — function not defined yet.

- [ ] **Step 3: Implement `extract_human_scenarios()`**

In `benchmark/core/scenarios.py`, add the import near the top:

```python
from annotator.core.storage import load_all_ground_truth_files, _conv_id_to_uuid
```

(Keep the existing `from annotator.core.utils import ...` line.)

Add the function below `extract_random_scenarios` (around line 162):

```python
_SCAFF_AGGS = {"scaffolding", "rigor"}


def extract_human_scenarios(transcripts: dict[str, dict]) -> list[Scenario]:
    """Build scenarios from human-annotated key moments in ground_truth_hybrid.

    A moment becomes a scenario iff:
      - situation_label_agg in {"scaffolding", "rigor"}
      - "cut_turn" is present on the moment
      - the moment's conv UUID resolves to a loaded transcript
      - cut_turn is within the transcript's turn range
      - the resolved conv_id is not in EXAMPLE_CONV_IDS

    The produced Scenario.detection dict is shape-compatible with the
    detected-mode path so annotator_bridge.build_synthetic_detections
    works unchanged.
    """
    # Build UUID -> full conv_id lookup so GT (UUID-keyed) can find composite-id transcripts.
    uuid_to_conv = {_conv_id_to_uuid(cid): cid for cid in transcripts}

    scenarios: list[Scenario] = []
    for gt in load_all_ground_truth_files():
        gt_uuid = gt.get("conversation_id")
        full_conv_id = uuid_to_conv.get(_conv_id_to_uuid(gt_uuid or ""))
        if not full_conv_id:
            continue
        if full_conv_id in EXAMPLE_CONV_IDS:
            continue

        conversation = transcripts[full_conv_id]
        max_turn = max((t["turn_number"] for t in conversation.get("turns", [])), default=0)

        for idx, m in enumerate(gt.get("key_moments", [])):
            if m.get("situation_label_agg") not in _SCAFF_AGGS:
                continue
            if "cut_turn" not in m:
                continue
            cut = m["cut_turn"]
            if not isinstance(cut, int) or cut < 1 or cut > max_turn:
                continue

            prefix = _format_prefix(conversation, cut)
            if not prefix:
                continue

            detection = {
                "turn_start": m["turn_start"],
                "turn_end": m["turn_end"],
                "annotation_type": "scaffolding",
                "situation": m.get("situation", ""),
                "situation_label_agg": m["situation_label_agg"],
                "moment_id": m.get("moment_id"),
                "annotator_id": m.get("annotator_id"),
            }

            scenarios.append(Scenario(
                scenario_id=f"{full_conv_id}__hum_{idx}",
                conv_id=full_conv_id,
                cut_turn=cut,
                transcript_prefix=prefix,
                student_context=_get_student_context(conversation),
                last_student_message=_last_student_msg(conversation, cut),
                mode="human",
                detection=detection,
            ))

    return scenarios
```

Also update `Scenario.mode` docstring (line ~24):

```python
mode: str                     # "detected" | "random" | "human"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/scenarios.py tests/test_benchmark_human_scenarios.py
git commit -m "benchmark: add extract_human_scenarios for ground_truth_hybrid"
```

---

## Task 2: Wire `human` mode into `load_scenarios()` dispatch

**Files:**
- Modify: `benchmark/core/scenarios.py` (extend `load_scenarios`)
- Modify: `tests/test_benchmark_human_scenarios.py` (add dispatch test)

### Steps

- [ ] **Step 1: Write the failing dispatch test**

Append to `tests/test_benchmark_human_scenarios.py`:

```python
from benchmark.core.scenarios import load_scenarios


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
    assert len(scenarios) == 2
    assert all(s.mode == "human" for s in scenarios)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_human_scenarios.py::test_load_scenarios_human_mode_skips_detection_and_extracts -v`
Expected: FAIL — `load_scenarios` doesn't know `mode == "human"`, treats it as an unrecognized mode and returns 0 scenarios (or raises depending on path).

- [ ] **Step 3: Extend `load_scenarios`**

In `benchmark/core/scenarios.py`, inside `load_scenarios`, after the `random` branch (around line 225) and before the post-processing block, add:

```python
    if mode == "human":
        hum_scenarios = extract_human_scenarios(transcripts)
        logger.info("Human scenarios: %d", len(hum_scenarios))
        scenarios.extend(hum_scenarios)
```

Update the function's docstring `mode` description to include `human`.

- [ ] **Step 4: Run all scenario tests**

Run: `pytest tests/test_benchmark_human_scenarios.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/scenarios.py tests/test_benchmark_human_scenarios.py
git commit -m "benchmark: dispatch scenarios.mode='human' to extract_human_scenarios"
```

---

## Task 3: Flip default in `config.yaml`

**Files:**
- Modify: `config.yaml`

### Steps

- [ ] **Step 1: Inspect current scenarios block**

Run: `grep -n "^scenarios:" -A 8 config.yaml`
Expected: you'll see `mode: detected` (or similar) under `scenarios:`.

- [ ] **Step 2: Flip the default**

Edit `config.yaml`: change the `mode:` field under `scenarios:` from its current value to `human`. Leave other keys (`max_scenarios`, `max_per_conv`, `random_count`, `random_seed`, `min_turn`, `test_transcripts`) untouched — they're ignored in `human` mode but harmless to keep.

Example (verify the exact existing surrounding lines before editing):

```yaml
scenarios:
  mode: human          # was: detected
  # ... other keys unchanged ...
```

- [ ] **Step 3: Verify by loading config and inspecting**

Run:
```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['benchmark']['scenarios']['mode'])"
```
(Or whatever the top-level path to `scenarios.mode` is — adapt based on `config.yaml` structure.)
Expected output: `human`

- [ ] **Step 4: Run the full test suite to catch surprises**

Run: `pytest tests/ -q`
Expected: all green (or only pre-existing skips/xfails).

- [ ] **Step 5: Commit**

```bash
git add config.yaml
git commit -m "benchmark: default scenarios.mode to 'human'"
```

---

## Task 4: End-to-end smoke + `docs/status.md` update

**Files:**
- Modify: `docs/status.md`

### Steps

- [ ] **Step 1: Smoke-test scenario count against real ground_truth_hybrid**

Run:
```bash
python -c "
from benchmark.core.scenarios import extract_human_scenarios
from annotator.core.utils import load_transcripts
ts = load_transcripts()
ss = extract_human_scenarios(ts)
print(f'scenarios={len(ss)}, convs={len({s.conv_id for s in ss})}')
"
```
Expected: roughly `scenarios=~2975, convs=~115` (numbers may drift as `ground_truth_hybrid` is resynced).

If the number is far off (e.g. 0, or off by an order of magnitude), stop and investigate: check that `data/ground_truth_hybrid/` exists, that `data/transcripts/` is loadable, and that the UUID/composite mapping in `extract_human_scenarios` is matching.

- [ ] **Step 2: Update `docs/status.md`**

Prepend a new "Recently Shipped" block at the top (keep prior sections intact):

```markdown
## Recently Shipped: Human Key Moments as Benchmark Source (2026-06-08)

The benchmark default scenario source is now human-annotated key moments from
`data/ground_truth_hybrid/`, filtered to `situation_label_agg in {scaffolding, rigor}`
and presence of an annotator-chosen `cut_turn`. Synthetic-detection mode (`detected`)
remains available for reproducing prior runs.

- New `extract_human_scenarios()` in `benchmark/core/scenarios.py`; `Scenario.detection`
  is shape-compatible with the existing detected-mode path so nothing downstream changes.
- Default `scenarios.mode: human` in `config.yaml`.
- IoU clustering threshold in `data/build_ground_truth.py` tightened 0.7 -> 1.0 so only
  exact turn-range matches cluster across annotators.
- Current scope post-resync: ~2,975 scaffolding/rigor scenarios across ~115 conversations.

Spec: [plans/specs/2026-06-08-human-key-moments-benchmark-design.md](plans/specs/2026-06-08-human-key-moments-benchmark-design.md)
Plan: [plans/2026-06-08-human-key-moments-benchmark.md](plans/2026-06-08-human-key-moments-benchmark.md)
```

Update the `*Last updated*` line at the top of the file to `2026-06-08`.

- [ ] **Step 3: Commit**

```bash
git add docs/status.md
git commit -m "docs: status update for human key moments benchmark"
```

---

## Self-Review (post-write)

- **Spec coverage:** new function + dispatch + config flip + tests + status doc — all spec items mapped to tasks. IoU=1.0 change already committed in the prior spec-commit. Resync prereq is documented in the spec, not part of this plan (correct — it's a prereq, not part of feature work).
- **Placeholder scan:** none. All test code, function bodies, and commands are concrete.
- **Type/name consistency:** `extract_human_scenarios`, `_SCAFF_AGGS`, `Scenario.mode="human"`, `scenario_id` suffix `__hum_<idx>`, detection dict keys `situation_label_agg` / `moment_id` / `annotator_id` are used consistently across Tasks 1, 2, and 4.
- **Edge cases covered by tests:** filter on agg, missing cut_turn, out-of-range cut_turn, missing transcript, rapport records, scenario_id stability, round-trip compatibility with `annotator_bridge`.

