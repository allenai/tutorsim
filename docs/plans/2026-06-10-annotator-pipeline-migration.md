# Annotator Pipeline Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate benchmark Phase 2 from the per-style profiles pipeline to Lucy's `annotate -> decompose -> structure` pipeline, score per-scenario action-appropriateness vs `situation_label_agg`, and update viewers + config + tests accordingly.

**Architecture:** Pull all annotator-side files from `insource/scaffolding_anno` (mechanical, with conflict resolution preferring our `client.py` / `storage.py` / `label.py` / `data/build_ground_truth.py` / `config.yaml`). Rewrite Phase 2 in `benchmark/run.py` to call `run_decompose` and `run_structure_label` after `prepare_bulk_entries / execute_and_parse_bulk`. Implement scoring against `situation_label_agg` in a new `benchmark/core/score.py`. Update both HTML viewers for the new annotation schema.

**Tech Stack:** Python 3.11, pytest. Touches `annotator/core/*`, `prompts/annotator/*`, `benchmark/core/annotator_bridge.py`, `benchmark/run.py`, `benchmark/eval/*`, `config.yaml`, `tests/*`.

**Spec:** [`docs/plans/specs/2026-06-10-annotator-pipeline-migration-design.md`](specs/2026-06-10-annotator-pipeline-migration-design.md)

---

## File Map

- **Pull from `insource/scaffolding_anno`** (mechanical, take theirs):
  - `annotator/core/annotate.py`, `decompose.py`, `structure.py`, `situate.py`, `embed.py`, `utils.py`, `iteration/advisor.py`, `iteration/structure_disagreements.py`, `eval/eval.py`, `run.py`, `core/README.md`
  - `prompts/annotator/v13/p2/scaffolding.md`, `prompts/annotator/v13/README.md`, `prompts/annotator/action_labeller/classify_action.md`, `prompts/annotator/student_result_classifier/classify_student_result.md`
- **Keep ours** (resolve in our favor):
  - `annotator/core/client.py` (caching + adaptive thinking)
  - `annotator/core/storage.py` (composite-id `_conv_id_to_uuid`)
  - `annotator/core/label.py` (composite-id split filter)
  - `data/build_ground_truth.py` (IoU 1.0)
- **Create:**
  - `benchmark/core/score.py` — new scoring module (action F1 + outcome rate)
- **Modify:**
  - `benchmark/core/annotator_bridge.py` — drop legacy label_bulk; add `run_decompose_for_benchmark` + `run_structure_for_benchmark` wrappers
  - `benchmark/run.py` — single Phase 2 pass (annotate → decompose → structure → score)
  - `benchmark/eval/view.py`, `benchmark/eval/view_replay.py` — render per-facet labels with color badges
  - `config.yaml` — drop `styles`, set `prompt_version: v13`, `context_window: 20`
  - Tests in `tests/` that reference old per-style paths

---

## Task 1: Pull Lucy's annotator-side files + reconcile conflicts

**Files:** see "Pull from" list above

### Steps

- [ ] **Step 1: Pull annotator-side code (take theirs, don't touch ours)**

```bash
git checkout insource/scaffolding_anno -- \
  annotator/core/annotate.py annotator/core/decompose.py annotator/core/structure.py \
  annotator/core/situate.py annotator/core/embed.py annotator/core/utils.py \
  annotator/core/README.md \
  annotator/iteration/advisor.py annotator/iteration/structure_disagreements.py \
  annotator/eval/eval.py annotator/run.py
```

Then explicitly restore our versions of the files Lucy's branch reverted:

```bash
# Restore from the commits on our branch that touched these
git checkout HEAD -- annotator/core/client.py annotator/core/storage.py \
                     annotator/core/label.py data/build_ground_truth.py
```

(Our `HEAD` already has the right versions of these. The `git checkout HEAD --` is a no-op safety against any side effects of the previous step.)

- [ ] **Step 2: Pull prompts**

```bash
git checkout insource/scaffolding_anno -- \
  prompts/annotator/v13 \
  prompts/annotator/action_labeller \
  prompts/annotator/student_result_classifier
```

- [ ] **Step 3: Verify pulled files import without error**

```bash
python -c "import annotator.core.decompose, annotator.core.structure, annotator.core.annotate"
```
Expected: clean (no ImportError).

- [ ] **Step 4: Run the existing test suite**

```bash
pytest tests/ -q --ignore=tests/test_eval_metrics.py
```

Some tests will fail at this stage (annotate.py signature changes, structure module expectations) — that's OK, Task 2+3 fixes them. The point is to see only annotate/structure-related failures, not random unrelated breakage. Note which tests fail; expect: `test_annotate_build`, `test_label_routing`, the existing benchmark Phase 2 tests.

If `tests/test_storage.py` or any other unrelated test breaks (composite-id, caching, thinking dispatch), STOP — the reconcile lost our work somewhere. Roll back the affected file and try again.

- [ ] **Step 5: Commit**

```bash
git add annotator/ prompts/annotator/
git commit -m "annotator: pull v13 + decompose + structure pipeline from scaffolding_anno"
```

---

## Task 2: Add benchmark bridge wrappers for decompose + structure (TDD)

**Files:**
- Modify: `benchmark/core/annotator_bridge.py`
- Create: `tests/test_benchmark_phase2_migration.py`

### Background

`annotator.core.decompose.run_decompose` and `annotator.core.structure.run_structure_label` both accept `annotations_data` (in-memory dict) and `profile` so the benchmark can reuse the existing storage path with a per-scenario namespacing identical to how `label_bulk` does today. We add two thin wrappers that take per-scenario annotation dicts from `execute_and_parse_bulk` and return them enriched.

### Steps

- [ ] **Step 1: Write a failing wrapper test**

Create `tests/test_benchmark_phase2_migration.py`:

```python
"""Tests for the benchmark Phase 2 migration to annotate -> decompose -> structure."""
from unittest.mock import MagicMock, patch

from benchmark.core.annotator_bridge import (
    decompose_bulk, structure_bulk,
)


def _annotations_payload():
    return {
        "version": "benchmark",
        "model": "claude-opus-4-6",
        "source": "benchmark_exchange",
        "results": {
            "s1": {
                "annotations": [
                    {"annotation_type": "scaffolding",
                     "turn_start": 5, "turn_end": 10,
                     "situation": "Student stuck on solving for x.",
                     "action": "Tutor broke the problem into two steps. Tutor asked a guiding question.",
                     "result": "Student followed the steps. Student arrived at the correct answer."},
                ],
            },
        },
    }


def test_decompose_bulk_enriches_each_annotation_with_facets(monkeypatch):
    """decompose_bulk wraps run_decompose; should add action_decomposed and
    result_decomposed to each annotation in the per-scenario results dict."""
    def fake_run_decompose(version, model, mode, phase_cfg, **kwargs):
        data = kwargs["annotations_data"]
        for sid, scen in data["results"].items():
            for ann in scen["annotations"]:
                ann["action_decomposed"] = ["facet a1", "facet a2"]
                ann["result_decomposed"] = ["facet r1"]
        return data

    monkeypatch.setattr(
        "benchmark.core.annotator_bridge.run_decompose",
        fake_run_decompose,
    )

    per_scenario_results = {"s1": _annotations_payload()["results"]}
    enriched = decompose_bulk(
        per_scenario_results=per_scenario_results,
        annotator_profile="anthropic",
        mode="sync",
    )
    ann = enriched["s1"]["s1"]["annotations"][0]
    assert ann["action_decomposed"] == ["facet a1", "facet a2"]
    assert ann["result_decomposed"] == ["facet r1"]


def test_structure_bulk_adds_action_and_result_labels(monkeypatch):
    """structure_bulk wraps run_structure_label; should add action_label and
    result_label to each annotation."""
    def fake_run_structure(version, model, mode, phase_cfg, **kwargs):
        data = kwargs["annotations_data"]
        for sid, scen in data["results"].items():
            for ann in scen["annotations"]:
                ann["action_label"] = ["scaffolding", "neither"]
                ann["result_label"] = ["pos"]
        return data

    monkeypatch.setattr(
        "benchmark.core.annotator_bridge.run_structure_label",
        fake_run_structure,
    )

    per_scenario_results = {
        "s1": {"s1": {"annotations": [
            {"annotation_type": "scaffolding",
             "turn_start": 5, "turn_end": 10,
             "action_decomposed": ["a1", "a2"],
             "result_decomposed": ["r1"]}
        ]}}
    }
    enriched = structure_bulk(
        per_scenario_results=per_scenario_results,
        annotator_profile="anthropic",
        mode="sync",
    )
    ann = enriched["s1"]["s1"]["annotations"][0]
    assert ann["action_label"] == ["scaffolding", "neither"]
    assert ann["result_label"] == ["pos"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmark_phase2_migration.py -v -k "decompose_bulk or structure_bulk"
```
Expected: ImportError on `decompose_bulk` / `structure_bulk`.

- [ ] **Step 3: Add `decompose_bulk` and `structure_bulk` wrappers**

In `benchmark/core/annotator_bridge.py`, after the existing `label_bulk` function (which we'll keep around for now for tests but no longer call from benchmark), add:

```python
from annotator.core.decompose import run_decompose
from annotator.core.structure import run_structure_label


def decompose_bulk(
    per_scenario_results: dict[str, dict],
    annotator_profile: str,
    mode: str = "batch",
) -> dict[str, dict]:
    """Run decompose on all scenarios' annotations in one in-memory pass.

    Input: {scenario_id: {scenario_id: {annotations: [...]}}}
    Returns: same shape, with action_decomposed / result_decomposed populated
             on each annotation.
    """
    if not per_scenario_results:
        return {}
    merged_results = {}
    for sid, results in per_scenario_results.items():
        merged_results.update(results)

    annotations_data = {
        "version": "benchmark",
        "source": "benchmark_exchange",
        "results": merged_results,
    }

    phase_cfg = get_phase_config("annotate", annotator_profile)
    enriched = run_decompose(
        version="benchmark",
        model=phase_cfg["model"],
        mode=mode,
        phase_cfg=phase_cfg,
        annotations_data=annotations_data,
        profile=annotator_profile,
    )
    if not enriched:
        return per_scenario_results

    out: dict[str, dict] = {}
    enriched_results = enriched.get("results", {})
    for sid in per_scenario_results:
        if sid in enriched_results:
            out[sid] = {sid: enriched_results[sid]}
        else:
            out[sid] = per_scenario_results[sid]
    return out


def structure_bulk(
    per_scenario_results: dict[str, dict],
    annotator_profile: str,
    mode: str = "batch",
) -> dict[str, dict]:
    """Run structure labelling on all scenarios in one in-memory pass.

    Input: {scenario_id: {scenario_id: {annotations: [...with action_decomposed/result_decomposed...]}}}
    Returns: same shape, with action_label / result_label added per annotation.
    """
    if not per_scenario_results:
        return {}
    merged_results = {}
    for sid, results in per_scenario_results.items():
        merged_results.update(results)

    annotations_data = {
        "version": "benchmark",
        "source": "benchmark_exchange",
        "results": merged_results,
    }

    phase_cfg = get_phase_config("annotate", annotator_profile)
    enriched = run_structure_label(
        version="benchmark",
        model=phase_cfg["model"],
        mode=mode,
        phase_cfg=phase_cfg,
        annotations_data=annotations_data,
        profile=annotator_profile,
        target="scaffolding",
        split="all",
    )
    if not enriched:
        return per_scenario_results

    out: dict[str, dict] = {}
    enriched_results = enriched.get("results", {})
    for sid in per_scenario_results:
        if sid in enriched_results:
            out[sid] = {sid: enriched_results[sid]}
        else:
            out[sid] = per_scenario_results[sid]
    return out
```

Make sure `get_phase_config` is already imported at the top (it's used by `label_bulk`). If not, add `from annotator.core.config import get_phase_config`.

If `run_decompose` or `run_structure_label` doesn't accept `split="all"`, inspect the function's actual split values and use whichever excludes the train-only filtering (probably no split filter applies when benchmark scenarios aren't in any split).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmark_phase2_migration.py -v -k "decompose_bulk or structure_bulk"
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/annotator_bridge.py tests/test_benchmark_phase2_migration.py
git commit -m "benchmark: add decompose_bulk + structure_bulk bridge wrappers"
```

---

## Task 3: Add `benchmark/core/score.py` with the new scoring function (TDD)

**Files:**
- Create: `benchmark/core/score.py`
- Modify: `tests/test_benchmark_phase2_migration.py`

### Steps

- [ ] **Step 1: Write failing scoring tests**

Append to `tests/test_benchmark_phase2_migration.py`:

```python
from benchmark.core.score import score_scenarios


def _scenario_score_input(sid, agg, action_labels, result_labels):
    """Build a (scenario_dict, annotation_dict) pair for the scorer."""
    scenario = {
        "scenario_id": sid,
        "conv_id": sid + "_conv",
        "mode": "human",
        "detection": {"situation_label_agg": agg},
    }
    annotation = {
        "annotations": [
            {"action_decomposed": ["a"] * len(action_labels),
             "action_label": list(action_labels),
             "result_decomposed": ["r"] * len(result_labels),
             "result_label": list(result_labels)},
        ],
    }
    return scenario, annotation


def test_score_scenarios_scaffolding_tp_fn():
    """gt=scaffolding & pred=scaffolding -> TP; gt=scaffolding & pred=none -> FN."""
    pairs = [
        _scenario_score_input("a", "scaffolding", ["scaffolding"], ["pos"]),
        _scenario_score_input("b", "scaffolding", ["neither"], ["neg"]),
    ]
    scen_dicts = [p[0] for p in pairs]
    ann_dicts = [p[1] for p in pairs]
    result = score_scenarios(scen_dicts, ann_dicts)
    assert result["scaffolding"]["tp"] == 1
    assert result["scaffolding"]["fn"] == 1
    assert result["scaffolding"]["fp"] == 0
    # F1 = 2*TP / (2*TP + FN + FP) = 2 / (2+1+0) = 0.667
    assert abs(result["scaffolding"]["f1"] - 2/3) < 1e-6


def test_score_scenarios_rigor_fp():
    """gt=scaffolding & pred=rigor -> FP for rigor (and FN for scaffolding)."""
    pairs = [
        _scenario_score_input("a", "scaffolding", ["rigor"], ["pos"]),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["rigor"]["fp"] == 1
    assert result["scaffolding"]["fn"] == 1


def test_score_scenarios_both_label_counts_for_both_classes():
    """An action labeled 'both' counts as containing scaffolding AND rigor."""
    pairs = [
        _scenario_score_input("a", "scaffolding", ["both"], ["pos"]),
        _scenario_score_input("b", "rigor", ["both"], ["pos"]),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding"]["tp"] == 1
    assert result["rigor"]["tp"] == 1
    assert result["scaffolding"]["fp"] == 1   # 'both' in scenario b (gt=rigor) counts as scaffolding FP
    assert result["rigor"]["fp"] == 1         # 'both' in scenario a (gt=scaffolding) counts as rigor FP


def test_score_scenarios_outcome_rate():
    """outcome_rate = fraction of scenarios with at least one 'pos' result label."""
    pairs = [
        _scenario_score_input("a", "scaffolding", ["scaffolding"], ["pos"]),
        _scenario_score_input("b", "scaffolding", ["scaffolding"], ["neg"]),
        _scenario_score_input("c", "scaffolding", ["scaffolding"], ["neg", "pos"]),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["outcome_pos_rate"] == 2/3


def test_score_scenarios_mixed_agg_excluded_from_action_f1_included_in_outcome():
    """Scenarios with agg in {mixed, both, neither, unknown} don't count in F1
    but still contribute to outcome rate."""
    pairs = [
        _scenario_score_input("a", "scaffolding", ["scaffolding"], ["pos"]),
        _scenario_score_input("b", "mixed", ["scaffolding"], ["pos"]),
        _scenario_score_input("c", "neither", ["rigor"], ["neg"]),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    # Only scenario a contributes to scaffolding F1:
    assert result["scaffolding"]["tp"] == 1
    assert result["scaffolding"]["fn"] == 0
    assert result["scaffolding"]["fp"] == 0
    # Outcome rate: 2 of 3 are pos
    assert result["outcome_pos_rate"] == 2/3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmark_phase2_migration.py -v -k "score_scenarios"
```
Expected: ImportError on `score_scenarios`.

- [ ] **Step 3: Implement `benchmark/core/score.py`**

Create `benchmark/core/score.py`:

```python
"""Per-scenario benchmark scoring under the new annotator pipeline.

Scores AI tutor performance against the human-tagged situation_label_agg
on each scenario:
  - scaffolding F1: TP when gt=scaffolding AND any action_label in
    {scaffolding, both}; FN when gt=scaffolding AND no scaffolding/both
    action; FP when gt!=scaffolding AND any scaffolding/both action.
  - rigor F1: same with scaffolding -> rigor.
  - outcome_pos_rate: fraction of scenarios with at least one 'pos'
    result_label.

Scenarios whose situation_label_agg is in {mixed, both, neither, unknown}
are excluded from F1 calculations (ambiguous ground truth) but still
contribute to outcome_pos_rate.
"""
from __future__ import annotations


_INFORMATIVE_GT = {"scaffolding", "rigor"}


def _action_labels_for_scenario(annotation_data: dict) -> list[str]:
    """Flatten the action_label list across all annotations for the scenario."""
    labels = []
    for ann in annotation_data.get("annotations", []) or []:
        al = ann.get("action_label") or []
        if isinstance(al, list):
            labels.extend(al)
        elif isinstance(al, str):
            labels.append(al)
    return labels


def _result_labels_for_scenario(annotation_data: dict) -> list[str]:
    labels = []
    for ann in annotation_data.get("annotations", []) or []:
        rl = ann.get("result_label") or []
        if isinstance(rl, list):
            labels.extend(rl)
        elif isinstance(rl, str):
            labels.append(rl)
    return labels


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def _precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def score_scenarios(scenarios: list[dict], annotations: list[dict]) -> dict:
    """Compute action F1 (scaffolding + rigor) and outcome rate.

    Args:
        scenarios: list of scenario dicts (must include detection.situation_label_agg).
        annotations: aligned list of annotation dicts (each has 'annotations' list
                     with 'action_label' + 'result_label' populated).

    Returns:
        {
          "scaffolding": {tp, fp, fn, precision, recall, f1},
          "rigor": {tp, fp, fn, precision, recall, f1},
          "outcome_pos_rate": float,
          "n_scenarios": int,
          "n_scored_for_f1": int,
        }
    """
    counts = {
        "scaffolding": {"tp": 0, "fp": 0, "fn": 0},
        "rigor": {"tp": 0, "fp": 0, "fn": 0},
    }
    outcome_pos = 0
    n_total = 0
    n_scored = 0

    for scenario, ann in zip(scenarios, annotations):
        n_total += 1
        gt = (scenario.get("detection") or {}).get("situation_label_agg")
        action_labels = set(_action_labels_for_scenario(ann))
        has_scaffolding = "scaffolding" in action_labels or "both" in action_labels
        has_rigor = "rigor" in action_labels or "both" in action_labels

        if gt in _INFORMATIVE_GT:
            n_scored += 1
            for cls in ("scaffolding", "rigor"):
                gt_pos = (gt == cls)
                pred_pos = has_scaffolding if cls == "scaffolding" else has_rigor
                if gt_pos and pred_pos:
                    counts[cls]["tp"] += 1
                elif gt_pos and not pred_pos:
                    counts[cls]["fn"] += 1
                elif not gt_pos and pred_pos:
                    counts[cls]["fp"] += 1

        # outcome rate uses ALL scenarios
        if "pos" in _result_labels_for_scenario(ann):
            outcome_pos += 1

    result: dict = {"n_scenarios": n_total, "n_scored_for_f1": n_scored}
    for cls in ("scaffolding", "rigor"):
        tp, fp, fn = counts[cls]["tp"], counts[cls]["fp"], counts[cls]["fn"]
        result[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _precision(tp, fp),
            "recall": _recall(tp, fn),
            "f1": _f1(tp, fp, fn),
        }
    result["outcome_pos_rate"] = (outcome_pos / n_total) if n_total else 0.0
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_benchmark_phase2_migration.py -v -k "score_scenarios"
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/score.py tests/test_benchmark_phase2_migration.py
git commit -m "benchmark: add action-F1 + outcome-rate scoring per scenario"
```

---

## Task 4: Rewrite Phase 2 in `benchmark/run.py`

**Files:**
- Modify: `benchmark/run.py`
- Modify: `config.yaml`
- Modify: `tests/test_benchmark_phase2_migration.py`

### Steps

- [ ] **Step 1: Write a failing end-to-end test**

Append to `tests/test_benchmark_phase2_migration.py`:

```python
import pytest


def test_phase2_e2e_produces_one_annotation_per_scenario_and_score(monkeypatch, tmp_path):
    """Mocked end-to-end: run_benchmark's Phase 2 produces a flat
    annotations/{profile}/{scenario_id}.json (no styles dir) and a scores/{profile}.json
    with action_f1 + outcome_rate."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear(); st._backend = None

    # Patch: prepare_bulk_entries returns 2 scenarios, execute_and_parse_bulk
    # returns annotations, decompose adds facets, structure adds labels.
    from benchmark.core import annotator_bridge as ab

    def fake_prepare(scenarios, exchanges, **kw):
        entries = []
        all_detections = {}
        for s in scenarios:
            entries.append({"key": f"{s.scenario_id}__0"})
            all_detections[s.scenario_id] = {
                s.scenario_id: {"detections": [{"annotation_type": "scaffolding",
                                                "turn_start": 5, "turn_end": 10}]}
            }
        return entries, all_detections, {}

    def fake_execute(entries, all_detections, annotator_profile, mode, existing_batch_id, on_batch_created):
        out = {}
        for sid in all_detections:
            out[sid] = {sid: {"annotations": [
                {"annotation_type": "scaffolding", "turn_start": 5, "turn_end": 10,
                 "situation": "S", "action": "A1. A2.", "result": "R1."}
            ]}}
        return out

    def fake_decompose_bulk(per_scenario_results, annotator_profile, mode):
        for sid, results in per_scenario_results.items():
            for inner_sid, scen in results.items():
                for ann in scen["annotations"]:
                    ann["action_decomposed"] = ["A1.", "A2."]
                    ann["result_decomposed"] = ["R1."]
        return per_scenario_results

    def fake_structure_bulk(per_scenario_results, annotator_profile, mode):
        for sid, results in per_scenario_results.items():
            for inner_sid, scen in results.items():
                for ann in scen["annotations"]:
                    ann["action_label"] = ["scaffolding", "neither"]
                    ann["result_label"] = ["pos"]
        return per_scenario_results

    monkeypatch.setattr(ab, "prepare_bulk_entries", fake_prepare)
    monkeypatch.setattr(ab, "execute_and_parse_bulk", fake_execute)
    monkeypatch.setattr(ab, "decompose_bulk", fake_decompose_bulk)
    monkeypatch.setattr(ab, "structure_bulk", fake_structure_bulk)

    # Build a minimal config + scenarios + exchanges directly and call the
    # Phase 2 + Phase 3 sub-functions (not the full pipeline -- we already
    # tested Phase 1 elsewhere).
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenarios = [
        Scenario(scenario_id="s1", conv_id="c1", cut_turn=4,
                 transcript_prefix="...", student_context="ctx",
                 last_student_message="hi", mode="human",
                 detection={"turn_start": 5, "turn_end": 10,
                            "annotation_type": "scaffolding",
                            "situation_label_agg": "scaffolding"}),
        Scenario(scenario_id="s2", conv_id="c2", cut_turn=4,
                 transcript_prefix="...", student_context="ctx",
                 last_student_message="hi", mode="human",
                 detection={"turn_start": 5, "turn_end": 10,
                            "annotation_type": "scaffolding",
                            "situation_label_agg": "rigor"}),
    ]
    exchanges = {
        "s1": Exchange(scenario_id="s1", tutor_model="m", generated_turns=[{"turn_number": 5, "role": "TUTOR", "text": "x"}], completed=True),
        "s2": Exchange(scenario_id="s2", tutor_model="m", generated_turns=[{"turn_number": 5, "role": "TUTOR", "text": "x"}], completed=True),
    }

    # Import + call the new Phase 2 driver.
    from benchmark.run import run_phase2_and_score
    summary = run_phase2_and_score(
        version="t1",
        profile="anthropic",
        annotator_profile="anthropic",
        annotator_mode="sync",
        prompt_version="v13",
        context_window=20,
        scenarios=scenarios,
        exchanges=exchanges,
    )

    # Score summary should have F1 fields:
    assert "scaffolding" in summary
    assert "rigor" in summary
    assert "outcome_pos_rate" in summary
    # s1: gt=scaffolding, pred=[scaffolding, neither] -> TP for scaffolding
    assert summary["scaffolding"]["tp"] == 1
    # s2: gt=rigor, pred=[scaffolding, neither] -> FP for scaffolding, FN for rigor
    assert summary["rigor"]["fn"] == 1
    # outcome rate: both are 'pos'
    assert summary["outcome_pos_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_phase2_migration.py::test_phase2_e2e_produces_one_annotation_per_scenario_and_score -v
```
Expected: ImportError on `run_phase2_and_score`.

- [ ] **Step 3: Extract `run_phase2_and_score` in `benchmark/run.py`**

Find the existing Phase 2 + Phase 3 code in `benchmark/run.py` (the `for style in styles:` loop and the per-style scoring loop after it). Replace both with a single new function call. First add the function near the top of the file:

```python
def run_phase2_and_score(
    version: str,
    profile: str,
    annotator_profile: str,
    annotator_mode: str,
    prompt_version: str,
    context_window: int,
    scenarios: list,
    exchanges: dict,
    with_screenshots: bool = False,
) -> dict:
    """Phase 2 (annotate -> decompose -> structure) + Phase 3 (score) in one shot.

    Replaces the old per-style profiles loop. Single annotator pass per
    scenario, then in-memory decompose + structure, then F1 + outcome score
    against situation_label_agg.

    Returns the summary dict saved to scores/{profile}.json.
    """
    from .core.annotator_bridge import (
        prepare_bulk_entries, execute_and_parse_bulk,
        decompose_bulk, structure_bulk,
    )
    from .core.score import score_scenarios

    # --- Annotate ---
    entries, all_detections, _ = prepare_bulk_entries(
        scenarios=scenarios,
        exchanges=exchanges,
        annotator_style=None,         # no per-style namespacing
        prompt_version=prompt_version,
        context_window=context_window,
        with_screenshots=with_screenshots,
    )
    logger.info("Phase 2: %d annotation entries across %d scenarios",
                len(entries), len(all_detections))

    if not entries:
        return {"scaffolding": {"tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,"f1":0.0},
                "rigor":       {"tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,"f1":0.0},
                "outcome_pos_rate": 0.0,
                "n_scenarios": 0,
                "n_scored_for_f1": 0}

    per_scenario_results = execute_and_parse_bulk(
        entries=entries,
        all_detections=all_detections,
        annotator_profile=annotator_profile,
        mode=annotator_mode,
        existing_batch_id=None,
        on_batch_created=lambda *_a, **_k: None,
    )
    logger.info("Phase 2: parsed %d scenario results", len(per_scenario_results))

    per_scenario_results = decompose_bulk(per_scenario_results, annotator_profile, mode=annotator_mode)
    logger.info("Phase 2: decomposed")
    per_scenario_results = structure_bulk(per_scenario_results, annotator_profile, mode=annotator_mode)
    logger.info("Phase 2: structured")

    # Save per-scenario annotations to disk (flat, no styles subdir).
    for scenario_id, results in per_scenario_results.items():
        save_benchmark_result(version, "annotations", profile,
                              f"{scenario_id}.json", data=results)

    # --- Phase 3: score ---
    scenario_dicts = [s.to_dict() for s in scenarios]
    annotation_dicts = []
    for s in scenarios:
        results = per_scenario_results.get(s.scenario_id, {})
        ann = results.get(s.scenario_id, {})
        annotation_dicts.append(ann)

    summary = score_scenarios(scenario_dicts, annotation_dicts)
    summary["profile"] = profile
    save_benchmark_result(version, "scores", f"{profile}.json", data=summary)
    logger.info("[%s] scaffolding F1=%.3f rigor F1=%.3f outcome_pos_rate=%.3f n=%d",
                profile,
                summary["scaffolding"]["f1"],
                summary["rigor"]["f1"],
                summary["outcome_pos_rate"],
                summary["n_scenarios"])
    return summary
```

- [ ] **Step 4: Replace the existing Phase 2 + Phase 3 loop in `run_benchmark`**

Find the `for style in styles:` loop and the per-style score loop that follow (roughly lines 270-430 -- inspect to confirm).

Replace ALL of that (from the loop start through the scoring summary save) with one call:

```python
        annotator_mode = annotator_cfg["mode"]
        run_phase2_and_score(
            version=version,
            profile=profile,
            annotator_profile=annotator_profile,
            annotator_mode=annotator_mode,
            prompt_version=annotator_cfg["prompt_version"],
            context_window=annotator_cfg["context_window"],
            scenarios=scenarios,
            exchanges=exchanges,
            with_screenshots=with_screenshots,
        )
```

This drops the `styles` iteration entirely. The `annotator_cfg["styles"]` key becomes unused.

- [ ] **Step 5: Update `config.yaml`**

In `benchmark.annotator:` block, remove the `styles` key (and its list items), set `prompt_version: v13`, set `context_window: 20`:

```yaml
  annotator:
    profile: anthropic
    prompt_version: v13            # was: profiles -- single-prompt v13 with decompose+structure pipeline
    context_window: 20             # was: 50 -- matches Lucy's revert
    mode: batch
    poll_interval: 60
```

(Keep `profile`, `mode`, `poll_interval` lines as they were.)

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_benchmark_phase2_migration.py -v
pytest tests/ -q --ignore=tests/test_eval_metrics.py
```
Expected: the new tests pass. Some existing tests that reference the old per-style schema will fail -- Task 5 cleans those up.

- [ ] **Step 7: Commit**

```bash
git add benchmark/run.py config.yaml tests/test_benchmark_phase2_migration.py
git commit -m "benchmark: rewrite Phase 2 to annotate -> decompose -> structure"
```

---

## Task 5: Update viewers + clean up legacy tests

**Files:**
- Modify: `benchmark/eval/view.py`
- Modify: `benchmark/eval/view_replay.py`
- Modify: `tests/` (legacy tests touching old per-style schema)

### Steps

- [ ] **Step 1: Update `view_replay.py` annotation column**

In `benchmark/eval/view_replay.py`, the annotations panel currently shows per-style sections. Replace with a single-section layout per scenario that renders:
- `situation` / `action` / `result` text (the SAR text from annotate).
- For each `action_decomposed[i]`, render the text + a colored badge for `action_label[i]` (scaffolding=blue, rigor=orange, neither=gray, both=purple).
- For each `result_decomposed[i]`, render the text + a colored badge for `result_label[i]` (pos=green, neg=red).
- A header strip showing the scenario's `situation_label_agg` (scaffolding / rigor / mixed / etc.) PLUS an "appropriate?" tag: green if action_label set contains the gt class, red if it doesn't, gray if gt is ambiguous (mixed/neither/unknown).

CSS additions (place near existing badge rules in the file):

```css
.facet {{ margin-top: 6px; font-size: 12px; line-height: 1.4; }}
.facet-text {{ color: #333; }}
.facet-badge {{
  display: inline-block; font-size: 10px; font-weight: 700;
  padding: 2px 7px; border-radius: 8px; margin-left: 6px;
  text-transform: uppercase; letter-spacing: 0.3px; vertical-align: middle;
}}
.facet-badge.scaffolding {{ background:#e3f2fd; color:#0d47a1; }}
.facet-badge.rigor {{ background:#fff3e0; color:#e65100; }}
.facet-badge.neither {{ background:#eceff1; color:#455a64; }}
.facet-badge.both {{ background:#f3e5f5; color:#6a1b9a; }}
.facet-badge.pos {{ background:#d4edda; color:#155724; }}
.facet-badge.neg {{ background:#f8d7da; color:#721c24; }}
.tag.appropriate-yes {{ background:#d4edda; color:#155724; border:1px solid #b1dfbb; }}
.tag.appropriate-no {{ background:#f8d7da; color:#721c24; border:1px solid #f1aeb5; }}
.tag.appropriate-amb {{ background:#e2e3e5; color:#383d41; border:1px solid #c6c8ca; }}
```

In `load_data`, replace the `per_style: dict[str, list[dict]]` block with a flat per-scenario annotation:

```python
        # Annotations live at annotations/{profile}/{scenario_id}.json (no styles subdir).
        annotation_data = None
        try:
            ann_root = get_benchmark_result_path(version, "annotations", profile)
            ann_path = ann_root / f"{scenario_id}.json"
            if ann_path and ann_path.exists():
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)
        except Exception:
            annotation_data = None
        # The annotation file shape: {results: {scenario_id: {annotations: [...]}}}
        anns = []
        if annotation_data:
            anns = ((annotation_data.get("results") or {}).get(scenario_id) or {}).get("annotations", [])
```

And add `anns` (instead of `per_style`) to the scenario dict:

```python
        scenarios.append({
            ...
            "annotations": anns,
        })
```

In the JS `renderAnnotations` function, replace the per-style iteration with a single-annotation render that walks `s.annotations[*]` and prints each annotation's SAR plus a `action_decomposed[i]` + label badge sequence. Compute the "appropriate?" verdict client-side:

```javascript
function appropriateClass(agg, allActionLabels) {{
  const informative = (agg === 'scaffolding' || agg === 'rigor');
  if (!informative) return 'amb';
  const set = new Set(allActionLabels);
  const pred = (agg === 'scaffolding') ? (set.has('scaffolding') || set.has('both'))
                                       : (set.has('rigor') || set.has('both'));
  return pred ? 'yes' : 'no';
}}
```

Use the verdict to add an `<span class="tag appropriate-{verdict}">appropriate: yes/no/?</span>` to the info bar next to the existing `agg-{label}` tag.

- [ ] **Step 2: Update `view.py` similarly**

Apply the same per-scenario flat annotation render to `benchmark/eval/view.py`'s AI Annotations sidebar. Drop the `style-group` per-archetype iteration; show one annotation per scenario with the same facet+badge treatment.

- [ ] **Step 3: Regenerate a viewer against an existing run (sanity check)**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark.eval.view_replay --version dyn_smoke_v12_2026_06_09 --profile anthropic
```

Expected: HTML regenerates without errors. Old run has no `action_decomposed` data, so the facets section will be empty for those scenarios; that's fine. No JS errors on inspection (open in browser briefly to verify).

- [ ] **Step 4: Update / remove legacy tests**

Run the test suite to find failures:

```bash
pytest tests/ -q --ignore=tests/test_eval_metrics.py 2>&1 | tail -40
```

Likely failing tests reference: per-style annotation paths, `label_to_score`, `extract_effectiveness_by_type`, old per-style score summary fields. For each:

- If the test exercises behavior we still support (e.g. `label_bulk` standalone), leave the test if `label_bulk` still works on its own. The benchmark just doesn't call it anymore.
- If the test exercises the per-style benchmark loop directly, delete it (the loop is gone).

Specific likely-affected tests:
- `tests/test_benchmark_resume.py` — may reference `annotations/{profile}/{style}/...` paths. Update to flat path or delete obsolete ones.
- `tests/test_benchmark_screenshots.py` — same.
- Any test that constructs a fake `annotator_cfg` with `styles=[...]` — drop `styles` from the fixture.

- [ ] **Step 5: Run the full test suite green**

```bash
pytest tests/ -q --ignore=tests/test_eval_metrics.py
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add benchmark/eval/view.py benchmark/eval/view_replay.py tests/
git commit -m "benchmark: viewers + tests for new annotation schema; drop per-style legacy"
```

---

## Task 6: End-to-end smoke + docs update

**Files:**
- Modify: `docs/status.md`

### Steps

- [ ] **Step 1: Tiny 2-scenario sync smoke**

```bash
PYTHONIOENCODING=utf-8 python -m benchmark --version pipeline_smoke_2026_06_10 --scenario-mode human --max-scenarios 2 --mode sync
```

Expected logs:
- Phase 1 (exchange) completes.
- Phase 2 logs:
  ```
  Phase 2: N annotation entries across 2 scenarios
  Phase 2: parsed 2 scenario results
  Phase 2: decomposed
  Phase 2: structured
  ```
- Phase 3: one summary line with `scaffolding F1`, `rigor F1`, `outcome_pos_rate`.

Output paths:
- `results/benchmark/pipeline_smoke_2026_06_10/annotations/anthropic/<scenario_id>.json` (flat, no styles subdir)
- `results/benchmark/pipeline_smoke_2026_06_10/scores/anthropic.json`

- [ ] **Step 2: Inspect one annotation file**

```bash
PYTHONIOENCODING=utf-8 python -c "
import json, os
d = os.listdir('results/benchmark/pipeline_smoke_2026_06_10/annotations/anthropic')
f = d[0]
data = json.load(open(f'results/benchmark/pipeline_smoke_2026_06_10/annotations/anthropic/{f}', encoding='utf-8'))
sid = list(data['results'].keys())[0]
a = data['results'][sid]['annotations'][0]
print('action_decomposed:', a.get('action_decomposed'))
print('action_label:', a.get('action_label'))
print('result_decomposed:', a.get('result_decomposed'))
print('result_label:', a.get('result_label'))
"
```

Expected: per-facet `action_label` (list of scaffolding/rigor/neither/both) and `result_label` (pos/neg) populated.

- [ ] **Step 3: Update `docs/status.md`**

Prepend a new block to `docs/status.md`:

```markdown
## Recently Shipped: Annotator Pipeline Migration (2026-06-10)

Benchmark Phase 2 migrated from the legacy per-style profiles SAR pipeline
to Lucy's new annotate -> decompose -> structure pipeline. Per-scenario
output now includes per-facet action labels (scaffolding/rigor/neither/both)
and result labels (pos/neg). Scoring is action-appropriateness F1 against
the human situation_label_agg tag (scaffolding F1 + rigor F1 + outcome rate).

- Pulled annotator-side code from scaffolding_anno; preserved our caching,
  composite-id, and adaptive-thinking patches.
- `benchmark/core/annotator_bridge.py` gains decompose_bulk + structure_bulk.
- `benchmark/core/score.py` (new) computes action F1 + outcome rate.
- Viewers updated for the new per-facet label schema.

Spec: [plans/specs/2026-06-10-annotator-pipeline-migration-design.md](plans/specs/2026-06-10-annotator-pipeline-migration-design.md)
Plan: [plans/2026-06-10-annotator-pipeline-migration.md](plans/2026-06-10-annotator-pipeline-migration.md)
```

Update the `*Last updated:*` line to `2026-06-10`.

- [ ] **Step 4: Commit**

```bash
git add docs/status.md
git commit -m "docs: status update for annotator pipeline migration"
```

---

## Self-Review

**Spec coverage:**
- Pull annotator-side code + conflict resolution — Task 1.
- New `decompose_bulk` + `structure_bulk` bridge — Task 2.
- Per-scenario scoring (`score.py`) — Task 3.
- Phase 2 rewrite in `run.py` — Task 4.
- Config (`styles` removed, prompt_version=v13, context_window=20) — Task 4.
- Viewer updates — Task 5.
- Test cleanup — Task 5.
- E2E smoke + docs — Task 6.

All spec items mapped.

**Placeholder scan:** no TBDs in step bodies. Step 4 of Task 5 says "find failures, update or delete" which is a judgment call — but the criteria for each decision are explicit.

**Type/name consistency:**
- `decompose_bulk` / `structure_bulk` / `score_scenarios` / `run_phase2_and_score` names consistent across Tasks 2-4.
- Annotation file paths: `annotations/{profile}/{scenario_id}.json` (flat) consistent in Tasks 4 and 5.
- Score file paths: `scores/{profile}.json` consistent.
- Score summary keys: `scaffolding`, `rigor`, `outcome_pos_rate`, `n_scenarios`, `n_scored_for_f1` consistent.

**Open uncertainty:** Task 2 Step 3 notes that `run_decompose` / `run_structure_label` may not accept `split="all"` — the implementer should inspect the function and use whatever bypasses train-only filtering. Documented; not a placeholder.
