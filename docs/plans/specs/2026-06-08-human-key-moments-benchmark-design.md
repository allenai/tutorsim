# Human Key Moments as Benchmark Source -- Design

*Status: spec / 2026-06-08*

## Goal

Replace synthetic key-moment detection as the benchmark's default scenario source with human-annotated moments from `data/ground_truth_hybrid/`, scoped to scaffolding-flavored situations only. Keep the synthetic-detection path runnable so prior runs remain reproducible.

## Why

The current benchmark runs synthetic v5 detection to find key moments, then has an AI tutor continue from each. Two problems:

1. Synthetic detection is noisy (precision ~32%) -- many "scenarios" aren't real teachable moments.
2. Detection covers both scaffolding and rapport, but the work we want to evaluate first is scaffolding (the moments humans flagged as `scaffolding` or `rigor`).

Human-curated moments from `ground_truth_hybrid` give us higher signal and let us scope cleanly to scaffolding/rigor.

## Selection rule

A key moment in `data/ground_truth_hybrid/<conv_id>.json` is included iff **all** hold:

- `situation_label_agg in {"scaffolding", "rigor"}`
- `cut_turn` field present on the moment (annotator-chosen benchmark cut point)
- `conv_id` resolves to a loadable transcript

No external pool intersection. `ground_truth_hybrid` *is* the benchmark source; whatever's in it after a resync is what runs.

### Scope today (post-resync, IoU=1.0)

- 207 hybrid files, 11,084 total moments
- 3,675 scaff/rigor moments; **2,975 with annotator-chosen cut_turn**
- **115 conversations** contribute at least one scenario

Moments without `cut_turn` (~700 / 19%) are dropped. They can be back-filled in the annotation UI later and will auto-appear after the next resync.

## Architecture

Additive change. No shape change to `Scenario`, no change to Phase 1 (exchange), Phase 2 (annotation), or scoring.

### `benchmark/core/scenarios.py`

New function:

```python
def extract_human_scenarios(transcripts: dict[str, dict]) -> list[Scenario]:
    """Build scenarios from human-annotated key moments in ground_truth_hybrid.

    Selection: situation_label_agg in {scaffolding, rigor} AND cut_turn present.
    Cut: annotator-chosen cut_turn.
    Scenario.detection mirrors the human moment so downstream
    annotator_bridge.build_synthetic_detections works unchanged.
    """
```

Implementation notes:

- Loads via `annotator.core.storage.load_all_ground_truth_files()` (existing; honors configured `storage.paths.ground_truth`).
- Skips when `conv_id in EXAMPLE_CONV_IDS`, conv missing from `transcripts`, or `cut_turn` falls outside the conversation's turn range.
- `scenario_id = f"{conv_id}__hum_{moment_idx}"` (index within the file -- stable as long as `key_moments` order is stable in the hybrid JSON).
- `Scenario.mode = "human"`.
- `Scenario.detection` is shape-compatible with the existing detected-mode dict so `annotator_bridge.build_synthetic_detections` runs unchanged:

```python
detection = {
    "turn_start": moment["turn_start"],
    "turn_end": moment["turn_end"],
    "annotation_type": "scaffolding",     # all selected moments are this type
    "situation": moment["situation"],
    # passthrough for traceability / downstream consumers:
    "situation_label_agg": moment["situation_label_agg"],
    "moment_id": moment.get("moment_id"),
    "annotator_id": moment.get("annotator_id"),
}
```

### `load_scenarios()` dispatch

`mode` accepts a new value `"human"`. Existing values (`detected`, `random`, `both`) keep working unchanged. Branches in `load_scenarios`:

```python
if mode in ("detected", "both"): ...    # existing
if mode in ("random", "both"):   ...    # existing
if mode == "human":
    hum = extract_human_scenarios(transcripts)
    scenarios.extend(hum)
```

`max_scenarios` / `max_per_conv` / `random_seed` continue to apply.

### `benchmark/run.py`

No structural change. Step 0 (detection) is already guarded by `if scenario_mode in ("detected", "both"):` -- `human` skips it naturally.

### `config.yaml`

Default flips:

```yaml
scenarios:
  mode: human          # was: detected
  # max_scenarios, max_per_conv unchanged
```

Synthetic-detection runs remain available by setting `mode: detected` (or `both` / `random`).

### Tests (`tests/`)

Unit tests for `extract_human_scenarios`:

- Filter correctness: keeps `situation_label_agg in {scaffolding, rigor}`, drops others (`mixed`, `neither`, `unknown`, `both`, `None`).
- Skips moments without `cut_turn`.
- Skips when `conv_id` missing from transcripts.
- Skips when `conv_id in EXAMPLE_CONV_IDS`.
- Scenario fields populated correctly: `cut_turn`, `mode="human"`, `detection` dict has required keys.
- `scenario_id` stability under fixture re-load.
- Round-trip: `Scenario` produced by `extract_human_scenarios` is accepted by `annotator_bridge.build_synthetic_detections` without error and yields a single scaffolding detection spanning `turn_start` to last-generated turn.

## Prerequisite (not automated in this spec)

Before running the benchmark, resync `ground_truth_hybrid` so it reflects the latest human annotations:

1. `aws s3 cp s3://kylel-alexisr-edu/deidentified/step_up_annotations.jsonl data/teacher_annotations/step_up_annotations.jsonl`
2. `python -m data.build_ground_truth --labeller hybrid`

**IoU note (2026-06-08):** `_cluster_by_iou` default changed from 0.7 -> 1.0. Only exact turn-range matches across annotators cluster. This was applied as part of this work.

When the step_up_benchmark pool grows (annotators add SAR records on newly-sampled transcripts), repeat the resync; new convs will appear in `ground_truth_hybrid` and auto-feed the benchmark.

## Out of scope

- Pulling new step_up_benchmark transcripts from S3 and triggering annotation -- separate workflow on the annotation interface side.
- Rapport-flavored benchmark scenarios -- this benchmark is scaffolding-only by design.
- Combined modes like `human+random` -- YAGNI, can add later.
- Composite/bench-bucket `conv_id` handling -- already addressed in prior storage work.

## Risk / open questions

- **Stability of `scenario_id`:** ties to moment order within the hybrid JSON. `build_ground_truth.py` preserves annotation input order; resyncs that only add new annotators (not reorder existing) keep IDs stable. If we ever sort key_moments, IDs shift -- worth a note in `build_ground_truth.py`.
- **Volume:** ~2,975 scenarios is comparable to the prior synthetic run (2,608). No performance concerns expected.
- **115 conv set:** smaller than prior 201-conv synthetic runs because IoU=1.0 + cut_turn requirement filters out convs with no agreement. Acceptable; the alternative is loosening the cut_turn requirement, which you ruled out.
