# Benchmark Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the benchmark pipeline up to the same operational floor as the annotator pipeline so a long-running batch can be ctrl-C'd, crashed, or interrupted overnight without losing work, then run the first full benchmark.

**Architecture:** Port the annotator's proven shard pre-filter + in-flight batch sidecar pattern into benchmark's Phase 2 (annotation). Per-scenario annotation files already exist on disk — they just need to gate batch submission. Add benchmark-namespaced sidecar helpers in `annotator/core/storage.py` mirroring the annotator versions. Wire the structured logger in (`benchmark/` currently does `setup_logging()` then prints). Delete dead code and a vestigial config-string fallback. Defer string-round-trip cleanup and student-slot rename to follow-ups — they're code smells, not blockers. Document the screenshot scoping limitation since it's not getting fixed in this pass.

**Tech Stack:** Python 3, existing `annotator` and `benchmark` packages, `pytest` + `moto` for tests.

**Reference:** Independent audit in conversation history (verified file:line citations against the source). Annotator equivalents to mirror: `annotator/core/annotate.py:313-402` (resume + sidecar), `annotator/core/storage.py:659-728` (storage helpers).

---

## File Structure

**Modified:**
- `annotator/core/storage.py` — add `save_benchmark_inflight_batch`, `load_benchmark_inflight_batch`, `clear_benchmark_inflight_batch` mirroring the annotator helpers at lines 705-728
- `benchmark/core/annotator_bridge.py` — `execute_and_parse_bulk` and `label_bulk` accept `existing_batch_id` + `on_batch_created` and forward to `run_batch`; remove dead `annotate_exchange` (lines 122-174)
- `benchmark/run.py` — Phase 2 pre-filters scenarios with existing annotation shards; wires sidecar around bulk submission; resolves version once and persists it; replaces `print()` with `logger`; drops `"annotator_profiles"` from the prompt-version branch at line 222
- `benchmark/core/exchange.py` — replace `print()` with `logger`
- `benchmark/core/scenarios.py` — replace `print()` with `logger`
- `tests/test_storage.py` — add 3 tests for benchmark sidecar lifecycle
- `tests/test_benchmark_resume.py` (created) — unit tests for the pre-filter + sidecar wire-through in `annotator_bridge`
- `docs/lessons_learned.md` — note that benchmark annotation runs in text-only mode (screenshot scoping)
- `docs/status.md` — update current state after run completes
- `docs/plans/_summary.md` — log this plan

**Not modified (deferred follow-ups, called out for traceability):**
- String round-trip in `annotator_bridge.build_synthetic_conversation` — defer
- `student_cfg = get_phase_config("tutor", student_profile)` slot rename — defer
- Per-style annotator-profile selection — defer (no current consumer)
- Hardcoded `prompts/` directory path — project-wide pattern, separate plan

---

## Task 1: Branch setup

**Files:** none (git ops)

- [ ] **Step 1: Confirm clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: Create the WIP branch off main**

Run:
```bash
git checkout main
git pull
git checkout -b wip/benchmark-production-readiness
```

- [ ] **Step 3: Verify benchmark tests baseline**

Run: `pytest tests/ -q`
Expected: 152 passed (current baseline per `_summary.md` 2026-04-27 entry).

---

## Task 2: Benchmark in-flight batch sidecar storage helpers

**Files:**
- Modify: `annotator/core/storage.py` — append a new public-API block after line 728
- Modify: `tests/test_storage.py` — append a `TestBenchmarkInflightBatch` class

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage.py`:

```python
class TestBenchmarkInflightBatch:
    def test_roundtrip(self, local_storage):
        from annotator.core.storage import (
            save_benchmark_inflight_batch, load_benchmark_inflight_batch,
        )
        save_benchmark_inflight_batch("v_test", "anthropic", "balanced", {
            "provider": "anthropic", "model": "claude-opus-4-6",
            "batch_id": "msgbatch_abc", "n_entries": 12,
            "entry_keys_hash": "abc123def456", "display_name": "annotate",
            "submitted_at": "2026-04-28T10:00:00",
        })
        loaded = load_benchmark_inflight_batch("v_test", "anthropic", "balanced")
        assert loaded["batch_id"] == "msgbatch_abc"
        assert loaded["n_entries"] == 12

    def test_load_missing_returns_none(self, local_storage):
        from annotator.core.storage import load_benchmark_inflight_batch
        assert load_benchmark_inflight_batch("v_nope", "anthropic", "generous") is None

    def test_clear_removes_sidecar(self, local_storage):
        from annotator.core.storage import (
            save_benchmark_inflight_batch, load_benchmark_inflight_batch,
            clear_benchmark_inflight_batch,
        )
        save_benchmark_inflight_batch("v_test", "anthropic", "balanced", {"batch_id": "x"})
        clear_benchmark_inflight_batch("v_test", "anthropic", "balanced")
        assert load_benchmark_inflight_batch("v_test", "anthropic", "balanced") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py::TestBenchmarkInflightBatch -v`
Expected: FAIL — `cannot import name 'save_benchmark_inflight_batch'`.

- [ ] **Step 3: Add the helpers**

Append to `annotator/core/storage.py` after line 728 (after `clear_inflight_batch`):

```python
# ===================================================================
# Public API -- Benchmark in-flight batch sidecars (per profile + style)
# ===================================================================
#
# Mirrors the annotator sidecar helpers above, but namespaced under
# results/benchmark/{version}/in_flight/{profile}_{style}.json so each
# (tutor profile, annotator style) batch tracks independently.

def _bench_inflight_rel(version: str, profile: str, style: str) -> str:
    base = _get_result_path("benchmark_results")
    return f"{base}/{version}/in_flight/{profile}_{style}.json"


def save_benchmark_inflight_batch(version: str, profile: str, style: str,
                                   data: dict) -> None:
    """Record an in-flight benchmark annotation batch's metadata."""
    _get_backend().write_json(_bench_inflight_rel(version, profile, style), data)


def load_benchmark_inflight_batch(version: str, profile: str,
                                   style: str) -> dict | None:
    """Return the recorded in-flight benchmark batch metadata, or None."""
    return _get_backend().read_json(_bench_inflight_rel(version, profile, style))


def clear_benchmark_inflight_batch(version: str, profile: str, style: str) -> None:
    """Delete the benchmark in-flight sidecar after a batch completes."""
    be = _get_backend()
    rel = _bench_inflight_rel(version, profile, style)
    if isinstance(be, LocalBackend):
        path = be.root / rel
        if path.exists():
            path.unlink()
    else:
        try:
            be.client.delete_object(Bucket=be.bucket, Key=be._key(rel))
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_storage.py::TestBenchmarkInflightBatch -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/storage.py tests/test_storage.py
git commit -m "feat(storage): benchmark in-flight batch sidecar helpers"
```

---

## Task 3: Thread `existing_batch_id` + `on_batch_created` through `annotator_bridge`

**Files:**
- Modify: `benchmark/core/annotator_bridge.py:235-272` (`execute_and_parse_bulk`)
- Modify: `benchmark/core/annotator_bridge.py:275-324` (`label_bulk`)
- Create: `tests/test_benchmark_resume.py`

**Why:** The bridge is the choke point that calls `run_batch`. Adding the two kwargs here lets `run.py` opt into resume without changing call signatures elsewhere. We thread through both the annotation step and the labeling step because labeling is also a batch submission.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_resume.py`:

```python
"""Resume + sidecar wire-through tests for the benchmark annotator bridge."""
from unittest.mock import patch, MagicMock


def test_execute_and_parse_bulk_forwards_existing_batch_id():
    from benchmark.core.annotator_bridge import execute_and_parse_bulk
    entries = [{"key": "scen1__scaffolding__0", "request": {"prompt": "p"}}]
    all_detections = {"scen1": {"scen1": {"detections": [
        {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding"}
    ]}}}

    with patch("benchmark.core.annotator_bridge.run_batch") as mock_rb, \
         patch("benchmark.core.annotator_bridge.ModelClient"), \
         patch("benchmark.core.annotator_bridge.get_phase_config",
               return_value={"model": "claude-opus-4-6", "poll_interval": 60}):
        mock_rb.return_value = {}
        execute_and_parse_bulk(
            entries=entries,
            all_detections=all_detections,
            annotator_profile="anthropic",
            mode="batch",
            existing_batch_id="msgbatch_resumed",
            on_batch_created=lambda bid: None,
        )

    kwargs = mock_rb.call_args.kwargs
    assert kwargs["existing_batch_id"] == "msgbatch_resumed"
    assert callable(kwargs["on_batch_created"])


def test_execute_and_parse_bulk_default_kwargs_are_none():
    from benchmark.core.annotator_bridge import execute_and_parse_bulk
    entries = [{"key": "scen1__scaffolding__0", "request": {"prompt": "p"}}]
    all_detections = {"scen1": {"scen1": {"detections": [
        {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding"}
    ]}}}
    with patch("benchmark.core.annotator_bridge.run_batch") as mock_rb, \
         patch("benchmark.core.annotator_bridge.ModelClient"), \
         patch("benchmark.core.annotator_bridge.get_phase_config",
               return_value={"model": "claude-opus-4-6", "poll_interval": 60}):
        mock_rb.return_value = {}
        execute_and_parse_bulk(
            entries=entries,
            all_detections=all_detections,
            annotator_profile="anthropic",
            mode="batch",
        )
    kwargs = mock_rb.call_args.kwargs
    assert kwargs.get("existing_batch_id") is None
    assert kwargs.get("on_batch_created") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_resume.py -v`
Expected: FAIL — `execute_and_parse_bulk() got an unexpected keyword argument 'existing_batch_id'`.

- [ ] **Step 3: Add the kwargs to `execute_and_parse_bulk`**

Replace `benchmark/core/annotator_bridge.py:235-272` with:

```python
def execute_and_parse_bulk(
    entries: list[dict],
    all_detections: dict,
    annotator_profile: str,
    mode: str = "batch",
    existing_batch_id: str | None = None,
    on_batch_created: callable = None,
) -> dict[str, dict]:
    """Execute bulk entries and parse results back to per-scenario annotations.

    When mode == "batch", existing_batch_id resumes polling on a previously
    submitted provider batch (skip submission). on_batch_created fires once
    immediately after submission with the new batch id, so the orchestrator
    can persist a sidecar before the poll loop starts.

    Returns: {scenario_id: parsed_results_dict}
    """
    if not entries:
        return {}

    annotate_cfg = get_phase_config("annotate", annotator_profile)
    client = ModelClient(annotate_cfg["model"])

    if mode == "batch":
        raw = run_batch(
            client, entries, display_name="benchmark_annotate",
            poll_interval=annotate_cfg["poll_interval"],
            existing_batch_id=existing_batch_id,
            on_batch_created=on_batch_created,
        )
    else:
        raw = run_sync_entries(client, entries)

    # Merge all detections into one dict and parse once (avoids repeated error logs)
    merged_detections = {}
    for scenario_id, remapped_detections in all_detections.items():
        merged_detections.update(remapped_detections)

    all_results = parse_and_merge(raw, merged_detections)

    # Split back to per-scenario
    per_scenario = {}
    for scenario_id in all_detections:
        if scenario_id in all_results:
            per_scenario[scenario_id] = {scenario_id: all_results[scenario_id]}
        else:
            per_scenario[scenario_id] = {}

    return per_scenario
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_benchmark_resume.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/core/annotator_bridge.py tests/test_benchmark_resume.py
git commit -m "feat(benchmark): forward existing_batch_id + on_batch_created in bridge"
```

---

## Task 4: Phase 2 pre-filter — skip already-annotated scenarios

**Files:** Modify: `benchmark/run.py:218-268` (Phase 2 style loop)

**Why:** The shards saved at `run.py:266-268` mean re-running can detect "already done" without any new helpers — `list_benchmark_result_files(version, "annotations", profile, style)` already returns the right list. We just need to filter `scenarios` before calling `prepare_bulk_entries`, then merge cached + new results before Phase 3.

- [ ] **Step 1: Inspect the Phase 2 loop you'll replace**

Read: `benchmark/run.py:218-268`. Confirm the style loop submits all scenarios every time. Confirm `save_benchmark_result(version, "annotations", profile, style, f"{scenario_id}.json", ...)` writes the per-scenario shard.

- [ ] **Step 2: Add a helper for loading existing annotation shards**

Insert just above the `Phase 2` print at `benchmark/run.py:215` (replace the `all_style_results = {}` initializer at 218 too):

```python
        # ---------------------------------------------------------------
        # Phase 2: Annotate all exchanges (batch per style)
        # ---------------------------------------------------------------
        logger.info("=== Phase 2: Annotate (%s mode, %d styles) ===", ann_mode, len(styles))

        all_style_results = {}

        def _load_cached_annotations(style):
            """Return {scenario_id: labeled_data} for shards already on disk."""
            cached = {}
            for fname in list_benchmark_result_files(version, "annotations", profile, style):
                if not fname.endswith(".json"):
                    continue
                sid = fname[:-5]
                data = load_benchmark_result(version, "annotations", profile, style, fname)
                if data is not None:
                    cached[sid] = data
            return cached
```

- [ ] **Step 3: Filter scenarios per style**

Replace the body of the `for style in styles:` loop at `benchmark/run.py:220-263` with:

```python
        for style in styles:
            if prompt_version_base == "profiles":
                prompt_version = f"profiles/{style}"
            else:
                prompt_version = prompt_version_base

            cached = _load_cached_annotations(style)
            missing = [s for s in scenarios if s.scenario_id not in cached]
            logger.info("[%s] %d cached, %d to annotate (prompts: %s)",
                        style, len(cached), len(missing), prompt_version)

            if not missing:
                all_style_results[style] = cached
                continue

            entries, all_detections, _ = prepare_bulk_entries(
                scenarios=missing,
                exchanges=exchanges,
                annotator_style=style,
                prompt_version=prompt_version,
                context_window=context_window,
            )
            logger.info("[%s] %d annotation entries across %d scenarios",
                        style, len(entries), len(all_detections))

            if not entries:
                all_style_results[style] = cached
                continue

            sidecar = load_benchmark_inflight_batch(version, profile, style)
            existing_batch_id = None
            if sidecar:
                expected = sidecar.get("entry_keys_hash")
                actual = _entries_keys_hash(entries)
                if expected == actual:
                    existing_batch_id = sidecar["batch_id"]
                    logger.info("[%s] resuming in-flight batch %s (submitted %s)",
                                style, existing_batch_id, sidecar.get("submitted_at", "?"))
                else:
                    logger.error(
                        "[%s] in-flight sidecar exists but entry-keys hash differs "
                        "(sidecar=%s, current=%s). Scenario set changed between runs. "
                        "Delete results/benchmark/%s/in_flight/%s_%s.json to start fresh.",
                        style, expected, actual, version, profile, style,
                    )
                    raise RuntimeError("entry-keys mismatch on benchmark in-flight resume")

            def _record(batch_id, _profile=profile, _style=style):
                save_benchmark_inflight_batch(version, _profile, _style, {
                    "provider": "unknown",
                    "model": annotator_cfg.get("model", ""),
                    "batch_id": batch_id,
                    "n_entries": len(entries),
                    "entry_keys_hash": _entries_keys_hash(entries),
                    "display_name": "benchmark_annotate",
                    "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })

            per_scenario_results = execute_and_parse_bulk(
                entries=entries,
                all_detections=all_detections,
                annotator_profile=annotator_profile,
                mode=ann_mode,
                existing_batch_id=existing_batch_id,
                on_batch_created=_record,
            )
            logger.info("[%s] parsed %d scenario results", style, len(per_scenario_results))

            annotate_cfg_full = get_phase_config("annotate", annotator_profile)
            per_scenario_labeled = label_bulk(
                per_scenario_results=per_scenario_results,
                annotator_style=style,
                annotator_profile=annotator_profile,
                annotator_model=annotate_cfg_full["model"],
                mode=ann_mode,
            )
            logger.info("[%s] labeled %d scenarios", style, len(per_scenario_labeled))

            for scenario_id, labeled_data in per_scenario_labeled.items():
                save_benchmark_result(version, "annotations", profile, style,
                                      f"{scenario_id}.json", data=labeled_data)

            clear_benchmark_inflight_batch(version, profile, style)

            merged = dict(cached)
            merged.update(per_scenario_labeled)
            all_style_results[style] = merged
```

- [ ] **Step 4: Add the imports**

In `benchmark/run.py`, replace the storage import block at lines 24-26:

```python
import datetime
import hashlib
import logging

from annotator.core.storage import (
    save_benchmark_result, load_benchmark_result, list_benchmark_result_files,
    save_benchmark_inflight_batch, load_benchmark_inflight_batch,
    clear_benchmark_inflight_batch,
)
```

And add at module level after the existing imports:

```python
logger = logging.getLogger(__name__)


def _entries_keys_hash(entries: list[dict]) -> str:
    """Stable short hash of an entries list, keyed on entry order + keys.
    Mirrors annotator/core/annotate.py:43 so the resume guard catches a
    changed scenario set between runs."""
    joined = "\n".join(e["key"] for e in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 5: Manual verification — pre-filter logic**

Run a quick sanity test (no API calls):
```bash
python -c "
from benchmark.run import _entries_keys_hash
e1 = [{'key': 'a__b__0'}, {'key': 'c__d__1'}]
e2 = [{'key': 'a__b__0'}]
assert _entries_keys_hash(e1) != _entries_keys_hash(e2)
assert _entries_keys_hash(e1) == _entries_keys_hash(list(e1))
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -q`
Expected: 157+ passed (152 baseline + 3 sidecar + 2 bridge).

- [ ] **Step 7: Commit**

```bash
git add benchmark/run.py
git commit -m "feat(benchmark): Phase 2 resume via shard pre-filter + in-flight sidecar"
```

---

## Task 5: Stable version resolution — resolve once, persist, reuse on resume

**Files:** Modify: `benchmark/run.py:331-375` (`main()`)

**Why:** `datetime.date.today()` runs every invocation. A re-run that crosses midnight produces a new version directory and orphans the previous Phase 1 shards. Fix: if `config.json` already exists for the auto-generated name from yesterday OR an in-flight sidecar exists for *any* recent date, prefer the existing version.

The simplest robust fix: stop auto-generating in `main()`. Persist the resolved version into `config.json` at first invocation, and on subsequent invocations look for the most recent matching `{tutor_profile}_*` directory before generating a new one.

- [ ] **Step 1: Replace the version resolution block**

Replace `benchmark/run.py:360-371` with:

```python
    if args.version:
        version = args.version
    else:
        version = config.get("version") or _resolve_or_create_version(config)
```

- [ ] **Step 2: Add the resolver helper**

Insert above `def main():` (around line 331):

```python
def _resolve_or_create_version(config: dict) -> str:
    """Find the most recent in-progress {tutor_profile}_* run, else generate new.

    A version is considered in-progress if its directory exists but no
    `_complete` marker is present. This keeps overnight resumes pointed at
    yesterday's run instead of jumping to today's date.
    """
    import datetime
    from annotator.core.storage import list_benchmark_result_files

    tutor_profile = config.get("tutor_profiles", ["anthropic"])[0]
    today = datetime.date.today().strftime("%Y-%m-%d")
    default = f"{tutor_profile}_{today}"

    # Look for existing run dirs matching the profile prefix
    base = list_benchmark_result_files("")  # lists top-level versions
    candidates = [v for v in base if v.startswith(f"{tutor_profile}_")]
    candidates.sort(reverse=True)  # most recent first by lexicographic date

    for candidate in candidates:
        complete_marker = list_benchmark_result_files(candidate, "_complete")
        if not complete_marker:
            logger.info("Resuming in-progress version: %s", candidate)
            return candidate

    logger.info("Auto-generated version: %s", default)
    return default
```

- [ ] **Step 3: Write the completion marker after a successful run**

At the very end of `run_benchmark()` (after the existing `print(f"\nResults saved (version: {version})")` line, which becomes `logger.info(...)` in Task 7), add:

```python
    save_benchmark_result(version, "_complete", data={
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
```

- [ ] **Step 4: Sanity-test the resolver**

```bash
python -c "
import datetime, json, os, tempfile
os.environ['STORAGE_BACKEND'] = 'local'
with tempfile.TemporaryDirectory() as td:
    os.environ['STORAGE_ROOT'] = td
    os.makedirs(f'{td}/results/benchmark/anthropic_2026-04-27', exist_ok=True)
    os.makedirs(f'{td}/results/benchmark/anthropic_2026-04-26', exist_ok=True)
    with open(f'{td}/results/benchmark/anthropic_2026-04-26/_complete.json', 'w') as f:
        json.dump({}, f)
    from benchmark.run import _resolve_or_create_version
    v = _resolve_or_create_version({'tutor_profiles': ['anthropic']})
    assert v == 'anthropic_2026-04-27', f'got {v}'
    print('OK')
"
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py
git commit -m "fix(benchmark): stable version resolution across midnight resumes"
```

---

## Task 6: Migrate benchmark prints to the structured logger

**Files:**
- Modify: `benchmark/core/exchange.py` (5 print sites)
- Modify: `benchmark/core/scenarios.py` (5 print sites)
- Modify: `benchmark/run.py` (~25 remaining print sites — many already replaced in Task 4)

**Why:** `setup_logging(version=version)` is called in `benchmark/run.py:373` but no module declares a logger and no events go through it. The `logs/{version}/run.log` file gets created and stays empty for benchmark runs. Mirror the migration done for annotator on 2026-04-27.

- [ ] **Step 1: Add the logger to `exchange.py`**

In `benchmark/core/exchange.py`, after the existing imports (around line 17), add:

```python
import logging

logger = logging.getLogger(__name__)
```

Then replace each `print(...)` call with the appropriate logger call:
- Line 194: `print(f"\n    Round {round_num + 1}/{num_turns} - Tutor batch ({len(active_ids)} scenarios)...")` → `logger.info("Round %d/%d - tutor batch (%d scenarios)", round_num + 1, num_turns, len(active_ids))`
- Line 214: `print(f"      WARN: tutor failed for {sid[:50]}")` → `logger.warning("tutor failed for %s", sid[:50])`
- Line 233: same pattern as 194 but with "Student" — use `logger.info`
- Line 253: same pattern as 214 but "student" — use `logger.warning`
- Line 278: `print(f"\n    Exchanges complete: {len(active_ids)}/{len(scenarios)} succeeded")` → `logger.info("Exchanges complete: %d/%d succeeded", len(active_ids), len(scenarios))`

- [ ] **Step 2: Add the logger to `scenarios.py`**

In `benchmark/core/scenarios.py`, add the same `import logging` + `logger = logging.getLogger(__name__)` block after imports.

Replace each `print(...)` with `logger.info(...)`:
- Line 193: `print(f"Loaded {len(transcripts)} transcripts")` → `logger.info("Loaded %d transcripts", len(transcripts))`
- Line 213, 221, 229, 234: same pattern.

- [ ] **Step 3: Replace remaining prints in `run.py`**

The Phase 2 logic was already migrated in Task 4. Convert the remaining print statements:
- `run.py:77`: `print("\n=== Step 0: ...")` → `logger.info("=== Step 0: Key Moment Detection ===")`
- `run.py:95`: `print("\n=== Step 1: ...")` → `logger.info("=== Step 1: Extract Scenarios ===")`
- `run.py:116-118` (banner): collapse to `logger.info("=== Evaluating: %s (%s) ===", profile, tutor_model)`
- `run.py:125`: `logger.info("--- Phase 1: Generate Exchanges (%d scenarios) ---", len(scenarios))`
- `run.py:152, 159, 181-182, 194, 196, 211`: convert to `logger.info`/`logger.warning`/`logger.error` based on severity
- `run.py:273`: `logger.info("--- Phase 3: Per-Style Scores ---")`
- `run.py:324-326`: `logger.info("[%s] mean=%.3f n=%d scaffolding=%.3f rapport=%.3f", style, overall_mean, n, type_means.get('scaffolding', 0), type_means.get('rapport', 0))`
- `run.py:328`: `logger.info("Results saved (version: %s)", version)`
- `run.py:371`: `logger.info("Auto-generated version: %s", version)`

- [ ] **Step 4: Verify no `print(` calls remain**

Run: `grep -rn "print(" benchmark/ --include='*.py'`
Expected: zero results (or only docstring examples — eyeball the output).

- [ ] **Step 5: Smoke-run the help text**

Run: `python -m benchmark --help`
Expected: argparse help prints; no exceptions.

- [ ] **Step 6: Commit**

```bash
git add benchmark/run.py benchmark/core/exchange.py benchmark/core/scenarios.py
git commit -m "feat(benchmark): migrate prints to structured logger"
```

---

## Task 7: Delete dead code and vestigial config check

**Files:**
- Modify: `benchmark/core/annotator_bridge.py` — remove `annotate_exchange` (lines 122-174) and its docstring fragment in lines 6-9
- Modify: `benchmark/run.py` — remove `annotate_exchange` from the import on line 31, drop the `"annotator_profiles"` branch from the prompt-version check

- [ ] **Step 1: Confirm `annotate_exchange` has no callers**

Run: `grep -rn "annotate_exchange" benchmark/ tests/`
Expected: only the import in `run.py:31` and the definition + docstring reference in `annotator_bridge.py`.

- [ ] **Step 2: Remove the function and its module docstring fragment**

In `benchmark/core/annotator_bridge.py`, delete lines 118-174 (the `# Per-scenario mode (sync)` divider through the end of `annotate_exchange`). Also trim the module docstring at lines 6-9 to:

```python
"""Bridge to the synthetic annotator pipeline.

Constructs in-memory transcripts and detections from benchmark exchanges,
then calls the existing annotation and labeling functions in bulk mode.
"""
```

- [ ] **Step 3: Drop the import in `run.py`**

In `benchmark/run.py:30-33`, remove `annotate_exchange` from the import:

```python
from .core.annotator_bridge import (
    prepare_bulk_entries, execute_and_parse_bulk, label_bulk,
)
```

- [ ] **Step 4: Drop the `"annotator_profiles"` vestigial check**

In `benchmark/run.py` (the prompt-version branch — was line 222 before Task 4 edits, now wherever the resolved style branch lives), the check should already be `if prompt_version_base == "profiles":` after Task 4. Confirm no `"annotator_profiles"` string remains in the file:

Run: `grep -n "annotator_profiles" benchmark/run.py`
Expected: zero results.

- [ ] **Step 5: Run tests**

Run: `pytest tests/ -q`
Expected: still 157+ passed.

- [ ] **Step 6: Commit**

```bash
git add benchmark/core/annotator_bridge.py benchmark/run.py
git commit -m "refactor(benchmark): drop dead annotate_exchange + vestigial annotator_profiles check"
```

---

## Task 8: Document the screenshot scoping limitation

**Files:**
- Modify: `docs/lessons_learned.md` — add a short note
- Modify: `docs/status.md` — section under "Benchmark Pipeline"

- [ ] **Step 1: Add a lesson**

Append to `docs/lessons_learned.md`:

```markdown
### Benchmark annotation runs in text-only mode (screenshots not threaded)

`annotator_bridge.prepare_bulk_entries` does not pass `with_screenshots=True` to
`build_analysis_entries`, and even if it did, the bridge remaps `conv_id` to
`scenario_id` (annotator_bridge.py:142-144) so `load_anchored_screenshots(conv_id, ...)`
would look up images under the scenario id and silently find nothing.

Implication: AI tutors are evaluated on text-only continuations, while the same
annotator pipeline run with `--with-screenshots` against the original transcript
would see images. Comparisons across the two pipelines are not apples-to-apples
when screenshots are involved.

Workaround for now: when comparing benchmark numbers to annotator gold numbers,
run the annotator side without `--with-screenshots`. Don't try to wire screenshots
into benchmark without first redesigning the conv_id → scenario_id remapping in
the bridge.
```

- [ ] **Step 2: Note it in the status doc**

Update `docs/status.md` under the "Benchmark Pipeline" section to add a "Known Limitations" subsection mentioning the same point in one line.

- [ ] **Step 3: Commit**

```bash
git add docs/lessons_learned.md docs/status.md
git commit -m "docs: document benchmark text-only annotation scoping"
```

---

## Task 9: Smoke-test the resume path end-to-end

**Files:** none (manual verification)

**Why:** The whole point of this work is that ctrl-C resumes cleanly. Verify before running for real.

- [ ] **Step 1: Run a 2-scenario sync benchmark**

Run:
```bash
python -m benchmark --tutor-profile anthropic --max-scenarios 2 --mode sync --version smoke_resume_test
```
Expected: completes, writes shards under `results/benchmark/smoke_resume_test/annotations/anthropic/{generous,balanced,demanding}/`.

- [ ] **Step 2: Verify shard structure**

Run:
```bash
ls results/benchmark/smoke_resume_test/annotations/anthropic/balanced/
```
Expected: 2 files, one per scenario.

- [ ] **Step 3: Re-run and confirm pre-filter kicks in**

Run: same command as Step 1.
Expected: log lines `[balanced] 2 cached, 0 to annotate (prompts: profiles/balanced)` for each style. No API calls hit.

- [ ] **Step 4: Delete one shard and re-run; only the missing one should run**

Run:
```bash
rm "results/benchmark/smoke_resume_test/annotations/anthropic/balanced/$(ls results/benchmark/smoke_resume_test/annotations/anthropic/balanced/ | head -1)"
python -m benchmark --tutor-profile anthropic --max-scenarios 2 --mode sync --version smoke_resume_test
```
Expected: log line `[balanced] 1 cached, 1 to annotate`.

- [ ] **Step 5: Cleanup**

Run: `rm -rf results/benchmark/smoke_resume_test logs/smoke_resume_test`

- [ ] **Step 6: Commit logs to summary**

No code changes — just confirm to yourself the resume works. Move to Task 10.

---

## Task 10: Run the full benchmark

**Files:** none (the actual run)

- [ ] **Step 1: Confirm config**

Read `config.yaml` benchmark section. Confirm:
- `tutor_profiles: [anthropic]` (or whichever models)
- `mode: batch`
- `styles: [generous, balanced, demanding]`
- `scenarios.mode: detected` and `max_scenarios: 0` (full run)

- [ ] **Step 2: Kick off the full run in batch mode**

Run:
```bash
python -m benchmark --tutor-profile anthropic
```
Expected: detection completes (or loads cached `detections.json`), exchange phase batches across rounds, annotation phase per style with logged "%d cached, %d to annotate" lines.

- [ ] **Step 3: Monitor**

Tail the run log: `tail -f logs/{version}/run.log` (replace `{version}` with the auto-generated string from the first log line).

- [ ] **Step 4: If interrupted, re-run the same command**

A ctrl-C, network blip, or laptop sleep should not lose work:
- Phase 1 (exchange) resumes per-scenario from cached files.
- Phase 2 (annotation) resumes per-style from shards + in-flight sidecar.
- Phase 3 (scoring) is local arithmetic — runs to completion at end.

- [ ] **Step 5: Inspect results**

Run:
```bash
ls results/benchmark/{version}/scores/
cat results/benchmark/{version}/scores/anthropic_balanced.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['n_scenarios'], d['mean_score'], d['by_type'])"
```

- [ ] **Step 6: Update `_summary.md` and `status.md` with results**

Append a "completed" entry to `docs/plans/_summary.md` for this plan with the final headline numbers. Update `docs/status.md`'s Benchmark Pipeline section with the run id and key outputs.

- [ ] **Step 7: Final commit**

```bash
git add docs/plans/_summary.md docs/status.md
git commit -m "docs: log first full benchmark run results"
```

---

## Self-Review Notes

**Spec coverage check:** Walking the original audit list:
- #1 Phase 2/3 resume → Tasks 3, 4
- #2 Inflight sidecar → Tasks 2, 3, 4
- #4 Logger migration → Task 6
- #6 Dead `annotate_exchange` → Task 7
- #7 Vestigial `"annotator_profiles"` → Task 7
- #10 Auto-version midnight flip → Task 5
- #12 Zero benchmark tests → partially addressed (Tasks 2, 3 add 5 tests covering the new resume code)
- #3 Screenshot scoping → Task 8 (documented, not fixed — explicit scoping decision)
- #5, #8, #11, #13 — deferred per the audit conclusion (code smells, not blockers)

**Type consistency check:** `_entries_keys_hash` defined in run.py:Task 4 with the same signature as `annotator/core/annotate.py:43`. Sidecar shape matches the annotator's: `provider, model, batch_id, n_entries, entry_keys_hash, display_name, submitted_at`. `existing_batch_id` and `on_batch_created` kwarg names match `run_batch`'s existing signature in `annotator/core/client.py`.

**Risk:** Task 5's `_resolve_or_create_version` assumes `list_benchmark_result_files("")` returns top-level version directories. Verify against `annotator/core/storage.py:751` (`list_benchmark_result_files`) — if that helper expects a non-empty version arg, Task 5 needs an alternative listing call. Adjust at execution time.
