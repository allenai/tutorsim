# Benchmark Screenshot Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the benchmark pipeline ingest screenshots end-to-end — Step 0 detection, Phase 1 exchange (tutor + student), and Phase 2 annotation — so AI tutors are evaluated under the same visual context the original human tutors had.

**Architecture:** The annotator side already supports `--with-screenshots` (delivered 2026-04-24). The benchmark inherits the helpers but doesn't wire them. Two structural changes unlock this:
1. Decouple screenshot *loading* from screenshot *use* in `build_analysis_entries` and `build_detection_entries` by accepting an optional pre-loaded `screenshots_by_conv` dict. The bridge (which remaps `conv_id → scenario_id`) loads screenshots once per scenario using the *original* `scenario.conv_id` and passes them keyed by `scenario_id`. The standalone annotator path keeps its existing load-by-conv_id fallback.
2. Add a single `with_screenshots` flag at the benchmark CLI/config layer. When on, all three phases load and attach images. Vision-support validation runs once at run start; runs fail fast on a non-vision model.

For Phase 1 specifically, the AI tutor and synthetic student see only screenshots whose `anchor_turn ≤ scenario.cut_turn` (i.e. visible at the cut point). No new screenshots arise during the synthetic exchange — the image set is fixed for the whole exchange.

**Tech Stack:** Python 3, existing `annotator` and `benchmark` packages, `pytest` + `unittest.mock` for tests. Anthropic vision API (Claude Opus 4.6 already vision-capable).

**Branch:** Continue on `wip/benchmark-production-readiness`. The resume + logger work from the prior plan is the foundation; screenshot ingestion is additive.

---

## File Structure

**Modified:**
- `annotator/core/annotate.py` — `build_analysis_entries` accepts optional `screenshots_by_conv: dict[str, list[dict]] | None`; when present, skip the `load_anchored_screenshots` call and use the dict directly.
- `annotator/core/detect.py` — `build_detection_entries` accepts the same optional kwarg with parallel semantics.
- `benchmark/run.py` — `--with-screenshots` CLI flag + `benchmark.with_screenshots` config key; vision validation; threads the flag into `run_detect`, exchange runners, and the bridge.
- `benchmark/core/annotator_bridge.py` — `prepare_bulk_entries` accepts `with_screenshots` flag; loads anchored screenshots per scenario keyed on `scenario.conv_id`, remaps to `scenario_id` for the dict it passes to `build_analysis_entries`.
- `benchmark/core/exchange.py` — `run_exchange` and `run_exchanges_batch` accept a per-scenario `images: list[str] | None` argument; tutor and student calls attach those images.
- `config.yaml` — add `benchmark.with_screenshots: false` (default off; opt-in).
- `tests/test_annotate_build.py` — assert `screenshots_by_conv` short-circuits the lookup.
- `tests/test_detect_parse.py` (or new `test_detect_build.py`) — same for `build_detection_entries`.
- `tests/test_benchmark_screenshots.py` (created) — bridge wiring + exchange filtering.
- `docs/lessons_learned.md` — replace the 2026-04-28 "text-only mode" entry with a "now supported" pointer.
- `docs/status.md` — update Benchmark Pipeline section.

---

## Task 1: `build_analysis_entries` accepts pre-loaded screenshots

**Files:**
- Modify: `annotator/core/annotate.py:111-185`
- Modify: `tests/test_annotate_build.py`

**Why:** Currently `build_analysis_entries` does `load_anchored_screenshots(conv_id, conversation["turns"])` keyed on the iteration's `conv_id`. The benchmark bridge remaps `conv_id` to `scenario_id`, so that lookup silently returns nothing. Decoupling load from use lets the caller (the bridge) own the lookup using the real `conv_id` while still passing a `scenario_id`-keyed dict downstream.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_annotate_build.py`:

```python
def test_build_analysis_entries_uses_provided_screenshots(temp_data):
    """When screenshots_by_conv is passed, the function uses it directly
    and does NOT call load_anchored_screenshots."""
    from annotator.core.annotate import build_analysis_entries

    detections_by_conv = {
        "scen_abc": {"detections": [
            {"turn_start": 5, "turn_end": 7,
             "annotation_type": "scaffolding", "brief_description": "x"}
        ]},
    }
    conversations_map = {
        "scen_abc": {
            "conversation_id": "scen_abc",
            "turns": [
                {"turn_number": i, "role": "TUTOR", "text": f"t{i}",
                 "type": "DIALOGUE", "timestamp": "", "start_seconds": float(i)}
                for i in range(1, 11)
            ],
        },
    }
    fake_screenshots = [
        {"filename": "s1.jpg", "anchor_turn": 6, "storage_path": "deidentified/screenshots/REAL_CONV/s1.jpg",
         "timestamp_seconds": 6.0},
    ]
    screenshots_by_conv = {"scen_abc": fake_screenshots}

    # Set up a v4 prompt to satisfy load_prompt
    import os
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    prompt_path = f"{base}/prompts/annotator/v4/p2/scaffolding.txt"
    assert os.path.exists(prompt_path), f"need real prompt at {prompt_path}"

    entries = build_analysis_entries(
        detections_by_conv, conversations_map,
        context_window=2, version="v4",
        with_screenshots=True,
        screenshots_by_conv=screenshots_by_conv,
    )
    assert len(entries) == 1
    request = entries[0]["request"]
    assert request.get("images") == ["deidentified/screenshots/REAL_CONV/s1.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_annotate_build.py::test_build_analysis_entries_uses_provided_screenshots -v`
Expected: FAIL — `build_analysis_entries() got an unexpected keyword argument 'screenshots_by_conv'`.

- [ ] **Step 3: Modify `build_analysis_entries`**

In `annotator/core/annotate.py:111-185`, change the signature and short-circuit screenshot loading when the dict is provided:

```python
def build_analysis_entries(detections_by_conv: dict, conversations_map: dict,
                           context_window: int, version: str,
                           dialogue_only: bool = False,
                           annotator_style: str | None = None,
                           with_screenshots: bool = False,
                           screenshots_by_conv: dict[str, list[dict]] | None = None) -> list[dict]:
    """Build batch entries for analysis.

    annotator_style is accepted for API compatibility but NOT injected into
    prompts. Style calibration is achieved by iterating the prompt against
    archetype-filtered ground truth, not by injecting style text.

    When with_screenshots=True, attaches per-moment images whose anchor turn
    falls inside the excerpt window (excerpt_start <= anchor_turn <= excerpt_end,
    inclusive). If screenshots_by_conv is provided, it overrides per-conv lookup
    by conv_id -- the caller has already done the loading, possibly with a
    different conv_id than the iteration key (e.g. the benchmark bridge passes
    scenario_id-keyed entries with screenshots loaded from the original conv_id).
    """
    from .screenshots import load_anchored_screenshots

    prompt_cache = {}
    entries = []

    for conv_id, conv_data in detections_by_conv.items():
        conversation = conversations_map.get(conv_id)
        if not conversation:
            logger.warning("No transcript found for %s, skipping", conv_id)
            continue

        if screenshots_by_conv is not None:
            all_screenshots = screenshots_by_conv.get(conv_id, [])
        elif with_screenshots:
            all_screenshots = load_anchored_screenshots(conv_id, conversation["turns"])
        else:
            all_screenshots = []

        # ... rest of the function unchanged ...
```

(Preserve every other line of the function — only the signature and the `all_screenshots = ...` block change.)

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_annotate_build.py -v`
Expected: all tests in file pass, including the new one.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/annotate.py tests/test_annotate_build.py
git commit -m "feat(annotate): build_analysis_entries accepts pre-loaded screenshots_by_conv"
```

---

## Task 2: `build_detection_entries` accepts pre-loaded screenshots

**Files:**
- Modify: `annotator/core/detect.py:74-107`
- Modify: `tests/test_detect_parse.py` (or create `tests/test_detect_build.py` if the existing file is parse-only)

**Why:** Same shape change as Task 1, for symmetry. The benchmark calls `run_detect` from Step 0; we want the same decoupled-loading hook so a future benchmark-side optimization (cached per-conv screenshot loads, S3-batched fetches) can pre-load and pass through.

For *Step 0 itself*, we don't strictly need this in this iteration — the benchmark passes real conv_ids straight through to `run_detect` without remapping. But matching `build_analysis_entries`'s contract avoids the parallel-API drift the project's audit flagged. Cheap to add now, expensive to retrofit later.

- [ ] **Step 1: Write the failing test**

In `tests/` (use existing detect-build test file if there is one; otherwise extend `test_detect_parse.py`):

```python
def test_build_detection_entries_uses_provided_screenshots(temp_data):
    from annotator.core.detect import build_detection_entries

    conversations = [{
        "conversation_id": "scen_abc",
        "turns": [{"turn_number": i, "role": "TUTOR", "text": f"t{i}",
                   "type": "DIALOGUE", "timestamp": "", "start_seconds": float(i)}
                  for i in range(1, 5)],
        "context": "ctx",
    }]
    fake = [{"filename": "s1.jpg", "anchor_turn": 2,
             "storage_path": "deidentified/screenshots/REAL/s1.jpg",
             "timestamp_seconds": 2.0}]
    screenshots_by_conv = {"scen_abc": fake}

    entries = build_detection_entries(
        conversations, targets=["scaffolding"], version="v4",
        with_screenshots=True, screenshots_by_conv=screenshots_by_conv,
    )
    assert len(entries) == 1
    assert entries[0]["request"].get("images") == ["deidentified/screenshots/REAL/s1.jpg"]
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_detect_parse.py::test_build_detection_entries_uses_provided_screenshots -v`
Expected: FAIL on unexpected kwarg.

- [ ] **Step 3: Modify `build_detection_entries`**

In `annotator/core/detect.py:74-107`:

```python
def build_detection_entries(conversations: list[dict], targets: list[str],
                            version: str, dialogue_only: bool = False,
                            with_screenshots: bool = False,
                            screenshots_by_conv: dict[str, list[dict]] | None = None) -> list[dict]:
    """Build batch entries for detection.

    When with_screenshots=True, attaches every image for the conversation.
    If screenshots_by_conv is provided, the function uses it directly instead
    of looking up by conv_id — symmetric with build_analysis_entries.
    """
    from .screenshots import load_anchored_screenshots

    prompt_cache = {}
    entries = []

    for conv in conversations:
        conv_id = conv["conversation_id"]

        if screenshots_by_conv is not None:
            screenshots = screenshots_by_conv.get(conv_id, [])
        elif with_screenshots:
            screenshots = load_anchored_screenshots(conv_id, conv["turns"])
        else:
            screenshots = []
        image_paths = [s["storage_path"] for s in screenshots]

        # ... rest unchanged ...
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_detect_parse.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/detect.py tests/test_detect_parse.py
git commit -m "feat(detect): build_detection_entries accepts pre-loaded screenshots_by_conv"
```

---

## Task 3: Add `with_screenshots` config + CLI flag in benchmark

**Files:**
- Modify: `config.yaml`
- Modify: `annotator/core/config.py:get_benchmark_config` (add CLI override mapping)
- Modify: `benchmark/run.py:main()` and `run_benchmark()`

**Why:** Single source of truth at the config layer, CLI override for ad-hoc runs. `--with-screenshots` is a project-standard flag name (matches the annotator CLI). Vision validation runs once when the flag is on; we fail fast before submitting any batches.

- [ ] **Step 1: Add the config key**

In `config.yaml`, in the `benchmark:` block (find it; there's only one), append after `aggregation:`:

```yaml
  with_screenshots: false  # opt-in; requires vision-capable models on tutor, student, annotator, detector, labeller
```

- [ ] **Step 2: Add CLI flag and override mapping**

In `benchmark/run.py:main()`, add the argparse flag near the other flags:

```python
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Attach anchored screenshots to detection, exchange, and annotation prompts. Requires vision-capable models.")
```

Add it to the overrides dict:

```python
    overrides = {
        ...
        "with_screenshots": args.with_screenshots if args.with_screenshots else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
```

In `annotator/core/config.py:get_benchmark_config`, add the override branch after the existing ones (around line 162):

```python
        if overrides.get("with_screenshots"):
            bm["with_screenshots"] = True
```

- [ ] **Step 3: Add vision validation at the top of `run_benchmark`**

Near the top of `benchmark/run.py:run_benchmark()`, just after `resolved_models` is populated and before `save_benchmark_result(version, "config.json", ...)`:

```python
    with_screenshots = config.get("with_screenshots", False)
    if with_screenshots:
        from annotator.core.client import validate_vision_support
        for role, model in resolved_models.items():
            validate_vision_support(model)
        logger.info("Screenshots: enabled -- validated vision support on all models (%s)",
                    ", ".join(sorted(set(resolved_models.values()))))
```

- [ ] **Step 4: Wire `with_screenshots` to Step 0**

In `benchmark/run.py`, update the `run_detect(...)` call to pass `with_screenshots=with_screenshots`:

```python
        detect_output = run_detect(
            version=f"benchmark_{version}",
            model=detect_model,
            mode=detect_mode,
            prompt_version=detect_prompt_version,
            targets=["scaffolding", "rapport"],
            phase_cfg=detect_phase_cfg,
            test=config.get("scenarios", {}).get("test_transcripts", 0),
            with_screenshots=with_screenshots,
        )
```

If `run_detect` doesn't already accept `with_screenshots`, check `annotator/core/detect.py:run_detect` — it should, per the screenshot-enrichment plan. If not, that's a separate fix and you should report BLOCKED with the missing signature.

- [ ] **Step 5: Smoke check the wiring (no API calls)**

Run:
```bash
python -c "
import os
os.environ['STORAGE_BACKEND'] = 'local'
from annotator.core.config import get_benchmark_config
cfg = get_benchmark_config({'with_screenshots': True})
assert cfg.get('with_screenshots') is True
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -q`
Expected: still 162 passed (157 baseline + 2 from Task 1 + 1 from Task 2 + 2 from new tests below if added; verify by count).

- [ ] **Step 7: Commit**

```bash
git add config.yaml annotator/core/config.py benchmark/run.py
git commit -m "feat(benchmark): --with-screenshots flag, vision validation, Step 0 wiring"
```

---

## Task 4: Phase 2 — bridge loads screenshots per scenario

**Files:**
- Modify: `benchmark/core/annotator_bridge.py:prepare_bulk_entries`
- Create: `tests/test_benchmark_screenshots.py`
- Modify: `benchmark/run.py` — pass `with_screenshots` into `prepare_bulk_entries`

**Why:** This is the core bridge fix. The bridge knows the original `scenario.conv_id` (it builds `synth_conv` from it). Load screenshots once per scenario using that real conv_id; build a dict keyed on `scenario.scenario_id` so it lines up with the remapped detections; pass it to `build_analysis_entries`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_screenshots.py`:

```python
"""Bridge loads screenshots per scenario when with_screenshots=True."""
from unittest.mock import patch, MagicMock


def test_prepare_bulk_entries_loads_screenshots_per_scenario():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0",
        conv_id="conv_xyz",
        mode="detected",
        cut_turn=10,
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello",
        student_context="ctx",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0",
        tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )
    fake_screenshots = [
        {"filename": "s1.jpg", "anchor_turn": 6, "storage_path": "deidentified/screenshots/REAL/s1.jpg", "timestamp_seconds": 6.0},
    ]

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots",
               return_value=fake_screenshots) as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
            with_screenshots=True,
        )

    # load_anchored_screenshots called with original conv_id, not scenario_id
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == "conv_xyz"

    # build_analysis_entries got screenshots_by_conv keyed by scenario_id
    kwargs = mock_build.call_args.kwargs
    sbc = kwargs.get("screenshots_by_conv")
    assert sbc == {"conv_xyz__det_0": fake_screenshots}
    assert kwargs.get("with_screenshots") is True


def test_prepare_bulk_entries_default_no_screenshots():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0", conv_id="conv_xyz", mode="detected",
        cut_turn=10, transcript_prefix="Turn 1. TUTOR: hi", student_context="ctx",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0", tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots") as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
        )

    mock_load.assert_not_called()
    assert mock_build.call_args.kwargs.get("screenshots_by_conv") is None
    assert mock_build.call_args.kwargs.get("with_screenshots") is False
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_benchmark_screenshots.py -v`
Expected: FAIL — `prepare_bulk_entries() got an unexpected keyword argument 'with_screenshots'` and import error for `load_anchored_screenshots` in the bridge.

- [ ] **Step 3: Modify the bridge**

In `benchmark/core/annotator_bridge.py`:

Add the import near the top (after the existing annotator imports):

```python
from annotator.core.screenshots import load_anchored_screenshots
```

Update `prepare_bulk_entries`:

```python
def prepare_bulk_entries(
    scenarios: list[Scenario],
    exchanges: dict[str, Exchange],
    annotator_style: str,
    prompt_version: str,
    context_window: int = 20,
    with_screenshots: bool = False,
) -> tuple[list[dict], dict, dict]:
    """Prepare annotation entries for many scenarios at once.

    When with_screenshots=True, loads anchored screenshots for each scenario
    using the *original* scenario.conv_id (not the remapped scenario_id) and
    passes them to build_analysis_entries via screenshots_by_conv.
    """
    all_entries = []
    all_detections = {}
    all_conversations = {}
    screenshots_by_conv: dict[str, list[dict]] | None = (
        {} if with_screenshots else None
    )

    for scenario in scenarios:
        exchange = exchanges.get(scenario.scenario_id)
        if not exchange:
            continue

        synth_conv = build_synthetic_conversation(scenario, exchange)
        detections = build_synthetic_detections(scenario, exchange)
        if not detections:
            continue

        remapped_conv = dict(synth_conv)
        remapped_conv["conversation_id"] = scenario.scenario_id
        remapped_conversations = {scenario.scenario_id: remapped_conv}
        remapped_detections = {scenario.scenario_id: detections[scenario.conv_id]}

        if with_screenshots:
            scenario_screenshots = load_anchored_screenshots(
                scenario.conv_id, synth_conv["turns"],
            )
            screenshots_by_conv[scenario.scenario_id] = scenario_screenshots

        entries = build_analysis_entries(
            remapped_detections, remapped_conversations,
            context_window, prompt_version,
            annotator_style=annotator_style,
            with_screenshots=with_screenshots,
            screenshots_by_conv=screenshots_by_conv,
        )

        all_entries.extend(entries)
        all_detections[scenario.scenario_id] = remapped_detections
        all_conversations[scenario.scenario_id] = remapped_conversations

    return all_entries, all_detections, all_conversations
```

- [ ] **Step 4: Wire `with_screenshots` from `run.py` into the bridge call**

In `benchmark/run.py`, in the Phase 2 style loop, find the existing `prepare_bulk_entries(...)` call and add `with_screenshots=with_screenshots`:

```python
            entries, all_detections, _ = prepare_bulk_entries(
                scenarios=missing,
                exchanges=exchanges,
                annotator_style=style,
                prompt_version=prompt_version,
                context_window=context_window,
                with_screenshots=with_screenshots,
            )
```

`with_screenshots` is read once at the top of `run_benchmark()` (Task 3 Step 3) — it's a local variable in scope.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_benchmark_screenshots.py tests/test_annotate_build.py -v`
Expected: all pass, including the 2 new bridge tests.

Run: `pytest tests/ -q`
Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add benchmark/core/annotator_bridge.py benchmark/run.py tests/test_benchmark_screenshots.py
git commit -m "feat(benchmark): Phase 2 ingests screenshots via bridge per-scenario lookup"
```

---

## Task 5: Phase 1 — exchange ingests screenshots

**Files:**
- Modify: `benchmark/core/exchange.py:run_exchange` and `run_exchanges_batch`
- Modify: `benchmark/run.py` — load screenshots once per scenario before calling exchange runners
- Modify: `tests/test_benchmark_screenshots.py` — add exchange-side tests

**Why:** AI tutor and synthetic student should see whatever was on screen at the cut point. Screenshots are filtered to `anchor_turn ≤ scenario.cut_turn` — nothing visible after the cut leaks into the actor side. Both `tutor_client` and `student_client` need vision support; that was already validated in Task 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark_screenshots.py`:

```python
def test_run_exchange_attaches_filtered_screenshots():
    from benchmark.core.exchange import run_exchange
    from benchmark.core.scenarios import Scenario
    from unittest.mock import MagicMock

    scenario = Scenario(
        scenario_id="conv_xyz__det_0", conv_id="conv_xyz", mode="detected",
        cut_turn=10, transcript_prefix="Turn 1. TUTOR: hi", student_context="ctx",
        detection=None,
    )
    images = ["deidentified/screenshots/REAL/s1.jpg",
              "deidentified/screenshots/REAL/s2.jpg"]

    tutor_resp = MagicMock(text="answer", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    student_resp = MagicMock(text="ok", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    tutor_client = MagicMock()
    tutor_client.model = "claude-opus-4-6"
    tutor_client.generate.return_value = tutor_resp
    student_client = MagicMock()
    student_client.model = "claude-opus-4-6"
    student_client.generate.return_value = student_resp

    run_exchange(
        scenario=scenario,
        tutor_client=tutor_client, student_client=student_client,
        num_turns=1, tutor_max_tokens=100, student_max_tokens=100,
        prompt_version="v1", images=images,
    )

    # All tutor and student calls received images=images
    assert tutor_client.generate.called
    for call in tutor_client.generate.call_args_list:
        assert call.kwargs.get("images") == images
    # Student is skipped on the last round; with num_turns=1 there's no student call.
    # Verify the one tutor call carried images.


def test_run_exchange_no_images_kwarg_default():
    """Exchange without images= kwarg works text-only (back-compat)."""
    from benchmark.core.exchange import run_exchange
    from benchmark.core.scenarios import Scenario
    from unittest.mock import MagicMock

    scenario = Scenario(
        scenario_id="x__0", conv_id="x", mode="detected",
        cut_turn=5, transcript_prefix="Turn 1. TUTOR: hi", student_context="ctx",
        detection=None,
    )
    tutor_resp = MagicMock(text="answer", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    tutor_client = MagicMock(); tutor_client.model = "x"; tutor_client.generate.return_value = tutor_resp
    student_client = MagicMock(); student_client.model = "x"
    student_client.generate.return_value = MagicMock(text="", usage={})

    run_exchange(
        scenario=scenario,
        tutor_client=tutor_client, student_client=student_client,
        num_turns=1, tutor_max_tokens=100, student_max_tokens=100,
        prompt_version="v1",
    )
    for call in tutor_client.generate.call_args_list:
        # Either images is absent (None default) or explicitly None
        assert call.kwargs.get("images") in (None, [])
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_benchmark_screenshots.py -v`
Expected: FAIL — `run_exchange() got an unexpected keyword argument 'images'`.

- [ ] **Step 3: Modify `run_exchange` (sync mode)**

In `benchmark/core/exchange.py:run_exchange` (around line 108-145):

```python
def run_exchange(
    scenario: Scenario,
    tutor_client: ModelClient,
    student_client: ModelClient,
    num_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    prompt_version: str,
    images: list[str] | None = None,
) -> Exchange:
    """Run a multi-turn exchange for a single scenario (sync mode).

    When images is provided, every tutor and student call receives those
    images attached. The image set is fixed for the duration of the exchange
    (no new screenshots emerge from synthetic dialogue).
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    running_transcript = scenario.transcript_prefix
    next_turn_num = scenario.cut_turn + 1

    for i in range(num_turns):
        prompt = _build_role_prompt("TUTOR", running_transcript, scenario.student_context, prompt_version)
        response = tutor_client.generate(
            prompt, json_mode=False, max_tokens=tutor_max_tokens,
            images=images,
        )
        _add_usage(exchange.tutor_usage, response.usage)

        messages = _split_messages(response.text) or ["..."]
        running_transcript, next_turn_num = _append_turns(
            exchange, messages, "TUTOR", running_transcript, next_turn_num,
        )

        if i < num_turns - 1:
            prompt = _build_role_prompt("STUDENT", running_transcript, scenario.student_context, prompt_version)
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

- [ ] **Step 4: Modify `run_exchanges_batch` similarly**

In `run_exchanges_batch`, accept `images_by_scenario: dict[str, list[str]] | None = None` (per-scenario images, since batch mode handles many scenarios at once). Pass `images=images_by_scenario.get(sid)` into each `build_batch_entry(...)` call (Anthropic's `build_batch_entry` should accept `images=` per the screenshot plan).

```python
def run_exchanges_batch(
    scenarios: list[Scenario],
    tutor_client: ModelClient,
    student_client: ModelClient,
    num_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    poll_interval: int,
    save_callback: callable = None,
    prompt_version: str = "v1",
    images_by_scenario: dict[str, list[str]] | None = None,
) -> dict[str, Exchange]:
    ...
    for round_num in range(num_turns):
        ...
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            prompt = _build_role_prompt("TUTOR", transcripts[sid], scenario.student_context, prompt_version)
            scenario_images = (images_by_scenario or {}).get(sid)
            tutor_entries.append(
                build_batch_entry(sid, prompt, json_mode=False,
                                  max_tokens=tutor_max_tokens,
                                  images=scenario_images)
            )
        ...
        # Same change for student_entries below
```

- [ ] **Step 5: Wire screenshot loading + filtering in `benchmark/run.py`**

Where Phase 1 starts (after `Phase 1: Generate Exchanges` log line, before the `existing_files = ...` lookup), build the per-scenario image dict:

```python
        images_by_scenario = None
        if with_screenshots:
            from annotator.core.screenshots import load_anchored_screenshots
            images_by_scenario = {}
            for scenario in scenarios:
                # Use only screenshots visible at or before the cut point.
                turns_for_anchor = [
                    {"turn_number": i + 1, "start_seconds": float(i + 1)}
                    for i in range(scenario.cut_turn)
                ] if scenario.cut_turn else []
                anchored = load_anchored_screenshots(scenario.conv_id, turns_for_anchor)
                visible = [s for s in anchored if s["anchor_turn"] <= scenario.cut_turn]
                images_by_scenario[scenario.scenario_id] = [s["storage_path"] for s in visible]
            logger.info("Screenshots: loaded for %d scenarios", len(images_by_scenario))
```

**NOTE:** the synthetic `turns_for_anchor` above is a stub used only for anchoring the *original* screenshot list. For correct anchoring you actually need the real transcript turns with their `start_seconds`. Better: load the real conversation once per scenario (via `load_all_transcripts()` cached at run start) and pass its turn list. Implementer: please do this — read `annotator/core/storage.load_all_transcripts()`, pull `conversations[scenario.conv_id]["turns"]`, pass that. This avoids fake start_seconds.

Then in the existing call sites:

```python
            if ann_mode == "batch":
                new_exchanges = run_exchanges_batch(
                    ...
                    images_by_scenario=images_by_scenario,
                )
            else:
                new_exchanges = {}
                for i, scenario in enumerate(missing):
                    ...
                    exchange = run_exchange(
                        scenario=scenario,
                        ...
                        images=(images_by_scenario or {}).get(scenario.scenario_id),
                    )
                    ...
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_benchmark_screenshots.py -v`
Expected: 4 tests pass (2 from Task 4 + 2 new).

Run: `pytest tests/ -q`
Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add benchmark/core/exchange.py benchmark/run.py tests/test_benchmark_screenshots.py
git commit -m "feat(benchmark): Phase 1 exchange attaches anchored screenshots up to cut_turn"
```

---

## Task 6: Smoke test with `--with-screenshots`

**Files:** none (manual verification)

- [ ] **Step 1: Run a tiny end-to-end with screenshots**

```bash
python -m benchmark --tutor-profile anthropic --test 2 --max-scenarios 2 --mode sync --version smoke_screenshots --with-screenshots
```

Expected log lines (in order):
- `Screenshots: enabled -- validated vision support on all models (...)`
- `Screenshots: loaded for 2 scenarios`
- Detection completes with images in `detections.json` (`total_images_sent > 0`)
- Phase 1 completes (exchanges include images in tutor + student calls — visible in `logs/{version}/run.log` if you check the API request payloads)
- Phase 2 logs `images_attached` per scenario in the saved annotation shards

- [ ] **Step 2: Inspect annotation shard for images metadata**

```bash
ls results/benchmark/smoke_screenshots/annotations/anthropic/balanced/
cat results/benchmark/smoke_screenshots/annotations/anthropic/balanced/<one>.json | python -c "import json,sys; d=json.load(sys.stdin); print('images_attached=', d.get('results', {}).get(list(d.get('results', {}).keys())[0], {}).get('images_attached'))"
```
Expected: a non-zero `images_attached` if the scenario's conv had any flagged-clean screenshots.

- [ ] **Step 3: Re-run, confirm pre-filter still works**

```bash
python -m benchmark --tutor-profile anthropic --test 2 --max-scenarios 2 --mode sync --version smoke_screenshots --with-screenshots
```
Expected: every style logs `2 cached, 0 to annotate` — same fast-path behavior we verified for text-only resume.

- [ ] **Step 4: Cleanup**

```bash
rm -rf results/benchmark/smoke_screenshots results/annotator/benchmark_smoke_screenshots logs/smoke_screenshots
```

---

## Task 7: Update docs

**Files:**
- Modify: `docs/lessons_learned.md` — replace the 2026-04-28 "text-only mode" entry with a "screenshots now wired" entry that points to this plan.
- Modify: `docs/status.md` — Benchmark Pipeline section, replace "Known limitation" with "Screenshots: opt-in via `--with-screenshots`."
- Modify: `docs/plans/_summary.md` — add the entry for this plan as Status: Implemented.

- [ ] **Step 1: Replace the lessons_learned entry**

Find the `## 2026-04-28: Benchmark annotation runs in text-only mode` block in `docs/lessons_learned.md` and replace its body with:

```markdown
## 2026-04-28: Benchmark annotation/exchange/detection were text-only (now opt-in)

**Original gap:** The annotator pipeline supported `--with-screenshots` (delivered 2026-04-24) but the benchmark — which reuses the annotator under the hood — never threaded the flag. Detection ran without images; tutor/student exchanges were text-only; annotation didn't see the screen even though the equivalent annotator-standalone run did.

**Why the flag was a no-op:** `build_analysis_entries` and `build_detection_entries` were keying screenshot lookup on `conv_id`. The benchmark bridge remaps `conv_id -> scenario_id` to namespace bulk batch keys, so the lookup silently returned [].

**Fix:** Decoupled screenshot loading from screenshot use. Both functions now accept an optional `screenshots_by_conv` dict — if provided, the function uses it directly instead of looking up by conv_id. The bridge loads screenshots using the original `scenario.conv_id` and passes a dict keyed on `scenario_id` so the function's iteration key still matches. Phase 1 exchange similarly accepts `images=` (sync) and `images_by_scenario=` (batch). Vision validation runs once at run start when `--with-screenshots` is on.

**Caveat:** Default off. Existing text-only benchmark runs are byte-for-byte unchanged. To compare two runs, keep the screenshot mode constant on both sides.
```

- [ ] **Step 2: Update `status.md`**

Replace the "Known limitation" block under "Benchmark Pipeline" with:

```markdown
**Screenshots**: opt-in via `--with-screenshots` (or `benchmark.with_screenshots: true` in config.yaml). When on, all three phases (detection, exchange, annotation) attach anchored screenshots from `deidentified/screenshots/{conv_id}/`. Default off — text-only runs reproduce prior numbers exactly.
```

- [ ] **Step 3: Add the entry to `_summary.md`**

Insert after the production-readiness entry:

```markdown
### 2026-04-28 — [Benchmark screenshot ingestion](2026-04-28-benchmark-screenshots.md)

**Goal**: The annotator side supported `--with-screenshots` but the benchmark didn't thread it through any phase. AI tutors were graded text-only while the same human-tutor moments could be graded with full visual context — apples-to-oranges. Wire screenshots into Step 0 (detection), Phase 1 (tutor + synthetic student), and Phase 2 (annotation) so the benchmark measures what the annotator was upgraded to measure.
**Status**: TBD — Implemented after Task 6 smoke passes.
**Result**: TBD.
```

- [ ] **Step 4: Commit**

```bash
git add docs/lessons_learned.md docs/status.md docs/plans/_summary.md
git commit -m "docs: benchmark screenshot ingestion plan + status update"
```

---

## Task 8: Run the full benchmark with screenshots

**Files:** none (the actual run)

- [ ] **Step 1: Confirm config is what you expect**

Read `config.yaml`. Confirm `benchmark.with_screenshots: true` (or plan to pass `--with-screenshots` on every invocation). Confirm `tutor_profiles`, mode, and styles match what you want.

- [ ] **Step 2: Kick off**

```bash
python -m benchmark --tutor-profile anthropic --with-screenshots
```

Detection batches first (longer with images), exchange runs in batch mode (per-scenario images), annotation runs per style. Resume infra from the prior plan handles ctrl-C and crashes.

- [ ] **Step 3: Monitor**

Tail the log: `tail -f logs/{version}/run.log`. Confirm `Screenshots: enabled` early. Confirm every batch has non-zero `images_attached` in its produced shards. If a batch shows zero images for every scenario, something's broken — stop and investigate before the run completes.

- [ ] **Step 4: Update `_summary.md` and `status.md` with final numbers**

When done, update the plan's entry in `_summary.md` and the status doc with headline numbers (mean score per style, total images sent, total tokens, total cost).

---

## Self-Review Notes

**Spec coverage:**
- Detection ingests screenshots → Tasks 2, 3.4
- Annotation ingests screenshots → Tasks 1, 4
- Exchange ingests screenshots → Task 5
- Vision validation fails fast → Task 3.3
- Resume still works with screenshots → Task 6.3 (re-run check)
- Backward compat (default off, text-only runs unchanged) → every task uses opt-in defaults

**Risks called out for execution:**
- Task 5 Step 5: the "synthetic turns_for_anchor" stub is wrong; implementer must replace with real transcript turns from `load_all_transcripts()`. I'm flagging this rather than papering over it.
- `validate_vision_support` is imported in Task 3 from `annotator.core.client`. If that helper doesn't exist there, the screenshot-enrichment plan didn't ship it where I expected — implementer should grep and adjust the import path. Report BLOCKED if no such helper exists anywhere.
- `build_batch_entry` accepting `images=` was promised by the screenshot-enrichment plan. Implementer must verify by reading the function signature before Task 5 Step 4. If it's missing, that's a separate fix in `annotator/core/client.py`.

**Type consistency:** `screenshots_by_conv: dict[str, list[dict]] | None` is the consistent shape across `build_analysis_entries`, `build_detection_entries`, and the dict the bridge constructs. Each list entry has `{"filename", "anchor_turn", "storage_path", "timestamp_seconds"}` per `screenshots.load_anchored_screenshots`'s contract.
