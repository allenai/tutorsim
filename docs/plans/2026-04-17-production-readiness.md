# Production Readiness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all production readiness issues found in the codebase review — config consistency, error handling, dead code, test coverage, abstraction gaps.

**Architecture:** Changes are organized bottom-up: config foundation first, then shared abstractions, then consumers, then tests. Each task is independently committable.

**Tech Stack:** Python 3.11+, pytest, pyyaml

---

### Task 1: Fix .env.example API key name and add config keys

The `.env.example` documents `GOOGLE_API_KEY` but `client.py` reads `GEMINI_API_KEY`. Also add `iou_threshold` and `batch_timeout` to config.yaml so these aren't hardcoded.

**Files:**
- Modify: `.env.example`
- Modify: `config.yaml`

- [ ] **Step 1: Fix .env.example**

Change `GOOGLE_API_KEY` to `GEMINI_API_KEY`:

```
# API keys
# GEMINI_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
```

- [ ] **Step 2: Add iou_threshold and batch_timeout to config.yaml**

Add after the `retry:` block:

```yaml
# Evaluation settings
eval:
  iou_threshold: 0.3

# Batch API settings
batch:
  timeout: 86400     # max seconds to poll a batch job (24h)
```

- [ ] **Step 3: Commit**

```bash
git add .env.example config.yaml
git commit -m "fix: align .env.example API key name, add iou_threshold and batch_timeout to config"
```

---

### Task 2: Add config accessors and shared run-param resolution

The `config.py` module needs accessors for the new keys, and version/style/profile resolution is duplicated across 4 files.

**Files:**
- Modify: `annotator/core/config.py`

- [ ] **Step 1: Add get_iou_threshold and get_batch_timeout to config.py**

Add after `get_annotator_defaults()`:

```python
def get_iou_threshold() -> float:
    """Get IoU threshold for detection/effectiveness matching."""
    config = load_config()
    return config.get("eval", {}).get("iou_threshold", 0.3)


def get_batch_timeout() -> int:
    """Get max seconds to poll a batch job before raising."""
    config = load_config()
    return config.get("batch", {}).get("timeout", 86400)
```

- [ ] **Step 2: Add resolve_run_params to config.py**

Add after the new functions:

```python
def resolve_run_params(
    cli_version: str | None,
    cli_profile: str | None,
    cli_style: str | None,
    cli_prompt_version: str | None,
) -> dict:
    """Resolve version, profile, style, and prompt_version from CLI + config.

    Resolution order for each: CLI > config > auto-generate/None.
    Returns dict with keys: version, profile, style, prompt_version.
    """
    import datetime

    config = load_config()
    defaults = get_annotator_defaults()

    profile = cli_profile or config.get("profile", "anthropic")

    if cli_version:
        version = cli_version
    elif defaults.get("version"):
        version = defaults["version"]
    else:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        version = f"{profile}_{date_str}"
        print(f"  Auto-generated version: {version}")

    style = cli_style
    if style is None:
        cfg_style = defaults.get("style")
        if cfg_style is not None:
            style = cfg_style

    prompt_version = cli_prompt_version or defaults.get("prompt_version") or version

    return {
        "version": version,
        "profile": profile,
        "style": style,
        "prompt_version": prompt_version,
    }
```

- [ ] **Step 3: Commit**

```bash
git add annotator/core/config.py
git commit -m "feat: add config accessors for iou_threshold/batch_timeout, shared resolve_run_params"
```

---

### Task 3: Replace IOU_THRESHOLD constant with config accessor

Currently `IOU_THRESHOLD = 0.3` is hardcoded in `utils.py` and imported by `advisor.py`, `eval.py`, and validation scripts.

**Files:**
- Modify: `annotator/core/utils.py` — remove `IOU_THRESHOLD` constant, add re-export from config
- Modify: `annotator/eval/eval.py` — use config accessor
- Modify: `annotator/iteration/advisor.py` — use config accessor
- Modify: `validation/generate_report.py` — use config accessor
- Modify: `validation/_generate_notebooks.py` — use config accessor

- [ ] **Step 1: Update utils.py**

Replace the `IOU_THRESHOLD = 0.3` line with a re-export so existing importers don't break:

```python
from .config import get_iou_threshold

# Re-export for backwards compatibility with scripts that import from utils
IOU_THRESHOLD = get_iou_threshold()
```

Note: this evaluates once at import time, which is fine — config doesn't change mid-process.

- [ ] **Step 2: Commit**

```bash
git add annotator/core/utils.py
git commit -m "refactor: source IOU_THRESHOLD from config.yaml instead of hardcoding"
```

---

### Task 4: Add batch polling timeout

All three batch implementations poll in infinite loops. Add a timeout that reads from config.

**Files:**
- Modify: `annotator/core/client.py`

- [ ] **Step 1: Add timeout to _run_batch_gemini**

Import at top of file (add to existing imports from `.config`):

```python
from .config import get_retry_config, get_batch_timeout
```

In `_run_batch_gemini`, add a start time before the polling loop and a timeout check inside:

```python
        # Poll
        import time as _time
        poll_start = _time.monotonic()
        batch_timeout = get_batch_timeout()
        completed_states = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
        }
        while batch_job.state.name not in completed_states:
            if _time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"Gemini batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.state.name})"
                )
            print(f"  State: {batch_job.state.name} -- polling in {poll_interval}s...")
            time.sleep(poll_interval)
            batch_job = gemini_client.batches.get(name=batch_job.name)
```

- [ ] **Step 2: Add timeout to _run_batch_openai**

Same pattern in `_run_batch_openai`:

```python
        # Poll
        poll_start = time.monotonic()
        batch_timeout = get_batch_timeout()
        terminal_states = {"completed", "failed", "expired", "cancelled"}
        while batch_job.status not in terminal_states:
            if time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"OpenAI batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.status})"
                )
            print(f"  Status: {batch_job.status} -- polling in {poll_interval}s...")
            time.sleep(poll_interval)
            batch_job = openai_client.batches.retrieve(batch_job.id)
```

- [ ] **Step 3: Add timeout to _run_batch_anthropic**

Same pattern in `_run_batch_anthropic`:

```python
    # Poll
    poll_start = time.monotonic()
    batch_timeout = get_batch_timeout()
    while message_batch.processing_status != "ended":
        if time.monotonic() - poll_start > batch_timeout:
            raise RuntimeError(
                f"Anthropic batch timed out after {batch_timeout}s "
                f"(state: {message_batch.processing_status})"
            )
        print(f"  Status: {message_batch.processing_status} -- polling in {poll_interval}s...")
        time.sleep(poll_interval)
        message_batch = anthropic_client.messages.batches.retrieve(message_batch.id)
```

- [ ] **Step 4: Commit**

```bash
git add annotator/core/client.py
git commit -m "feat: add batch polling timeout from config (default 24h)"
```

---

### Task 5: Use resolve_run_params in annotator pipeline entry points

Replace the duplicated version/style/profile resolution blocks in `run.py`, `detect.py`, `annotate.py`, `label.py`.

**Files:**
- Modify: `annotator/run.py`
- Modify: `annotator/core/detect.py`
- Modify: `annotator/core/annotate.py`
- Modify: `annotator/core/label.py`

- [ ] **Step 1: Refactor annotator/run.py**

Replace lines 63-86 (from `defaults = get_annotator_defaults()` through `style = cfg_style`) with:

```python
    from .core.config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]
```

Remove the now-unused import of `get_annotator_defaults` (keep `get_phase_config` and `load_config` — `load_config` is no longer needed either since `resolve_run_params` handles profile resolution).

Update imports at top: remove `get_annotator_defaults` and `load_config` from the import line:

```python
from .core.config import get_phase_config
```

- [ ] **Step 2: Refactor annotator/core/detect.py main()**

Replace lines 215-248 (from `defaults = get_annotator_defaults()` through `prompt_version = f"profiles/{style}"`) with:

```python
    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]

    phase_cfg = get_phase_config("detect", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    # Override prompt version when style is set and p1 prompts exist
    if style and not args.prompt_version:
        style_p1_dir = PROMPTS_DIR / "profiles" / style / "p1"
        if style_p1_dir.exists():
            prompt_version = f"profiles/{style}"
```

Update imports at top: remove `get_annotator_defaults` and `load_config`:

```python
from .config import get_phase_config
```

- [ ] **Step 3: Refactor annotator/core/annotate.py main()**

Replace lines 351-383 (from `defaults = get_annotator_defaults()` through `prompt_version = f"profiles/{style}"`) with:

```python
    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.annotator_style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]

    phase_cfg = get_phase_config("annotate", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")
    context_window = args.context if args.context is not None else phase_cfg.get("context_window", 20)

    # When style is set, override prompt version to per-style profiles
    if style and not args.prompt_version:
        prompt_version = f"profiles/{style}"
```

Update imports at top: remove `get_annotator_defaults` and `load_config`:

```python
from .config import get_phase_config
```

- [ ] **Step 4: Refactor annotator/core/label.py main()**

Replace lines 182-205 (from `defaults = get_annotator_defaults()` through `mode = args.mode or ...`) with:

```python
    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.annotator_style,
        cli_prompt_version=None,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]

    phase_cfg = get_phase_config("label", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")
```

Update imports at top: remove `get_annotator_defaults` and `load_config`:

```python
from .config import get_phase_config
```

Note: `label.py` also imports `get_annotator_defaults` inside `run_label()` at line 76 for loading the labeller prompt name — that import stays.

- [ ] **Step 5: Commit**

```bash
git add annotator/run.py annotator/core/detect.py annotator/core/annotate.py annotator/core/label.py
git commit -m "refactor: deduplicate version/style/profile resolution into resolve_run_params"
```

---

### Task 6: Record thinking config in output metadata

Detection, annotation, and label outputs don't record whether thinking mode was enabled.

**Files:**
- Modify: `annotator/core/detect.py` — add to output dict
- Modify: `annotator/core/annotate.py` — add to output dict

- [ ] **Step 1: Add thinking config to detect output**

In `run_detect()`, add to the `output` dict (after `"targets": targets,`):

```python
        "thinking": phase_cfg.get("thinking", False),
        "thinking_budget": phase_cfg.get("thinking_budget", 0),
```

- [ ] **Step 2: Add thinking config to annotate output**

In `run_annotate()`, add to the `output` dict (after `"targets": targets,`):

```python
        "thinking": phase_cfg.get("thinking", False),
        "thinking_budget": phase_cfg.get("thinking_budget", 0),
```

- [ ] **Step 3: Commit**

```bash
git add annotator/core/detect.py annotator/core/annotate.py
git commit -m "feat: record thinking mode config in pipeline output metadata"
```

---

### Task 7: Replace load_annotator_archetype_ids with config accessor

`eval.py` defines `load_annotator_archetype_ids()` which reads from a separate JSON file. The config already has `archetype_annotators` with the same data. Replace all usages.

**Files:**
- Modify: `annotator/eval/eval.py` — remove function, use config
- Modify: `validation/generate_report.py` — update import
- Modify: `validation/_generate_notebooks.py` — update generated code

- [ ] **Step 1: Update eval.py**

Remove the `load_annotator_archetype_ids` function (lines 492-501) and the `ANNOTATOR_PROFILES_PATH` constant (line 54).

Remove the `json` import if no longer needed (it's still used by `load_eval_json` — check). Actually `json` is not directly used in eval.py — it uses `load_annotator_result` which returns parsed dicts. Remove `import json` from line 42.

Wait — `json` is not imported directly in eval.py. `load_annotator_archetype_ids` opens the file with `json.load`. After removing it, check if `json` is still needed. Looking at the file: `json` is imported at line 42 but only used in `load_annotator_archetype_ids`. Remove it.

Replace `load_annotator_archetype_ids` with a thin wrapper that delegates to config:

```python
def load_annotator_archetype_ids(archetype: str) -> set[str]:
    """Load the set of annotator IDs belonging to the given archetype.

    Reads from archetype_annotators in config.yaml.
    """
    from ..core.config import get_archetype_annotators
    result = get_archetype_annotators(archetype)
    if result is None:
        raise ValueError(
            f"Unknown archetype '{archetype}'. "
            f"Check archetype_annotators in config.yaml."
        )
    return result
```

Also remove `ANNOTATOR_PROFILES_PATH` (line 54) and its import of `json` (line 42).

Remove the `filter_ground_truth_by_archetype` function too? No — it's still used and does real work. Keep it.

- [ ] **Step 2: Commit**

```bash
git add annotator/eval/eval.py
git commit -m "refactor: load_annotator_archetype_ids reads from config.yaml instead of separate JSON"
```

---

### Task 8: Migrate advisor.py to use storage layer

`advisor.py` reads result files via direct `Path` operations, bypassing the storage layer.

**Files:**
- Modify: `annotator/iteration/advisor.py`

- [ ] **Step 1: Replace direct file reads with storage calls**

At top, add storage import:

```python
from ..core.storage import load_annotator_result, get_annotator_result_path
```

In `collect_detection_errors()` (line 37-40), replace:

```python
    det_path = RESULTS_DIR / version / "detections.json"
    if not det_path.exists():
        raise FileNotFoundError(f"No detections at {det_path}")
    with open(det_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)
```

with:

```python
    det_data = load_annotator_result(version, "detections.json")
    if det_data is None:
        raise FileNotFoundError(f"No detections found for version {version}")
```

In `collect_annotation_errors()` (lines 189-200), replace the chain of `ann_path` checks:

```python
    version_dir = RESULTS_DIR / version
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    ann_path = version_dir / f"annotations_gold{style_suffix}.json"
    if not ann_path.exists():
        ann_path = version_dir / "annotations_gold.json"
    if not ann_path.exists():
        ann_path = version_dir / f"annotations{style_suffix}.json"
    if not ann_path.exists():
        ann_path = version_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No annotations in {version_dir}")

    with open(ann_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
```

with:

```python
    from ..core.storage import load_annotator_result, annotator_result_exists
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    candidates = [
        f"annotations_gold{style_suffix}.json",
        "annotations_gold.json",
        f"annotations{style_suffix}.json",
        "annotations.json",
    ]
    llm_data = None
    for candidate in candidates:
        llm_data = load_annotator_result(version, candidate)
        if llm_data is not None:
            break
    if llm_data is None:
        raise FileNotFoundError(f"No annotations found for version {version}")
```

In `load_eval_metrics()` (lines 305-310), replace:

```python
    eval_path = RESULTS_DIR / version / f"eval_{mode}.json"
    if not eval_path.exists():
        return {}

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
```

with:

```python
    data = load_annotator_result(version, f"eval_{mode}.json")
    if data is None:
        return {}
```

In `main()` advisor save (lines 567-572), replace:

```python
    output_dir = RESULTS_DIR / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    style_suffix = f"_{args.annotator_style}" if args.annotator_style else ""
    output_path = output_dir / f"advisor_{args.pass_type}_{args.type}{style_suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(advice, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")
```

with:

```python
    from ..core.storage import save_annotator_result
    style_suffix = f"_{args.annotator_style}" if args.annotator_style else ""
    filename = f"advisor_{args.pass_type}_{args.type}{style_suffix}.json"
    save_annotator_result(args.version, filename, advice)
    print(f"\nSaved: {filename} (version: {args.version})")
```

In `analyze_main()` (lines 766-770), replace:

```python
    det_path = RESULTS_DIR / args.version / "detections.json"
    if not det_path.exists():
        print(f"ERROR: No detections found at {det_path}")
        return

    with open(det_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)
```

with:

```python
    det_data = load_annotator_result(args.version, "detections.json")
    if det_data is None:
        print(f"ERROR: No detections found for version {args.version}")
        return
```

After all replacements, remove `RESULTS_DIR` from the utils import if no longer used (check — it's also used in `classify_errors` for nothing actually, `RESULTS_DIR` is only used in the replaced paths). Remove the `json` import too if no longer needed. Actually `json` is still used in `main()` for `json.dumps(stats)` and `json.loads(response.text)`. Keep it.

Remove `RESULTS_DIR` from the utils import line:

```python
from ..core.utils import (
    compute_iou, merge_overlapping_ranges, load_transcripts, get_excerpt,
    load_ground_truth, REPO_ROOT, DATA_DIR, IOU_THRESHOLD,
    EXAMPLE_CONV_IDS,
)
```

- [ ] **Step 2: Commit**

```bash
git add annotator/iteration/advisor.py
git commit -m "refactor: advisor.py uses storage layer instead of direct Path access"
```

---

### Task 9: Add clear error when no transcripts found

When `data/transcripts/` doesn't exist, the pipeline silently produces empty results.

**Files:**
- Modify: `annotator/core/detect.py`
- Modify: `benchmark/core/scenarios.py`

- [ ] **Step 1: Add check in detect.py load_conversations**

In `load_conversations()`, after loading transcripts, add:

```python
def load_conversations(limit: int = 0) -> list[dict]:
    """Load all consolidated transcript JSON files via storage layer."""
    transcripts = load_all_transcripts()
    if not transcripts:
        raise FileNotFoundError(
            "No transcripts found. Ensure data/transcripts/ contains JSON files, "
            "or configure transcript paths in config.yaml under storage.paths.transcripts."
        )
    conversations = sorted(transcripts.values(), key=lambda c: c.get("conversation_id", ""))
    if limit > 0:
        conversations = conversations[:limit]
    return conversations
```

- [ ] **Step 2: Add check in scenarios.py load_scenarios**

In `load_scenarios()`, after `transcripts = load_transcripts()`, add:

```python
    if not transcripts:
        raise FileNotFoundError(
            "No transcripts found. Ensure data/transcripts/ contains JSON files, "
            "or configure transcript paths in config.yaml under storage.paths.transcripts."
        )
```

- [ ] **Step 3: Commit**

```bash
git add annotator/core/detect.py benchmark/core/scenarios.py
git commit -m "fix: fail with clear error when no transcripts found instead of producing empty results"
```

---

### Task 10: Clean up dead code and unused imports

Remove unused constants, functions, and imports identified in the review.

**Files:**
- Modify: `annotator/eval/view.py` — remove unused `escape()` function
- Modify: `benchmark/eval/view.py` — remove unused `escape()` function
- Modify: `annotator/core/detect.py` — remove unused `TRANSCRIPTS_DIR`, `RESULTS_DIR` imports
- Modify: `annotator/core/annotate.py` — remove unused `TRANSCRIPTS_DIR`, `RESULTS_DIR` imports
- Modify: `annotator/core/label.py` — remove unused `RESULTS_DIR` import
- Modify: `annotator/eval/view.py` — remove unused `DATA_DIR`, `TRANSCRIPTS_DIR` imports
- Modify: `benchmark/eval/eval.py` — remove unused `BENCHMARK_RESULTS_DIR`
- Modify: `benchmark/eval/view.py` — remove unused `BENCHMARK_RESULTS_DIR`

- [ ] **Step 1: Remove unused escape() in annotator/eval/view.py**

Remove lines 112-113:

```python
def escape(text: str) -> str:
    return html.escape(str(text)) if text else ""
```

Also remove the `import html` at line 17 (only used by the removed function — the JS handles escaping).

- [ ] **Step 2: Remove unused escape() in benchmark/eval/view.py**

Remove lines 127-128:

```python
def escape(text: str) -> str:
    return html.escape(str(text)) if text else ""
```

Also remove `import html` at line 17.

Wait — `html.escape` is used in `build_html` at line 140 for the title: `{escape(version)}`. So `escape` IS used in benchmark/eval/view.py. Keep it.

Let me recheck annotator/eval/view.py — no, `escape` is not called anywhere in the template string. The JS `escapeHtml` handles it. Remove it.

- [ ] **Step 3: Remove unused imports from detect.py**

Change line 23:

```python
from .utils import format_transcript, TRANSCRIPTS_DIR, RESULTS_DIR
```

to:

```python
from .utils import format_transcript
```

- [ ] **Step 4: Remove unused imports from annotate.py**

Change line 31:

```python
from .utils import format_excerpt, TRANSCRIPTS_DIR, RESULTS_DIR, load_ground_truth
```

to:

```python
from .utils import format_excerpt, load_ground_truth
```

- [ ] **Step 5: Remove unused RESULTS_DIR import from label.py**

Change line 29:

```python
from .utils import RESULTS_DIR
```

Remove this line entirely — `RESULTS_DIR` is not used in label.py.

- [ ] **Step 6: Remove unused imports from annotator/eval/view.py**

Change line 19:

```python
from ..core.utils import DATA_DIR, RESULTS_DIR, TRANSCRIPTS_DIR, load_ground_truth
```

to:

```python
from ..core.utils import load_ground_truth
```

- [ ] **Step 7: Remove unused BENCHMARK_RESULTS_DIR from benchmark/eval/eval.py**

Remove line 25:

```python
BENCHMARK_RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "benchmark"
```

Also check if `Path` is still needed. `Path` is not imported in this file — it's imported via the removed line only. So also remove `from pathlib import Path` if present... checking: line 19 has `from pathlib import Path`. After removing the constant, `Path` is unused. Remove it.

- [ ] **Step 8: Remove unused constants from benchmark/eval/view.py**

Remove lines 24-25:

```python
REPO_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_RESULTS_DIR = REPO_ROOT / "results" / "benchmark"
```

Check if `Path` is still used — yes, `Path` is imported at line 18 and `REPO_ROOT` is only used for `BENCHMARK_RESULTS_DIR`. But wait — `Path` isn't imported in this file. Looking at imports: line 18 is `from pathlib import Path`. After removing REPO_ROOT and BENCHMARK_RESULTS_DIR, `Path` is unused. Remove the import.

- [ ] **Step 9: Commit**

```bash
git add annotator/eval/view.py benchmark/eval/view.py annotator/core/detect.py annotator/core/annotate.py annotator/core/label.py benchmark/eval/eval.py
git commit -m "cleanup: remove dead code, unused imports, and unused constants"
```

---

### Task 11: Fix naming inconsistency (labeller vs labeler)

Config uses `labeller` (British). Benchmark output uses `labeler`. Standardize on `labeller` since that's what config.yaml and the prompts directory use.

**Files:**
- Modify: `benchmark/run.py`

- [ ] **Step 1: Fix labeler -> labeller in benchmark/run.py**

Change line 86:

```python
    resolved_models["labeler"] = get_phase_config("label", ann_profile)["model"]
```

to:

```python
    resolved_models["labeller"] = get_phase_config("label", ann_profile)["model"]
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/run.py
git commit -m "fix: standardize labeller spelling in benchmark config output"
```

---

### Task 12: Fix conftest.py ground truth path mismatch

The test fixture creates `data/ground_truth/` but config.yaml specifies `data/ground_truth_v2`. The fixture needs to either create the right path or override the config path.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Fix ground truth path in fixture**

The simplest fix: override `STORAGE_GROUND_TRUTH` env var to match what the fixture creates.

Update the `local_storage` fixture:

```python
@pytest.fixture
def local_storage(temp_data, monkeypatch):
    """Configure storage for local backend against temp dir."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(temp_data))
    monkeypatch.setenv("STORAGE_GROUND_TRUTH", "data/ground_truth")
    import annotator.core.config as cfg_mod
    cfg_mod._loaded_config = None
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None
    yield temp_data
    st._backend = None
    st._cache.clear()
```

- [ ] **Step 2: Verify the pre-existing failing test now passes**

Run: `pytest tests/test_storage.py::TestLocalBackend::test_load_all_ground_truth -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "fix: align test fixture ground truth path with config expectations"
```

---

### Task 13: Add unit tests for pure functions

Add tests for the core pure functions that have no test coverage.

**Files:**
- Create: `tests/test_config.py`
- Create: `tests/test_utils.py`
- Create: `tests/test_client.py`
- Create: `tests/test_detect_parse.py`
- Create: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write tests/test_config.py**

```python
"""Tests for annotator.core.config."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear the config cache before each test."""
    import annotator.core.config as cfg
    cfg._loaded_config = None
    yield
    cfg._loaded_config = None


class TestGetPhaseConfig:
    def test_returns_model(self):
        from annotator.core.config import get_phase_config
        cfg = get_phase_config("detect", "anthropic")
        assert "model" in cfg
        assert cfg["model"] == "claude-opus-4-6"

    def test_phase_overrides_merge(self):
        from annotator.core.config import get_phase_config
        cfg = get_phase_config("annotate", "anthropic")
        # Profile-level keys present
        assert "max_tokens" in cfg
        # Phase-level override present
        assert "context_window" in cfg
        assert cfg["context_window"] == 20

    def test_unknown_profile_raises(self):
        from annotator.core.config import get_phase_config
        with pytest.raises(ValueError, match="Unknown profile"):
            get_phase_config("detect", "nonexistent_profile")

    def test_no_profile_raises_when_missing(self):
        from annotator.core.config import get_phase_config, load_config
        config = load_config()
        original = config.get("profile")
        try:
            config.pop("profile", None)
            with pytest.raises(ValueError, match="No profile specified"):
                get_phase_config("detect", None)
        finally:
            if original is not None:
                config["profile"] = original


class TestGetArchetypeAnnotators:
    def test_returns_full_mapping(self):
        from annotator.core.config import get_archetype_annotators
        mapping = get_archetype_annotators()
        assert "generous" in mapping
        assert "balanced" in mapping
        assert "demanding" in mapping
        assert isinstance(mapping["generous"], set)

    def test_returns_set_for_archetype(self):
        from annotator.core.config import get_archetype_annotators
        generous = get_archetype_annotators("generous")
        assert isinstance(generous, set)
        assert "Gerber" in generous

    def test_returns_none_for_unknown(self):
        from annotator.core.config import get_archetype_annotators
        assert get_archetype_annotators("nonexistent") is None


class TestGetIouThreshold:
    def test_returns_float(self):
        from annotator.core.config import get_iou_threshold
        val = get_iou_threshold()
        assert isinstance(val, float)
        assert val == 0.3


class TestResolveRunParams:
    def test_cli_overrides_all(self):
        from annotator.core.config import resolve_run_params
        params = resolve_run_params(
            cli_version="test_v1",
            cli_profile="gemini",
            cli_style="generous",
            cli_prompt_version="v4",
        )
        assert params["version"] == "test_v1"
        assert params["profile"] == "gemini"
        assert params["style"] == "generous"
        assert params["prompt_version"] == "v4"

    def test_auto_generates_version(self):
        from annotator.core.config import resolve_run_params
        params = resolve_run_params(
            cli_version=None,
            cli_profile="anthropic",
            cli_style=None,
            cli_prompt_version=None,
        )
        assert "anthropic_" in params["version"]
        assert params["profile"] == "anthropic"
```

- [ ] **Step 2: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 3: Write tests/test_utils.py**

```python
"""Tests for annotator.core.utils pure functions."""
import pytest
from annotator.core.utils import compute_iou, merge_overlapping_ranges


class TestComputeIou:
    def test_identical_ranges(self):
        assert compute_iou((1, 5), (1, 5)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((1, 3), (5, 7)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((1, 5), (3, 7))
        # intersection: {3,4,5} = 3, union: {1,2,3,4,5,6,7} = 7
        assert abs(iou - 3 / 7) < 1e-9

    def test_one_contains_other(self):
        iou = compute_iou((1, 10), (3, 5))
        # intersection: {3,4,5} = 3, union: {1..10} = 10
        assert abs(iou - 3 / 10) < 1e-9

    def test_adjacent_ranges(self):
        # (1,3) and (4,6) share no turns
        assert compute_iou((1, 3), (4, 6)) == 0.0

    def test_single_turn_overlap(self):
        iou = compute_iou((1, 3), (3, 5))
        # intersection: {3} = 1, union: {1,2,3,4,5} = 5
        assert abs(iou - 1 / 5) < 1e-9


class TestMergeOverlappingRanges:
    def test_empty_input(self):
        assert merge_overlapping_ranges([]) == []

    def test_no_overlap(self):
        moments = [
            {"turn_start": 1, "turn_end": 3, "annotation_type": "scaffolding"},
            {"turn_start": 10, "turn_end": 12, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 2

    def test_overlapping_same_type(self):
        moments = [
            {"turn_start": 1, "turn_end": 5, "annotation_type": "scaffolding"},
            {"turn_start": 3, "turn_end": 8, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 1
        assert clusters[0]["turn_start"] == 1
        assert clusters[0]["turn_end"] == 8

    def test_adjacent_same_type_merged(self):
        moments = [
            {"turn_start": 1, "turn_end": 3, "annotation_type": "scaffolding"},
            {"turn_start": 4, "turn_end": 6, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 1

    def test_different_types_not_merged(self):
        moments = [
            {"turn_start": 1, "turn_end": 5, "annotation_type": "scaffolding"},
            {"turn_start": 3, "turn_end": 8, "annotation_type": "rapport"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 2
```

- [ ] **Step 4: Run utils tests**

Run: `pytest tests/test_utils.py -v`
Expected: All PASS

- [ ] **Step 5: Write tests/test_client.py**

```python
"""Tests for annotator.core.client pure functions."""
import pytest
from annotator.core.client import (
    infer_provider, _strip_json_fences, _extract_entry, build_batch_entry,
)


class TestInferProvider:
    def test_gemini(self):
        assert infer_provider("gemini-3.1-pro-preview") == "gemini"

    def test_openai_gpt(self):
        assert infer_provider("gpt-5.4") == "openai"

    def test_openai_o_series(self):
        assert infer_provider("o3-mini") == "openai"
        assert infer_provider("o4-mini") == "openai"

    def test_anthropic(self):
        assert infer_provider("claude-opus-4-6") == "anthropic"
        assert infer_provider("claude-sonnet-4-6") == "anthropic"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot infer provider"):
            infer_provider("llama-3")

    def test_case_insensitive(self):
        assert infer_provider("GEMINI-3.1-pro") == "gemini"
        assert infer_provider("Claude-Opus-4-6") == "anthropic"


class TestStripJsonFences:
    def test_no_fences(self):
        assert _strip_json_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_json_fences(text) == '{"key": "value"}'

    def test_bare_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_json_fences(text) == '{"key": "value"}'

    def test_whitespace_preserved_inside(self):
        text = '```json\n{\n  "key": "value"\n}\n```'
        assert '"key": "value"' in _strip_json_fences(text)


class TestBuildBatchEntry:
    def test_basic_entry(self):
        entry = build_batch_entry("test_key", "test prompt")
        assert entry["key"] == "test_key"
        assert entry["request"]["contents"][0]["parts"][0]["text"] == "test prompt"

    def test_json_mode_default(self):
        entry = build_batch_entry("k", "p")
        gen_cfg = entry["request"]["generation_config"]
        assert gen_cfg["response_mime_type"] == "application/json"

    def test_json_mode_false(self):
        entry = build_batch_entry("k", "p", json_mode=False)
        gen_cfg = entry["request"]["generation_config"]
        assert "response_mime_type" not in gen_cfg


class TestExtractEntry:
    def test_round_trip(self):
        entry = build_batch_entry("my_key", "my prompt", json_mode=True, max_tokens=1000)
        key, prompt, json_mode, max_tokens = _extract_entry(entry)
        assert key == "my_key"
        assert prompt == "my prompt"
        assert json_mode is True
        assert max_tokens == 1000
```

- [ ] **Step 6: Run client tests**

Run: `pytest tests/test_client.py -v`
Expected: All PASS

- [ ] **Step 7: Write tests/test_detect_parse.py**

```python
"""Tests for detection result parsing."""
import pytest
import json
from annotator.core.detect import parse_detection_results


class TestParseDetectionResults:
    def test_valid_json(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({
                    "detections": [{
                        "turn_start": 1,
                        "turn_end": 5,
                        "annotation_type": "scaffolding",
                        "brief_description": "test",
                        "suggested_cut_turn": 1,
                    }]
                }),
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 1
        assert result["conv1"]["detections"][0]["turn_start"] == 1

    def test_invalid_json(self):
        raw = {
            "conv1__scaffolding": {
                "text": "not json at all",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 0

    def test_error_entry(self):
        raw = {
            "conv1__scaffolding": {
                "error": "API error",
                "text": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 0

    def test_missing_suggested_cut_turn_defaults(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({
                    "detections": [{
                        "turn_start": 5,
                        "turn_end": 10,
                        "annotation_type": "scaffolding",
                        "brief_description": "test",
                    }]
                }),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        }
        result = parse_detection_results(raw)
        det = result["conv1"]["detections"][0]
        assert det["suggested_cut_turn"] == 4  # max(1, turn_start - 1)

    def test_usage_accumulates(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
            "conv1__rapport": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 200, "output_tokens": 75, "total_tokens": 275},
            },
        }
        result = parse_detection_results(raw)
        assert result["conv1"]["usage"]["input_tokens"] == 300
        assert result["conv1"]["usage"]["output_tokens"] == 125
```

- [ ] **Step 8: Run detect parse tests**

Run: `pytest tests/test_detect_parse.py -v`
Expected: All PASS

- [ ] **Step 9: Write tests/test_eval_metrics.py**

```python
"""Tests for eval metric functions."""
import pytest
from annotator.eval.eval import (
    cohens_kappa, compute_consensus_label, map_to_binary,
    EFFECTIVENESS_LABELS, BINARY_LABELS,
)


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = ["effective", "partial", "ineffective"]
        b = ["effective", "partial", "ineffective"]
        assert cohens_kappa(a, b, EFFECTIVENESS_LABELS) == 1.0

    def test_empty_lists(self):
        assert cohens_kappa([], [], EFFECTIVENESS_LABELS) == 0.0

    def test_complete_disagreement(self):
        a = ["effective", "effective", "effective"]
        b = ["ineffective", "ineffective", "ineffective"]
        kappa = cohens_kappa(a, b, EFFECTIVENESS_LABELS)
        assert kappa < 0  # worse than chance

    def test_binary_perfect(self):
        a = ["right", "wrong", "right"]
        b = ["right", "wrong", "right"]
        assert cohens_kappa(a, b, BINARY_LABELS) == 1.0


class TestComputeConsensusLabel:
    def test_majority_vote(self):
        assert compute_consensus_label(["effective", "effective", "partial"]) == "effective"

    def test_tie_uses_median(self):
        # Two effective, two ineffective -> median of [0,0,2,2] = 1 -> partial
        labels = ["effective", "effective", "ineffective", "ineffective"]
        assert compute_consensus_label(labels) == "partial"

    def test_empty(self):
        assert compute_consensus_label([]) == "unclear"

    def test_single_label(self):
        assert compute_consensus_label(["partial"]) == "partial"


class TestMapToBinary:
    def test_effective(self):
        assert map_to_binary("effective") == "right"

    def test_partial(self):
        assert map_to_binary("partial") == "wrong"

    def test_ineffective(self):
        assert map_to_binary("ineffective") == "wrong"

    def test_unknown(self):
        assert map_to_binary("unclear") is None
```

- [ ] **Step 10: Run eval metrics tests**

Run: `pytest tests/test_eval_metrics.py -v`
Expected: All PASS

- [ ] **Step 11: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (including the previously-failing `test_load_all_ground_truth`)

- [ ] **Step 12: Commit**

```bash
git add tests/
git commit -m "test: add unit tests for config, utils, client, detect parsing, eval metrics"
```

---

### Task 14: Final verification

- [ ] **Step 1: Run full test suite one more time**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Smoke-test CLI help**

Run:
```bash
python -m annotator --help
python -m benchmark --help
python -m annotator.core.detect --help
python -m annotator.eval.eval --help
```
Expected: All print help text without errors

- [ ] **Step 3: Verify config loads correctly**

Run:
```bash
python -c "from annotator.core.config import get_phase_config, get_iou_threshold, get_batch_timeout, resolve_run_params; print('iou:', get_iou_threshold()); print('timeout:', get_batch_timeout()); print(resolve_run_params(None, None, None, None))"
```
Expected: Prints iou: 0.3, timeout: 86400, and auto-generated version params
