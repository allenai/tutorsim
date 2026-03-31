# Project Status

*Last updated: 2026-03-31*

## Current State

### Repo Structure

- `annotator/` -- 3-pass annotation pipeline (detect, annotate, label)
- `benchmark/` -- tutor evaluation pipeline (detect, scenarios, exchange, annotate, score)
- `prompts/` -- all prompt templates (annotator v4/v5, profiles, labeller, benchmark)
- `config.yaml` -- unified config (model profiles, benchmark settings, storage)
- `data/` -- private student data (gitignored)
- `results/` -- all pipeline outputs (gitignored)
- `history/` -- archived iteration data (gitignored)
- `tests/` -- storage layer tests (pytest + moto)

### Annotator Pipeline -- Complete

The 3-pass annotation pipeline is stable and validated:

| Version | Model | Detection | Annotation | Eval | Notes |
|---------|-------|-----------|------------|------|-------|
| v4 | Gemini | Yes | Yes | Yes | Previous canonical detection prompts |
| v5 | Claude | Yes | -- | -- | Iterated detection (51.5% recall, 32.3% precision) + cut point guidance for benchmark |
| v3_gemini | Gemini | Yes | Yes | Yes | Previous best |
| v3_claude | Claude | Yes | Yes | Yes | Best recall (64.9%) |

**Per-archetype annotator profiles** (final, all exceed human ceiling):
- Generous (5 annotators, n=297): 3-way kappa 0.4061 (ceiling 0.3350)
- Balanced (3 annotators, n=510): 3-way kappa 0.5364 (ceiling 0.5049)
- Demanding (1 annotator, n=79): 3-way kappa 0.6283

### Benchmark Pipeline -- Redesigned, First Full Run In Progress

The benchmark is now fully ground-truth-free. It uses synthetic detection to find key moments and cut points, then evaluates how an AI tutor handles each moment.

**Pipeline flow:**
1. v5 detection on all transcripts (finds key moments + `suggested_cut_turn`)
2. Each detection becomes a scenario (cut at suggested_cut_turn)
3. Synthetic tutor + student exchange (20 rounds)
4. 3-style annotation (generous/balanced/demanding profiles)
5. Per-style scoring (no composite aggregation -- user picks their perspective)

**First run** (`claude-opus-4-6_2026-03-26`): detection complete (2,608 detections from old v5 prompts, avg 25.1/conv). Exchange phase stalled (0 completed rounds). Needs re-run with updated v5 prompts.

**Results naming**: `{tutor_model}_{date}` with `config.json` capturing all run conditions (resolved model names, prompt versions, turn counts).

### Key Technical Decisions

- **Detection ceiling is model-limited**, not prompt-limited. v4 detection prompts are final.
- **v5 prompts = v4 detection verbatim + cut point guidance appended.** Cut point sections don't affect detection quality -- they add boundary and cut point instructions after the v4 content.
- **Annotation profiles use per-archetype prompts** (`profiles/{generous,balanced,demanding}/p2/`).
- **Benchmark is ground-truth-agnostic.** It only reads transcripts. Detection, annotation, and scoring are all synthetic.
- **Scores are per-style, not composite.** No weighted aggregation across styles. Each annotator perspective is a separate result.
- **Storage layer is Factor IV compliant.** Backend ABC pattern, env var overrides for all paths. `STORAGE_BACKEND=s3` for production, `local` for development.

---

## Completed Work

### 2026-03-31: v5 Detection Prompt Iteration

Iterated v5 detection prompts through 4 rounds (3 advisor patches + 1 mandatory rewrite). The original v5 prompts replaced v4's "cast a wide net" with a 3-criteria "key moment" test that cratered recall (64.2% -> 22.8%).

**Results progression:**

| Version | Recall | Precision | IoU | Avg/conv |
|---------|--------|-----------|-----|----------|
| v4 (baseline) | 64.2% | 23.4% | 0.616 | 16.6 |
| v5 (pre-iteration) | 22.8% | 24.6% | 0.588 | 5.3 |
| v5r1 | 39.6% | 39.1% | 0.676 | 5.6 |
| v5r3 (limit=30) | 45.9% | 35.3% | 0.660 | 7.5 |
| **v5r4 (winner)** | **51.5%** | **32.3%** | **0.657** | **9.2** |

**Key findings:**
- The 3-criteria test was the structural recall bottleneck. Replacing it with "cast a wide net" + count targets recovered the most recall.
- Advisor `--limit 30` (vs default 10) surfaced patterns that 10 examples missed: academic scaffolding as rapport (30% of rapport misses), rigor-push on correct answers (25% of scaffolding misses).
- v5r4 keeps all v5 improvements (definitions, boundary guidance, cut points, false positive list, consolidation rules) while restoring v4-style broad detection.
- Detection count reduced from 25/conv (original v5) to 9.2/conv (v5r4) -- 63% reduction while maintaining 51.5% recall.

Full iteration log: `history/v5_detection_iteration/iteration_log.md`

### 2026-03-26: Storage Layer (Factor IV)

Refactored `annotator/core/storage.py` from `if backend == "s3"` branching to a `StorageBackend` ABC with `LocalBackend` and `S3Backend` implementations. All public functions delegate to a singleton backend instance -- zero branching.

Key changes:
- **Backend protocol**: `StorageBackend` ABC with `read_json`, `write_json`, `list_files`, `exists`, `get_local_path`
- **Env var overrides**: `STORAGE_BACKEND`, `STORAGE_ROOT`, `S3_BUCKET`, `S3_PREFIX`, `STORAGE_TRANSCRIPTS`, `STORAGE_GROUND_TRUTH`, `STORAGE_ANNOTATOR_RESULTS`, `STORAGE_BENCHMARK_RESULTS`
- **Single config**: collapsed 3 sections (local/s3/paths) into 1 `paths:` section
- **Tests**: 13 tests (11 local, 2 S3 with moto), all passing
- **dotenv**: storage.py loads `.env` so env vars work without explicit export
- `.env.example` created documenting all env vars

Blocked on S3 testing: need AWS access keys from IT admin + cross-account bucket policy from Ai2.

### 2026-03-26: Benchmark Decoupled from Ground Truth

The benchmark pipeline no longer touches ground truth data. Previously it used human-annotated key moments as scenarios. Now:

- **Step 0**: Runs v5 detection on all transcripts (synthetic key moment detection)
- **Step 1**: Each detection becomes a scenario, cut at `suggested_cut_turn`
- `Scenario.detection` replaces `Scenario.ground_truth_moment`
- `annotator_bridge.py` uses detection turn ranges, not ground truth
- `benchmark/eval/eval.py` and `view.py` removed all ground truth references

### 2026-03-26: Per-Style Scoring (No Composite)

Removed weighted composite score aggregation. Each annotator style (generous/balanced/demanding) produces its own independent score file:
- `scores/anthropic_generous.json`
- `scores/anthropic_balanced.json`
- `scores/anthropic_demanding.json`

Each contains mean score, per-type breakdown (scaffolding/rapport), and per-scenario labels. No style weights, no blending. The user picks which perspective they care about.

### 2026-03-26: Run Traceability

`config.json` in each benchmark run now records resolved model names:
```json
{
  "resolved_models": {
    "tutor_anthropic": "claude-opus-4-6",
    "student": "claude-opus-4-6",
    "annotator": "claude-opus-4-6",
    "labeler": "claude-opus-4-6",
    "detector": "claude-opus-4-6"
  }
}
```

### 2026-03-26: v5 Prompt Design

v5 detection prompts = v4 detection content preserved verbatim + cut point sections appended:
- "How to Define Moment Boundaries" -- guidance on turn_start/turn_end scoping
- "How to Choose a Cut Point" -- three criteria (context, genuine decision, no preview) + examples
- `suggested_cut_turn` field in output JSON

Important: the v4 detection content (Research Context, What to Look For, Cast a Wide Net) is untouched. Adding the cut point sections after the v4 content does not affect detection quality.

### 2026-03-26: Cleanup

- Deleted stale `annotator/eval/histogram_compare.py`, `annotator/eval/view_compare.py`, `benchmark/README.md`
- Archived old benchmark results (v1, v2, smoke tests) to `history/`
- Data leak inventory created (`data/data_in_git_history.txt`): 117 transcript IDs + 10 annotator names in git history
- `.pytest_cache` added to `.gitignore`

---

## Previous Completed Work

### ARW-62: Per-profile detection iteration -- DONE (no improvement)

Tested per-style detection prompts. 2 rounds, 12 advisor calls. Detection is model-limited. Details: [profile_detection_iteration.md](profile_detection_iteration.md)

### ARW-63: Cut after key moments -- DONE (superseded)

Originally changed benchmark to cut after key moments using ground truth. Now superseded by the synthetic detection approach -- benchmark cuts at `suggested_cut_turn` from v5 detection, no ground truth involved.

### ARW-64: --style flag -- DONE

Added `--style` CLI flag across the annotator pipeline for per-profile runs.

---

## Token Usage Tracking

All results contain per-call token usage (`input_tokens`, `output_tokens`, `total_tokens`). Results are gitignored and exist only on disk -- see [lessons_learned.md](lessons_learned.md) for why this matters.
