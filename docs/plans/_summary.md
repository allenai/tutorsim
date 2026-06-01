# Plans Summary

Index of planned work and change log for the project. Plans live in this directory as `YYYY-MM-DD-<slug>.md`; specs in `specs/`.

## Plans

### 2026-06-01 — [Benchmark student modes](specs/2026-06-01-benchmark-student-modes-design.md)

**Goal**: Replace the benchmark's single hardcoded synthetic student prompt with selectable student "modes" ported from Alexis's synth-students repo, making `imitate_example` (her strongest realism mode) the shipping default.
**Status**: Shipped.
**Result**: Added `prompts/benchmark/v2/` with `tutor_system.txt` (copy of v1) and `students/{imitate_example,simple,expert,paraphrase_with_example}.txt`. `benchmark.student.mode` config field selects which student prompt to load; null falls back to legacy `student_system.txt` so v1 still works. `_build_role_prompt` in `benchmark/core/exchange.py` dispatches on `student_mode`; `run.py` threads it through both sync and batch call sites and records it in `resolved_models`. Default config flips to `prompt_version: v2` + `student.mode: imitate_example`. Trait-based modes (`trait_*`) deferred -- they need a separate trait-generator phase. 6 new tests pass, full 175-test suite green.

### 2026-05-22 — [Labeller headroom check](2026-05-22-labeller-headroom.md)

**Goal**: Decide whether the per-type hybrid labeller (test_v2 kappa 0.782) has real headroom without commissioning new human ratings.
**Status**: Done -- at ceiling, no change to canonical hybrid.
**Result**: Three independent angles all confirm the per-type hybrid (v2 sc + v6 ra) sits at the practical ceiling on this data. (1) Error analysis on the 21 test_v2 hybrid errors found 81% are human-disagreement (16 A) or junk-annotation (1 C); only 4 errors are prompt-fixable. (2) Oracle ceiling over v2/v4/v5/v6 is 0.833 but unreachable -- the 4 Claude-designed prompts agree on 91.2% of items, so naive majority vote regresses to 0.750 and no simple router captures the +0.051 headroom. (3) v7 (Gemini-designed) underperformed both incumbents on train_v2 (sc 0.694 vs v2 0.778, ra 0.772 vs v6 0.808); v8 (gpt-5.4) blocked on OpenAI quota and not generated. v7's cross-architecture loss is evidence that the classifier (Claude Opus 4.6) -- not the prompt -- is the bottleneck. To move further: new human ratings with cross-reviewer overlap, a different classifier model held-constant on prompt, or upstream annotation-input fixes.

### 2026-05-20 — [Labeller validation](2026-05-20-labeller-validation.md)

**Goal**: The EC2 validation app has been collecting human ratings on SAR annotations (~490 done ratings from 4 reviewers). This is ground truth for the labeller itself. Use it to measure the current labeller honestly, then iterate.
**Status**: Shipped. Per-type routed labeller is the new active configuration.
**Result**: v2 split (343 train / 147 test, stratified). Tried four prompt variants -- rule refinements (regressed), 4 hand-picked examples (kappa 0.741), full-data meta-prompt with my priming (0.739), full-data meta-prompt with no priming (0.741). All single-prompt routes plateau at ~0.74 due to inter-rater inconsistency on scaffolding's "worked + improvement note" pattern. Per-type routing (v2 for scaffolding, v6 for rapport) lifts test kappa **0.725 -> 0.782 with no binary regression** (still 0.796). Rapport kappa lifts +0.111 (0.664 -> 0.775); scaffolding kappa unchanged at 0.783. Shipped as a dict-valued `annotator.labeller` config; same routing wired into `build_ground_truth.py --labeller hybrid`.

### 2026-03-26 — [Factor IV storage refactor](2026-03-26-factor-iv-storage-refactor.md)

**Goal**: The same pipeline needs to run on a laptop against local files and in production against S3. Before this, `if backend == "s3"` branching was smeared through the storage layer and paths were hardcoded — swapping environments required code edits.
**Status**: Implemented. S3 access is unblocked as of 2026-04-28 — `aws sts get-caller-identity` works locally via the boto3 default credential chain, and `s3://kylel-alexisr-edu/` is reachable. Earlier "blocked on IT credentials" status no longer applies.
**Result**: `StorageBackend` ABC with `LocalBackend` and `S3Backend` implementations, singleton delegate, zero branching. Every path (transcripts, ground truth, results) is overridable via env var. `STORAGE_BACKEND=local|s3` flips environments without touching code or config files. 13 tests (11 local, 2 with moto) passing.

### 2026-04-17 — [Codebase cleanup & CLI simplification](2026-04-17-codebase-cleanup.md)

**Goal**: Day-to-day pipeline runs required too many flags and tripped over verified bugs and hardcoded constants. Reduce friction so common operations (run a pass, label gold, re-run with a style) are short and correct by default.
**Status**: Implemented.
**Result**: Bugs fixed, hardcoded constants moved into `config.yaml`, and an `annotator:` config section defaults common CLI flags (version auto-generates when omitted). No new abstractions or files.

### 2026-04-17 — [Labeller V2](2026-04-17-labeller-v2.md) · [spec](specs/2026-04-17-labeller-v2-design.md)

**Goal**: Ground-truth labels and pipeline labels were not on the same scale — four divergent labeller prompts with different criteria and different inputs. The v1 labeller also over-applied "partial" (~32.6%) to anything with hedged language, masking real human disagreement and making kappa comparisons misleading.
**Status**: Implemented.
**Result**: Single shared `classify_v2.txt` prompt used by both `build_ground_truth.py` and the pipeline labeller, with outcome-anchored criteria and all four annotation fields fed in. Ground truth versioned at `data/ground_truth_v2/`. 13.1% of ground-truth labels shifted (mostly partial → effective/ineffective as intended); spot-check 94% correct. Balanced human ceiling dropped 0.5049 → 0.2310, exposing the real disagreement v1 was hiding.

### 2026-04-17 — [Production readiness fixes](2026-04-17-production-readiness.md)

**Goal**: Before a public research release, the repo needs to be usable by someone new to the project — without tripping on missing env docs, dead code, silent config mismatches, or untested edges.
**Status**: Partially implemented — see the plan file for per-task checkboxes.
**Result**: `.env.example` corrected (`GEMINI_API_KEY` vs. `GOOGLE_API_KEY`), `iou_threshold` and `batch_timeout` moved into `config.yaml`, bottom-up task order (config → shared abstractions → consumers → tests). Remaining tasks are individually committable.

### 2026-04-24 — Shared logging module

**Goal**: The repo had 357 `print()` calls and zero `logging` infrastructure — long batch runs left no structured artifact, and there was no way to dial verbosity without editing source. Also no obvious home for cross-cutting code (annotator/core/ had been doing double duty as the shared library).
**Status**: Implemented (infra + thin migration slice). 17 prints in `annotator/core/annotate.py` migrated to logger calls; the remaining ~340 prints across `annotator/`, `benchmark/`, `data/`, `validation/` migrate incrementally in follow-up PRs.
**Result**: New top-level `common/` package as the documented home for shared infrastructure. `common.logging_setup.setup_logging()` is idempotent, env-var driven (`LOG_LEVEL`, `LOG_FILE`), and writes per-run logs to `logs/{version}/run.log` for reproducibility. Two-phase init: console handler at process start, file handler attaches once `version` is resolved. Wired into both `annotator` and `benchmark` runners. 11 unit tests covering idempotency, two-phase init, file-handler gating, and level resolution.

**Follow-ups**: migrate remaining prints incrementally; add `--log-level` CLI flag if useful; upload `run.log` to S3 results dir at end-of-run when `STORAGE_BACKEND=s3`.

### 2026-04-28 — [Benchmark screenshot ingestion](2026-04-28-benchmark-screenshots.md)

**Goal**: The annotator side supported `--with-screenshots` (delivered 2026-04-24) but the benchmark didn't thread it through any phase. AI tutors were being graded text-only while the same human-tutor moments could be graded with full visual context — apples-to-oranges. Wire screenshots into Step 0 (detection), Phase 1 (tutor + synthetic student exchange), and Phase 2 (annotation) so the benchmark measures what the annotator was upgraded to measure.
**Status**: Implemented (code + tests + smoke), validation blocked on data.
**Result**: `build_analysis_entries` and `build_detection_entries` accept optional `screenshots_by_conv` so callers can inject pre-loaded screenshots keyed by any id (decouples lookup from use, fixing the conv_id -> scenario_id remap blocker). Bridge loads per-scenario screenshots using `scenario.conv_id` and passes them keyed by `scenario_id`. Phase 1 exchange attaches images filtered to `anchor_turn <= cut_turn` to both tutor and student in sync + batch modes. Vision validation runs once at run start. 6 new tests covering all three integration points. **End-to-end validation blocked**: S3's `deidentified/screenshots/` has 3 conv UUIDs that have no matching transcripts anywhere accessible — every smoke run currently degrades to text-only because the loader honestly returns `[]` for every conv. Code is correct and unit-tested; real-data verification awaits the deidentification pipeline producing transcripts paired with screenshot UUIDs.

### 2026-04-28 — [Benchmark production readiness](2026-04-28-benchmark-production-readiness.md)

**Goal**: The benchmark pipeline lags behind the annotator on operational basics — Phase 2 has no resume, no in-flight batch sidecar, prints instead of logger, dead `annotate_exchange` code, vestigial `"annotator_profiles"` config branch, and auto-generated versions that flip across midnight. A long batch run can lose hours of work to a single ctrl-C. Bring benchmark up to the same floor as the annotator pipeline before attempting the first full run.
**Status**: Implemented and smoke-verified.
**Result**: Phase 2 resumable via shard pre-filter + per-(profile, style) in-flight sidecar (mirrors `annotator/core/annotate.py:313-402`). Stable version pointer at `_active_runs/{profile}.json` survives midnight resumes. ~35 `print()` calls migrated to structured logger; ~67 LOC of dead `annotate_exchange` and vestigial config branches removed. Smoke test verified resume in three modes: full cache hit (7s, no API), partial recovery (delete one shard, only that one re-runs), first run (18 min for 2 scenarios sync). Plus a latent config-mutation bug fixed in `get_benchmark_config` (added `deepcopy`). 5 new tests covering sidecar + bridge wiring.

### 2026-04-24 — [Screenshot enrichment](2026-04-24-screenshot-enrichment.md) · [spec](specs/2026-04-24-screenshot-enrichment-design.md)

**Goal**: Detection and annotation judge pedagogy from text alone, but transcripts often contain bare `[SCREEN UPDATE]` placeholders or narration-only enrichment turns. When a tutor says "look at this" or a student reacts to something visual, the pipeline is missing real context. Screenshots already exist on S3; wire them into the prompt path for the annotator pipeline only (benchmark stages stay text-only this iteration).
**Status**: Implemented (PR #7, opt-in via `--with-screenshots`); not yet evaluated at scale.
**Result**: New `annotator/core/screenshots.py` anchors each image to the latest transcript turn whose `start_seconds <= image timestamp`. `ModelClient.generate` and `build_batch_entry` accept `images=[storage_path, ...]` and resolve per-provider (base64 inline for local/Gemini, presigned URL for S3 + Anthropic/OpenAI). `[SCREEN @ turn N: image K]` markers in the rendered transcript drive content interleaving so each image lands next to its anchor turn instead of at the end. Default off everywhere -- existing runs are byte-for-byte unchanged. Anthropic prompt caching is enabled by default for annotation runs to amortize images that repeat across overlapping excerpt windows.

## Change log

Reverse chronological. Stuff that shipped but didn't have a dedicated plan file, or non-obvious deltas worth recording.

### 2026-04-27: Per-conv resume + finish-the-print-migration

**Problem**: Long batch runs had no recovery point — a Ctrl-C or crash threw away every result that had already streamed back from the provider, and ~50 `print()` calls across `annotator/{core,run.py}` were still bypassing the shared logging infra, so structured `logs/{version}/run.log` artifacts only captured a fraction of what actually happened during a run.

**Changes (resume)**:
- New `save_annotator_shard` / `list_annotator_shard_ids` / `load_annotator_shards` in `annotator/core/storage.py`. Shards live at `results/annotator/{version}/shards/{basename}/{conv_id}.json`. `basename` is the resolved output filename without `.json` (e.g. `detections`, `annotations_generous`, `annotations_gold`), so each profile/source variant gets its own namespace.
- `run_detect` and `run_annotate` now pre-filter conv_ids that already have a shard for the same `(version, basename)` and only send the remainder to the model. Per-conv shards are written as soon as `parse_*_results` produces them; the monolithic `detections.json` / `annotations*.json` is then assembled from the union of all shards on disk.
- `parse_detection_results` records `images_attached` (sum across targets, == API attachments) alongside `images_seen` (per-conv max, == unique images). The aggregate `total_images_sent` is computed from `images_attached` so the metric holds across multi-run aggregation.
- Caveat: in batch mode, Ctrl-C *during polling* abandons the in-flight provider batch — that compute is lost, but no on-disk state is corrupted. A re-run starts a fresh, smaller batch with only the un-sharded conv_ids.

**Changes (logging)**:
- `client.py`, `storage.py`, `label.py`, `config.py`, `detect.py`, `run.py` now declare a module-level `logger = logging.getLogger(__name__)`. All non-docstring `print()` calls are migrated.
- Per-entry chatter (sync `[N/total]` progress, batch poll status) drops to `logger.debug` — gone from default INFO terminals, opt-in via `LOG_LEVEL=DEBUG`. Lifecycle events (uploaded, batch created, finished, downloading) stay at `logger.info`. Retries are `warning`; terminal failures are `error`.
- `run.py` banner separators (`'=' * 60` between passes) collapse to single `logger.info("=== PASS N: ... ===")` lines — the file handler's timestamp + module already provide structure.
- The two `print()` calls in `client.py`'s module docstring stay; they're code examples, not runtime output.

**Coverage**: 4 new shard-helper tests in `test_storage.py`, 1 new test in `test_detect_parse.py` for the `images_seen` / `images_attached` distinction. All 149 tests pass.

### 2026-04-27: Logger format string -- em-dash to pipe

**Problem**: `common/logging_setup.py` `_FORMAT` used a U+2014 em-dash separator (`name — message`). Project memory explicitly flags this: Windows cp1252 stderr can mojibake or raise `UnicodeEncodeError`, and the file handler/stream handler don't necessarily share encoding. Pre-existing in the shared-logging plan; flagged after a code review of the screenshot-enrichment PR routed more traffic through the format.

**Change**: One-character format-string update -- `—` to `|`. File-handler output (utf-8) is unaffected; stderr handler output now stays inside the cp1252 subset on Windows. Per the project's "log format = public contract" rule, this is recorded here so anyone parsing log lines knows the separator changed.

### 2026-04-27: Real ctrl-C resume -- per-conv sync checkpoint + batch-ID persistence

**Problem**: The first cut of "ctrl-C resume" only worked across *completed* runs -- shards were only written after `run_sync_entries` / `run_batch` returned, so a ctrl-C in the middle of either lost everything for that run. The original ask was "if I ctrl-C and start again, it'll start where it left off," which the previous implementation did not actually deliver in practice.

**Changes (sync mode)**:
- `run_detect` and `run_annotate` now group entries by `conv_id` and call `run_sync_entries` per conv, sharding after each conv's entries return. A ctrl-C between convs leaves a clean per-conv checkpoint on disk; the next run skips already-sharded convs and picks up at the next un-sharded one.

**Changes (batch mode)**:
- New `save_inflight_batch` / `load_inflight_batch` / `clear_inflight_batch` helpers in `storage.py`. Sidecar lives at `results/annotator/{version}/in_flight/{basename}.json` (subdir keeps it out of `list_annotator_result_files`).
- `run_batch` and the three provider runners (Gemini, OpenAI, Anthropic) accept `existing_batch_id` (skip submission, retrieve and continue polling) and `on_batch_created` (called with the new batch id immediately after submission so the orchestrator can persist it before the poll loop starts).
- `run_detect` / `run_annotate` check the sidecar before submitting. They hash the current entries' keys against `entry_keys_hash` recorded at submission. Match -> resume polling on the saved batch id, no double-submit. Mismatch (user changed conv set between runs) -> error loudly with instructions to delete the sidecar.
- After successful parse + shard write, the sidecar is cleared.
- Anthropic-specific: `id_to_key` short-id mapping is rebuilt deterministically from `entries` order on resume; the entry-keys hash check guarantees the input is the same, so we don't need to persist the mapping itself.

**Coverage**: 3 sidecar lifecycle tests in `test_storage.py`. End-to-end ctrl-C smoke run in sync mode against `claude-opus-4-6` (test=2 convs, ctrl-C between convs, re-run completes from conv 2). All 152 tests pass.

**Caveat**: changing the conv set between runs (adding/removing transcripts, switching `--with-screenshots`) shifts entry indices and produces an `entry_keys_hash` mismatch -- the resume aborts loudly rather than silently mis-mapping results. The fix is documented at the error site.

### 2026-04-27: Annotator per-pass CLIs now call `setup_logging()`

**Problem**: After migrating prints in `detect.py` / `annotate.py` / `label.py` to `logger.info(...)`, INFO records emitted by these modules vanished when invoked directly via `python -m annotator.core.detect` (and the equivalents). Python doesn't run `annotator/__main__.py` for dotted module targets, so the only `setup_logging()` call upstream of those CLIs (`__main__.py`) never executed -- the root logger had no handlers and INFO propagated to the stdlib `lastResort` handler at WARNING-level.

**Change**: `setup_logging(version=version)` is now called in each per-pass `main()` immediately after `resolve_run_params`. Append-mode file handler means a per-pass invocation appends to the same `logs/{version}/run.log` the unified runner uses, so a workflow of `python -m annotator.core.detect` followed by `python -m annotator.core.annotate` produces a single coherent log file. The review-patterns skill checklist was updated to reflect this expanded entry-point list.

### 2026-04-17: Labeller V2 — Unified Prompt + Outcome-Anchored Criteria

**Problem**: Found 4 divergent labeller prompts with different criteria and different inputs. Ground truth script (`build_ground_truth.py`) only passed result text; pipeline labeller (`classify.txt`) passed situation+action+result. The v1 labeller overused "partial" for anything with hedged language (~690/2115 = 32.6%), masking real disagreement.

**Changes**:
- New `classify_v2.txt` prompt: outcome-anchored criteria, all 4 fields (annotation_type, situation, action, result), explicit guidance that situation context is not evidence, specific partial signals
- `build_ground_truth.py`, `extract_ground_truth.py`, `label.py` all load the shared prompt
- Ground truth versioned: `data/ground_truth_v1/` (baseline), `data/ground_truth_v2/` (v2 labeller)

**Ground truth label shifts (v1 -> v2)**:
- Unchanged: 1838/2115 (86.9%)
- Changed: 277/2115 (13.1%)
- partial -> ineffective: 162, partial -> effective: 83 (v1 was inflating partial)
- ineffective -> partial: 10, effective -> partial: 9 (the known misclassifications, mostly fixed)

**Eval comparison (3-way kappa, v1 labeller -> v2 labeller)**:

| Style | v1 Kappa | v1 Ceiling | v2 Kappa | v2 Ceiling | Notes |
|-------|----------|------------|----------|------------|-------|
| Balanced | 0.5364 | 0.5049 | 0.4574 | 0.2310 | Ceiling dropped — v1 was masking human disagreement |
| Generous | 0.4061 | 0.3350 | 0.3740 | 0.4081 | Kappa slightly down, ceiling up |
| Demanding | 0.6283 | -- | 0.4810 | -- | Dropped, small n=172 |

**Spot-check validation** (50 random transitions + 13 targeted by-annotator samples):
- partial -> ineffective: 27/30 correct (90%). The 3 errors are genuine edge cases (e.g., "not effective for tutor but effective for student" — explicitly mixed dimensions).
- partial -> effective: 20/20 correct (100%). Every case was a fundamentally positive assessment with minor improvement suggestions.
- Combined accuracy: **94% (47/50)**. The v1 labeller was inflating partial by treating courtesy hedging ("but the tutor could also...") and improvement suggestions ("to make this more effective...") as mixed signals.
- Per-annotator check: Forbes accounts for 141/277 changes (29.9% change rate) because she writes detailed improvement suggestions in every assessment. V2 correctly reads these as "positive with suggestions" or "negative with intent acknowledged" rather than "mixed." Padgett (59 changes), Gerber (36), Mann (16) all checked — same pattern confirmed.
- AI annotation labelling: 0 verdict-label mismatches on 583 AI annotations with explicit verdicts. V2 is consistent on AI text.

**Interpretation**: The v2 labeller is more polarized — it resolves ambiguous assessments to effective/ineffective instead of defaulting to partial. This reveals real disagreement that v1 was hiding behind inflated partial counts. The balanced human ceiling dropped from 0.5049 to 0.2310 because different annotators' hedged narratives now resolve to different poles instead of all landing on partial. The AI kappa dropped correspondingly because it's being measured against a more discriminating ground truth.

**Spec**: [specs/2026-04-17-labeller-v2-design.md](specs/2026-04-17-labeller-v2-design.md)

### 2026-04-16: Qualitative Review + Prompt Fix + Full Regeneration

**Qualitative review** of "Human vs AI annotation comparison" PDF (external expert comparison of human vs AI annotations). The PDF compared against v3_gemini results. Key findings:

1. **AI never evaluated whether timing was appropriate for rapport** -- 0% of v3_gemini situation fields flagged bad timing, vs human annotators doing this routinely
2. **AI judged rapport by strategy quality, not student engagement** -- v3 prompt literally said "focus on the quality of the tutor's strategy"
3. **Human label-narrative inconsistency** -- humans sometimes labeled "effective" but wrote narratives describing partial effectiveness. Strengthens case for AI annotation.

**Prompt changes** (applied to all 3 profiles + v4 base):
- Added "poorly timed rapport" example showing tutor interrupting focused student mid-problem. Based on real human annotator patterns (Padgett, Mann, Flick, Forbes). Calibrated per profile style.
- Strengthened situation field instruction to explicitly require timing evaluation before describing context.

**Results after prompt fix** (balanced profile, anthropic):
- Situation mentions timing: 25.6% -> 89.6%
- Situation flags BAD timing: 0.0% -> 1.4%
- Rapport ineffective rate: 16.8% -> 17.9% (slight increase, as intended)

**Cleanup**: Deleted stale `v5/p2/` prompts (duplicated v4 base, never iterated). Archived duplicate result directories. Fixed version naming. All `v5_gold/` results regenerated with updated prompts.

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

S3 access verified working 2026-04-28 (boto3 default credential chain + bucket `kylel-alexisr-edu`). The earlier "blocked on IT credentials" line was stale.

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

## Earlier work

### ARW-62: Per-profile detection iteration — no improvement

Tested per-style detection prompts. 2 rounds, 12 advisor calls. Detection is model-limited. Details: [../profile_detection_iteration.md](../profile_detection_iteration.md)

### ARW-63: Cut after key moments — superseded

Originally changed benchmark to cut after key moments using ground truth. Now superseded by the synthetic detection approach — benchmark cuts at `suggested_cut_turn` from v5 detection, no ground truth involved.

### ARW-64: --style flag — done

Added `--style` CLI flag across the annotator pipeline for per-profile runs.
