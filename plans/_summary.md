# Data Investigation Summary

Overview of investigations into video-transcript alignment quality for the Step Up tutoring dataset (109 sessions, ~45-65 min each).

## Bottom Line

**Gold transcripts are correct and can be used as-is.** The timestamp "drift" observed during manual QuickTime spot-checks was a VFR video playback artifact, not a real data quality issue. Plans 001-004 were chasing a non-problem. Plan 005 confirmed this across 10 files, and Plan 007 confirmed again using direct same-timepoint window matching (no alignment step).

---

## Plan 001: Alignment Validator

**Plan:** `001_video_transcript_alignment_validator_cli_with_stt_sampling.md`
**Script:** `scripts/data/validate_alignment.py`
**Output:** `output/` (109 per-file markdown reports)

Built a CLI tool that matches video/transcript pairs, samples 5 segments per file, extracts audio at transcript timestamps, runs Whisper STT, and computes WER. Processed all 109 files with parallel workers.

**Result:** 74.7% mean alignment rate, 0.35 mean WER across 109 files. Some files showed low alignment at later timestamps, which initially suggested drift. This triggered the subsequent investigations.

## Plan 002: Drift Detection and Correction

**Plan:** `002_transcript_timestamp_drift_detection_and_correction.md`
**Script:** `scripts/data/fix_alignment.py`

Designed a system to detect drift curves via anchor-based fuzzy matching: extract audio windows around transcript timestamps, run Whisper with word timestamps, use fuzzy text matching to find where transcript text actually occurs. Fit piecewise-linear corrections.

**Result (002a):** `002a_failed_drift_matching_test.md` — Failed. Tested 4 matching algorithms, 2 anchor lengths, 3 window sizes, 2 Whisper models via `scripts/data/test_matching.py`. Best configuration (fuzz_ratio, long anchors, +/-60s window, base model) had mean error 10.5s on file 1 and 16.1s on file 2. Tutoring audio has too many repeated common phrases, causing false matches. The approach cannot reliably recover drift within the +/-3s target.

## Plan 003: Whisper Re-transcription

**Plan:** `003_whisper_retranscription_feasibility_test.md`
**Results:** `003_whisper_retranscription_results.md`
**Script:** `scripts/data/test_whisper_quality.py`
**Output:** `output/whisper_quality/`

Since drift correction failed, tested whether Whisper could replace gold transcripts entirely (correct timestamps by construction).

**Result:** Failed. Whisper `base` achieves ~0.19 WER on tutor speech but ~0.40-0.48 WER on student speech. 33-46% of segments have WER > 0.5. Dominant failure: students speak quietly/far from mic; Whisper hallucinates "You" or fabricates plausible but wrong content. `small` model only improves WER by 0.014 for 2.3x runtime — bottleneck is audio quality, not model capacity. Wholesale replacement not viable.

## Plan 004: Hybrid Alignment (Whisper Timestamps + Gold Text)

**Plan:** `004_hybrid_alignment_whisper_timestamps_gold_text.md`
**Results:** `004_hybrid_alignment_results.md`
**Script:** `scripts/data/test_hybrid_alignment.py`
**Output:** `output/hybrid_alignment/`

Combined best of both: keep gold transcript text (accurate), replace timestamps with Whisper-derived ones (correct by construction). Used Needleman-Wunsch alignment between Whisper words and gold words to transfer timestamps.

**Result:** The NW alignment itself worked well (80%+ coverage), but **Whisper timestamps closely agreed with gold timestamps** — near-zero delta even at points where manual QuickTime checks showed 10-31s of drift. Applying the "correction" made WER worse (0.19 -> 0.45 for file 1). This was the key finding that led to Plan 005.

## Plan 005: AV Desync Investigation + Extended Validation

**Plan:** `005_av_desync_investigation.md`
**Results:** `005_av_desync_results.md`
**Scripts:** `scripts/data/test_av_desync.py`, `scripts/data/whisper_transcribe.py`, `scripts/data/test_timestamp_agreement.py`
**Output:** `output/av_desync/`, `output/timestamp_agreement/`, `data/stepup/transcripts/2_13/Whisper/`

Investigated whether the observed drift was a QuickTime VFR playback artifact rather than real transcript error.

**Part 1 (2 files):** ffprobe confirmed both test files are VFR (variable frame rate) with wildly irregular frame intervals (16ms to 670ms). PTS analysis showed cumulative drift of 34-80s between actual frame timestamps and constant-FPS-assumed time. Audio extracted at gold transcript timestamps matched gold text (4/6 spot-checks). Track durations matched within 0.05s.

**Part 2 (10 files):** Extended to 10 files with full Whisper transcription + NW alignment. 9 of 9 valid files showed gold and Whisper timestamps agree within 2-9s (within Whisper `base` accuracy). 1 file had a Whisper transcription failure (hallucinated "You" tokens due to long silences) — not a drift issue.

**Result:** Confirmed. Gold transcripts are correctly aligned to the audio track. The "drift" was QuickTime displaying the wrong video frame for a given audio position due to VFR.

## Plan 007: Same-Timepoint Window Overlap (No Alignment)

**Plan:** `007_same_timepoint_window_overlap_validation.md`
**Results:** `007_same_timepoint_window_overlap_results.md`
**Script:** `scripts/data/test_same_timepoint_overlap.py`
**Output:** `output/same_timepoint_overlap_window/`

Ran a stricter validation that does **not** use Needleman-Wunsch or any timestamp correction:
- Sample many timepoints per file
- Pull text windows around each same timestamp from gold and Whisper transcripts
- Compare those windows using overlap and keyword-anchor signals

Important keywords were derived from **gold transcript-only per-file IDF**, so rare terms in gold serve as anchors for matching the corresponding Whisper window.

**Result:** 7 PASS, 2 MIXED, 0 FAIL, 1 INSUFFICIENT_DATA across 10 cached files. Manual review of mixed/failure examples showed transcription wording noise, not temporal shift. This provides direct same-timepoint confirmation that gold and Whisper timestamps align.

## Plan 008: Screenshot Diagnostic with Tiny Clips

**Plan:** `008_screenshot_diagnostic_with_tiny_clips.md`
**Results:** `008_screenshot_diagnostic_results.md`
**Script:** `scripts/data/test_screenshot_diagnostic.py`
**Output:** `output/screenshot_diagnostic_008/`

Revisited screenshot extraction from Plan 006 with a stricter diagnostic workflow:
- Sample 12 timestamps per file
- Extract both screenshot and 10-second playable MP4 clip (video+audio)
- Transcribe clip audio and score local transcript-vs-audio agreement
- Label each point (`ALIGNED`, `NOISY_BUT_USABLE`, `UNALIGNED`, `INSUFFICIENT_TEXT`)

This made manual visual checking practical: each screenshot now has nearby playback context and an explicit confidence label.

**Result:** 7 GOOD, 2 REVIEW, 0 BAD, 1 INSUFFICIENT_DATA across 10 files. The pipeline now supports confident screenshot verification without relying on player-clock interpretation.

### Plan 008b: Follow-up on Previously Disputed File (`2024-t23285...`)

**Results:** `008b_problematic_file_followup_results.md`
**Output:** `output/screenshot_diagnostic_008_23285/`

Ran the same Plan 008 diagnostic on the previously disputed file that was not in the 10-file set.

**Result:** `GOOD` (12 valid points, 11 trustworthy, 91.67% trust rate, longest unaligned streak = 1).
After visual deduplication, 7 informative checkpoints remained for manual review.

## Plan 009: Visual Moment Selection (Stable Post-Change States)

**Plan:** `009_visual_moment_selection_stable_states.md`
**Results:** `009_visual_moment_selection_stable_states_results.md`
**Scripts:** `scripts/data/select_visual_moments.py`, `scripts/data/test_screenshot_diagnostic.py`
**Output:** `output/visual_moment_selection_009_v3/`, `output/screenshot_diagnostic_009_visual_v3/`

Built a visual-only moment selector for screenshot timing that prefers stable states after visual activity:
- Detect frame-change bursts from low-FPS video signal
- Select post-burst plateaus (not mid-drawing)
- Add explicit dark->content transition capture so "problem first appears" is not missed
- Apply adaptive dedupe to reduce redundant frames without collapsing meaningful intermediate states

Then re-used Plan 008 diagnostics to render screenshot + tiny clip + transcript window for manual verification at selected points.

**Result:** On the fixed 10-file set, selection runtime was practical (197.3s total; 19.7s mean per file), with 27-62 selected moments per file. Downstream diagnostic verification yielded 8 GOOD, 2 REVIEW, 0 BAD, 0 INSUFFICIENT_DATA using the recomputed full-10 summary (`_summary_full10.md`). Manual review confirmed materially better coverage of intermediate board states.

## Plan 010: LLM Transcript Semantic Moment Selection

**Plan:** `010_llm_transcript_semantic_moment_selection.md`
**Results:** `010_llm_transcript_semantic_moment_selection_results.md`
**Scripts:** `scripts/data/select_transcript_moments.py`, `scripts/data/run_010_llm_diagnostics.py`
**Output:** `output/llm_moment_selection_010/`, `output/screenshot_diagnostic_010_llm/`

Introduced a transcript-first screenshot moment selector using three LLMs (OpenAI, Anthropic, Gemini) to identify timestamps where visual context is needed to understand the dialogue (drawing/writing/showing work/deictic references). Implemented two-pass extraction (chunk-level candidate mining + global refinement), timestamp normalization, per-model manifests, and merged manifests (union + consensus).

Reused Plan 008 diagnostic rendering to produce the same review artifacts (screenshot + tiny clip + local gold text + confidence verdict) for apples-to-apples comparison against Plan 009.

Added per-transcript token and USD accounting by model, plus full-dataset (109 matched sessions) cost projection.

**Result:** On the fixed 10 files, selector run completed for all providers. Diagnostic verdicts were strong across all produced sets (OpenAI 10/10 GOOD, Anthropic 10/10 GOOD, Gemini 9/9 GOOD for non-empty manifests, merged union 10/10 GOOD, merged consensus 10/10 GOOD). Projected full-run cost for 109 sessions is approximately: OpenAI $0.74, Anthropic $7.98, Gemini $0.76 (with CIs logged in output reports).

## Plan 011: Timeline Comparison Visualization

**Plan:** `011_timeline_comparison_visualization.md`
**Results:** `011_timeline_comparison_visualization_results.md`
**Script:** `scripts/data/plot_timeline_comparison_011.py`
**Output:** `output/timeline_comparison_011/`

Added per-video timeline plots to compare screenshot moments selected by:
- Plan 009 visual selector (`v3`)
- Plan 010 provider selectors (`openai`, `anthropic`, `gemini`)
- Plan 010 merged selectors (`merged_consensus`, `merged_union`)

Each figure uses a shared horizontal time axis, row-per-method tick marks, and selected thumbnails to expose both high-agreement clusters and method-unique picks.

**Result:** Generated 10 timeline plots (fixed 10 stems), one aggregate markdown report, and summary JSON with counts/overlap metadata. This made visual overlap and misses across 009 vs 010 immediately inspectable.

## Plan 012: Screenshot-Conditioned Transcript Understanding Benchmark

**Plan:** `012_screenshot_conditioned_transcript_understanding.md`
**Results:** `012_screenshot_conditioned_transcript_understanding_results.md`
**Script:** `scripts/data/run_transcript_understanding_benchmark.py`
**Output:** `output/transcript_understanding_012/`

Implemented a benchmark to test whether screenshot context materially improves transcript understanding at selected moments:
- Build selected moments from Plan 010 `merged_union`
- Build paired control moments away from selected timestamps
- Generate image-only dense captions per selected screenshot (OpenAI/Anthropic/Gemini)
- Generate 4 MCQs per moment (2 comprehend + 2 predict) with iterative hardening
- Evaluate solver matrix across conditions:
  - `selected_text`
  - `control_text`
  - `selected_caption_{openai|anthropic|gemini}`
- Score exact MCQ accuracy and compute rank-order pass checks:
  - `selected+caption > selected-text > control-text`

**Result:** Pipeline implementation completed and syntax-validated; benchmark run not executed in this session (requires live provider API keys/calls).

## Plan 013: Reusable Video+Transcript Toolkit Modules

**Plan:** `013_reusable_video_transcript_toolkit_modules.md`
**Results:** `013_reusable_video_transcript_toolkit_modules_results.md`
**Modules:** `tutor_bench/toolkit/`

Extracted reusable primitives from script-local implementations into shared modules:
- time utilities (`ts_to_seconds`, `fmt_ts`, tolerance matching)
- transcript parsing/context extraction
- stems/jsonl I/O
- LLM JSON parsing + retry sleep + cost computation
- timestamp overlap/unmatched operations

Refactored high-leverage scripts (`select_transcript_moments.py`, `run_transcript_understanding_benchmark.py`, `plot_timeline_comparison_011.py`, `test_screenshot_diagnostic.py`) to consume shared toolkit APIs while keeping CLI/output shapes stable. Added unit tests for the new toolkit modules.

**Result:** Shared library base is now in place for ongoing plans, reducing duplicate reimplementation risk and making future experiment scripts easier to maintain.

## Plan 014: Prompt Registry and Prompt Observability

**Plan:** `014_prompt_registry_and_observability.md`
**Results:** `014_prompt_registry_and_observability_results.md`
**Module:** `tutor_bench/toolkit/prompts.py`

Moved Plan 010/012 prompt templates into a shared prompt registry with stable prompt IDs and refactored both scripts to consume those prompt builders.

Added prompt observability in outputs:
- prompt IDs embedded in summaries/manifests
- emitted `_prompts.json` snapshots for reporting/audit
- 012 per-record prompt IDs for captions/questions/responses

**Result:** Prompt provenance is now explicit and machine-readable, enabling later reporting by prompt version and reducing hidden prompt drift.

## Plan 015: MMTutor Keyframe Baseline (Uncapped)

**Plan:** `015_mmtutor_keyframe_baseline.md`
**Results:** `015_mmtutor_keyframe_baseline_results.md`
**Scripts:** `scripts/data/select_mmtutor_keyframes.py`, `scripts/data/run_mmtutor_diagnostics.py`, `scripts/data/plot_timeline_comparison_015.py`
**Outputs:**
- `output/mmtutor_keyframe_selection_015/`
- `output/screenshot_diagnostic_015_mmtutor/`
- `output/timeline_comparison_015/`

Implemented an MMTutor-comparable baseline adapted to tutor-bench workflows:
- SSIM boundary-based candidate discovery
- premium multi-provider VLM semantic pruning over frame+transcript context
- uncapped key-step extraction with quality/novelty/time-gap filtering
- key-step + previous-step pairing metadata
- merged set synthesis (`merged_union`, `merged_consensus`)
- timeline comparison against 009/010 plus qualitative markdown including screenshot path, clip path, and gold transcript context
- added async batch-mode support for OpenAI/Anthropic/Gemini provider paths
- added monitoring helper script for long-running batch executions

**Result:** End-to-end scripts and output contracts are implemented. Single-stem smoke-fast run completed with qualitative comparison outputs; 10-stem all-provider batch run launched and monitoring is in progress.

## Plan 016: Benchmark Port — rebuild the tutoring benchmark cleanly in tutor-bench

**Plan/Spec:** `016_benchmark_port_design.md`
**Status:** design (spec) — not yet implemented

Port the tutoring **replay benchmark** (cut a real transcript at a decisive moment → model tutor
continues vs. a simulated student → calibrated LLM judge scores scaffolding & rigor) from the old
`Insource-Services/ai2-synthetic-annotations` repo (the `tutorsim` package) into this repo, cleanly
and tutor-bench-native, while emitting **byte-identical on-disk artifacts** to the shipped Insource
benchmark.

Dual anchor: outputs = Insource byte-for-byte; organization/naming = clean. Approach C (hybrid):
cohesive `tutor_bench/benchmark/` subpackage, reuse `toolkit/` where it can't perturb bytes, replace
the mock `Annotator`/`Evaluator` stubs, apply Kyle's terminology (LM step = "scoring"/"judge", not
"annotation"; `Annotation`→`Judgment`; frozen wire keys like `annotation_type` preserved as a
released contract). Offline only — live 520 reproduction explicitly deferred. Phased: foundations →
scoring → rollout → orchestration/reporting → human baseline + docs.

**Result:** Phases 0–5 DONE + committed on branch `feat/benchmark-port` (each verified byte-identical via
ported tests + AST logic-diff vs `insource/main`; ruff check + format clean). The full benchmark package is
ported and integrated:
- Phase 0: scaffold + packaged prompts/`default_config.yaml` (checksum-verified, LF-pinned via `.gitattributes`).
- Phase 1: `resources, client, config` (env var `TUTOR_BENCH_CONFIG`), `scenarios`.
- Phase 2: `scoring` (3-pass judge; `Annotation`→`Judgment`; wire keys incl. `annotation_type` kept).
- Phase 3: `tutor, student, conversation` (full rollout).
- Phase 4: `results` (byte-identical JSON), `report`, `cli` (run/report/view/dataset{build,validate};
  dropped dup `build-scenarios`; config.json provenance key `tutorsim_version`→`tutor_bench_version`).
- Phase 5: human baseline (`human.py` + runner script, serializer routed through `results.write_score`);
  top-level wiring (`tutor_bench.__init__` exports register_tutor/student; `tutor_bench/cli.py` → benchmark CLI);
  removed mock `Annotator`/`Evaluator` stubs + demo scripts + 1 mock test; README rewritten (benchmark-first,
  scorer/judge terminology, no `tutorsim`, verified accurate to code).
- **Suite: 344 passed / 10 skipped** (10 skip = external `balanced_520` absent; expected). Full `tests/` (benchmark + toolkit).

**Remaining — Phase 6 (data pipeline + configs):** port the dataset-build pipeline from `insource/main`
(`data/build_ground_truth.py`, `data/split_ground_truth.py`, `data/build_consolidated.py`,
`data/sort_ground_truth.py`, `data/README.md`) into a clean tutor-bench home (recommend `scripts/dataset/`),
with the same transformation discipline (rename `tutorsim`→`tutor_bench.benchmark`; strip provenance/internal
refs; deterministic — `split_ground_truth` is already `sorted()`+seeded). Decide placement of `split.json`
(tracked, reproducibility artifact) and `stats.ipynb` (tracked, rerunnable stats) with explicit `.gitignore`
allowlist entries. This also resolves Ryan's Insource hygiene questions (no orphan top-level `prompts/`,
scripts out of `data/`, intentional split.json/stats.ipynb). Then open the PR.

Source-of-truth = `insource/main` (+ uncommitted Insource working-tree `human.py`/`score_human_baseline.py`).
Live 520 reproduction against real APIs still out of scope (offline); it remains the one decisive check before
publishing leaderboard numbers.

---

## Chronology

| # | Plan | Approach | Outcome |
|---|------|----------|---------|
| 001 | Validate alignment via STT sampling | Spot-check WER at transcript timestamps | 74.7% alignment, suggested possible drift |
| 002 | Detect/correct drift via fuzzy matching | Anchor text + sliding window match | Failed — repeated phrases cause false matches |
| 003 | Replace transcripts with Whisper | Full re-transcription | Failed — student speech WER too high (0.40-0.48) |
| 004 | Hybrid: gold text + Whisper timestamps | NW alignment + timestamp transfer | Revealed Whisper agrees with gold timestamps |
| 005 | Investigate AV desync | ffprobe VFR + PTS + 10-file validation | **Confirmed: drift is QuickTime artifact, transcripts are correct** |
| 007 | Same-timepoint window overlap | Direct same-time comparisons, no alignment | **Confirmed again: timestamp differences are not the issue** |
| 008 | Screenshot diagnostic + tiny clips | Screenshot + playable local context + audio gate | **Made screenshot verification actionable and confidence-scored** |
| 008b | Disputed-file follow-up (`2024-t23285...`) | Same Plan 008 diagnostic on previously excluded file | **Passed as GOOD with high trust rate** |
| 009 | Visual moment selection (stable states) | Burst/plateau visual selector + dark->content capture + Plan 008 diagnostics | **Best current screenshot timing method; better intermediate-state recall with practical runtime** |
| 010 | LLM transcript semantic moment selection | 3-model semantic timestamp extraction + merged sets + Plan 008 diagnostics + cost projection | **Operational and strong diagnostics; enables per-model cost/performance tradeoff analysis** |
| 011 | Timeline comparison visualization | Multi-method per-video timeline plots with thumbnail callouts | **Clear visual overlap/divergence view across 009 v3 and all 010 sets** |
| 012 | Screenshot-conditioned transcript understanding benchmark | MCQ-based understanding test with/without screenshot-derived caption context | **Implemented and ready to run; quantitative benchmark pending execution** |
| 013 | Reusable toolkit module extraction | Shared `tutor_bench/toolkit` primitives + script refactors + unit tests | **Core reusable base established for future plan velocity and consistency** |
| 014 | Prompt registry + observability | Shared prompt builders + prompt IDs emitted in outputs | **Prompt provenance now explicit for reproducibility and reporting** |
| 015 | MMTutor keyframe baseline (uncapped) | SSIM candidate generation + premium VLM pruning + 009/010/015 timeline + qualitative artifacts | **Implemented and ready for API-backed benchmark/inspection runs** |
| 016 | Benchmark port (clean rebuild in tutor-bench) | Cohesive `tutor_bench/benchmark/` subpackage; byte-identical outputs vs Insource; Kyle terminology; phased + offline | **Design stage — spec written** |
