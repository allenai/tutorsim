# CLAUDE.md

## Project Overview

This project answers: **"How good are AI tutors at the human side of teaching?"** It measures whether AI tutor models can replicate the pedagogical strategies real human tutors use -- specifically scaffolding (guiding students to answers without giving them away) and rapport (building trust, reading emotions, making learning feel safe).

The dataset is 104 real K-12 math tutoring transcripts. Human expert annotators labeled "key moments" as effective, partial, or ineffective.

The system has two pipelines:
1. **Annotator pipeline** (`annotator/`) -- uses LLMs to replicate what human annotators do (detect moments, analyze tutor strategies, label effectiveness). Validated to exceed human inter-rater agreement.
2. **Benchmark pipeline** (`benchmark/`) -- fully ground-truth-free. Runs synthetic detection to find key moments + cut points in transcripts, has an AI tutor continue from the cut point with a synthetic student, then scores the AI's pedagogical quality using the annotator pipeline with 3 calibrated styles.

## Commands

```bash
# Annotator pipeline
python -m annotator --version v4 --profile anthropic
python -m annotator --version v4 --profile anthropic --style generous  # per-style run
python -m annotator.eval.view --version v4
python -m annotator.eval.eval --version v4 --style generous            # per-style eval

# Benchmark pipeline (full run)
python -m benchmark --version claude-opus-4-6_2026-03-27 --tutor-profile anthropic

# Benchmark smoke test (10 transcripts, uses config num_turns)
python -m benchmark --version smoke_test --tutor-profile anthropic --test 10 --mode sync

# Benchmark flags
#   --test N          Limit detection to N transcripts (default: all 104)
#   --max-scenarios N Limit scenarios after detection (default: all)
#   --mode sync       Use sync API instead of batch (slower but no queue waits)
#   --tutor-profile   Which model to evaluate (anthropic, gemini, openai)
```

## Project Structure

- `annotator/` -- 3-pass annotation pipeline (detect, annotate, label). Entry: `run.py`
- `annotator/core/storage.py` -- Factor IV storage layer (S3/local backends, env var config)
- `benchmark/` -- tutor evaluation pipeline (detect -> scenarios -> exchange -> annotate -> score). Entry: `run.py`
- `prompts/` -- all prompt templates
  - `annotator/v4/` -- canonical detection + annotation prompts
  - `annotator/v5/` -- v4 detection + cut point guidance (used by benchmark)
  - `annotator/profiles/{generous,balanced,demanding}/p2/` -- per-archetype annotation prompts
  - `annotator/labeller/` -- Pass 3 classification prompts
- `config.yaml` -- unified config (model profiles, benchmark settings, storage)
- `data/` -- private student transcripts and ground truth (gitignored, 104 conversations)
- `results/` -- all pipeline outputs (gitignored, exists only on disk)
- `history/` -- archived iteration data (gitignored)
- `tests/` -- storage layer tests (pytest + moto)

## Architecture

### Annotator Pipeline

3-pass pipeline: detect -> annotate -> label. Each pass uses `annotator/core/client.py` (ModelClient) which wraps Gemini, OpenAI, and Anthropic APIs with unified batch and sync modes. Model profiles are in `config.yaml` under `profiles:`.

- **Pass 1 (detect)**: Find turn ranges where notable pedagogical moments occur. v4 prompts are final (model-limited ceiling, not prompt-limited). v5 adds cut point guidance for benchmark use.
- **Pass 2 (annotate)**: Analyze each moment (situation/action/result). Per-archetype profiles in `profiles/{style}/p2/`.
- **Pass 3 (label)**: Classify as effective/partial/ineffective.

### Benchmark Pipeline

Fully ground-truth-free. Steps:
1. **Step 0: Detection** -- Run v5 detection on transcripts (finds key moments + `suggested_cut_turn`)
2. **Step 1: Scenarios** -- Each detection becomes a scenario, cut at `suggested_cut_turn`
3. **Phase 1: Exchanges** -- Synthetic tutor + student alternate for N turns (config: `exchange.num_turns`). Saved incrementally after each round.
4. **Phase 2: Annotation** -- 3-style annotation (generous/balanced/demanding profiles)
5. **Phase 3: Scoring** -- Per-style scores saved separately (no composite aggregation)

Results naming: `{tutor_model}_{date}`. Each run's `config.json` records all resolved model names for traceability.

### Storage Layer

`annotator/core/storage.py` provides Factor IV compliant S3/local abstraction:
- `STORAGE_BACKEND=local` (default): reads from `data/`, writes to `results/`
- `STORAGE_BACKEND=s3`: reads/writes to S3 with in-memory caching
- All paths overridable via env vars (`STORAGE_TRANSCRIPTS`, `STORAGE_GROUND_TRUTH`, etc.)
- See `.env.example` for all env vars

All data I/O in the codebase routes through the storage layer. Callers never touch file paths directly.

## Critical Rules

- **NEVER run `git checkout HEAD -- .` or `git reset --hard` without verifying `results/` is safe.** Results are gitignored and exist only on disk -- there is no backup in git.
- **NEVER commit `data/` or `results/`.** These contain private student data and generated outputs.
- **Data was leaked in git history** (112 transcript IDs + annotator names). See `data/data_in_git_history.txt`. Needs `git filter-repo` to scrub.
- Use absolute imports throughout (e.g., `from annotator.core.config import get_phase_config`). No `sys.path` hacks.
- Detection prompts (v4) are finalized. Do not modify existing v4 detection content -- only append new sections (like v5's cut point guidance).
- Annotation profiles are finalized and exceed human inter-annotator ceilings. Do not re-iterate unless you have a specific hypothesis.
- Scores are per-style, not composite. Do not aggregate across styles -- the user picks their perspective.

## Code Style

- Python 3.10+. Type hints encouraged.
- JSON for all pipeline I/O via storage layer (`save_annotator_result`, `load_annotator_result`, etc.)
- Batch mode preferred for large runs (cost-efficient). Sync mode (`--mode sync`) for debugging/smoke tests.
- Pipeline outputs include token usage per API call -- preserve for cost tracking.
- `load_prompt` functions try `.md` first, then `.txt` (v5 uses .md, v4 uses .txt).

## Key Context

- See @docs/status.md for current project state, completed work log, and what's in progress.
- See @docs/lessons_learned.md for pitfalls encountered during development.
- See @docs/profile_detection_iteration.md for ARW-62 detection iteration findings.
- See `annotator/iteration/ITERATION_INSTRUCTIONS.md` for prompt iteration framework.

## Current State (2026-03-27)

- **Annotator pipeline**: stable, v4 prompts final, 3 archetype profiles validated.
- **Benchmark pipeline**: redesigned (ground-truth-free), smoke test running. First full run pending.
- **Storage layer**: Factor IV refactored, local backend working, S3 credentials configured but bucket data layout not yet set up.
- **AWS access**: credentials configured (`~/.aws/credentials`), bucket `kylel-alexisr-edu` accessible. Data needs to be uploaded to `synth-students/tutor-bench/` prefix.
- **Config**: `num_turns` currently set to 5 for smoke testing. Change back to 20 for full runs.
