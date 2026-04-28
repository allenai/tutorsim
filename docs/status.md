# Project Status

*Last updated: 2026-04-16*

For the change log and planned work, see [plans/_summary.md](plans/_summary.md).

## Current State

### Repo Structure

- `annotator/` -- 3-pass annotation pipeline (detect, annotate, label)
- `benchmark/` -- tutor evaluation pipeline (detect, scenarios, exchange, annotate, score)
- `prompts/` -- all prompt templates (annotator v4 base, v5 detection, profiles, labeller, benchmark)
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

**Per-archetype annotator profiles** (labeller v2, ground_truth_v2):
- Generous (5 annotators, n=705): 3-way kappa 0.3740 (ceiling 0.4081)
- Balanced (3 annotators, n=1123): 3-way kappa 0.4574 (ceiling 0.2310)
- Demanding (1 annotator, n=172): 3-way kappa 0.4810

### Benchmark Pipeline -- Redesigned, First Full Run In Progress

The benchmark is now fully ground-truth-free. It uses synthetic detection to find key moments and cut points, then evaluates how an AI tutor handles each moment.

**Pipeline flow:**
1. v5 detection on all transcripts (finds key moments + `suggested_cut_turn`)
2. Each detection becomes a scenario (cut at suggested_cut_turn)
3. Synthetic tutor + student exchange (num_turns: 2, configurable)
4. 3-style annotation (generous/balanced/demanding profiles)
5. Per-style scoring (no composite aggregation -- user picks their perspective)

**First run** (`claude-opus-4-6_2026-03-26`): detection complete (2,608 detections from old v5 prompts, avg 25.1/conv). Exchange phase stalled (0 completed rounds). Needs re-run with updated v5 prompts.

**Results naming**: `{tutor_model}_{date}` with `config.json` capturing all run conditions (resolved model names, prompt versions, turn counts).

**Screenshots**: opt-in via `--with-screenshots` (or `benchmark.with_screenshots: true` in `config.yaml`). When on, all three phases (Step 0 detection, Phase 1 exchange, Phase 2 annotation) attach anchored screenshots from `deidentified/screenshots/{conv_id}/`. Default off — text-only runs reproduce prior numbers exactly. Wired but not yet validated end-to-end against real images: S3's `deidentified/screenshots/` has 3 conv UUIDs that have no matching transcripts anywhere accessible, so every screenshot-enabled run currently degrades to text-only. See `docs/lessons_learned.md` for the data-pairing gap.

**Resume**: Phase 1 (exchange) and Phase 2 (annotation) both resumable via per-scenario shard pre-filter + in-flight batch sidecar. A ctrl-C or crash mid-batch is recoverable — re-run with the same `--version` and the pipeline picks up where it left off. Stable version pointer at `_active_runs/{profile}.json` survives midnight resumes.

### Key Technical Decisions

- **Detection ceiling is model-limited**, not prompt-limited. v4 detection prompts are final.
- **v5 prompts = v4 detection verbatim + cut point guidance appended.** Cut point sections don't affect detection quality -- they add boundary and cut point instructions after the v4 content.
- **Annotation profiles use per-archetype prompts** (`profiles/{generous,balanced,demanding}/p2/`).
- **Benchmark is ground-truth-agnostic.** It only reads transcripts. Detection, annotation, and scoring are all synthetic.
- **Scores are per-style, not composite.** No weighted aggregation across styles. Each annotator perspective is a separate result.
- **Storage layer is Factor IV compliant.** Backend ABC pattern, env var overrides for all paths. `STORAGE_BACKEND=s3` for production, `local` for development.

### Prompt Organization (cleaned up 2026-04-16)

- `v4/p1/` + `v4/p2/` -- base detection + annotation prompts (fallback for non-styled runs)
- `v5/p1/` -- detection with cut point guidance (for benchmark). No p2 -- annotation prompts live in profiles.
- `profiles/{balanced,generous,demanding}/p2/` -- **canonical annotation prompts** (most evolved, used by all `--style` runs)
- `profiles/{balanced,generous,demanding}/p1/` -- per-style detection prompts

### Current Gold Results (`v5_gold/`)

All regenerated 2026-04-16 with updated profile prompts (inappropriate-timing example added):

| Style | Convs | Annotations | Effective | Partial | Ineffective |
|-------|-------|-------------|-----------|---------|-------------|
| No style | 201 | 1688 | 21.6% | 38.7% | 39.3% |
| Generous | 110 | 756 | 40.7% | 28.0% | 31.2% |
| Balanced | 161 | 1176 | 43.3% | 28.1% | 28.6% |
| Demanding | 16 | 172 | 20.3% | 26.2% | 53.5% |

Gemini balanced results also regenerated in `annotator_profiles/balanced/`.

## Token Usage Tracking

All results contain per-call token usage (`input_tokens`, `output_tokens`, `total_tokens`). Results are gitignored and exist only on disk -- see [lessons_learned.md](lessons_learned.md) for why this matters.
