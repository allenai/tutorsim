# Video+Transcript Toolkit Quick Start

This guide describes the reusable data-workflow components used by recent plans (008–012) for:

- video + gold transcript pairing
- screenshot moment selection
- screenshot validation with tiny clips
- multi-provider LLM workflows (selection, captioning, QA generation, QA evaluation)
- cost and metrics tracking

## Who This Is For

Use this page if you are new to the repo and want to quickly answer:

- What are the main reusable components?
- Where are they organized?
- How do I run a basic workflow?
- How do I plug in different LLM providers/models/prompts?

## Current Status

The reusable module layer is being standardized.
Today, most workflows are script-based under `scripts/data/`.

This document shows:

1. the target reusable module APIs, and
2. the current script entrypoints that already implement those capabilities.

## Repository Organization (Toolkit-Focused)

- `scripts/data/`
  - `test_screenshot_diagnostic.py`: clip+screenshot extraction + transcript-audio validation
  - `select_visual_moments.py`: visual signal-based moment selection
  - `select_transcript_moments.py`: transcript-semantic moment selection (multi-provider)
  - `run_010_llm_diagnostics.py`: run diagnostics for 010 manifest sets
  - `plot_timeline_comparison_011.py`: timeline overlap/unique visualization
  - `run_transcript_understanding_benchmark.py`: QA benchmark with/without screenshot caption context
- `configs/`
  - `visual_selection_fixed10.txt`: fixed stem list
  - `llm_pricing.json`: pricing table used for cost rollups
- `plans/`
  - plan and results docs for reproducible experiments
- `output/`
  - generated manifests, screenshots/clips, summaries, and benchmark outputs

## Target Reusable Library Modules (Planned)

- `tutorbench.core.time`
  - timestamp parse/format and tolerance matching utilities
- `tutorbench.core.transcript`
  - parse gold transcript format into typed segments
  - build context windows around a target timestamp `t`
- `tutorbench.core.video`
  - `ffprobe` duration
  - `ffmpeg` clip/frame extraction
- `tutorbench.core.io`
  - stems/json/jsonl read/write helpers
- `tutorbench.moments.schema`
  - typed moment manifest + diagnostic point schemas
- `tutorbench.moments.ops`
  - dedupe/clustering/overlap/unmatched timestamp operations
- `tutorbench.llm.providers`
  - unified text/image call interface for OpenAI/Anthropic/Gemini
  - retries, fallback aliases, cache support
- `tutorbench.llm.prompts`
  - prompt templates + versioned prompt IDs
- `tutorbench.llm.costing`
  - usage normalization + per-provider/model/stem/stage cost rollups
- `tutorbench.diagnostics.screenshot`
  - reusable screenshot diagnostic primitives
- `tutorbench.eval.qa`
  - QA generation/solving/scoring for transcript understanding experiments
- `tutorbench.reporting`
  - markdown/json summary builders

## Main Data Contracts

### 1) Moment Manifest (`*_moments.json`)

Typical fields:

- metadata: session stem, provider/model, optional usage/cost
- `moments[]`: timestamp (`t`, `timestamp`), reason/evidence/tags/confidence

### 2) Diagnostic Points (`_points.json`)

Typical fields:

- `timestamp_seconds`
- `screenshot_path`
- `clip_path`
- local transcript context + alignment/verdict fields

### 3) QA Benchmark Files

- `moments/selected_moments.jsonl`
- `moments/control_moments.jsonl`
- `questions/iter_XX/questions.jsonl`
- `responses/iter_XX/responses.jsonl`
- `_summary.json`, `_summary.md`
- `_costs_by_stem.json`

## Quick Start Workflows

### A) Generate transcript-semantic screenshot moments (multi-provider)

```bash
UV_CACHE_DIR=/tmp/uvcache uv run scripts/data/select_transcript_moments.py \
  data/stepup/transcripts/2_13/Transcripts \
  --stems-file configs/visual_selection_fixed10.txt \
  --output-dir output/llm_moment_selection_010 \
  --providers openai,anthropic,gemini \
  --pricing-config configs/llm_pricing.json
```

What you get:

- per-provider manifests (`openai/`, `anthropic/`, `gemini/`)
- merged manifests (`merged_union/`, `merged_consensus/`)
- selector cost summaries/projections

### B) Validate screenshot moments with tiny clips

```bash
UV_CACHE_DIR=/tmp/uvcache uv run scripts/data/run_010_llm_diagnostics.py \
  data/stepup/videos/2_13/2_13 \
  data/stepup/transcripts/2_13/Transcripts \
  --whisper-dir data/stepup/transcripts/2_13/Whisper \
  --stems-file configs/visual_selection_fixed10.txt \
  --manifest-root output/llm_moment_selection_010 \
  --output-root output/screenshot_diagnostic_010_llm
```

What you get:

- screenshot + short clip per sampled timestamp
- transcript-audio alignment verdicts
- per-set summary markdown/json

### C) Compare visual vs text-based selectors on a timeline

```bash
UV_CACHE_DIR=/tmp/uvcache uv run scripts/data/plot_timeline_comparison_011.py \
  --stems-file configs/visual_selection_fixed10.txt \
  --output-dir output/timeline_comparison_011
```

What you get:

- per-video timeline plot with colored method ticks + thumbnail callouts
- summary markdown/json with overlap and unique moments

### D) Run transcript understanding QA benchmark with/without screenshot context

```bash
UV_CACHE_DIR=/tmp/uvcache uv run scripts/data/run_transcript_understanding_benchmark.py \
  --stems-file configs/visual_selection_fixed10.txt \
  --transcripts-dir data/stepup/transcripts/2_13/Transcripts \
  --selected-manifest-dir output/llm_moment_selection_010/merged_union \
  --selected-diag-dir output/screenshot_diagnostic_010_llm/merged_union \
  --output-dir output/transcript_understanding_012 \
  --providers openai,anthropic,gemini \
  --pricing-config configs/llm_pricing.json
```

What you get:

- selected vs control moments
- dense captions per provider
- generated MCQs (comprehend + predict)
- solver responses across conditions
- accuracy + rank-order summaries + cost rollups

## API Usage Examples (Target Library Style)

### Parse transcript and build context at time `t`

```python
from tutorbench.core.transcript import parse_gold_transcript, context_prefix

segments = parse_gold_transcript("data/.../Transcripts/<stem>_transcript.txt")
ctx = context_prefix(segments, t=1418.98, pre_s=90.0)
print(ctx)
```

### Extract short clip + screenshot for a moment

```python
from tutorbench.core.video import extract_clip, extract_frame

extract_clip(video_path, start_s=t - 2.5, duration_s=10.0, out_mp4=clip_path)
extract_frame(clip_path, at_s=2.5, out_jpg=shot_path)
```

### Call multi-provider LLM uniformly (text or image)

```python
from tutorbench.llm.providers import call_llm, LLMRequest

req = LLMRequest(
    provider="openai",
    model="gpt-5.2",
    mode="text",
    prompt=prompt_text,
    timeout_s=120,
)
resp = call_llm(req, cache_key="qa-author::<id>", cache_dir=".cache/llm")
print(resp.text, resp.usage, resp.model_used)
```

### Compute cost rollups

```python
from tutorbench.llm.costing import load_pricing, compute_cost

pricing = load_pricing("configs/llm_pricing.json")
cost = compute_cost(resp.usage, provider="openai", model=resp.model_used, pricing=pricing)
print(cost.total_cost_usd)
```

## Design Rules For New Experiments

- Reuse schema and ops modules before adding new script-local helpers.
- Keep prompts versioned; do not silently overwrite prompt behavior.
- Always emit both machine-readable (`.json/.jsonl`) and skim-friendly (`.md`) summaries.
- Include provider/model, prompt version, and pricing config path in outputs.
- Treat tolerance windows (for example, `5s` equivalence) as explicit parameters in reports.

## Common Pitfalls

- Comparing timestamps exactly instead of with a tolerance window.
- Mixing cached and newly executed calls without separating estimated vs executed cost.
- Letting script-local parsers drift from canonical transcript/moment schemas.
- Evaluating caption-conditioned runs without checking whether captions introduce distractor bias.

## Minimal Reading Order

1. `plans/011_timeline_comparison_visualization_results.md`
2. `plans/012_screenshot_conditioned_transcript_understanding_results.md`
3. `scripts/data/run_transcript_understanding_benchmark.py`
4. `configs/llm_pricing.json`
