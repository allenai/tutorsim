# Video+Transcript Toolkit Quick Start

This walkthrough takes you from raw tutoring-session data to a benchmark that measures whether screenshots improve transcript understanding. Each section builds on the last and runs a real script you can follow along with.

**Prerequisites**: `ffmpeg`/`ffprobe` installed, LLM provider API keys in your environment (OpenAI, Anthropic, and/or Google), and `uv` for running scripts.

## What we're working with

Each tutoring session in the Step Up dataset has:

- a video (`.mp4`) in `data/stepup/videos/2_13/2_13/`
- a gold transcript (`.txt`) in `data/stepup/transcripts/2_13/Transcripts/`
- optionally, Whisper transcripts in `data/stepup/transcripts/2_13/Whisper/`

A **stem** identifies a session. The file `configs/visual_selection_fixed10.txt` lists 10 stems for repeatable experiments — we'll use it throughout.

Start by reading one transcript to see the format:

```python
from pathlib import Path
from tutor_bench.toolkit.transcript_utils import parse_gold_transcript

transcripts_dir = Path("data/stepup/transcripts/2_13/Transcripts")
stem = "2021-t12757_2025-s11381_3f675e00-ee9e-4306-ab28-271016fb4a98"
segments = parse_gold_transcript(transcripts_dir / f"{stem}_transcript.txt")

# Each segment has start/end times and speaker/text
for seg in segments[:5]:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}: {seg.text[:80]}")
```

You can also build a transcript context window around any timestamp — this is what the LLM pipelines see:

```python
from tutor_bench.toolkit.transcript_utils import context_prefix

# 90 seconds of transcript leading up to t=1419s
ctx = context_prefix(segments, center_t=1419.0, pre_s=90.0)
print(ctx)
```

## Step 1: Select screenshot moments from transcripts

The first pipeline reads transcripts and asks LLMs to identify timestamps where a screenshot would help explain what's happening — moments where the tutor says "look at this" or references something visual.

```bash
uv run scripts/data/select_transcript_moments.py \
  data/stepup/transcripts/2_13/Transcripts \
  --stems-file configs/visual_selection_fixed10.txt \
  --output-dir output/llm_moment_selection_010 \
  --providers openai,anthropic,gemini \
  --pricing-config configs/llm_pricing.json
```

Under the hood, this:
1. Chunks each transcript into ~8-minute windows
2. Sends each chunk through a two-pass LLM pipeline (identify candidates, then refine)
3. Deduplicates and snaps timestamps to actual transcript boundaries
4. Merges across providers (union and consensus)

**Outputs** in `output/llm_moment_selection_010/`:

```
openai/         # per-provider moment manifests
anthropic/
gemini/
merged_union/   # all moments from any provider
merged_consensus/  # moments where ≥2 providers agree
```

Each manifest is a JSON file per stem. A moment looks like:

```json
{
  "t": 1418.98,
  "timestamp": "00:23:38.980",
  "reason": "Tutor asks student to look at diagram",
  "evidence_quote": "can you see the triangle I drew?",
  "confidence": 0.85,
  "tags": ["visual_reference"],
  "source": "openai",
  "chunk_id": 2
}
```

Cost summaries are also emitted per provider so you can track spend.

## Step 2: Extract screenshots and clips for review

Now take those moment timestamps and pull actual frames and short clips from the videos:

```bash
uv run scripts/data/run_010_llm_diagnostics.py \
  data/stepup/videos/2_13/2_13 \
  data/stepup/transcripts/2_13/Transcripts \
  --whisper-dir data/stepup/transcripts/2_13/Whisper \
  --stems-file configs/visual_selection_fixed10.txt \
  --manifest-root output/llm_moment_selection_010 \
  --output-root output/screenshot_diagnostic_010_llm
```

This runs `test_screenshot_diagnostic.py` over each provider's manifest set. For each sampled timestamp it:
- extracts a screenshot (JPEG)
- extracts a ~10s clip around the moment
- checks transcript-audio alignment and writes a verdict

**Outputs** per manifest set: screenshot images, clip files, `_points.json` diagnostic records, and `_summary.md` for quick review.

You can also extract frames directly in Python:

```python
from pathlib import Path
from tutor_bench.toolkit.video_utils import extract_clip, extract_frame

video = Path("data/stepup/videos/2_13/2_13/<stem>.mp4")
t = 1418.98

extract_clip(video, clip_start=t - 2.5, clip_duration=10.0, out_mp4=Path("clip.mp4"))
extract_frame(Path("clip.mp4"), at_seconds=2.5, out_jpg=Path("frame.jpg"))
```

## Step 3: Compare selection methods on a timeline

If you want to see where different selectors agree or disagree:

```bash
uv run scripts/data/plot_timeline_comparison_011.py \
  --stems-file configs/visual_selection_fixed10.txt \
  --output-dir output/timeline_comparison_011
```

**Outputs**: per-video timeline plots with colored ticks per method plus thumbnail callouts, and a summary of overlap vs unique moments. Requires the `eval` extras (`matplotlib`, `pillow`).

## Step 4: Run the understanding benchmark

This is the main experiment. It tests whether adding screenshot-derived captions to transcript context actually helps LLMs answer questions better.

```bash
uv run scripts/data/run_transcript_understanding_benchmark.py \
  --stems-file configs/visual_selection_fixed10.txt \
  --transcripts-dir data/stepup/transcripts/2_13/Transcripts \
  --selected-manifest-dir output/llm_moment_selection_010/merged_union \
  --selected-diag-dir output/screenshot_diagnostic_010_llm/merged_union \
  --output-dir output/transcript_understanding_012 \
  --providers openai,anthropic,gemini \
  --pricing-config configs/llm_pricing.json
```

The pipeline has several stages:

1. **Select + control moments** — pairs each selected moment with a matched control timestamp from elsewhere in the transcript
2. **Dense captioning** — generates image-only captions for each screenshot using multiple providers
3. **QA authoring** — creates multiple-choice questions (2 comprehension + 2 prediction, 4 choices each)
4. **Solver matrix** — each provider answers under three conditions: `selected_text` (transcript only at selected moment), `selected_caption_{provider}` (transcript + caption), `control_text` (transcript at control moment)
5. **Scoring** — exact-match accuracy, no judge model needed
6. **Iteration** — regenerates questions that were too easy, to sharpen the signal

**Outputs** in `output/transcript_understanding_012/`:

```
moments/selected_moments.jsonl
moments/control_moments.jsonl
questions/iter_00/questions.jsonl
responses/iter_00/responses.jsonl
_summary.json          # machine-readable results
_summary.md            # human-readable results
_costs_by_stem.json    # per-stem cost breakdown
```

The key result is whether the rank order holds: `selected+caption > selected-text > control-text`.

## Toolkit reference

The shared utilities live in `tutor_bench/toolkit/`. Here's what each module does:

| Module | Purpose |
|--------|---------|
| `transcript_utils` | Parse gold transcripts, build context windows around timestamps |
| `video_utils` | `ffprobe` duration, `ffmpeg` clip/frame extraction |
| `time_utils` | Timestamp parsing (`"00:23:38.980"` ↔ `1418.98`), tolerance matching |
| `io_utils` | Read stem files, load/save JSONL |
| `moment_ops` | Overlap and unmatched-timestamp operations between moment sets |
| `llm_utils` | JSON extraction from model responses, retry backoff, cost computation |
| `prompts` | Versioned prompt builders for moment selection, captioning, QA authoring/solving |

## Tips

- Always start with the small `configs/visual_selection_fixed10.txt` stems file before scaling up.
- Keep output directories separated by experiment so runs don't mix.
- Cost summaries are part of the output — check them, especially for multi-provider runs.
- Timestamps should be compared with tolerance windows (typically 5s), not exact equality.
- Experiment plans and prior results live in `plans/` if you want rationale and history.
