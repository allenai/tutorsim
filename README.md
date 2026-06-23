# AI Tutor Benchmark

Evaluates AI tutor models on real K-12 tutoring transcripts using an LLM-based synthetic annotation pipeline.

## How It Works

1. **Extract scenarios** from real tutoring transcripts -- cut at pedagogically interesting moments
2. **Generate exchanges** -- AI tutor continues the conversation with a synthetic student
3. **Annotate** -- 3-pass pipeline detects key moments, analyzes tutor strategies, labels effectiveness
4. **Score** -- Three annotator styles (generous/balanced/demanding) produce a weighted composite score
5. **Leaderboard** -- Rank tutor models across all scenarios

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
echo "GEMINI_API_KEY=..." >> .env
echo "OPENAI_API_KEY=..." >> .env
echo "ANTHROPIC_API_KEY=..." >> .env

# Run benchmark (small test)
python -m benchmark --version test --tutor-profile anthropic --max-scenarios 2 --num-turns 2 --mode sync

# Run full benchmark
python -m benchmark --version v1

# View results
python -m benchmark.eval.eval --version v1 --profile anthropic
python -m benchmark.eval.view --version v1 --profile anthropic
```

## Annotator Pipeline (standalone)

The annotation pipeline can also run independently on raw transcripts:

```bash
# Full 3-pass pipeline
python -m annotator --version v4 --profile anthropic

# Evaluate against human ground truth
python -m annotator.eval.eval --version v4 --mode full
```

## Configuration

All model profiles and benchmark settings live in `config.yaml` at the repo root.

Supported providers: Gemini, OpenAI, Anthropic. Each supports batch API for efficient large-scale runs.

## Project Structure

- `annotator/` -- Synthetic annotation pipeline (detect, annotate, label)
- `benchmark/` -- Tutor model evaluation (scenarios, exchange, scoring)
- `prompts/` -- All prompt templates (annotator, benchmark, archived iterations)
- `config.yaml` -- Model profiles and benchmark settings
- `data/` -- Private student transcripts and ground truth (gitignored)
- `results/` -- Generated outputs (gitignored)
