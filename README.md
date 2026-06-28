# tutor-bench

[![CI](https://github.com/allenai/tutor-bench/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/tutor-bench/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**tutor-bench measures how well language models tutor.** It replays a model under
test against a simulated student on frozen, human-annotated tutoring moments,
then uses a calibrated LLM judge to score the generated continuation for two
pedagogical behaviors:

- **Scaffolding** -- guiding a student toward an answer without handing it over.
- **Rigor** -- pushing a student to do the cognitive work when they are ready
  for it, rather than over-helping.

Each benchmark moment is cut from a real K-12 tutoring transcript at a point
where a *team of human teachers* annotated what good tutoring looks like. The
model continues the conversation from the cut; the scorer judges whether it did
the pedagogically appropriate thing for that moment.

## What this repo optimizes for

- A minimal, readable codebase for NLP researchers.
- A fast contributor loop with a small quality gate: `ruff`, `pyright`, `pytest`.
- Reproducible runs: a sha256-pinned dataset, deterministic sampling, and a
  full reproducibility record written into every run.
- Local experiment artifacts (`data/`, `output/`, `results/`) are intentionally
  untracked by default.

## Installation

tutor-bench requires Python 3.11 or newer.

```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `tutor-bench` command and the `tutor_bench` Python package.
Runtime assets that affect benchmark behavior are packaged under
`tutor_bench/benchmark/`:

- `tutor_bench/benchmark/default_config.yaml`
- `tutor_bench/benchmark/prompts/tutor/`
- `tutor_bench/benchmark/prompts/student/`
- `tutor_bench/benchmark/prompts/scorer/`

## Setup

Tutor, student, and scorer models are called through hosted provider APIs. Set
the API keys for the providers you intend to use as environment variables (a
local `.env` file is loaded automatically):

```bash
# .env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
TOGETHER_API_KEY=...
```

Each provider in `default_config.yaml` declares which environment variable holds
its key; you only need keys for the providers your run touches.

## Quickstart

```bash
# Score a tutor model on the first 10 scenarios of the default dataset
tutor-bench run --tutors claude-opus-4-8 --sample 10

# Aggregate completed runs into a leaderboard (leaderboard.md + leaderboard.csv)
tutor-bench report

# Build a self-contained HTML viewer of all runs
tutor-bench view
```

`run` writes one run directory per `(tutor x mode)` cell under `results/`.
`report` and `view` read the `summary.json` of every run directory and render a
combined leaderboard.

## The benchmark / dataset

The official dataset is **`balanced_520`**: 520 scenarios, balanced as **260
scaffolding-appropriate + 260 rigor-appropriate** moments. Each scenario is a
frozen cut point in a real tutoring transcript, paired with the aggregate
judgment of a team of human teachers about which behavior the moment calls for.

The dataset is distributed via HuggingFace, external to git, and pinned by a
sha256 content hash recorded in its manifest. Validate an installed dataset
with:

```bash
tutor-bench dataset validate --set balanced_520
```

Validation checks the manifest name, schema version, record count, and content
hash against the installed `scenarios.jsonl`. Until a dataset is installed,
`tutor-bench run` fails fast with an actionable missing-dataset error.

### Scenario record schema

Each line of `scenarios/<set>/scenarios.jsonl` is one `Scenario`:

```json
{
  "id": "balanced_520:<conv_id>__hum_<turn_start>_<turn_end>",
  "context": [
    { "turn_number": 12, "role": "tutor",   "text": "..." },
    { "turn_number": 13, "role": "student", "text": "..." }
  ],
  "dimension": "scaffolding",
  "student": {
    "mode": "oracle",
    "reference": "Turn 14. TUTOR: ...\nTurn 15. STUDENT: ...",
    "context": "Grade 7, Math"
  },
  "rubric": {
    "gold": "scaffolding",
    "hint": "..."
  },
  "provenance": {
    "conv_id": "...",
    "cut_turn": 13,
    "turn_start": 12,
    "turn_end": 13,
    "moment_id": "...",
    "annotator_id": "...",
    "chosen_cut_turn": 13,
    "cut_votes": { "13": 3, "14": 1 },
    "cluster_size": 4
  }
}
```

- `context` -- the pre-cut transcript prefix the model sees. `turn_number` is the
  real (non-sequential) turn number from the source transcript; `role` is
  lowercase.
- `dimension` / `rubric.gold` -- the behavior the teacher team judged appropriate
  for this moment (`scaffolding` or `rigor`).
- `student.reference` -- the real post-cut human turns, used by the oracle
  student to stay faithful to the original conversation.
- `provenance` -- how the cut point was chosen (modal vote across annotators,
  ties broken toward the smaller turn) plus source identifiers.

## Testing a tutor model

### Hosted models (config roster)

Hosted tutors are referenced by model id; the provider is inferred from the id.
The roster and per-model call settings live in `default_config.yaml`:

```yaml
models:                          # tutor roster; provider inferred from id
  claude-opus-4-8:             { thinking: true, effort: xhigh }
  claude-sonnet-4-6:           { thinking: true, effort: high }
  gemini-2.5-pro:              { thinking: true, thinking_budget: -1 }
  gemini-3.5-flash:            { thinking: true, thinking_budget: -1 }
  gpt-5.4-mini-2026-03-17:     { thinking: true, reasoning_effort: high }
  gpt-5.5-2026-04-23:          { thinking: true, reasoning_effort: high }
  deepseek-ai/DeepSeek-V4-Pro: {}

student: { model: claude-opus-4-6, mode: oracle, thinking: false }
scorer:  { model: claude-opus-4-6, thinking: adaptive }

defaults: { seed: 10, trials: 1, max_turns: 5 }
```

Run one or more tutors across one or more prompt modes:

```bash
tutor-bench run \
  --tutors claude-opus-4-8 gpt-5.5-2026-04-23 \
  --modes plain scaffolding_rigor \
  --sample 50
```

Supported tutor prompt modes:

| Mode | Description |
|---|---|
| `plain` | Baseline tutor prompt. |
| `scaffolding_rigor` | Tutor prompt that explicitly targets scaffolding and rigor. |
| `oracle` | Reference-aware tutor prompt; requires a scenario reference transcript. |

`run` options:

| Option | Purpose |
|---|---|
| `--tutors MODEL_ID ...` | **Required.** Tutor model id(s) or registered custom tutor name(s). |
| `--modes MODE ...` | Prompt mode(s). Default: `plain scaffolding_rigor`. |
| `--dataset NAME` | Scenario set under `scenarios/`. Default: `balanced_520`. |
| `--sample N` | Use the first `N` scenarios from the dataset (deterministic). Default: all. |
| `--trials N` | Run each tutor/mode cell `N` times and summarize mean + spread. |
| `--seed N` | Seed recorded in the run config for reproducibility. |
| `--max-turns N` | Maximum generated tutor/student turns per conversation. |
| `--trait-cache-dir DIR` | Cache directory for generated student traits. |
| `--config FILE` | Explicit config file (highest precedence). |

Tutors are scheduled by provider: cells in different provider lanes run in
parallel, while cells in the same lane run sequentially. Runs are resumable --
a scenario whose transcript and score are already on disk is skipped -- and
skip-on-error, so one bad scenario is logged and skipped rather than crashing
the run. A run where zero scenarios complete raises an error instead of writing
an empty summary.

### Custom models (in-process callables)

Models that are not in the hosted roster can be registered as Python callables
with `register_tutor` / `register_student` from `tutor_bench`. Each callable
takes the conversation so far and returns the next turn:

```python
from tutor_bench import register_tutor, register_student
from tutor_bench.cli import main


@register_tutor("my-tutor")
def my_tutor(conversation):
    return "next tutor turn"


@register_student("my-student")
def my_student(conversation):
    return "next student turn"


if __name__ == "__main__":
    main()
```

Because registration happens in-process, import your registrations before
invoking the CLI -- the simplest path is a small wrapper script like the one
above, then:

```bash
python my_wrapper.py run --tutors my-tutor --sample 1
```

There is no plugin auto-discovery mechanism yet.

## Swapping the student

The simulated student is fully swappable. By default it is an **oracle**
student: a hosted model (`student` in the config) that uses each scenario's
reference transcript to stay faithful to how the real student behaved after the
cut. To change the student model or mode, edit the `student` block in your
config; to supply a custom student, register one with `register_student` as
shown above. Student personas can be enriched along trait dimensions
(`tutor_bench/benchmark/prompts/student/dimensions/`), with generated traits
cached under `--trait-cache-dir`.

## Configuration

Config is resolved in this precedence order:

1. `tutor-bench run --config path/to/config.yaml`
2. `TUTOR_BENCH_CONFIG` environment variable
3. `./config.yaml` in the working directory
4. packaged `tutor_bench/benchmark/default_config.yaml`

Top-level config sections:

| Key | Purpose |
|---|---|
| `providers` | Maps each provider name to the environment variable holding its API key. |
| `models` | Tutor roster; provider inferred from the model id. |
| `student` | Hosted model and mode for the simulated student. |
| `scorer` | Hosted model used for the scoring passes. |
| `defaults` | Default `seed`, `trials`, and `max_turns` for `tutor-bench run`. |
| `retry` | Retry behavior for direct model calls. |
| `batch` | Batch API polling timeout for scorer/provider batch paths. |

## Results & reproducibility

Each `(tutor x mode)` cell writes one run directory:

```text
results/<run_id>/
  config.json                     # resolved config + reproducibility record
  transcripts/<scenario_id>.json  # the generated conversation
  scores/<scenario_id>.json       # the judge's per-scenario score (one Judgment)
  summary.json                    # aggregate metrics over completed scenarios
```

`run_id` is self-documenting: `{tutor}_{mode}_{dataset}_{date}`.

### Per-scenario score schema

Each `scores/<scenario_id>.json` is one `Judgment` produced by the calibrated
LLM judge. The scorer runs three passes (annotate, decompose, structure) over
the model's continuation and emits:

```json
{
  "scenario_id": "balanced_520:...",
  "annotation_type": "scaffolding",
  "turn_start": 14,
  "turn_end": 18,
  "situation": "...",
  "action": "...",
  "result": "...",
  "action_decomposed": ["..."],
  "result_decomposed": ["..."],
  "overscaffold_decomposed": ["..."],
  "action_label": "scaffolding",
  "result_label": "pos",
  "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }
}
```

- `annotation_type` -- the scored dimension for this moment
  (`scaffolding` | `rapport`); the benchmark scores along the scaffolding lens.
- `action_label` -- what the tutor's move did
  (`scaffolding` | `rigor` | `both` | `neither` | `unclear`).
- `result_label` -- the student outcome
  (`pos` | `neg` | `unclear` | `no_evidence`).
- `overscaffold_decomposed` -- a non-empty list means over-scaffolding was
  detected.
- `usage` -- tokens summed across all three scoring passes.

(The field is named `annotation_type` on disk for schema continuity; it denotes
the *scored dimension*. In tutor-bench, "annotation" refers to what the human
teachers did to the source transcripts -- the model measurement step is the
**scorer**.)

### Summary metrics

`summary.json` aggregates the per-scenario judgments into leaderboard metrics:

| Metric | Interpretation |
|---|---|
| `scaffolding_did.rate` | Of scaffolding-appropriate scenarios, how often the tutor scaffolded. Higher is better. |
| `rigor_did.rate` | Of rigor-appropriate scenarios, how often the tutor pushed for rigor. Higher is better. |
| `overscaffold.rate` | Fraction of scenarios where over-scaffolding was detected. Lower is better. |
| `outcome_pos_rate` | Fraction of scenarios with a positive student-outcome label. |
| `scaffold_calibrated.score` | Scaffolding success after excluding over-scaffolded cases. |
| `rigor_calibrated.score` | Rigor success after excluding over-scaffolded cases. |

`summary.json` also records `run_counts` (attempted / succeeded / failed /
resumed), any `failed_scenarios`, and `latency` / `tokens` blocks for the tutor
and student calls. With `--trials N`, metrics are reported as `mean` + `spread`
(standard deviation) across trials.

### What is pinned

Every run's `config.json` carries a `reproducibility` block: the `tutor-bench`
package version, the git commit, a config hash, hashes of every packaged prompt,
and the dataset manifest. Combined with the sha256-pinned dataset and
deterministic first-N sampling, this makes a fresh-clone run reproducible at the
configuration level. Live LLM scores still vary run to run due to model
nondeterminism.

## Repository layout

```text
tutor-bench/
├── tutor_bench/
│   ├── cli.py                 # console-script entry point (re-exports benchmark CLI)
│   ├── benchmark/             # the replay benchmark (primary entry point)
│   │   ├── cli.py             # run | report | view | dataset {build,validate}
│   │   ├── config.py          # config loading, precedence, model registries
│   │   ├── client.py          # provider model client + batch API
│   │   ├── scenarios.py       # Scenario schema, dataset load/validate/build
│   │   ├── tutor.py           # tutor prompt modes
│   │   ├── student.py         # simulated student + trait generation
│   │   ├── conversation.py    # tutor/student rollout from the cut point
│   │   ├── scoring.py         # three-pass LLM judge -> Judgment
│   │   ├── report.py          # leaderboard + HTML viewer
│   │   ├── results.py         # results/<run_id>/ on-disk store
│   │   ├── human.py           # human-baseline scoring helpers
│   │   ├── default_config.yaml
│   │   └── prompts/           # packaged tutor / student / scorer prompts
│   └── toolkit/               # multimodal video+transcript primitives (see below)
├── tests/                     # test suite and fixtures
├── scripts/                   # utility / experiment scripts (not in the CI gate)
├── configs/                   # config assets
├── plans/_summary.md          # developer log summary
├── .github/workflows/         # CI config
└── pyproject.toml             # packaging + tool config
```

Local-only directories are gitignored by default: `data/`, `output/`,
`results/`, `scenarios/`, and most of `plans/` (except `plans/_summary.md`).

## Multimodal toolkit

`tutor_bench/toolkit/` holds reusable primitives for video + transcript
workflows (transcript and moment operations, timing, LLM and I/O helpers,
video utilities). The replay benchmark is now the primary entry point, but the
toolkit remains for the data and video work that supports it.

## Development

Run the full local quality gate before opening a PR:

```bash
make run-checks
```

Individual checks:

```bash
make lint           # ruff check
make format-check   # ruff format --check
make typecheck      # pyright
make test-fast      # pytest, excluding slow/integration/gpu
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.

## License

Apache-2.0. See [LICENSE](LICENSE).
