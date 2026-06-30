# Tutorsim

Tutorsim is a benchmark for measuring how well language models tutor. It runs a
model-under-test against a simulated student on frozen tutoring moments, then
scores the generated continuation for scaffolding and rigor.

## Installation

Tutorsim requires Python 3.11 or newer. Install from a checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the `tutorsim` command and the `tutorsim` Python package. Runtime
assets that affect benchmark behavior are packaged under `src/tutorsim/`:

- `src/tutorsim/default_config.yaml`
- `src/tutorsim/prompts/tutor/`
- `src/tutorsim/prompts/student/`
- `src/tutorsim/prompts/scorer/`

Historical prompt archives remain under the root `prompts/` directory. They are
kept for research traceability and are not loaded by the packaged runtime.

## Configuration

Tutorsim loads config in this order:

1. `tutorsim run --config path/to/config.yaml`
2. `TUTORSIM_CONFIG`
3. `./config.yaml`
4. packaged `src/tutorsim/default_config.yaml`

Use `configs/local.example.yaml` as a starting point for local overrides.

Config files have these top-level sections:

| Key | Purpose |
|---|---|
| `providers` | Maps provider names to the environment variable that contains the API key. |
| `models` | Model roster for hosted tutor models. The provider is inferred from the model ID. |
| `student` | Hosted model and mode used for the simulated student. |
| `scorer` | Hosted model used for the scoring passes. |
| `defaults` | Default `seed`, `trials`, and `max_turns` values for `tutorsim run`. |
| `retry` | Retry behavior for direct model calls. |
| `batch` | Batch API polling timeout for scorer/provider paths that use batch APIs. |

## Dataset Status

The official `balanced_520` dataset is external to git and may not be published
yet. The expected installed layout is:

```text
scenarios/
  balanced_520/
    scenarios.jsonl
    manifest.json
    CHANGELOG.md
```

`manifest.json` records the dataset name, version, schema version, record count,
content hash, creation date, and provenance. Until the official dataset is
installed, `tutorsim run` fails fast with an actionable missing-dataset error.

You can validate the tiny committed fixture used for plumbing tests:

```bash
tutorsim dataset validate --set mini_set --root tests/tutorsim/fixtures
```

Build a dataset from local ground-truth inputs:

```bash
tutorsim dataset build \
  --set balanced_520 \
  --ids data/balanced_520_scenario_ids.json \
  --ground-truth data/ground_truth \
  --transcripts data/transcripts \
  --version 2026-06-27 \
  --created 2026-06-27
```

Build inputs are expected to be private/local files:

- `--ids`: JSON list of scenario IDs in the form `{conv_id}__hum_{turn_start}_{turn_end}`.
- `--ground-truth`: directory of ground-truth JSON files keyed by conversation.
- `--transcripts`: directory of transcript JSON files keyed by conversation.
- `--step-up-jsonl`: optional normalized transcript JSONL source used to fill transcript lookups.
- `--out-root`: output root; defaults to `scenarios/`.

The build command writes `scenarios/<set>/scenarios.jsonl`,
`scenarios/<set>/manifest.json`, and a stub
`scenarios/<set>/CHANGELOG.md`.

The legacy alias `tutorsim build-scenarios` still works, but new docs and scripts
should use `tutorsim dataset build`.

## Running

Once a dataset is installed:

```bash
tutorsim run \
  --tutors claude-opus-4-8 \
  --modes plain scaffolding_rigor \
  --sample 10
```

Useful run options:

| Option | Purpose |
|---|---|
| `--tutors MODEL_ID ...` | Tutor model IDs or registered custom tutor names to evaluate. |
| `--modes MODE ...` | Tutor prompt modes. Defaults to `plain scaffolding_rigor`. |
| `--dataset NAME` | Scenario set under `scenarios/`. Defaults to `balanced_520`. |
| `--sample N` | Use the first `N` scenarios from the dataset. |
| `--trials N` | Run each tutor/mode cell multiple times and summarize mean/spread. |
| `--seed N` | Store the seed in the run config for reproducibility. |
| `--max-turns N` | Maximum generated tutor/student turns per conversation. |
| `--trait-cache-dir DIR` | Cache directory for generated student traits. |

Supported tutor prompt modes are:

| Mode | Description |
|---|---|
| `plain` | Baseline tutor prompt. |
| `scaffolding_rigor` | Tutor prompt that explicitly targets scaffolding and rigor. |
| `oracle` | Reference-aware tutor prompt that requires a scenario reference transcript. |

Each run writes under `results/`. Re-running skips scenarios whose transcript and
score are already present. Partial failures are recorded in `summary.json`; a
run where zero scenarios complete raises an error and does not produce a
leaderboard-ready empty summary.

Run output layout:

```text
results/<run_id>/
  config.json
  transcripts/<scenario_id>.json
  scores/<scenario_id>.json
  summary.json
```

Each run's `config.json` includes the resolved config, package version when
available, git commit when available, config hash, prompt hashes, and dataset
manifest if present.

## Reports And Viewers

Aggregate completed run summaries into leaderboard files:

```bash
tutorsim report --results-root results --out leaderboard
```

This writes `leaderboard.md` and `leaderboard.csv`.

Build a self-contained HTML viewer:

```bash
tutorsim view --results-root results --out viewer.html
```

The report and viewer commands read `summary.json` files from run directories.

## Scoring

Tutorsim scores generated continuations with model-backed annotation passes and
then aggregates per-scenario scores into leaderboard metrics:

| Metric | Interpretation |
|---|---|
| `scaffolding_did.rate` | Of scaffolding-appropriate scenarios, how often the tutor scaffolded. |
| `rigor_did.rate` | Of rigor-appropriate scenarios, how often the tutor pushed for rigor. |
| `overscaffold.rate` | Fraction of scenarios where over-scaffolding was detected. Lower is better. |
| `outcome_pos_rate` | Fraction of scenarios with a positive student outcome label. |
| `scaffold_calibrated.score` | Scaffolding success after excluding over-scaffolded cases. |
| `rigor_calibrated.score` | Rigor success after excluding over-scaffolded cases. |

## Action Taxonomy

Classifies each `action_decomposed` facet into a 13-letter A–M scheme and
writes macro/moment headline tables. Install the optional `taxonomy` extra
(pandas), then:

```bash
tutorsim taxonomy classify --kind key_moments --input key_moments.jsonl --output ./human
tutorsim taxonomy classify --kind tutorsim --input results/<run_id>/ \
  --scenarios scenarios/balanced_520/scenarios.jsonl --output ./lm
tutorsim taxonomy headline --human ./human/classified.csv --lm ./lm/classified.csv --output ./headline
```

`tutorsim taxonomy run` chains all three. Frozen scheme + prompt live under
`src/tutorsim/`; figure notebooks live under `analysis/working-paper-*`.

## Custom Models

Hosted models are referenced by ID and configured in the model roster. Custom
callables can be registered for in-process use:

```python
from tutorsim import register_tutor, register_student

@register_tutor("my-tutor")
def my_tutor(conversation):
    return "next tutor turn"

@register_student("my-student")
def my_student(conversation):
    return "next student turn"
```

For CLI use, import the registration module before calling `tutorsim.cli.main()`
from a small wrapper script. There is not yet a plugin discovery mechanism.

Example wrapper:

```python
from tutorsim import register_tutor, register_student
from tutorsim.cli import main


@register_tutor("my-tutor")
def my_tutor(conversation):
    return "next tutor turn"


@register_student("my-student")
def my_student(conversation):
    return "next student turn"


if __name__ == "__main__":
    main()
```

Then run the wrapper with normal CLI arguments, for example:

```bash
python my_tutorsim_wrapper.py run --tutors my-tutor --sample 1
```

## Repository Layout

```text
.
├── pyproject.toml
├── configs/
│   └── local.example.yaml
├── src/tutorsim/
│   ├── cli.py
│   ├── config.py
│   ├── client.py
│   ├── scenarios.py
│   ├── tutor.py
│   ├── student.py
│   ├── conversation.py
│   ├── scoring.py
│   ├── results.py
│   ├── report.py
│   ├── taxonomy.py
│   ├── default_config.yaml
│   └── prompts/
│       ├── tutor/
│       ├── student/
│       ├── scorer/
│       └── taxonomy/
├── prompts/                 historical prompt archives
├── docs/                    design notes, plans, and research docs
├── analysis/                notebooks and exploratory analysis
├── scenarios/               local dataset installs, gitignored
├── data/                    raw/private build inputs, mostly gitignored
├── results/                 run outputs, gitignored
└── tests/tutorsim/
```

## Development

Run the test suite without real API calls:

```bash
uv run --extra dev pytest -q
```

When testing against real model providers, export the provider API keys first.
