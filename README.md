# tutor-bench

[![CI](https://github.com/allenai/tutor-bench/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/tutor-bench/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A lightweight benchmarking and evaluation framework for AI tutoring systems.

## What this repo optimizes for

- Minimal, readable codebase for NLP researchers.
- Fast contributor loop with a small quality gate: `ruff`, `pyright`, `pytest`.
- Local experiment artifacts are intentionally untracked by default.

## Installation

```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick usage

```python
from tutor_bench import Annotator, Evaluator

annotator = Annotator(model="gpt-4")
annotations = annotator.process_transcripts("path/to/transcripts.jsonl")
annotations.save("path/to/annotations.jsonl")

evaluator = Evaluator()
metrics = evaluator.evaluate(
    transcripts="path/to/transcripts.jsonl",
    annotations="path/to/annotations.jsonl",
)
print(metrics.summary())
```

## Development commands

```bash
# Run all local checks
make run-checks

# Individual checks
make lint
make format-check
make typecheck
make test-fast
```

## Repository layout

```text
tutor-bench/
├── tutor_bench/           # library code
├── tests/                 # test suite and fixtures
├── scripts/               # utility scripts
├── configs/               # config assets
├── plans/_summary.md      # developer log summary
├── .github/workflows/     # CI config
└── pyproject.toml         # packaging + tool config
```

## Notes on local artifacts

The following directories are ignored by default and intended for local experimentation only:

- `data/`
- `output/`
- most of `plans/` (except `plans/_summary.md`)

## Testing

```bash
python -m pytest -q tests -m "not slow and not integration and not gpu"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the minimal contributor workflow.
