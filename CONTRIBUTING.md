# Contributing to tutor-bench

Thanks for contributing. This project prioritizes a minimal, research-friendly workflow.

## Development setup

```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Required quality checks

Before opening a PR, run:

```bash
make run-checks
```

Equivalent individual checks:

```bash
python -m ruff check tutor_bench tests
python -m ruff format --check tutor_bench tests
python -m pyright tutor_bench
python -m pytest -q tests -m "not slow and not integration and not gpu"
```

## Scope conventions

- Keep library-quality code in `tutor_bench/`.
- Keep tests in `tests/`.
- Experimental scripts in `scripts/` are allowed but are not part of the required CI quality gate.
- Large local artifacts should stay out of git (`data/`, `output/`, and most `plans/`).
- Keep `plans/_summary.md` as the persistent developer log.

## Pull requests

- Include tests for behavior changes in `tutor_bench/` when possible.
- Keep PRs focused and small.
- No mandatory changelog update is required.

## Need help

Open a GitHub issue: https://github.com/allenai/tutor-bench/issues
