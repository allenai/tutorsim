# AGENTS.md

TutorSim is a language-model benchmarking framework. It replays real
human-student tutoring transcripts from expert-annotated cut points using LM
tutors and simulated students, scores the generated continuations, and reports
the results. See README.md for project goals, usage, and the full repository
layout.

## Layout

- `src/tutorsim/` — the installable benchmark runtime (`tutorsim` CLI): runs
  and scores a released dataset. Everything a run needs ships in the package.
- `tutorsim_build/` — maintainer-only dataset construction (`tutorsim-build`
  CLI): ground-truth building and release writing.
- `analysis/` — paper notebooks, plots, and the action taxonomy.
- `tests/` — `tutorsim/` (runtime), `tutorsim_build/`, `analysis/`.

Import rule: build and analysis code may import `tutorsim`; the runtime never
imports them.

## Invariants

- This is research code: reproducibility comes first. Doing it right beats
  doing it fast.
- Benchmark-defining code lives in `tutorsim_build/` — `moments_build.py`,
  `moments.schema.json`, and the frozen `balanced_520_ids.json` used in the 
  June 2026 Preview paper. These determine what the benchmark *is*. 
  Changes there can change published results; treat them with care.
- The `src/tutorsim` runtime consumes released datasets only. It never constructs, filters, or regenerates benchmark data (including student traits — those are frozen
  in the release).
- `data/` and `results/` are gitignored and must stay that way. Never commit
  datasets, transcripts, or run outputs. The datasets are de-identified, but still take care to never commit anything containing student data.
- All LLM prompts live under consolidated `prompts/{my prompt}.md` directories as standalone markdown files, never inline in Python source. Templates are loaded from disk and filled at call time.
- Every LLM call path records token usage (input/output/total tokens) — this
  is the project's cost-tracking mechanism.
- Never make choices about which language model to use or language model configuration for any API call without consulting the user; this is a language model benchmark, the choice of model matters

## Tests

```bash
pytest tests/tutorsim -q   # runtime only (needs the [dev] extra)
pytest tests -q            # full suite (needs [dev,build-dev,taxonomy]; missing extras skip)
```

The suite runs without real API calls. New features and bug fixes need
accompanying tests; test business logic, not boilerplate.
