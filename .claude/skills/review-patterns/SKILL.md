---
name: review-patterns
description: Review code changes against this project's established patterns for S3 data boundaries, LLM usage, storage abstraction, logging, testing, and code style. Use when reviewing a branch before PR or after making changes.
disable-model-invocation: true
argument-hint: "[--staged | module-name]"
---

Review code changes on the current branch against this project's established patterns.

## Setup

1. Determine the diff to review:
   - If `$ARGUMENTS` contains `--staged`, run `git diff --cached`
   - Otherwise run `git diff main...HEAD` to get all changes on this branch
   - If `$ARGUMENTS` contains a module name (e.g. `deidentify`, `normalize`, `receive_pii`), filter the diff to only files under that path
2. Also run `git diff main...HEAD --stat` to get the list of changed files
3. Read the full diff content and the list of changed files before proceeding

If there are no changes, say so and stop.

## Review Checklist

Work through each category below. For each, inspect the diff for violations. Only report findings that are actually present in the diff — do not speculate. Reference specific files and line numbers from the diff.

### 1. Plans & Documentation
If any files under `docs/plans/` are changed:
- Plan filenames are `YYYY-MM-DD-<descriptive-name>.md` — dashes between all parts, lowercase. The descriptive name must be a human-readable summary of what the plan does (e.g. `2026-03-26-factor-iv-storage-refactor.md`). Flag auto-generated-looking slugs, ticket IDs alone, or vague names like `update`, `fix`, `plan-1`. Specs go in `docs/plans/specs/` with the same naming convention.
- New or completed plans must add an entry to [docs/plans/_summary.md](../../../docs/plans/_summary.md) using the project's per-plan format: `### YYYY-MM-DD — [Title](file.md)` followed by `**Goal**:`, `**Status**:`, `**Result**:` lines. Goal is the "so what" (the problem), not a restatement of changes. Do not use a markdown table.
- `docs/status.md` is *current state only* — historical entries belong in `_summary.md`. Flag any change-log content added to `status.md`.

### 2. Logging
The project's logging setup lives in [common/logging_setup.py](../../../common/logging_setup.py). Flag any of:

- New `print()` calls for status, progress, errors, or warnings in non-test code under `annotator/`, `benchmark/`, `common/`, `data/`, or `validation/`. The pattern is `import logging` + `logger = logging.getLogger(__name__)` at module top, then `logger.info/debug/warning/error/exception(...)` at call sites. Existing prints in modules other than `annotator/core/annotate.py` are tolerated (incremental migration), but **net new** prints in any module are a violation.
- Modules that emit logs without declaring a module-level `logger = logging.getLogger(__name__)`. Don't use the root logger directly (`logging.info(...)`) and don't hardcode logger names.
- Calls to `setup_logging()` outside the wired entry points. Allowed callers: [annotator/__main__.py](../../../annotator/__main__.py), [annotator/run.py](../../../annotator/run.py) `main()`, [annotator/core/detect.py](../../../annotator/core/detect.py) `main()`, [annotator/core/annotate.py](../../../annotator/core/annotate.py) `main()`, [annotator/core/label.py](../../../annotator/core/label.py) `main()`, [benchmark/__main__.py](../../../benchmark/__main__.py), [benchmark/run.py](../../../benchmark/run.py) `main()`. Per-pass `core/*.py` mains are listed because `python -m annotator.core.detect` is a documented entry point — without `setup_logging()` there, INFO records propagate to a handler-less root and silently disappear. Library code (everything that's not a `main()` of a wired entry point) must not configure logging.
- Manual `logging.basicConfig()`, `logger.addHandler(...)`, or any `FileHandler`/`StreamHandler` construction in business logic. The handler set is owned by `common/logging_setup.py`.
- Files named `logging.py` anywhere in the tree (shadows the stdlib). Use `logging_setup.py` or another disambiguated name.
- Changes to the log format string, level resolution, or env-var names (`LOG_LEVEL`, `LOG_FILE`, `LOG_REPO_ROOT`) without an accompanying entry in [docs/plans/_summary.md](../../../docs/plans/_summary.md) — these are public contract for ops.
- f-string formatting in log calls when the level might be filtered: prefer `logger.info("count=%d", n)` over `logger.info(f"count={n}")` for hot paths. Treat as a suggestion, not a violation.

### 3. Code Style
- `__main__.py` files should be thin orchestrators: parse args, load config, dispatch to functions. Flag substantial business logic in `__main__.py`, or a `__main__.py` that is a 1 liner passing to a junk drawer `run.py`
- No temporal/historical names: flag any new identifier or filename containing "New", "Old", "Legacy", "Improved", "Enhanced", "Unified", "Refactored", or a bare version number (`v2`, `v3`).

  **Carve-out — experiment iterations:** prompt versions, ground-truth dumps, detection iteration logs, and other artifacts that intentionally coexist at different revisions may use versioned names, but the name must convey *what changed*, not just increment a counter. Prefer a descriptive slug (e.g. `ground_truth_outcome_anchored/`, `prompts/annotator/v5r4_cast_wide_net/`) over bare `v1`/`v2`/`v3` (e.g. `ground_truth_v2/` is opaque the moment you forget what "v2" was). If bare versioning is unavoidable (tight iteration loop), the parent directory must carry a short mapping (in a `README.md`, an adjacent iteration log, or an entry in [docs/plans/_summary.md](../../../docs/plans/_summary.md)) that says what each version represents.
- Import ordering: stdlib, then third-party, then local. Flag out-of-order imports.
- Module length versus file proliferation: 
   - Lightly flag any file that exceeds ~200 lines. Longer files may be OK if they group similar logic. Don't create junk drawer files just to hit an arbitrary file length constraint.
   - Also flag newly created files that may not be needed / may be junk drawers.
   - Ensure that within a file the content is semantically similar, e.g., detect.py should not contain `def run_apply:` if there is also an apply.py     
- No PII in git: scan the diff for anything that looks like real names, email addresses, phone numbers, or student IDs. Ignore test fixtures with obviously fake data.
- This project is well-organized with meaningful filenames names. Long file names or file names so short as to be meaningless are a hint that the code may be an anti-pattern.
- Flag orphan code

### 4. Prompts
- All prompt content lives under [prompts/](../../../prompts/) as `.md` files. Flag new prompts authored inline in Python source (multi-line strings, f-strings constructed at call time, string constants longer than a line or two) or as new `.txt` files — the repo has legacy `.txt` prompts and is migrating to `.md`, so `.txt` is tolerated for edits but not for new files.
- Prompt templates are loaded from disk, then filled with `.replace()` / `.format()` at call time (see the existing pattern in [annotator/core/annotate.py](../../../annotator/core/annotate.py) `load_prompt()` and `build_analysis_entries()`). Flag code that defines prompt bodies in Python and loads substitution values from a file — that's backwards.

### 5. Testing
- New features or bug fixes should have corresponding test changes. If the diff adds substantial new logic but no test files are modified, flag it.
- Check that no tests are deleted. Modified tests are fine; removed `def test_*` functions are a violation.
- Tests should cover custom business logic, not trivial getters/setters.

### 6. Idempotency & Resumability
For any new long-running operation (LLM batch, multi-conversation pass, scenario generation, etc.):
- Must be safe to re-run. Check for "if exists, skip" before expensive work — use `annotator_result_exists()` for annotator outputs and `load_benchmark_result()` + completion-flag checks for benchmark outputs (see the existing pattern around lines 123-156 in [benchmark/run.py](../../../benchmark/run.py)).
- All result writes must go through the storage layer ([annotator/core/storage.py](../../../annotator/core/storage.py)) — `save_annotator_result()`, `save_benchmark_result()`, `write_json()`. Flag any direct `open(..., "w")` or `json.dump()` to a results path; the storage layer does atomic write-and-rename to avoid corrupted files on crash mid-write.
- LLM JSON parsing: model outputs occasionally include leading prose before the JSON object. Use the existing `parse_and_merge()` style (try/except `json.JSONDecodeError`, record errors, continue) — see lines 147-179 in [annotator/core/annotate.py](../../../annotator/core/annotate.py). Flag bare `json.loads(response.text)` on raw model output without error handling.
- Per-call token usage must be recorded on every result (input_tokens, output_tokens, total_tokens). This is the project's only cost-tracking mechanism. Flag any new LLM call path that drops the usage fields.

### 7. Security
- No PII or sensitive information should be stored in git

## Output Format

Structure your response as follows:

### Pattern Review: `<branch name>`

**Files reviewed:** (list from --stat)

**Violations** (must fix):
For each violation, make a numbered list:
1. `file:line` — **[Category]** Description of the violation.

**Suggestions** (consider fixing):
For each suggestion, make a numbered list:
2. `file:line` — **[Category]** Description of the suggestion.

Limit nits to 5. If you see more than 5 nits, say "and N+ similar Nits".

**Compliant patterns observed:**
Briefly note (2-3 bullets max) patterns that are correctly followed — this confirms the review was thorough.

If there are no violations and no suggestions, say:
> No issues found. All changed code follows established patterns.

Keep the review concise. Do not explain what the patterns are — just flag deviations.
