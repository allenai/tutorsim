---
name: review-patterns
description: Review code changes against this project's established patterns for LLM usage, logging, testing, and code style. Use when reviewing a branch before PR or after making changes.
disable-model-invocation: true
argument-hint: "[--staged | module-name]"
---

Review code changes on the current branch against this project's established patterns.

## Setup

1. Determine the diff to review:
   - If `$ARGUMENTS` contains `--staged`, run `git diff --cached`
   - Otherwise run `git diff main...HEAD` to get all changes on this branch
   - If `$ARGUMENTS` contains a module name (e.g. `annotator`, `evaluator`, `toolkit`), filter the diff to only files under that path
2. Also run `git diff main...HEAD --stat` to get the list of changed files
3. Read the full diff content and the list of changed files before proceeding

If there are no changes, say so and stop.

## Review Checklist

Work through each category below. For each, inspect the diff for violations. Only report findings that are actually present in the diff — do not speculate. Reference specific files and line numbers from the diff.

### 1. Plans & Documentation
- New or completed plans must add an entry to `plans/_summary.md` using the project's per-plan format: `## Plan [###] — [Title]` followed by `**Plan**:`. `**Script**:` and `**Output**:` are optional. `**Goal**` is the "so what" (the problem), not a restatement of changes, similarly `**Result**` captures the bottom line impact of the changes, not a list of changes.

### 2. Logging
Flag any of:

- New `print()` calls 
- Modules that emit logs without declaring a module-level `logger = logging.getLogger(__name__)`. Don't use the root logger directly (`logging.info(...)`) and don't hardcode logger names.
- Manual `logging.basicConfig()`, `logger.addHandler(...)`, or any `FileHandler`/`StreamHandler` construction in business logic.
- Changes to the log format string, level resolution, or env-var names (`LOG_LEVEL`, `LOG_FILE`, `LOG_REPO_ROOT`) without an accompanying entry in `plans/_summary.md` — these are public contract for ops.

### 3. Code Style
- No temporal/historical names: flag any new identifier or filename containing "New", "Old", "Legacy", "Improved", "Enhanced", "Unified", "Refactored", or a bare version number (`v2`, `v3`).

  **Carve-out — experiment iterations:** prompt versions, ground-truth dumps, evaluation iteration logs, and other artifacts that intentionally coexist at different revisions may use versioned names, but the name must convey *what changed*, not just increment a counter. Prefer a descriptive slug (e.g. `prompts/annotator/v5r4_cast_wide_net/`) over bare `v1`/`v2`/`v3` (e.g. `ground_truth_v2/` is opaque the moment you forget what "v2" was). If bare versioning is unavoidable (tight iteration loop), the parent directory must carry a short mapping (in `docs/`, an adjacent iteration log, or an entry in `plans/_summary.md`) that says what each version represents.
- Import ordering: stdlib, then third-party, then local. Flag out-of-order imports.
- Module length versus file proliferation: 
   - Lightly flag any file that exceeds ~200 lines. Longer files may be OK if they group similar logic. Don't create junk drawer files just to hit an arbitrary file length constraint.
   - Also flag newly created files that may not be needed / may be junk drawers.
   - Ensure that within a file the content is semantically similar, e.g., detect.py should not contain `def run_apply:` if there is also an apply.py     
- No PII or secrets in git: scan the diff for anything that looks like real names, email addresses, phone numbers, API keys, or credentials. Ignore test fixtures with obviously fake data.
- This project is well-organized with meaningful filenames names. Long file names or file names so short as to be meaningless are a hint that the code may be an anti-pattern.
- Flag orphan code

### 4. Prompts
- All prompt content lives as `.md` files in `tutor_bench/prompts/` or `scripts/prompts`. Flag new prompts authored inline in Python source (multi-line strings, f-strings constructed at call time, string constants longer than a line or two).

### 5. Testing
- New features or bug fixes should have corresponding test changes. If the diff adds substantial new logic but no test files are modified, flag it.
- Check that no tests are deleted. Modified tests are fine; removed `def test_*` functions are a violation.
- Tests should cover custom business logic, not trivial getters/setters.
- Temporary scripts in `scripts/` do not need testing.

### 6. Idempotency & Resumability
For any new long-running operation (LLM batch, multi-conversation pass, scenario generation, etc.):
- Must be safe to re-run. Check for "if exists, skip" before expensive work
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
1. `file:line` — **[Category]** Description of the suggestion.

Limit nits to 5. If you see more than 5 nits, say "and N+ similar Nits".

**Compliant patterns observed:**
Briefly note (2-3 bullets max) patterns that are correctly followed — this confirms the review was thorough.

If there are no violations and no suggestions, say:
> No issues found. All changed code follows established patterns.

Keep the review concise. Do not explain what the patterns are — just flag deviations.
