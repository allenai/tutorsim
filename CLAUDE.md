# CLAUDE.md

Note: this agent handoff file predates the `tutorsim` package refactor. It may
still contain legacy references to the old `annotator/` and `benchmark/`
package layout. The supported user-facing workflow is documented in
`README.md`; current runtime code lives under `src/tutorsim/`.

## Project Overview

This project answers: **"How good are AI tutors at tutoring?"** It measures whether AI tutor models can replicate the pedagogical strategies real human tutors use -- specifically scaffolding (guiding students to answers without giving them away) and rapport (building trust, reading emotions, making learning feel safe).

The system has two pipelines:
1. **Annotator pipeline** (`annotator/`) -- uses LLMs to replicate what human annotators do (detect moments, analyze tutor strategies, label effectiveness). Validated to exceed human inter-rater agreement.
2. **Benchmark pipeline** (`benchmark/`) -- fully ground-truth-free. Runs synthetic detection to find key moments + cut points in transcripts, has an AI tutor continue from the cut point with a synthetic student, then scores the AI's pedagogical quality using the annotator pipeline with 3 calibrated styles.

Because this is a research project, we are interested in creating REPRODUCIBLE RESULTS.

### Project Docs

Memory lives in `docs/`. Key docs:

- `@docs/status.md` — Read at the start of every session. Contains current project state and next steps.
- `@docs/lessons_learned.md` — Read when encountering bugs or unexpected behavior. May contain a fix we've already found.
- `@docs/plans/_summary.md` - A log of our plans.

Keep all docs concise and current. They're working documents, not archives — overwrite stale content rather than letting files grow indefinitely. Prune what's no longer relevant.

**_summary.md**

Add a record each time we start a new plan, and each time we complete a plan. Add a title, the goal of the plan (the "so what" behind the plan, not a list of changes), the status, and the result.

**Status** — the structured handoff between sessions. Update it at the end of every session, when the context alert fires, and after meaningful milestones. Include: current branch, what was accomplished, what's in progress, what's next, any open decisions or blockers.

**Lessons Learned** — when we catch a mistake or discover unexpected behavior:

1. Check if the lesson already exists before adding a duplicate.
2. Add an entry with: what went wrong, why it happened, and what the fix was.
3. Keep entries to a few lines each.
4. If an existing entry is wrong or incomplete, update it.

Write lessons as soon as they're learned — don't wait until end of session.

## Core Principles

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- YAGNI — don't build what we don't need yet. Solve the problem in front of us.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive — abandon it only if it's technically wrong.
- Prefers direct, honest feedback over diplomacy.
- Values systematic debugging over quick fixes.
- Wants architectural discussions before major changes.
- Speak up about bad ideas — don't just go along with them.


## Working with LLMs
- This project is reseraching the capabilities of LLMs. We therefore need to be strategic about how we implement LLM calls.
- All prompts go in `prompts/` and .md files
- Ask before selecting which LLM model to use, and before setting model parameters (max tokens, temperature, etc.). I care a lot about model selection and configuration.
- Don't limit tokens or context sent to LLMs in early prototypes. We will manage long context purposefully if and when it becomes an issue.
- Always use context7 to fetch up-to-date documentation for libraries, frameworks, and APIs — both when writing our own code and when building LLM prompts or tool configurations.
- Track tokens in / tokens out and cost for all LLM calls

## Git Workflow

After every meaningful change (bug fix, feature, config update), commit. Don't let work pile up uncommitted.

- Check for uncommitted code before starting any new plan. Ask how to handle it.
- When starting work without a clear branch for the current task, create a WIP branch.
- When you think we're ready to PR a feature, ask.
- Don't merge PRs — we have a separate workflow for this.

## IMPORTANT: Security & Sensitive Data

### Before EVERY Commit

Review the full staged diff for secrets, credentials, PII, and any sensitive data. Think about what counts as sensitive in the specific project context — don't just pattern-match a generic checklist. Check for accidental data files, logs, or database dumps. If ANY concern is found, stop and flag it before committing.

### Repository Hygiene

- When writing or updating `.gitignore`, explore the full repository structure first (`find`, `ls -R`, `tree`, etc.). Do not guess — understand what actually exists before deciding what to ignore.
- Data directories should be gitignored by default. Use your judgment about what constitutes data, output, or artifacts based on the actual repo contents.
- API keys must never appear anywhere — not in code, config, logs, or output. Use environment variables or secret managers.

## Context Management

You have context awareness — you receive token budget updates automatically. Alert the user at these thresholds:

- **~60k tokens** — Mention it briefly. Note we're warming up.
- **~120k tokens** — Flag it clearly. Suggest wrapping up the current task and updating `docs/status.md`.
- **~240k tokens** — Strong warning. Update `docs/status.md` with full handoff state. Recommend `/compact` or `/clear`.
- **~480k tokens** — Stop and write the handoff. Don't start new work.

Do NOT wait for auto-compaction. Proactive handoff > silent context loss.

## Test Driven Development
- FOR EVERY NEW FEATURE OR BUGFIX, YOU MUST follow Red / Green Test Driven Development
- ALL TEST FAILURES ARE YOUR RESPONSIBILITY, even if they're not your fault. The Broken Windows theory is real.
    - Never delete a test because it's failing. Instead, raise the issue.
    - Don't ignore a failing test. If you are SURE that the failing test is not your fault, write a task to `docs/` to fix it
    - Never say "All tests pass" when they do not
- Do not waste time writing very basic / low level tests. Test only custom business logic.
