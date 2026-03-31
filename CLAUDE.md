# CLAUDE.md

## Core Principles

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- YAGNI — don't build what we don't need yet. Solve the problem in front of us.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive — abandon it only if it's technically wrong.
- Prefers direct, honest feedback over diplomacy.
- Values systematic debugging over quick fixes.
- Wants architectural discussions before major changes.
- Speak up about bad ideas — don't just go along with them.
- Ask before selecting which LLM model to use, and before setting model parameters (max tokens, temperature, etc.). I care a lot about model selection and configuration.
- Don't limit tokens or context sent to LLMs in early prototypes. We will manage long context purposefully if and when it becomes an issue.
- Always use context7 to fetch up-to-date documentation for libraries, frameworks, and APIs — both when writing our own code and when building LLM prompts or tool configurations.


## Git Workflow

After every meaningful change (bug fix, feature, config update), commit. Don't let work pile up uncommitted.

- Check for uncommitted code before starting any new plan. Ask how to handle it.
- When starting work without a clear branch for the current task, create a WIP branch.
- Ask before pushing code.
- When you think we're ready to PR a feature, ask.
- Don't merge PRs — we have a separate workflow for this.

## IMPORTANT: Security & Sensitive Data

### Before EVERY Commit

Review the full staged diff for secrets, credentials, PII, and any sensitive data. Think about what counts as sensitive in the specific project context — don't just pattern-match a generic checklist. Check for accidental data files, logs, or database dumps. If ANY concern is found, stop and flag it before committing.

### Repository Hygiene

- When writing or updating `.gitignore`, explore the full repository structure first (`find`, `ls -R`, `tree`, etc.). Do not guess — understand what actually exists before deciding what to ignore.
- Data directories should be gitignored by default. Use your judgment about what constitutes data, output, or artifacts based on the actual repo contents.
- API keys must never appear anywhere — not in code, config, logs, or output. Use environment variables or secret managers.

### New Repository Checklist

When a new repo is introduced, verify before doing any work:

1. Is the repo private?
2. Is it in the correct org/tenant (user's own, client's, etc.)? Ask if unclear.
3. Does `.gitignore` exist and cover secrets, data, and environment files?
4. Are there any existing secrets already committed? If so, flag immediately.

## Project Setup

### Project-Specific Context

When starting a new project, build out this section together. This is the stuff Claude can't infer from reading code:

- **Why** — the purpose, the user, the problem being solved. This is the north star. When ambiguous decisions come up, this is what we use to break ties.
- **What** — the tech stack, what the major components are and what they do. Not a folder listing (explore the repo for that), but the reasoning behind structural decisions. What's legacy vs. active? What patterns are intentional? What looks wrong but is correct?
- **How** — the workflows, gotchas, and non-obvious constraints. Not exact commands (figure those out from the repo), but how things fit together, external dependencies, and things that break silently.

Don't document things here that can be discovered by exploring the repo. This section is for context that lives in people's heads, not in files.


## Project Overview

This project answers: **"How good are AI tutors at the human side of teaching?"** It measures whether AI tutor models can replicate the pedagogical strategies real human tutors use -- specifically scaffolding (guiding students to answers without giving them away) and rapport (building trust, reading emotions, making learning feel safe).

The dataset is 104 real K-12 math tutoring transcripts. Human expert annotators labeled "key moments" as effective, partial, or ineffective.

The system has two pipelines:
1. **Annotator pipeline** (`annotator/`) -- uses LLMs to replicate what human annotators do (detect moments, analyze tutor strategies, label effectiveness). Validated to exceed human inter-rater agreement.
2. **Benchmark pipeline** (`benchmark/`) -- fully ground-truth-free. Runs synthetic detection to find key moments + cut points in transcripts, has an AI tutor continue from the cut point with a synthetic student, then scores the AI's pedagogical quality using the annotator pipeline with 3 calibrated styles.

Because this is a research project, we are interested in creating REPRODUCIBLE RESULTS. 

### Project Docs

Every project should have a `docs/` folder. When starting a new project, create it and these files if they don't exist. These files are managed by you (Claude), committed to git, and loaded on demand:

- `@docs/status.md` — Read at the start of every session. Contains current project state and next steps.
- `@docs/lessons_learned.md` — Read when encountering bugs or unexpected behavior. May contain a fix we've already found.

Keep all docs concise and current. They're working documents, not archives — overwrite stale content rather than letting files grow indefinitely. Prune what's no longer relevant.

**Status** — the structured handoff between sessions. Update it at the end of every session, when the context alert fires, and after meaningful milestones. Include: current branch, what was accomplished, what's in progress, what's next, any open decisions or blockers.

**Lessons Learned** — when we catch a mistake or discover unexpected behavior:

1. Check if the lesson already exists before adding a duplicate.
2. Add an entry with: what went wrong, why it happened, and what the fix was.
3. Keep entries to a few lines each.
4. If an existing entry is wrong or incomplete, update it.

Write lessons as soon as they're learned — don't wait until end of session.

## Context Management

You have context awareness — you receive token budget updates automatically. Alert the user at these thresholds:

- **~60k tokens** — Mention it briefly. Note we're warming up.
- **~120k tokens** — Flag it clearly. Suggest wrapping up the current task and updating `docs/status.md`.
- **~240k tokens** — Strong warning. Update `docs/status.md` with full handoff state. Recommend `/compact` or `/clear`.
- **~480k tokens** — Stop and write the handoff. Don't start new work.

Do NOT wait for auto-compaction. Proactive handoff > silent context loss.