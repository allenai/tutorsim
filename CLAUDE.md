# CLAUDE.md

## Project Overview

This project answers: **"How good are AI tutors at tutoring?"** It measures whether AI tutor models can replicate the pedagogical strategies real human tutors use -- specifically when to scaffold for students vs when to push for rigor, and how to build rapport.

Because this is a research project, we are interested in creating REPRODUCIBLE RESULTS.

## Scope conventions

- Keep library-quality code in `tutor_bench/`
- Keep tests in `tests/`
- Experimental scripts in `scripts/` are allowed but are not part of the required CI quality gate
- Large local artifacts should stay out of git (`data/`, `output/`, and most `plans/`)
- Keep `plans/_summary.md` as the persistent developer log
    - Update `plans/_summary.md` when we make a new plan and when we finish implementation of a plan
- Logs go in `logs/`, do not use print statement
- Documentaiton in `docs/`.
- Save plans in `plans/` with the format `YYYY-MM-DD-<descriptive-name>.md`

## Pull requests

- Include tests for behavior changes in `tutor_bench/` when possible
- Keep PRs focused and small
- No mandatory changelog update is required
- Read CONTRIBUTING.md
- Assert there are no secrets, API Keys, credentials, PII or other sensitive data before pushing to remote
- Check that relevant `docs/` are up to date

## Core Principles

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts
- YAGNI — don't build what we don't need yet. Solve the problem in front of us
- Prefer direct, honest feedback over diplomacy
- Value systematic debugging over quick fixes
- Surface architectural discussions before major changes
- Speak up about bad ideas — don't just go along with them

## Working with LLMs

This project is reseraching the capabilities of LLMs. We therefore need to be strategic about how we implement LLM calls. We care a lot about model selection and configuration

- All prompts go in dedicated .md files, never inline
- Ask before selecting which LLM model to use
- Ask before setting model parameters (max tokens, temperature, limits, etc.)
- Always use context7 to fetch up-to-date documentation for libraries, frameworks, and APIs
- Track tokens in / tokens out and cost for all LLM calls

## Test Driven Development
- FOR EVERY NEW FEATURE OR BUGFIX, YOU MUST follow Red / Green Test Driven Development
- ALL TEST FAILURES ARE YOUR RESPONSIBILITY, even if they're not your fault. The Broken Windows theory is real
    - Never delete a test because it's failing. Instead, raise the issue
    - Don't ignore a failing test. If you are SURE that the failing test is not your fault, write a task to `docs/` to fix it
    - Never say "All tests pass" when they do not
- Do not waste time writing very basic / low level tests. Test only custom business logic
- Temporary / early stage work in scripts/ does not to follow red / green tdd
