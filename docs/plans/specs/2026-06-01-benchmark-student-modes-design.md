# Benchmark student modes (port from synth-students)

**Date:** 2026-06-01
**Status:** Design — pending review

## Goal

Replace the benchmark's single hard-default synthetic student prompt with selectable student "modes" ported from Alexis's `synth-students` repo. Make `imitate_example` (her strongest realism mode) the shipping default. Skip trait-based modes for now — they require a separate trait-generator phase and substantial infra.

## Scope

In scope (prompt-only modes):
- `imitate_example` — model is told to imitate the specific real student from the transcript prefix.
- `simple` — generic elementary-school student persona, no example reference.
- `expert` — strong student who makes no mistakes (control / upper-bound persona).
- `paraphrase_with_example` — paraphrase the real student's style, content kept similar.

Out of scope:
- `trait_*` family (cognitive, learning_efficiency, distractedness, affect, all, joined). These require a new trait-generator LLM phase that reads the real transcript and produces a trait description, then conditions the student prompt. Deferred.
- Changes to the tutor prompt.
- Changes to the annotator pipeline.

## File layout

Add a new `v2` benchmark prompt directory so `v1` remains reproducible:

```
prompts/benchmark/
  v1/                              # untouched
    tutor_system.txt
    student_system.txt
  v2/                              # new
    tutor_system.txt               # copy of v1 (no tutor-side changes)
    students/
      imitate_example.txt
      simple.txt
      expert.txt
      paraphrase_with_example.txt
```

## Config

Add one new key under `benchmark.student`:

```yaml
benchmark:
  exchange:
    prompt_version: v2             # was: v1
  student:
    profile: anthropic             # unchanged — selects the model
    mode: imitate_example          # NEW — selects students/{mode}.txt
```

Default behavior change: ships with `prompt_version: v2` and `student.mode: imitate_example`. Acknowledged that this changes benchmark numbers vs prior runs — user explicitly opted in.

Back-compat: if `student.mode` is null or unset, loader falls back to `student_system.txt` for the named `prompt_version`. This keeps v1 runs (and any future single-prompt versions) working without code branching tied to `prompt_version`.

## Code change

Only two files need to change.

### `benchmark/core/exchange.py`

`_build_role_prompt` gains a `student_mode: str | None` parameter. When `role == "STUDENT"` and `student_mode` is provided, load `students/{student_mode}.txt` under the prompt version. Otherwise load the legacy `student_system.txt`. Tutor branch is unchanged.

Both call sites — `run_exchange` (sync) and `run_exchanges_batch` (batch) — accept `student_mode` and pass it through every place they currently pass `prompt_version`. No other plumbing changes.

### `benchmark/run.py`

Read `config["student"].get("mode")` once at the top of `run_benchmark`. Pass it into both `run_exchange` and `run_exchanges_batch`. The resolved value is also recorded in `resolved_models`-style traceability for the run config snapshot (so `results/benchmark/{version}/config.json` captures which student mode ran).

## Prompt adaptations from Alexis's source

Alexis's prompts assume full conversation generation with starter-from-example. Ours is cutpoint continuation. Each ported prompt:

- Drops length-matching / "do not end early" / `convo_length` instructions. We cap with `num_turns=2`.
- Drops her `[STUDENT]:` output-format requirement.
- Keeps our `[NEXT]` multi-message delimiter convention (mentioned in the prompt).
- Substitutes `{student_context}` at runtime (grade/subject), same as v1.
- Replaces "Now you will generate a conversation" with "Continue as the student from the conversation so far."

Mode-specific:
- **`imitate_example`**: the example to imitate IS the transcript prefix the model already sees. Tell it so explicitly. Carry over her aggressive imitation instructions verbatim where they apply: match length, mistakes, spelling, capitalization, conceptual level; rewrite any PII; goal is to be indistinguishable from the human student. Drop her "First describe what the student does" scratchpad step (it's tuned for full-conversation generation, not continuation).
- **`simple`**: persona-only. K-12 student, makes natural mistakes, respond to current turn.
- **`expert`**: persona-only. Strong student, no mistakes, plain text (no LaTeX). Useful as an upper-bound for tutor-friendliness experiments.
- **`paraphrase_with_example`**: tell the model the real student's turns are in the prefix; paraphrase their style/content for the continuation rather than copying.

## Testing

- Unit-level: a smoke test that `_build_role_prompt` loads the correct file per mode and substitutes `{student_context}`.
- End-to-end: a 1-scenario sync-mode run for each of the four modes to confirm the loader works and the model produces non-empty output. Done locally; not committed.
- No automated metric test — student-mode quality is what the benchmark itself measures.

## Out-of-scope follow-ups

- Trait-based modes (`trait_*`) — separate spec when wanted. Requires:
  - A `TraitGenerator` class porting `synth-students/src/students/traits.py`.
  - A per-scenario trait-generation phase (cached, like detections), with its own model in config.
  - Two prompts (trait-generator system + trait-conditioned student).
- CLI flag for `--student-mode` to override config — trivial when wanted.
- Multi-mode runs (one tutor evaluated against many synthetic students) — needs an output-dir layer change. Deferred.
