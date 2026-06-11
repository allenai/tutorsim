# Oracle Tutor Mode -- Design

*Status: spec / 2026-06-10*

## Goal

Add a tutor mode where the AI sees the full real conversation -- including
post-cut human turns -- and is instructed to mimic the real human tutor as
closely as possible. Functions as a control / ceiling cell in the experimental
matrix: "if an AI tutor had perfect oracle access to what the real tutor did,
how close to the real continuation can it get?"

## Why

The benchmark currently runs a single tutor variant (the v5 simple prompt).
Without a ceiling condition, we have no upper bound on what's achievable on
this task with this model -- only whatever score the simple prompt produces.
Oracle gives us that bound: if oracle scores far above simple, the headroom
is real and prompt iteration / fine-tuning is worth pursuing. If oracle barely
beats simple, the ceiling is low and we're chasing diminishing returns.

It also sets up cells C / D in the planned 4-cell matrix:
- C: oracle tutor x "best" student (per A/B comparison)
- D: prompt-maxed rigor/scaffolding-aware tutor x "best" student

## Config

Add a new sibling block to `benchmark.student`:

```yaml
benchmark:
  tutor:
    mode: null        # null | oracle (default null = standard v5 tutor)
```

Default stays null (existing v5 behavior unchanged). Opt in by setting
`mode: oracle` for the oracle cell.

## Prompts

New file: `prompts/benchmark/v5/tutors/oracle.txt`. Substitutes
`{student_context}` (same as today) and `{reference_transcript}` (new --
the real human turns from after `cut_turn`, formatted the same way as
`transcript_prefix`).

Content sketch (final wording to be drafted during implementation, but the
spec fixes the structure):

```
You are an online tutor in a live tutoring session with a K-12 student.

## Goal
The conversation continued after this point in the real session. Your task
is to continue the conversation as the tutor, matching the real tutor's
style, strategy, length, register, and pedagogical moves as closely as
possible.

## Context
{student_context}

## What the real tutor did from this point on
{reference_transcript}

## Sending multiple messages
Use [NEW_MESSAGE] on its own line to split your reply into multiple messages.

## Moving on
Return [NEXT_PROBLEM] when the real tutor would have moved on. Our system
will end the replay there.
```

Default `tutor_system.txt` is unchanged; oracle is its own file under
`tutors/`.

## Code change

### `_build_role_prompt` (`benchmark/core/exchange.py`)

Two new optional kwargs (parallel to the student-side trait wiring):

```python
def _build_role_prompt(
    role: str,
    transcript_prefix: str,
    extra: str,
    student_context: str,
    prompt_version: str = "v1",
    student_mode: str | None = None,
    scenario=None,
    trait_client=None,
    trait_model: str | None = None,
    tutor_mode: str | None = None,                   # NEW
    reference_transcript: str | None = None,         # NEW
) -> tuple[str, str]:
```

When `role == "TUTOR"` and `tutor_mode` is set:

1. Load `tutors/{tutor_mode}.txt` instead of `tutor_system.txt`.
2. Substitute `{reference_transcript}` (raises if it's None, since the
   placeholder will be in the prompt).

The reference goes into the cacheable head, so per-scenario caching still
hits.

### `run_exchange` / `run_exchanges_batch`

Both grow two new kwargs:

```python
tutor_mode: str | None = None,
transcripts: dict[str, dict] | None = None,
```

Just before the tutor turn, if `tutor_mode == "oracle"`, compute the
reference once per scenario (memoize on a per-call dict):

```python
def _build_reference_transcript(conversation: dict, cut_turn: int) -> str:
    lines = []
    for turn in conversation.get("turns", []):
        if turn["turn_number"] <= cut_turn:
            continue
        n = turn["turn_number"]
        lines.append(f"Turn {n}. {turn['role']}: {turn['text']}")
    return "\n".join(lines)
```

Pass that into `_build_role_prompt`.

In sync mode, compute reference once before the loop. In batch mode,
compute reference once per scenario at the top of the function and stash
in a dict keyed by `scenario_id`.

### `benchmark/run.py`

Read `config.get("tutor", {}).get("mode")`. When `tutor_mode == "oracle"`,
unconditionally load transcripts (today they're only loaded conditionally
for `with_screenshots`) and pass both `tutor_mode` and `transcripts`
through to the two call sites.

Guard: if `tutor.mode == "oracle"` and transcripts can't be loaded, raise
a clear error at the top of `run_benchmark` rather than silently failing
inside the exchange loop.

## Tests

In `tests/test_benchmark_oracle_tutor.py` (new):

- `_build_role_prompt(role="TUTOR", tutor_mode="oracle",
  reference_transcript="<text>")` loads `tutors/oracle.txt`, substitutes
  `{reference_transcript}`, places the reference in the head (not the
  tail).
- Missing-reference guard: `tutor_mode="oracle"` + `reference_transcript=None`
  raises a clear `ValueError`.
- `_build_reference_transcript(conversation, cut_turn=N)` returns the
  formatted post-cut turns and nothing before / equal to N.
- `run_exchange` with `tutor_mode="oracle"` and a `transcripts` dict
  produces a tutor prompt whose `cacheable_prefix` contains the post-cut
  reference.
- `run_exchange` with `tutor_mode="oracle"` and `transcripts=None` raises
  a clear error.
- `run_exchange` with `tutor_mode=None` (legacy) ignores `transcripts` --
  no reference substituted, default `tutor_system.txt` loaded.

## What's NOT changing

- Standard v5 tutor (`tutor.mode = null`) keeps shipping; oracle is opt-in.
- Student side untouched.
- `[END]` / `[NEXT_PROBLEM]` / `max_turns` / `ended_via` semantics same.
- Prompt caching wiring untouched -- reference lives in the head, hits
  cache on round 2+.
- Annotator / labeller pipelines unaffected.

## Risk / open questions

- **Reference-transcript token weight.** Late-cut scenarios may have a
  long post-cut continuation -- e.g. a cut at turn 22 in a 297-turn
  conversation includes ~275 turns of reference. This balloons the
  cacheable head from ~10k tokens (prefix) to ~80k+ tokens (prefix +
  reference). Cache savings still apply per round, but the cache-creation
  cost on round 1 is large. Worth noting; mitigation if it bites: cap the
  reference at some lookahead window (e.g. next 30 turns) -- defer until
  observed.
- **Mimicry vs spirit.** Oracle is told to mimic style + pedagogy, not
  copy verbatim. The annotator pipeline scores pedagogical effectiveness,
  not surface-level similarity to the reference. If mimicry is too literal
  (model just rephrases real turns), scoring becomes uninformative. If
  observed, tighten the prompt to encourage "in the spirit of" rather
  than "as close as possible."
- **Cell C "best student" dependency.** The oracle cell is useful on its
  own as a tutor ceiling, regardless of which student it pairs with. But
  the matrix calls for "oracle x best student" where "best" comes from
  Cell A/B comparison. That's a sequencing dependency outside this spec;
  noted for context.
- **Annotator may flag oracle as "too perfect"** -- if the AI essentially
  reproduces the real tutor, the SAR fields become trivially "effective."
  This is what the ceiling test is supposed to surface; not a bug.
