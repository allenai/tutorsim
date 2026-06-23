# Dynamic Exchange Length -- Design

*Status: spec / 2026-06-08*

## Goal

Let the AI tutor decide when to end a benchmark scenario by emitting an `[END]` token at the end of its reply, capped by a hard `max_turns: 100` ceiling on generated turns. Replace the current fixed `num_turns: 2` exchange. New prompt version `v3` carries the `[END]` instruction; v1/v2 remain frozen.

## Why

The current benchmark always runs exactly `num_turns: 2` rounds (tutor + student + tutor). That's an arbitrary length: some moments resolve in one tutor reply, others need a half-dozen back-and-forths to play out. Fixed length under-evaluates short scenarios (no breathing room) and over-evaluates long ones (forces filler). Let the tutor judge when the scenario is done.

## Behavior

### Stop signal

The tutor signals end-of-scenario by ending its last message with `[END]` on its own (with surrounding whitespace allowed). The scenario stops after that tutor turn — no student reply follows. The `[END]` token is stripped before the message is saved; the wrap-up text before it stays as the final tutor turn.

Example tutor reply that ends the scenario:
```
Great work figuring that out! Let me know if you want to try another one.
[END]
```
Saved as a single tutor turn: `"Great work figuring that out! Let me know if you want to try another one."`

If `[END]` appears in the middle of a message (rare), strip it and still treat the scenario as ended.

If the tutor never emits `[END]`, the scenario runs until the hard `max_turns` ceiling.

### Hard cap

`max_turns` (default `100`) counts *generated* turns -- prefix turns are excluded. Each round adds one tutor + one student turn (modulo multi-message turns split by `[NEXT]`, which add multiple turns each). Loop terminates when `len(exchange.generated_turns) >= max_turns` regardless of `[END]`.

No minimum turn count -- the tutor may end on its first reply.

### Batch mode

Scenarios that emit `[END]` (or hit `max_turns`) are removed from `active_ids` and don't participate in subsequent rounds. The loop continues until `active_ids` is empty or the maximum number of rounds is reached. The maximum round count is `ceil(max_turns / 2)` (each round contributes up to 2 turns: tutor + student).

## Prompt: `prompts/benchmark/v3/`

Create `prompts/benchmark/v3/` as a verbatim copy of `prompts/benchmark/v2/`, then append two paragraphs to `v3/tutor_system.txt` (only that file changes; `v3/students/*.txt` are byte-identical to v2):

```
When the moment has played out and there's nothing useful left to add (the student
has reached an answer they can run with, the misconception is resolved, or the
problem is finished), end your final message with `[END]` on its own line.
Include a brief, natural wrap-up before the token -- e.g. "Great work, let me
know if you want to try another. [END]". Don't use `[END]` until the scenario
genuinely feels resolved; it's fine for some moments to take several exchanges.
```

The default `prompt_version` in `config.yaml` flips to `v3`. v1 and v2 are unchanged.

## Code changes

### `benchmark/core/exchange.py`

**New helper:**

```python
END_TOKEN = "[END]"

def _check_end_token(text: str) -> tuple[str, bool]:
    """Strip [END] token from text and report whether it was present.

    The token may appear at the end of the message (most common) or mid-text
    (rare); either way the scenario should end. Whitespace around the token
    is stripped.
    """
    if END_TOKEN not in text:
        return text, False
    cleaned = text.replace(END_TOKEN, "").rstrip()
    return cleaned, True
```

**Sync mode (`run_exchange`):** loop replaces `for i in range(num_turns)`:

```python
ended = False
while not ended and len(exchange.generated_turns) < max_turns:
    # Tutor turn
    response = tutor_client.generate(...)
    text, ended = _check_end_token(response.text)
    messages = _split_messages(text) or (["..."] if not ended else [])
    running_transcript, next_turn_num = _append_turns(
        exchange, messages, "TUTOR", running_transcript, next_turn_num,
    )

    if ended or len(exchange.generated_turns) >= max_turns:
        break

    # Student turn
    response = student_client.generate(...)
    messages = _split_messages(response.text) or ["..."]
    running_transcript, next_turn_num = _append_turns(
        exchange, messages, "STUDENT", running_transcript, next_turn_num,
    )
```

Note the special case: if `[END]` is the entire tutor reply (no wrap-up text), `_split_messages("")` returns `[]` and no tutor turn is appended -- avoids saving an empty turn. If there's wrap-up text, it's saved normally.

`num_turns` parameter is renamed to `max_turns`.

**Batch mode (`run_exchanges_batch`):** the outer `for round_num in range(num_turns)` becomes `for round_num in range(ceil(max_turns / 2))`, with early-exit when `active_ids` is empty. Tutor result processing strips `[END]`, and IDs are removed from `active_ids` when ended OR when `len(exchanges[sid].generated_turns) >= max_turns`. Student result processing similarly checks the max-turns ceiling.

### `benchmark/run.py`

Threads `max_turns` from config to `run_exchange` / `run_exchanges_batch` instead of `num_turns`.

### `config.yaml`

```yaml
benchmark:
  exchange:
    max_turns: 100        # was: num_turns: 2
    prompt_version: v3    # was: v2 (v3 = v2 + [END] instruction in tutor prompt)
```

No back-compat: old config files that still say `num_turns` will fail config validation (intentional; surfaces the migration).

## Tests (`tests/`)

Extend `tests/test_benchmark_student_modes.py` or add `tests/test_benchmark_exchange_dynamic.py`:

- `_check_end_token`: trailing token stripped; mid-text token stripped + ended; absent token returns `ended=False` + text unchanged.
- `run_exchange`: tutor emitting `[END]` on first reply ends the exchange after exactly that tutor turn (no student turn after; `len(generated_turns)` matches the tutor's message count).
- `run_exchange`: tutor never emits `[END]`; exchange runs until `len(generated_turns) >= max_turns`.
- `run_exchange`: `[END]` token alone (no wrap-up) does not add an empty tutor turn.
- `run_exchanges_batch`: one scenario emits `[END]` round 1, another doesn't; the first drops out of active_ids; the second continues until max_turns.

(Implementer can mock `ModelClient.generate` with side_effect lists per scenario, matching the existing test patterns in `test_benchmark_student_modes.py`.)

## What's NOT changing

- Student-side loop: only the tutor can end a scenario.
- Scoring, annotation, scenario extraction.
- v1/v2 prompts (frozen for reproducibility). v1/v2 runs will always hit `max_turns` since they don't know about `[END]`.

## Out of scope

- Student-triggered stops (e.g., student says "thanks bye").
- Smarter judge for "is the scenario actually resolved" -- relying on the tutor's own judgment + max cap is enough for a first pass.
- Per-scenario or per-style overrides of `max_turns`.

## Risk / open questions

- **Tutors might emit `[END]` too eagerly,** ending scenarios after a one-line reply that doesn't actually exercise the pedagogical strategy we want to score. Mitigations: the prompt explicitly says "don't use `[END]` until the scenario genuinely feels resolved." If results show pathological short exchanges, we can revisit (add min_turns, tighten prompt language, or fall back to judge-based termination).
- **Tutors might never emit `[END]`** and always hit the 100-turn cap. Cost concern at scale. Smoke test after implementation: run 5-10 scenarios in sync mode and inspect the distribution of generated-turn counts before committing to a large batch run.
- **`[END]` could collide with legitimate tutor text** (e.g. tutor literally typing "[END]" as part of an explanation). Unlikely in practice; if it surfaces, switch to a less-likely token like `<<END-SCENARIO>>`.
