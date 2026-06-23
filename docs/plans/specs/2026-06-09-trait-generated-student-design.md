# Trait-Generated Student Mode -- Design

*Status: spec / 2026-06-09*

## Goal

Add a new synthetic student mode where, per scenario, a generator LLM reads
only the **transcript prefix** (turns 1..cut_turn) and writes a student persona
description. The Phase 1 student then embodies that persona. The generator MUST
NOT see any post-cut human turns -- no oracle leak.

## Mode name

`trait` -- selected via `benchmark.student.mode: trait`. Matches the family of
trait-* modes in Alexis's `synth-students` repo. We wire one variant here (joined
persona); per-dimension variants (`trait_cognitive`, `trait_affect`, etc.) stay
deferred.

## New prompts (under `prompts/benchmark/v5/`)

- **`trait_generator.txt`** -- generator system prompt. Asks the model to read
  the conversation prefix and write a 2-3 paragraph student persona covering:
  math skill level, common mistakes, affect / emotional state, attention
  patterns, learning style. Specific and grounded in what's visible in the
  prefix. Closes with: "This persona will be used to simulate the same student
  continuing the session."

- **`students/trait.txt`** -- student-mode prompt with a new `{trait_persona}`
  placeholder. Tells the model: "You are role-playing the following student.
  Embody them closely -- match how they reason, ask questions, and respond.
  {trait_persona}. Continue the conversation naturally as this student."
  The existing `{student_context}` substitution still happens (grade / subject
  metadata).

## New module: `benchmark/core/traits.py`

Single public function:

```python
def get_or_generate_trait(
    scenario: Scenario,
    prompt_version: str,
    model_client: ModelClient,
    model_name: str,
) -> str:
    """Return cached persona for (conv_id, cut_turn), else generate + cache.

    Generator input: scenario.transcript_prefix and scenario.student_context
    ONLY. Never references post-cut turns. Cache path:
        results/benchmark/_trait_cache/<conv_id>__<cut_turn>.json
    """
```

Cache file shape:
```json
{
  "conv_id": "...",
  "cut_turn": 22,
  "persona": "<the generated text>",
  "generator_model": "claude-opus-4-8",
  "prompt_version": "v5",
  "prefix_length_chars": 12345,
  "usage": {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...},
  "generated_at": "2026-06-09T..."
}
```

Reads/writes go through the storage backend (`save_benchmark_result` /
`load_benchmark_result` style). The cache directory is treated as a top-level
sibling of versioned result dirs so traits survive across benchmark versions.

## Exchange integration (`benchmark/core/exchange.py`)

`_build_role_prompt` gets an additional path when `role == "STUDENT"` and
`student_mode == "trait"`:

1. Load `students/trait.txt` (no special-case file path -- the existing
   `students/{mode}.txt` convention covers it).
2. Look up / generate persona via `get_or_generate_trait(scenario, ...)`.
3. Substitute `{trait_persona}` in the loaded prompt.

This means `_build_role_prompt` needs access to the `Scenario` object (currently
it only gets `transcript_so_far` and `student_context` strings). Either:

- **Option A (recommended):** pass `scenario` to `_build_role_prompt` as an
  optional kwarg. Existing string-only callers ignore it; the new trait path
  uses it. Minimal signature change.
- Option B: pass the persona string itself. Requires callers to resolve it
  before calling `_build_role_prompt`. More plumbing in both sync and batch
  loops.

We'll go with Option A.

In-memory cache inside the exchange loop: once a persona is resolved for a
scenario, store it on a local dict so subsequent student turns in the same
scenario don't re-hit the storage backend.

## Trait generation -- lazy, sync-first

The first student turn for each scenario triggers `get_or_generate_trait`.
For sync mode this is one inline LLM call (cheap given the cache reuses across
runs). For batch mode the first round has serial cost during prompt-build, but
the cache makes subsequent rounds (and subsequent runs) free.

Pre-batched trait generation as a Phase 0.5 is left out of scope; we can add
it if the inline cost becomes a real problem on a full benchmark run.

## Sync vs batch call-site changes

Both `run_exchange` and `run_exchanges_batch` need to forward the `scenario`
object into `_build_role_prompt`. Sync already has the scenario in scope.
Batch builds prompts per active_id from `scenario_map[sid]`, so it has the
scenario too. No new kwargs on the public exchange functions.

The model client used for trait generation reuses the **student profile's**
ModelClient (already constructed in run.py). Pass it through to
`_build_role_prompt` -- same place `student_client` already lives in the call.

The simplest plumbing: `_build_role_prompt(..., scenario=None,
trait_client=None, trait_model=None)`. When `student_mode == "trait"`, the
caller must supply both.

## Oracle-leak guard

Hard constraint: the generator prompt is built from
`scenario.transcript_prefix` only. Test asserts:

- The text passed to the generator LLM contains every prefix turn.
- The text passed to the generator LLM does NOT contain any post-cut text
  pulled from `transcripts[conv_id].turns` when `turn_number > cut_turn`.

The `Scenario` dataclass already discards post-cut turns from
`transcript_prefix` (via `_format_prefix(conv, cut_turn)`). The guard test
just confirms we never reach back to the conversation object during trait
generation.

## Config change

```yaml
benchmark:
  student:
    profile: anthropic            # unchanged -- reused for trait generation
    mode: trait                   # NEW value; default stays imitate_example
```

We do NOT flip the default. `imitate_example` keeps shipping; `trait` is an
opt-in mode the user selects per run (e.g., for the next varied smoke).

## Tests

- `get_or_generate_trait`:
  - cache hit returns saved persona without LLM call (mock the client; assert
    it isn't called when the cache file exists).
  - cache miss invokes client, writes file with the correct shape, returns
    the generated text.
  - cache file path uses (conv_id, cut_turn) as key.

- Oracle-leak test:
  - Construct a scenario with `transcript_prefix` covering turns 1..cut.
  - Mock the trait-gen client to record the prompt text it receives.
  - Assert: the recorded prompt contains every prefix turn's text and contains
    nothing from a synthetic post-cut turn we plant on the same `conversation`
    object.

- `_build_role_prompt` integration:
  - With `student_mode="trait"` and a mocked trait persona, output contains the
    persona text and does NOT contain the literal `{trait_persona}` placeholder.

## What's NOT changing

- `imitate_example` / `simple` / `expert` / `paraphrase_with_example` modes:
  untouched. Default stays `imitate_example`.
- Tutor side: unchanged.
- Annotator pipeline: unchanged.
- Cut-point selection: unchanged.

## Risk / open questions

- **Persona length blowup:** if the generator writes 500+ words per scenario,
  each student turn pays that cost on every round. Mitigations: cap
  `max_tokens` on the generator call (e.g., 800), and a short instruction in
  `trait_generator.txt` to stay within 2-3 paragraphs.
- **Persona drift:** the generated persona is fixed across all rounds of a
  scenario. If the AI tutor adapts in unexpected ways, the persona may
  stop fitting. Acceptable for first pass -- mirrors how a real student's
  general traits hold through a session even as the immediate context shifts.
- **Cache invalidation:** if `trait_generator.txt` is rewritten, old cached
  personas become stale. We don't auto-invalidate -- the cache key is
  `(conv_id, cut_turn)`, not `(conv_id, cut_turn, prompt_version)`. If we
  iterate the generator prompt, the user manually deletes
  `results/benchmark/_trait_cache/` before re-running. This is a documented
  knob, not a bug.
