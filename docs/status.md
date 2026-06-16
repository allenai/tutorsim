# Project Status

*Last updated: 2026-06-16*

## Active experimental loop

User (Albert) drives iteration: posts a diagnostic from a viewer or a comparison, names a small targeted change (prompt edit, code edit, new metric), Claude makes the change and re-runs the relevant cells, surfaces the new scores + viewer link. Repeat.

Current cells: **default-tutor × oracle-student** and **rigor-tutor × oracle-student**, both on **v9 prompts**, **max_turns=5**, same 10-scenario varied_smoke corpus (seed=42, 5 scaffolding + 5 rigor gold).

## Hard rules for prompt edits

1. **Use the user's exact words.** When the user gives you a prompt or a sentence to add to a prompt, paste it verbatim. Do not add embellishments, paraphrases, or "stay within the conceptual ceiling..." style restatements. Albert called this fluff and it cost a full re-run.
2. **Verbatim from synth-students otherwise.** Student prompts (`prompts/benchmark/v*/students/*.txt`, `trait_generator/*.txt`, `dimensions/*.txt`) come from `C:\Users\azhang\OneDrive - Insource Services Inc\Desktop\Ai2\synth-students/src/students/`. If the user asks for a new student variant, model it on the closest synth-students class verbatim, then add only the minimum structural changes our pipeline needs.
3. **Minimal structural mods only — and tell the user exactly what you added.** Two known necessary additions:
   - `{student_context}` substitution (system passes student metadata via this placeholder)
   - `[NEW_MESSAGE]` instruction (our exchange loop splits multi-utterance turns on this token)
   No other additions without asking. Don't invent "When to scaffold" / "When to push for rigor" / "Stay within the conceptual ceiling" sections.
4. **Strip synth-students instructions that don't apply to our pipeline.** We use single-call turns, not their two-call workflow. We don't have a JSON `'end'` key. Specifically:
   - "First start by describing what the student does..." in ImitateExample / Oracle → strip
   - "set the 'end' key to True" in `get_shared_generation_instructions` → stripped
   - Length-matching-the-example-conversation → stripped
5. **Bump prompt version, never overwrite.** New prompt content goes in a new `prompts/benchmark/v{N+1}/` dir. Run dirs include `prompt_version` in the name (`{tutor_model}_{prompt_version}_{tutor_mode}_tutor_{student_mode}_student_{date}`).

## Current scoring

`benchmark/core/score.py` — Lucy's 3-axis (proposed 2026-06-15):

- `scaffolding_did` rate: of scaffolding-gold scenarios, fraction where LM action_label ∈ {scaffolding, both}. Higher better.
- `rigor_did` rate: of rigor-gold scenarios, fraction where LM action_label ∈ {rigor, both}. Higher better.
- `overscaffold` rate: scenarios with non-empty `overscaffold_decomposed` / total. Lower better. This is the real discriminator between tutors — the always-both cheat is caught here, not by the did-rates.
- `outcome_pos_rate`: scenarios with any `pos` result facet / total.

Per-dim F1 is gone. The 5-turn cap doesn't reduce "both" labeling (both rate stays 0.40–0.60 regardless of max_turns) — the over-scaffold rate does the discriminating work.

## Tutor modes (v9)

- `default` (`v9/tutors/default.txt`): NEW "Expert K-12 math tutor" prompt (Albert wrote it). Frames the scaffold↔rigor dial; names over-scaffolding as the most common failure. With this prompt, default tutor's over-scaffold rate dropped 0.30 → 0.10, basically matching the dedicated rigor prompt.
- `oracle` (`v9/tutors/oracle.txt`): unchanged from v8. Tutor sees full post-cut real transcript via `{reference_transcript}`.
- `rigor` (`v9/tutors/rigor.txt`): unchanged from v8. Socratic, withhold answers.

## Student modes (v9)

- `oracle` (`v9/students/oracle.txt`): persona (TraitGenerator-generated joined-3 from pre-cut) + full post-cut transcript verbatim. **Includes Albert's "DO NOT provide the correct answer..." rule as a bullet (his exact words, no embellishment).**
- `trait` / `imitate_example` / `simple` / `expert` / `paraphrase_with_example` / `trait_with_example`: verbatim synth-students.
- `_PROMPT_VERSION` hardcoded in `benchmark/synth_students/prompts.py` decouples student version from tutor version. Currently at v8; v9 students are identical to v8.

## Open bug just fixed (re-run needed)

**Moment scope leaked across the cut.** `benchmark/core/annotator_bridge.py:build_synthetic_detections` set the annotator's moment window to `human_moment_turn_start → last_AI_turn`, so the "ONLY tutor actions between START and END" instruction in `prompts/annotator/v13/p2/scaffolding.md` was scoring the human tutor's pre-cut turns AND the AI replay as one combined strategy. Concretely: a moment where the human tutor pushed for rigor before the cut and the AI scaffolded after came out as `scaffolding + rigor`, inflating both-rates and shifting action_labels off the AI's actual behavior. Lucy's "guided questioning mistaken for rigor" was partly this — the rigor was the human tutor's, not the AI's.

Fix on main: moment now spans `first_AI_turn → last_AI_turn`. Pre-cut human context still reaches the annotator via the surrounding excerpt window (context_window=20), so the situation read still has the setup; only the action/result are scoped to the AI.

**All "Latest results" below are stale** — they were produced with the buggy bridge. Re-run before reading them.

## Latest results (n=10, v9, max_turns=5, oracle student) — STALE, pre-bridge-fix

| | default × oracle (new Expert prompt) | rigor × oracle | 
|---|---|---|
| did_scaffold | 0.80 (4/5) | 0.80 (4/5) |
| did_rigor | 1.00 (5/5) | 0.80 (4/5) |
| overscaffold (lower=better) | 0.10 (1/10) | 0.00 (0/10) |
| outcome+ | 1.00 | 1.00 |
| both-rate | 0.60 | 0.40 |

Notable: the new default closes most of the gap with rigor. If this holds at larger n the rigor-specific mode might be subsumable into a well-written default.

Run dirs:
- `results/benchmark/claude-opus-4-8_v9_default_tutor_oracle_student_20260616/`
- `results/benchmark/claude-opus-4-8_v9_rigor_tutor_oracle_student_20260616/`
- Older runs archived under `results/benchmark/archive/`.

## How to launch a run

```
PYTHONPATH=. python scripts/varied_smoke.py \
  --tutor-mode {default|rigor|oracle} \
  --student-mode {oracle|trait|...} \
  --prompt-version v9 \
  --max-turns 5
```

Auto-naming: `{tutor_model}_{prompt_version}_{tutor_mode}_tutor_{student_mode}_student_{date}`. Trait personas are cached at `results/benchmark/_trait_cache/` keyed by `(conv_id, cut_turn, trait_mode)` so re-runs are free on persona generation.

## Viewer

`PYTHONPATH=. python -m benchmark.eval.view_replay --version <dir-name> --profile anthropic`

Surfaces: did-rates, over-scaffold rate, outcome+, action label distribution, per-scenario verdict, tutor + student system prompts (with persona + reference substituted in), moment box, annotation card with action/result prose + facets + over-scaffold facets.

The "scored N/N" block hides when nothing was dropped. Verdict badge is direction-match, not quality (quality lives in over-scaffold + outcome).

## Known open questions

- Pin the "missed scaffolding scenario" in v9 default (4/5 scaffolding instead of 5/5) — is the new prompt over-rotating toward rigor?
- Whether to also extend the v9 default's "Expert K-12 math tutor" framing to the rigor mode (rigor was unchanged in v9; might benefit from the same scaffold↔rigor explicit framing).
- n=10 caveat throughout. None of these comparisons are at confirmation sample sizes.

## Git state

Branch: `main` at HEAD. All multi-PR work from 2026-06-15 (v6/v7/v8 prompts, oracle student, 3-axis scoring, viewer rewrite, decompose in-memory fix) is shipped on `insource/main`. v9 prompts + dir-naming fix are uncommitted on local main — ready to commit when Albert says ship.
