# Annotator Pipeline Migration -- Design

*Status: spec / 2026-06-10*

## Goal

Migrate the benchmark Phase 2 from the legacy per-style profiles pipeline
(`generous` / `balanced` / `demanding` SAR + effectiveness label) to Lucy's
new annotator pipeline (`annotate` -> `decompose` -> `structure` with
per-facet action and outcome labels). Replace the scalar effectiveness score
with an action-appropriateness F1 against the human `situation_label_agg`
tag.

## Why

Lucy's pipeline is now the team direction:
- Single calibrated scaffolding prompt (v13) replaces three per-archetype
  prompts.
- `decompose` splits action and result text into discrete facets so each
  facet can be classified independently.
- `structure` labels:
  - each action facet -> `scaffolding | rigor | neither | both`
  - each result facet -> `pos | neg` (student outcome)
- Annotator-side validation reports test F1 of 0.94 / 0.62 / 0.79
  (scaffolding / rigor / student outcome) against teacher ground truth.

Our benchmark currently sits on the prior path. Migrating gives us results
that align with the annotator validation numbers and matches what Lucy's
F1 numbers measure on the *real* transcripts. The benchmark question we
care about -- "did the AI tutor take the right kind of action for this
moment?" -- maps cleanly onto her action labels vs the human
`situation_label_agg` tag.

## Scope

In scope (single PR):

1. **Pull annotator-side code from `insource/scaffolding_anno`** and
   reconcile with our local changes (caching, composite-id fix, thinking
   adaptive).
2. **Rewrite benchmark Phase 2** in `benchmark/run.py` to call
   `annotate -> decompose -> structure` per scenario.
3. **New scoring** -- per-scenario action-appropriateness against
   `situation_label_agg`, aggregated as scaffolding F1 + rigor F1 +
   student outcome rate.
4. **Update viewers** (`view.py`, `view_replay.py`) to show the new
   annotation schema (per-facet action and result labels).
5. **Update `config.yaml`** -- drop `styles` from `benchmark.annotator`,
   set `prompt_version: v13`, set `context_window: 20`.
6. **Update tests** that reference the old pipeline.

Out of scope:
- Annotator-side iteration / advisor changes (already in scaffolding_anno;
  we pull but don't modify).
- Standalone annotator validation runs (separate workflow).
- Per-archetype reporting / lens -- the new pipeline is single-track.

## Reconcile strategy

Files Lucy touched that we ALSO touched on `feat/conv-id-bench-lookups`:

| File | Our change | Lucy's change | Strategy |
|---|---|---|---|
| `annotator/core/client.py` | prompt caching + adaptive thinking | (her changes) | **Take ours**; cherry-pick any new helpers from hers |
| `annotator/core/storage.py` | composite-id `_conv_id_to_uuid` fix | (older, no fix) | **Take ours** |
| `annotator/core/label.py` | restored `_conv_id_to_uuid` split filter | (older) | **Take ours** |
| `annotator/core/annotate.py` | none (we pulled hers earlier) | latest v13-aware version | **Take hers** |
| `annotator/core/decompose.py` | none | new | **Take hers** |
| `annotator/core/structure.py` | none | new | **Take hers** |
| `annotator/core/situate.py` | none | (her version) | **Take hers** |
| `annotator/core/embed.py` | none | new | **Take hers** |
| `annotator/core/utils.py` | none | (her version) | **Take hers** |
| `annotator/eval/eval.py` | none | (her version) | **Take hers** |
| `annotator/iteration/advisor.py` | none | (her version) | **Take hers** |
| `annotator/iteration/structure_disagreements.py` | none | new | **Take hers** |
| `annotator/run.py` | none | (her version) | **Take hers** |
| `prompts/annotator/*` | none new from us | v13 + action_labeller + student_result_classifier | **Take hers** |
| `data/build_ground_truth.py` | IoU 0.7 -> 1.0 | (her version, IoU 0.7) | **Take ours** |
| `config.yaml` | model split, context_window 50, new benchmark.tutor block | context_window 20, no benchmark.tutor block | **Take ours with context_window 20** |

Mechanical merge per the table, then run tests + spot-check that our
caching / adaptive thinking / composite-id paths still work.

## New benchmark Phase 2

Today:
```
for style in styles:
    annotate(prompt_version=f"profiles/{style}") -> SAR per scenario
    label() -> effectiveness label per annotation
    save annotations/{profile}/{style}/{scenario_id}.json
    score: mean(label_to_score(label)) per scenario, per style
```

After:
```
annotate(prompt_version="v13") -> SAR per scenario
decompose() -> action_decomposed + result_decomposed facets per annotation
structure() -> action_label + result_label per facet
save annotations/{profile}/{scenario_id}.json    # one file per scenario
score: per-scenario action-appropriateness + outcome rate (see below)
```

One pass, no styles loop. The annotator config block drops `styles` and
sets `prompt_version: v13`.

## Scoring

Per scenario:

- **Action prediction:** the set of action labels emitted across all
  action_decomposed facets, e.g. `{scaffolding, neither}` or
  `{rigor}` or `{both}`. (`both` is treated as containing both
  scaffolding AND rigor.)
- **Ground truth:** the scenario's `situation_label_agg` value
  (`scaffolding` | `rigor` | `mixed` | `neither` | `both` | `unknown`).
  For F1 we focus on the two informative tags: `scaffolding` and `rigor`.
- **Per-scenario verdicts:**
  - `appropriate_scaffolding`: ground truth is `scaffolding` AND predicted
    set contains `scaffolding` (or `both`).
  - `appropriate_rigor`: ground truth is `rigor` AND predicted set
    contains `rigor` (or `both`).
  - `student_outcome_pos`: any result_decomposed facet labeled `pos`.

- **Aggregate metrics** per tutor profile / per cell:
  - **Scaffolding precision/recall/F1** across all scenarios:
    - TP: gt=scaffolding AND prediction includes scaffolding/both
    - FN: gt=scaffolding AND prediction doesn't
    - FP: gt!=scaffolding AND prediction includes scaffolding/both
    - TN: rest
  - **Rigor precision/recall/F1**: same, swapping scaffolding -> rigor.
  - **Student outcome rate**: fraction of scenarios with at least one
    `pos` result facet.
  - Scenarios with `gt in {mixed, both, neither, unknown}` are excluded
    from action F1 calculations (no clear ground truth) but still
    contribute to the outcome rate.

Saved as `results/benchmark/{version}/scores/{profile}.json` (one file
per profile, no `_<style>` suffix).

## Viewer updates

Both `view.py` and `view_replay.py` annotation panels:
- Drop the per-style section grouping.
- Show one annotation per scenario with: `situation`, `action`, `result`
  text PLUS `action_decomposed[]` (each with label badge) PLUS
  `result_decomposed[]` (each with label badge).
- Color the action label badges: scaffolding=blue, rigor=orange,
  neither=gray, both=purple.
- Color the result label badges: pos=green, neg=red.
- Header info bar gains a "appropriate?" tag per scenario:
  - Green if action prediction matched the ground-truth tag.
  - Red if it didn't.
  - Gray for ambiguous ground truth (`mixed` / `neither` / `unknown`).

## Tests

- `tests/test_benchmark_phase2_migration.py` (new):
  - End-to-end mocked: a 2-scenario run produces one annotation file per
    scenario, with `action_decomposed` and `result_decomposed` populated
    and labels attached.
  - Scoring: a scenario with `gt=scaffolding` and prediction
    `[scaffolding, neither]` counts as appropriate_scaffolding=True;
    F1 across multiple scenarios is computed correctly (hand-checked).
  - Ground truth `mixed` -> scenario excluded from action F1, still
    included in outcome rate.

- Update existing benchmark tests that reference `styles` or per-style
  result paths.

## Config changes

```yaml
benchmark:
  annotator:
    profile: anthropic
    prompt_version: v13            # was: profiles
    # styles: removed (no per-archetype iteration in the new pipeline)
    context_window: 20             # was: 50; matches Lucy's revert
    mode: batch
    poll_interval: 60
```

## What's NOT changing

- `benchmark.scenarios.mode: human` -- still pulls human moments.
- `benchmark.exchange.*` -- tutor / student / [NEXT_PROBLEM] / max_turns
  unchanged.
- `benchmark.tutor.mode` -- oracle / null tutor mode unchanged.
- `benchmark.student.mode` -- imitate_example / simple / trait etc.
  unchanged.
- Prompt caching, adaptive thinking, composite-id handling -- preserved.

## Migration safety net

After landing, run a 2-scenario sync smoke to verify the new pipeline
produces results we can inspect. Old smokes (cellA_simple_2026_06_08,
dyn_smoke_v12_2026_06_09, etc.) stay readable on disk for retrospective
comparison; new runs save under different scoring schema.

## Risks / open questions

- **Lucy's annotator side may evolve further** between our pull and our
  PR. We're pulling at her current `insource/scaffolding_anno` HEAD; if
  she pushes more before we PR, we re-reconcile.
- **F1 sensitivity to ground-truth coverage.** Modal-cut scenarios skew
  scaffolding (2,468) vs rigor (1,207) in `ground_truth_hybrid`. F1 on
  the underrepresented rigor class will have higher variance on small
  smokes. Documented; not a blocker.
- **`both` and `mixed` interpretation.** When the model emits `both` as
  an action facet label, we treat it as containing both scaffolding and
  rigor. When the human ground truth is `mixed`, we exclude from F1.
  These are different concepts at different layers; spec'd to keep them
  separate.
- **Coordinating with Lucy's PR #16.** Her PR is open against main with
  the same files we're modifying. After we land ours, hers becomes
  redundant. Plan: communicate before PRing; she closes her PR or we
  merge it first and rebase. Tracked separately.
