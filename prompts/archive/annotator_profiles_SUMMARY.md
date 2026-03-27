# Annotator Profile Iteration Summary

## Problem

Human annotators don't agree with each other uniformly -- they cluster into archetypes by labeling tendency. A single prompt can't match all of them simultaneously. We needed per-archetype prompts to maximize agreement with each group.

## Archetypes

| Archetype | Annotators | n | Tendency |
|---|---|---|---|
| Generous | Gerber, Jones, Shields, Stobbe, Trujillo | 297 | More likely to rate effective |
| Balanced | Forbes, Mann, Padgett | 510 | Middle ground |
| Demanding | Flick | 79 | More likely to rate ineffective |

## Method

Iterate separate annotation prompts per archetype using gold mode (human-detected moments, isolating annotation quality from detection). Each round: run pipeline -> evaluate against archetype-filtered ground truth -> ask LLM advisor to analyze disagreements and propose prompt edits -> validate & apply -> re-evaluate -> accept if improved, revert if regressed.

No "style text" injected into prompts. Calibration comes purely from iterating prompt content against each archetype's ground truth subset.

Baseline: v3 prompts (v1 p1 detection + v2 p2 annotation) with Gemini gemini-3.1-pro-preview.

## Results

| Archetype | Baseline 3-Way Kappa | Final 3-Way Kappa | Delta | Human Ceiling | Exceeds Ceiling |
|---|---|---|---|---|---|
| Generous | 0.3691 | 0.4061 | +3.7pp | 0.3350 | Yes |
| Balanced | 0.4576 | 0.5364 | +7.9pp | 0.5049 | Yes |
| Demanding | 0.6283 | 0.6283 | 0pp | -- | -- |

### Generous (final prompt: v5_generous_r1)

| Round | Type | Delta | Decision |
|---|---|---|---|
| Scaffolding R1 (v4_generous) | scaffolding | +2.5pp | Accept |
| Scaffolding R2 (v5_generous) | scaffolding | +2.5pp | Accept |
| Scaffolding R3 (v6_generous) | scaffolding | -5.4pp | Revert |
| Rapport R1 (v5_generous_r1) | rapport | +28.5pp binary | Accept |
| Rapport R2 (v5_generous_r2) | rapport | -3.5pp | Revert |
| Claude attempt (v7_generous) | both | -5.4pp | Revert (effective rate hit 63%, too generous) |

Final breakdown: scaffolding 3-way kappa 0.3822, rapport 3-way kappa 0.5000.

### Balanced (final prompt: v5_balanced_r1)

| Round | Type | Delta | Decision |
|---|---|---|---|
| Scaffolding R1 (v4_balanced) | scaffolding | +1.1pp | Accept |
| Scaffolding R2 (v5_balanced) | scaffolding | +2.2pp | Accept |
| Scaffolding R3 (v6_balanced) | scaffolding | -4.8pp | Revert |
| Rapport R1 (v5_balanced_r1) | rapport | +5.7pp | Accept |
| Rapport R2 (v5_balanced_r2) | rapport | -5.2pp | Revert |

Final breakdown: scaffolding 3-way kappa 0.4217, rapport 3-way kappa 0.5724.

### Demanding (final prompt: v3 unchanged)

| Round | Type | Delta | Decision |
|---|---|---|---|
| Scaffolding R1 (v4_demanding) | scaffolding | -26.0pp | Revert |
| Rapport R1 (v4_demanding_r1) | rapport | -21.2pp | Revert |

Only 1 annotator, 79 annotations (28 after matching). Too thin for stable iteration.

## Key Findings

1. **All iterable archetypes exceed human inter-annotator ceiling.** The LLM agrees with its target humans better than those humans agree with each other.

2. **Rapport iteration gave the biggest single-round gains** -- generous +28.5pp, balanced +5.7pp binary kappa in round 1.

3. **2 rounds was the sweet spot.** Round 3 consistently regressed for both generous and balanced. The prompts had absorbed what the advisor could teach.

4. **Demanding was too thin to iterate.** With 28 matched annotations, any change swung metrics wildly.

5. **No style injection needed.** Calibration came entirely from iterating prompt content against filtered ground truth.

6. **Claude's advisor was too aggressive.** Pushed generous effective rate past the 60% guardrail, regressing overall kappa.

## File Locations

- Final prompts: `synthetic_annotator/pipeline/prompts/annotator_profiles/{generous,balanced,demanding}/p2/`
- Final results: `synthetic_annotator/results/annotator_profiles/{generous,balanced,demanding}/`
- Iteration history: `synthetic_annotator/results/annotator_profiles/summary.json`
- Archived iteration attempts (prompts): `synthetic_annotator/pipeline/prompts/iteration/archetype_attempts/`
- Archived iteration attempts (results): `synthetic_annotator/results/iteration/archetype_attempts/`
