# v4 Prompt Summary

## What v4 Is

A clean rewrite of the v3 prompts for all 4 prompt files (p1 scaffolding, p1 rapport, p2 scaffolding, p2 rapport). v3 had accumulated directional nudges across multiple iteration rounds that contradicted each other. v4 went back to the original research framing and rewrote from scratch.

## What Changed from v3

### Structural: Separation of Concerns

v3 violated the pass separation -- the annotator (Pass 2) prompt included full effectiveness criteria and examples with labels (`"effectiveness": "effective"`). This made the labeller (Pass 3) redundant since the model was already classifying during annotation.

v4 removed all effectiveness criteria and labels from the annotator prompt. The annotator now only analyzes (situation/action/result). The labeller reads the full annotation and classifies. Each pass does one job.

### Content: Research Grounding

v3 told the model to look for "scaffolding-related pedagogical events" without defining what scaffolding means in this study's context.

v4 added:
- **Research context**: The specific research question ("how tutors decide to push for rigor vs. introduce scaffolds")
- **Construct definitions**: What scaffolding, rigor, and rapport mean with named strategies (breaking down, hinting, modeling, check-ins, emotional validation, etc.)
- **Strategy taxonomies**: Concrete list of strategies the model can reference by name
- **The core tradeoff**: The specific decision the tutor faces at each moment

### Detection (p1): More Specific Guidance

v3 p1 scaffolding was 44 lines, v4 is 54 lines. Key additions:
- Defined scaffolding and rigor with concrete strategies
- Articulated the core tradeoff explicitly
- Expanded "what to look for" checklist (student errors, coasting, tutor strategy shifts)

v3 p1 rapport was 47 lines, v4 is 53 lines. Key additions:
- Defined rapport and its role in tutoring
- Expanded detection triggers (power dynamics, academic content as rapport, failed rapport attempts)

### Annotation (p2): Analytical vs. Evaluative

v3 p2 scaffolding (103 lines) included effectiveness criteria (effective/partial/ineffective definitions) and examples with `"effectiveness"` field in JSON output. The model produced labels alongside analysis.

v4 p2 scaffolding (118 lines) removed all effectiveness criteria. Examples show analysis only -- no labels. Added:
- Step-by-step reasoning instructions ("First understand the student's state, then identify the strategy, then analyze calibration")
- Richer examples with substantive analysis (not just summaries)
- "Using Context" guidance on referencing surrounding turns
- Explicit instruction for substantive analysis vs. transcript summaries

Same pattern for rapport: v3 (109 lines) had effectiveness criteria and labeled examples; v4 (117 lines) removed them and added research-grounded definitions, strategy taxonomy, and step-by-step reasoning.

### Examples

v3 examples were short and included effectiveness labels:
```json
{"situation": "Student was intimidated...", "result": "Effective. The tutor correctly...", "effectiveness": "effective"}
```

v4 examples are longer with substantive analysis and no labels:
```json
{"situation": "The student was intimidated by large numbers in division (5600 / 70) and said 'I don't know how to do this.' This is an appropriate time to scaffold -- the student's block is intimidation, not lack of knowledge.", "result": "The scaffold was well-calibrated. The tutor correctly identified that the student already knew the underlying fact and just needed a bridge to see it..."}
```

The examples set the tone more strongly than instructions -- v4 examples model the analytical depth expected.

## Results

| Metric | v3 | v4 | Delta |
|---|---|---|---|
| Cluster Recall | 56.4% | 64.2% | +7.8pp |
| Moment Precision | 26.6% | 23.4% | -3.1pp |
| 3-Way Kappa | 35.0% | 32.0% | -2.9pp |
| Within Human Range | 53.0% | 52.3% | -0.7pp |
| Effective Rate | 40.5% | 36.8% | -3.7pp |

v4 significantly improved detection recall (+7.8pp) by better defining what to look for. Annotation kappa slightly regressed (-2.9pp 3-way) because the separation of concerns shifted how the model analyzed moments -- the annotator wrote balanced analysis (strengths and weaknesses) which the labeller sometimes read as "partial" even when the overall assessment was positive.

The v4 prompts became the baseline for per-archetype iteration, which addressed the remaining kappa gap through targeted prompt tuning per annotator group.

## Lessons Documented

The v3-to-v4 rewrite produced 11 lessons captured in ITERATION_INSTRUCTIONS.md:

1. Internal consistency over metric chasing
2. Separation of concerns between passes
3. Prompts need real definitions, not just instructions
4. Check the full pipeline, not just the prompt you're iterating
5. Don't blindly apply all advisor changes
6. Chain-of-thought framing helps
7. Spot-check actual outputs, not just metrics
8. Changing one pass can break downstream passes
9. Examples are more powerful than instructions
10. Go back to the source material
11. The advisor optimizes for metrics, not prompt quality
