# Labeller V2 Design

## Problem

The labeller classifies annotation result text as effective/partial/ineffective. It has three problems:

1. **Four divergent prompts.** `extract_ground_truth.py` and `build_ground_truth.py` each have an inline prompt (result-only, outcome-ish criteria). `classify.txt` has a different prompt (situation+action+result, strategy-centric criteria). `classify_binary.txt` is a fourth variant. These produce labels on different scales, but we compare them as if they're equivalent. Note: `build_ground_truth.py` is the script that produces the per-conversation ground truth files the pipeline actually reads; `extract_ground_truth.py` produces a legacy single-file format.

2. **Ground truth labeller only receives result text.** The pipeline labeller (`classify.txt`) receives situation, action, and result. The ground truth labeller (`extract_ground_truth.py`) only receives result text. In ~4 cases, the action field contains positive signals (e.g., "these are all good strategies") that the result text contradicts — the ground truth labeller never sees this context.

3. **Poor partial detection.** The labeller rounds hedged language to the nearest pole instead of recognizing it as partial. ~18 annotations out of 2,115 are misclassified:
   - 6 labeled effective that should be partial ("somewhat effective," "hard to tell," "one-word answers" with positive framing)
   - 8 labeled ineffective that should be partial ("good start but didn't build," "good that the student... but without guidance")
   - 4 labeled ineffective where action has positive signals the result-only labeller never saw

These errors are concentrated in the borderline cases that most affect kappa scores.

## What Doesn't Need Fixing

The labeller correctly classifies the vast majority of annotations. Specific things that work and should be preserved:

- **Unambiguous verdicts are handled correctly.** "Not effective" → ineffective, "effective because..." → effective. These cover ~95%+ of annotations.
- **AI verdict-first text is always classified correctly.** 0 mismatches between AI result verdicts and labeller labels.
- **Situation context ("appropriate time") is correctly not treated as evidence.** 169 correctly-labeled ineffective annotations have "appropriate time" or "good place" in their situation field — this is context-setting, not a positive signal.

## Design

### Single prompt: `classify_v2.txt`

One prompt file used by both `extract_ground_truth.py` and `label.py`. Receives four fields: annotation_type, situation, action, result text.

```
You are classifying a teaching coach's analysis of a tutoring moment.

Type: {annotation_type}
Situation: {situation}
Action: {action}
Assessment: {result_text}

Read the full analysis, then classify the tutor's effectiveness.

## Classification

- "effective": The assessment describes a positive outcome overall. The strategy worked, the student engaged or learned, or the annotator's overall verdict is clearly positive. Minor caveats or suggestions for improvement do not make an effective assessment partial.

- "partial": The assessment describes genuinely mixed results. Look for these specific signals:
  - Explicit hedge language: "somewhat effective," "partially effective," "semi-effective," "seems effective but..."
  - A trajectory that starts positive but ends negative: "good start but didn't build on it," "began well but..."
  - Uncertainty about whether the strategy worked: "unclear if effective," "hard to tell," "not sure if the student understood"
  - The student got the right answer but understanding is unclear
  - A meaningful positive AND a meaningful negative described in the same assessment, without one clearly dominating

- "ineffective": The assessment describes a failed outcome overall. The strategy didn't work, the student didn't engage or learn, or the annotator's overall verdict is clearly negative. Incidental positives like "good idea but..." or "the tutor tried but..." do not make a negative assessment partial — these acknowledge intent, not results.

## Important

The situation field provides context (timing, session flow, student state). Phrases like "appropriate time" or "good place to build rapport" describe WHEN something happened, not whether it worked. Do not treat timing context as a positive signal for classification.

The action field describes what the tutor did. If the action describes a well-chosen strategy but the assessment describes a negative outcome, weight the outcome.

Base your classification on the overall weight of the assessment. Do not anchor on individual words like "but" or "however" — these appear in effective, partial, and ineffective assessments alike.

Respond with ONLY one word: effective, partial, or ineffective
```

### Pipeline integration

**`build_ground_truth.py` changes (primary -- produces per-conversation files the pipeline reads):**
- Delete inline `CLASSIFICATION_PROMPT`, load `classify_v2.txt` from `prompts/annotator/labeller/`
- Pass all four fields: annotation_type, situation, action, result text
- Add `--labeller` CLI arg (default: `v2`). Output goes to `data/ground_truth_{labeller}/`
- Keep using Anthropic batch API via `ModelClient`

**`extract_ground_truth.py` changes (legacy single-file format):**
- Delete inline `CLASSIFICATION_PROMPT`, load `classify_v2.txt` from `prompts/annotator/labeller/`
- Pass all four fields
- Keep using Gemini API directly via `google.genai`

**`label.py` changes:**
- Load `classify_v2.txt` instead of `classify.txt`
- Pass `annotation_type` into the prompt template (already available in annotation dict)
- No other structural changes

**Ground truth versioning:**
- `data/ground_truth/` renamed to `data/ground_truth_v1/` (current baseline, untouched)
- `data/ground_truth_v2/` created by re-running `build_ground_truth.py --labeller v2`
- `config.yaml` `paths.ground_truth` updated to point at active version
- Both directories gitignored (data dirs already are)

**Existing files preserved:**
- `classify.txt` and `classify_binary.txt` stay in repo for reproducibility
- No changes to eval, storage, or any other pipeline component

### Re-run sequence

After implementation:
1. `python data/extract_ground_truth.py --labeller v2` -- regenerate ground truth with v2 labeller
2. Update `config.yaml` ground_truth path to `data/ground_truth_v2`
3. Re-label v5_gold results: `python -m annotator.core.label --version v5_gold --style balanced` (etc.)
4. Re-run eval: `python -m annotator.eval.eval --version v5_gold --mode annotations --style balanced`
5. Compare v1 vs v2 kappa scores

### Expected impact

~18 ground truth labels will shift (mostly ineffective→partial and effective→partial). This may move kappa scores, especially for demanding (n=79) where a few label changes have outsized effect. The shifts are corrections, not regressions.

## Out of Scope

- Binary labeller (`classify_binary.txt`) -- separate use case, not addressed here
- Annotation prompt changes -- already done in v5 profile prompts
- Detection prompt changes -- unrelated
- New human comparison study -- would validate but is a separate initiative
