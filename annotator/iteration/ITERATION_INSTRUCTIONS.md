# Prompt Iteration Instructions

## Overview

Iterate on detection (Pass 1), annotation (Pass 2), and labeller (Pass 3) prompts using an LLM-powered semantic advisor. The advisor reads error examples with full transcript context and proposes specific prompt edits. Claude Code handles the mechanical pipeline steps. The model used for each phase (including the advisor) is configured in `config.yaml` via profiles (gemini, openai, anthropic).

## Lessons Learned (v3 -> v4)

These principles were discovered during the v3-to-v4 rewrite and should guide all future iteration:

### 1. Internal consistency over metric chasing

v3 was an accumulation of directional nudges ("Critical Check: What Could Be Better?", "Before rating, ask yourself...") that individually targeted specific metric weaknesses. The result was a prompt that contradicted itself -- telling the annotator to be balanced while also telling it to look harder for flaws. A clean, coherent prompt (v4) outperformed the patched one.

**Rule: Before adding any instruction, check whether it contradicts or undermines existing instructions.** If a proposed change says "be more critical of X" but the prompt already says "give credit for reasonable attempts," you have a conflict. Resolve it by rewriting the relevant section, not by appending a nudge.

### 2. Separation of concerns between passes

Each pass has a specific job. When responsibilities leak across passes, both get worse:

- **Pass 1 (Detector)**: Find turn ranges where notable moments occur. Cast a wide net. Don't analyze or judge -- just flag.
- **Pass 2 (Annotator)**: Analyze the flagged moment. Describe situation, action, and result with substantive analysis of strengths and weaknesses. Don't classify using labels (effective/partial/ineffective).
- **Pass 3 (Labeller)**: Read the full annotation (situation + action + result) and classify as effective/partial/ineffective. Don't re-analyze -- just weigh what the annotator wrote.

v3 violated this: the annotator prompt included full effectiveness criteria and examples with labels, making the labeller redundant. When we removed the label overlap and let each pass do its own job, metrics improved.

**Rule: When iterating a prompt, check that proposed changes don't pull in responsibilities from another pass.** If the advisor suggests adding effectiveness criteria to the annotator prompt, reject it -- that's the labeller's job. If the advisor suggests making the detector more selective, reject it -- the detector should cast a wide net.

### 3. Prompts need real definitions, not just instructions

v3 told the annotator to look for "scaffolding-related pedagogical events" without defining what scaffolding or rigor actually mean. The model was working from its general training rather than the study's specific constructs.

v4 added:
- Research context: what we're studying and why
- Construct definitions: what scaffolding, rigor, and rapport mean in this context
- Strategy taxonomies: named strategies (breaking down, hinting, modeling, etc.) the annotator can reference
- The core tradeoff: the specific decision the tutor faces

**Rule: Every prompt should ground the model in the research definitions. If the advisor proposes changes that reference concepts not defined in the prompt, add the definitions first.**

### 4. Check the full pipeline, not just the prompt you're iterating

The biggest v4 win came from fixing the labeller -- it was only reading the result field, not the full annotation. This had nothing to do with the P1 or P2 prompts. If we'd only iterated P2, we'd have missed it entirely.

**Rule: Before starting a prompt iteration cycle, run the full pipeline end-to-end and check ALL passes for bottlenecks.** Look at:
- Labeller prompt: does it read enough context to classify well?
- Labeller distribution: is it biased toward one label?
- Detection count: is the detector finding too many or too few moments?
- Annotation quality: are the S/A/R fields substantive or just summaries?

### 5. Don't blindly apply all advisor changes

The v3 iteration instructions said "Apply every change the advisor proposes. Do not filter, cherry-pick, or second-guess." This led to accumulated nudges that made the prompt incoherent.

**Updated rule: Apply the advisor's changes, but validate each one against the principles above.** Reject changes that:
- Add directional nudges without resolving contradictions with existing text
- Pull in responsibilities from another pass (e.g., adding effectiveness criteria to the annotator)
- Reference undefined concepts
- Just append instructions to the end of the prompt rather than integrating them

If a change targets a real problem but the proposed fix is a nudge, rewrite the relevant section of the prompt instead.

### 6. Chain-of-thought framing helps

Adding step-by-step reasoning instructions to the annotator ("First understand the student's state, then identify the strategy, then analyze calibration") improved analysis quality. This is well-supported by research and costs nothing.

**Rule: Annotator and labeller prompts should include step-by-step reasoning instructions tailored to the specific analysis task.**

### 7. Spot-check actual outputs, not just metrics

The labeller was reading one field out of three and nobody noticed until we read the raw code. The in-memory pipeline ran Pass 3 successfully (printed correct label stats) but never saved to disk -- eval read stale data and showed all labels as invalid. Both bugs were invisible from metrics alone.

**Rule: After running the pipeline, spot-check before trusting eval numbers.**
- Read a few raw annotations in `annotations.json` -- do they have the `effectiveness` field?
- Are the S/A/R fields substantive analysis or just summaries of the transcript?
- Does the label distribution make sense? (e.g., 0% effective rate is a red flag)

### 8. Changing one pass can break downstream passes

When we removed effectiveness labels from P2's output format, the labeller (which was designed for evaluative text starting with "Effective. The tutor correctly...") suddenly had to classify analytical text with mixed strengths/weaknesses. It wasn't equipped for that -- it defaulted to "partial" for everything because analytical text always mentions both sides.

**Rule: When you change a pass's output format or style, re-check that downstream passes still work with the new input.** Specifically:
- If you change P1 output: check that P2 receives the right fields via `{brief_description}`, `{excerpt}`, etc.
- If you change P2 output style: check that the labeller prompt is calibrated for the new kind of text.
- If you change the labeller prompt: check that it still handles the current annotation style.

### 9. Examples are more powerful than instructions

v3's examples included `"effectiveness": "effective"` in the JSON output, so the model produced effectiveness labels regardless of what the instructions said. v4's examples don't include that field, so the model doesn't produce it. The examples set the tone, voice, level of detail, and output format more strongly than any instruction paragraph.

**Rule: When changing a prompt, update the examples to match.** If you want the annotator to stop doing X, remove X from the examples. If you want a different analysis style, rewrite the examples in that style. Don't just add an instruction -- the model will follow the examples over the instructions when they conflict.

### 10. Go back to the source material

The original research instructions ("We are studying how tutors in K-12 math tutoring conversations decide to push for rigor...") were clear and well-scoped. v3 had drifted away from them through iteration -- the research framing was gone, replaced by generic "identify ALL scaffolding-related pedagogical events." Going back to the original research question improved the prompts more than any advisor cycle.

**Rule: When a prompt has drifted through iteration, don't patch it further. Go back to the original research instructions and rewrite from scratch.** The source material for this study:
- Scaffolding: "We are studying how tutors in K-12 math tutoring conversations decide to push for rigor in a session by asking students to do harder tasks / tasks that involve more metacognition, versus when they choose to introduce scaffolds that make problems more accessible for students."
- Rapport: "We are studying rapport building behaviors in K-12 math tutoring conversations. In this study, you will annotate tutoring transcripts with observations of how tutors attempt to build rapport with students."

### 11. The advisor optimizes for metrics, not prompt quality

The advisor reads error examples and proposes changes to fix them. It has no concept of prompt coherence, internal consistency, or separation of concerns. It will happily add "be more critical" alongside "give credit for reasonable attempts" if that's what the error examples suggest. It is a useful tool for identifying error patterns, but its proposed fixes need human judgment.

**Rule: Use the advisor to understand what's going wrong (error patterns), but don't treat its proposed prompt edits as authoritative.** The advisor is good at diagnosis, unreliable at prescription.

## Structural Guardrails

### Maximum 3 advisor rounds before mandatory rewrite

After 3 rounds of advisor-driven patches on a prompt, stop. Do not apply more patches. Instead:

1. Read the patched prompt end-to-end. Does it still read as a coherent document?
2. If yes, continue iterating.
3. If no (and it usually won't after 3 rounds), rewrite the prompt from scratch:
   - Start from the research framing and construct definitions
   - Incorporate the lessons from the 3 rounds (what error patterns did the advisor find?)
   - Write a clean prompt that addresses those patterns structurally, not with nudges

This is what the v3-to-v4 transition was. v3 was 3+ rounds of patches. v4 was a rewrite that incorporated the lessons. The rewrite outperformed the patches.

### Diff against the clean baseline

After each round of advisor changes, diff the current prompt against the last clean version (the last rewrite, not the last patch). This shows how much the prompt has drifted. If the diff is mostly appended paragraphs at the end of sections, you're accumulating nudges. If the diff shows integrated changes throughout, you're probably fine.

```bash
diff prompts/annotator/v4/p2/scaffolding.txt prompts/annotator/<current>/p2/scaffolding.txt
```

### Ask the advisor to flag inconsistencies

Before asking the advisor to propose fixes, ask it to review the current prompt for internal consistency. Add to the advisor workflow:

1. First call: "Read this prompt. Are there any instructions that contradict each other? Any undefined terms? Any places where the prompt tells the model to do two incompatible things?"
2. Fix inconsistencies first.
3. Then run the normal error analysis and iteration.

## The Iteration Cycle

```
1. Audit full pipeline  -->  2. Run & evaluate  -->  3. Ask advisor  -->  4. Validate & apply
      ^                                                                           |
      |               6. Regressed? Revert. Improved? Next round.                 |
      +----------------------------  5. Evaluate  <------------------------------+
```

### Step 0: Audit the full pipeline (before starting iteration)

Before iterating any prompt, run the full pipeline and check each pass:

1. **Run**: `python -m annotator --version <v> --profile <profile>`
2. **Evaluate**: `python -m annotator.eval.eval --version <v>`
3. **Check the labeller**: Is the label distribution healthy? Is the labeller reading enough context? Check `prompts/annotator/labeller/classify.txt`.
4. **Check annotations**: Read a few raw annotations in `results/<v>/annotations.json`. Are the S/A/R fields substantive analysis or just transcript summaries?
5. **Check detections**: Is the detector finding a reasonable number of moments (8-15 scaffolding, 5-12 rapport per session)?
6. **Identify the bottleneck**: Is the problem in detection (recall), annotation (quality of analysis), or labeling (classification accuracy)?

Only iterate the pass that is the bottleneck.

### Step 1: Run the pipeline

All commands accept `--profile <name>` to select the model provider. If omitted, uses the default profile from `config.yaml`.

For detection iteration:
```bash
python -m annotator.core.detect --version <v> --prompt-version <pv> --profile <profile>
```

For annotation iteration (gold mode isolates annotation quality from detection):
```bash
python -m annotator.core.annotate --version <v>_gold --prompt-version <pv> --gold --profile <profile>
python -m annotator.core.label --version <v>_gold --gold --profile <profile>
```

For per-archetype annotation iteration (iterate the prompt against a specific annotator subset):
```bash
python -m annotator.core.annotate --version <v>_gold --prompt-version <pv> --gold --annotator-style <style> --profile <profile>
python -m annotator.core.label --version <v>_gold --gold --annotator-style <style> --profile <profile>
```
The `--annotator-style` flag filters gold moments to that archetype's annotators only. No style text is injected into prompts -- calibration is achieved by iterating the prompt against archetype-filtered ground truth.

For per-archetype detection iteration:
```bash
# Step 1: Run detection ONCE with the current baseline prompt (same for all styles)
python -m annotator.core.detect --version <v> --prompt-version <pv> --profile <profile>

# Step 2: Evaluate the SAME detections against each archetype's ground truth
python -m annotator.eval.eval --version <v> --mode detections --style generous
python -m annotator.eval.eval --version <v> --mode detections --style balanced
python -m annotator.eval.eval --version <v> --mode detections --style demanding

# Step 3: Use advisor per archetype to diagnose style-specific error patterns
python -m annotator.iteration.advisor --pass detection --version <v> --type scaffolding --annotator-style generous
python -m annotator.iteration.advisor --pass detection --version <v> --type scaffolding --annotator-style demanding

# Step 4: Iterate each style's p1 prompt independently based on advisor diagnosis
# Per-style prompts live at: prompts/annotator/profiles/{style}/p1/{scaffolding,rapport}.txt
```

**Critical: Do NOT pre-bake biases into per-style prompts.** All style prompts must start as identical copies of the current canonical detection prompt (v4). The differentiation between styles comes entirely from evaluating against archetype-filtered ground truth and iterating based on what the advisor finds -- not from injecting assumptions about what "generous" or "demanding" detection should look like. The data tells you how each archetype's annotators actually detect moments; the prompt iteration converges to match that behavior.

For full pipeline:
```bash
python -m annotator --version <v> --prompt-version <pv> --profile <profile>
```

### Step 2: Evaluate

```bash
# Detection only
python -m annotator.eval.eval --version <v> --mode detections

# Annotation only (gold mode)
python -m annotator.eval.eval --version <v>_gold --mode annotations

# Annotation only, filtered to archetype ground truth
python -m annotator.eval.eval --version <v>_gold --mode annotations --annotator-style generous

# Full pipeline
python -m annotator.eval.eval --version <v> --mode full

# Compare two versions
python -m annotator.eval.eval --compare <v1> <v2> --mode full
```

### Step 3: Ask the advisor

The advisor loads its meta-prompt from `prompts/annotator/iteration/<profile>/`. If the profile directory doesn't exist, it falls back to `gemini/`.

```bash
# Detection iteration
python -m annotator.iteration.advisor --pass detection --version <v> --type scaffolding --profile <profile>
python -m annotator.iteration.advisor --pass detection --version <v> --type rapport --profile <profile>

# Annotation iteration
python -m annotator.iteration.advisor --pass annotation --version <v> --type scaffolding --profile <profile>
python -m annotator.iteration.advisor --pass annotation --version <v> --type rapport --profile <profile>

# Per-archetype annotation iteration (errors filtered to archetype's annotators)
python -m annotator.iteration.advisor --pass annotation --version <v> --type scaffolding --annotator-style generous --profile <profile>
```

The advisor sends the model:
- The current prompt (verbatim)
- 10 correct matches (what works)
- 10 complete misses, 10 near-misses, 10 false positives (detection) or 10 agreements + 10 disagreements per confusion type (annotation)
- Each with transcript excerpts and human annotations

The advisor returns structured JSON with:
- Semantic error patterns (grouped by root cause)
- Proposed prompt edits (exact text replacements)
- Risk/directional assessments

Output saved to `results/<v>/advisor_<pass>_<type>.json`.

### Step 4: Validate and apply proposed changes

**Do not blindly apply all changes.** Review each proposed change against these checks:

1. **Consistency check**: Does this change contradict existing instructions? If yes, rewrite the section rather than appending.
2. **Responsibility check**: Does this change pull in work from another pass? (e.g., adding label definitions to the annotator, adding analysis to the detector). If yes, reject.
3. **Definition check**: Does this change reference concepts not defined in the prompt? If yes, add definitions first.
4. **Nudge check**: Is this change just appending "be more X" or "watch out for Y" to the end? If yes, integrate it into the relevant section or reject.

To apply:
1. Copy the current prompt version to a new version:
   ```bash
   cp -r prompts/annotator/<current>/ prompts/annotator/<new>/
   ```
2. Open the specific prompt file for the type being iterated: `prompts/annotator/<new>/p{1,2}/{scaffolding,rapport}.txt`. Only edit the file matching the `--type` that was passed to the advisor.
3. For each entry in `proposed_changes` that passes validation:
   - If `current_text` and `proposed_text` are provided: find and replace.
   - If `current_text` is null: integrate `proposed_text` into the relevant section (don't just append).
   - If `proposed_text` is null: delete `current_text`.
4. After applying, re-read the full prompt and check it reads as a coherent document, not a list of patches.

### Step 5: Evaluate the new version

Run the pipeline and evaluate (Steps 1-2 again). Compare metrics to the previous best.

### Step 6: Accept or revert

- **If metrics improved (or held steady)**: The new version becomes the current best. Go to Step 3 for another round.
- **If metrics regressed**: Revert to the previous best version. Consider whether the regression is from a specific change and try removing just that one. If the whole batch regressed, the iteration is done for this type.

## Prompt Architecture

### Pass 1 (Detection): `prompts/v{N}/p1/{scaffolding,rapport}.txt`
- Research context and construct definitions
- "What to look for" checklist
- Target count (8-15 scaffolding, 5-12 rapport)
- "Cast a wide net" instruction
- Output format: `{turn_start, turn_end, annotation_type, brief_description}`

### Pass 2 (Annotation): `prompts/v{N}/p2/{scaffolding,rapport}.txt`
- Research context and expanded construct definitions with strategy taxonomies
- "Using Context" guidance
- "Your Task" with field definitions:
  - Situation: why this is/isn't an appropriate moment for the construct
  - Action: what strategy the tutor used (named from taxonomy)
  - Result: how well it worked, strengths, weaknesses, missed alternatives
- Step-by-step reasoning instructions
- Examples (3, covering good/mixed/poor moments -- WITHOUT effectiveness labels)
- Output format: `{annotation_type, turn_start, turn_end, situation, action, result}`

### Pass 3 (Labeller): `prompts/labeller/classify.txt`
- Reads situation + action + result (full annotation context)
- Classifies as effective/partial/ineffective
- Includes calibration guidance (minor weaknesses don't downgrade, minor strengths don't upgrade)
- Output: single word

## Versioning

### Pipeline prompts (canonical)
- `prompts/annotator/v{N}/p1/` -- detection prompts (scaffolding.txt, rapport.txt)
- `prompts/annotator/v{N}/p2/` -- annotation prompts (scaffolding.txt, rapport.txt)
- `prompts/annotator/labeller/` -- labeller prompts (shared across versions)
- v4 is the current best (clean rewrite from v3 with research grounding and separation of concerns)

### Iteration artifacts
- `prompts/annotator/iteration/<profile>/detection.txt` -- advisor meta-prompt for detection
- `prompts/annotator/iteration/<profile>/annotation.txt` -- advisor meta-prompt for annotation
- `prompts/annotator/iteration/<profile>/attempts/` -- iterated prompt versions (named by original version + pass)
- `prompts/annotator/iteration/archetype_attempts/` -- archived per-archetype iterations from v3
- `results/iteration/<profile>/` -- baseline runs and iteration results per model

### Pipeline results (canonical)
- `results/v{N}/` -- pipeline outputs, eval results, and advisor results

## Variance Checks

Re-run the same prompt to measure batch-to-batch noise:
- Detection: ~1pp overall, ~3pp per type
- Annotation: ~7pp kappa

Improvements within the variance band may be noise. After finding a promising change, re-run to confirm.

## Stopping Criteria

You've reached the ceiling when:
- The advisor keeps proposing changes that target the same patterns you've already tried
- Remaining errors are fundamentally hard (counterfactual reasoning, ground truth noise)
- Variance checks show improvements are within the noise band
- The prompt reads as incoherent from accumulated changes (time for a rewrite, not more iteration)

## Post-Iteration Cleanup

After an iteration cycle is complete (you've hit the ceiling or found a winner), clean up so the repo stays navigable. Iteration generates many intermediate artifacts -- prompts, results, advisor outputs, eval JSONs -- across multiple directories. Left uncleaned, these accumulate and make the file structure hard to reason about.

### What to keep

1. **The winning prompt**: Promote it to its canonical location.
   - New version: copy to `prompts/annotator/v{N+1}/p{1,2}/`
   - Profile update: copy to `prompts/annotator/profiles/{style}/p2/`
2. **The winning results**: Keep in `results/annotator/v{N+1}/` (or the appropriate profile results directory).
3. **The iteration log**: Write a summary of what was tried, what worked, and what didn't. Add it to the relevant `history/` subdirectory (e.g., `history/claude_annotation_iteration/annotation_iteration.md`).

### What to archive

Move intermediate artifacts out of working directories into archive locations:

| Artifact | During iteration | Archive to |
|----------|-----------------|------------|
| Attempt prompts | `prompts/annotator/iteration/<profile>/attempts/<name>/` | `prompts/archive/iteration/<profile>/attempts/<name>/` |
| Attempt results | `results/annotator/iteration/<profile>/<name>/` | `history/<model>_<pass>_iteration/results_<name>/` |
| Advisor outputs | `results/annotator/<version>/advisor_*.json` | Keep with the attempt results they belong to |

### What to delete

- `*_requests.jsonl` files (batch request payloads) -- these are large, reproducible, and never needed after the run completes.
- Failed or abandoned attempts that produced no useful signal. If an attempt errored out or was immediately reverted, delete rather than archive.

### Cleanup checklist

After completing an iteration cycle:

1. **Promote the winner** to its canonical prompt and results location.
2. **Move all non-winning attempt prompts** from `prompts/annotator/iteration/` to `prompts/archive/iteration/`.
3. **Move all non-winning attempt results** from `results/annotator/iteration/` to `history/`.
4. **Delete `*_requests.jsonl`** files from all attempt results (they're large and reproducible).
5. **Update the baselines table** at the bottom of this document with the new best metrics.
6. **Verify** that `prompts/annotator/iteration/` and `results/annotator/iteration/` contain only active work, not completed cycles.

The goal: at any point, `prompts/annotator/iteration/` should contain only in-progress work. Completed iteration history lives in `prompts/archive/` and `history/`. Canonical prompts live in `prompts/annotator/v{N}/` or `prompts/annotator/profiles/`.

## Pipeline Commands Reference

```bash
# All pipeline commands accept --profile <name> to override config.yaml default.
# Profiles: gemini (gemini-3.1-pro-preview), openai (gpt-5.4), anthropic (claude-opus-4-6)

# Full pipeline
python -m annotator --version <v>
python -m annotator --version <v> --profile anthropic

# Detect
python -m annotator.core.detect --version <v> --prompt-version <pv>
python -m annotator.core.detect --version <v> --prompt-version <pv> --profile anthropic

# Annotate
python -m annotator.core.annotate --version <v> --prompt-version <pv>
python -m annotator.core.annotate --version <v>_gold --prompt-version <pv> --gold
python -m annotator.core.annotate --version <v>_gold --prompt-version <pv> --gold --annotator-style generous

# Label
python -m annotator.core.label --version <v>
python -m annotator.core.label --version <v>_gold --gold
python -m annotator.core.label --version <v>_gold --gold --annotator-style generous
python -m annotator.core.label --version <v> --binary

# Evaluate
python -m annotator.eval.eval --version <v> --mode detections
python -m annotator.eval.eval --version <v> --mode annotations
python -m annotator.eval.eval --version <v> --mode full
python -m annotator.eval.eval --compare v3_gemini v4 --mode full

# Advisor
python -m annotator.iteration.advisor --pass detection --version <v> --type scaffolding
python -m annotator.iteration.advisor --pass annotation --version <v> --type rapport
python -m annotator.iteration.advisor --pass annotation --version <v> --type scaffolding --annotator-style generous

# Error analysis (manual deep dives)
python -m annotator.iteration.advisor analyze --version <v> --error-type miss --type scaffolding --limit 10
python -m annotator.iteration.advisor analyze --version <v> --summary-only

# HTML comparison view
python -m annotator.eval.view --version <v>
```

## File Paths

- Pipeline prompts: `prompts/annotator/v{N}/p{1,2}/{scaffolding,rapport}.txt`
- Labeller prompts: `prompts/annotator/labeller/{classify,classify_binary}.txt`
- Iteration meta-prompts: `prompts/annotator/iteration/<profile>/{detection,annotation}.txt`
- Iteration prompt attempts: `prompts/annotator/iteration/<profile>/attempts/<name>/`
- Iteration results: `results/iteration/<profile>/<name>/`
- Config: `config.yaml`
- Results: `results/v{N}/`
- Ground truth: `data/ground_truth/`
- Transcripts: `data/transcripts/`
- Advisor output: `results/<version>/advisor_{detection,annotation}_{scaffolding,rapport}.json`
- Example exclusion logic: `annotator/core/utils.py` (`EXAMPLE_CONV_IDS`)

## Current Baselines

### Best prompt: v4 (clean rewrite with research grounding)

| Metric | v3_gemini | v4 | Delta |
|--------|-----------|-----|-------|
| Cluster Recall | 56.4% | 64.2% | +7.8pp |
| Moment Precision | 26.6% | 23.4% | -3.1pp |
| Binary Kappa | 34.7% | 35.9% | +1.2pp |
| 3-Way Kappa | 35.0% | 32.0% | -2.9pp |
| Within Human Range | 53.0% | 52.3% | -0.7pp |
| Effective Rate | 40.5% | 36.8% | -3.7pp |

Human ceiling: Binary kappa 0.30, 3-way kappa 0.31-0.33

### Known issues in v4
- Scaffolding effective rate is very low (9.8%) -- annotator is too critical of scaffolding moments
- 3-way kappa slightly below v3 -- the partial/effective boundary needs tuning for scaffolding
- Rapport is strong (binary kappa 0.37, effective rate 55.2%)

### Historical baselines

| Run | Model | Prompt | Recall | Precision | 3-Way Kappa | Notes |
|---|---|---|---|---|---|---|
| v1 | gemini-3.1-pro-preview | v1 | 54.1% | 30.0% | 27.0% | Gemini full pipeline |
| v2 | gemini-3.1-pro-preview | v2 | 51.7% | 28.2% | 36.6% | Best annotation kappa |
| v8 | claude-opus-4-6 | v1 | 64.9% | 22.7% | 24.2% | Best recall |
| v3_gemini | gemini-3.1-pro-preview | v3 | 56.4% | 26.6% | 35.0% | Pre-rewrite baseline |
| v4 | gemini-3.1-pro-preview | v4 | 64.2% | 23.4% | 32.0% | Current best |
