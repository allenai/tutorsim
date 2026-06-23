# ARW-62: Per-Profile Detection Iteration

*Completed: 2026-03-24*

## Objective

Evaluate whether detection (Pass 1) benefits from per-style prompts (generous/balanced/demanding), following the same per-archetype iteration approach used successfully for annotation (Pass 2) profiles.

## Background

The annotator profiles (`prompts/annotator/profiles/{style}/p2/`) were created by iterating annotation prompts against archetype-filtered ground truth. Each profile matches a different subset of human annotators: generous (Gerber, Jones, Shields, Stobbe, Trujillo, n=297), balanced (Forbes, Mann, Padgett, n=510), and demanding (Flick, n=79). All profiles exceed human inter-annotator agreement ceilings.

Detection (Pass 1) had always used a single v4 prompt regardless of style. Prior work (12 iterations v1-v11 with Gemini, 1 iteration v9 with Claude) suggested detection is model-limited, not prompt-limited. ARW-62 tested whether the per-archetype approach could find style-specific detection improvements that generic iteration missed.

## Method

### Infrastructure (ARW-64 prerequisite)

Added `--style` CLI flag to the full pipeline and detection pass:
- `python -m annotator --version v4 --style generous` uses per-style prompts for both detection (p1) and annotation (p2)
- `python -m annotator.core.detect --version v4 --style generous` uses `profiles/generous/p1/` prompts
- Graceful fallback: if `profiles/{style}/p1/` doesn't exist, uses the default prompt version
- Created `profiles/{style}/p1/` directories for scaffolding and rapport detection prompts

### Iteration Cycle

**Baseline**: Used existing v8_baseline results (v4 prompts, Claude claude-opus-4-6, 104 transcripts, 1726 detections). Evaluated the same detections against each archetype's filtered ground truth.

**Baseline per-archetype results:**

| Metric | Generous (32 convs) | Balanced (73 convs) | Demanding (6 convs) | Overall (98 convs) |
|--------|:---:|:---:|:---:|:---:|
| Cluster Recall | 59.9% | 68.8% | 64.4% | 64.9% |
| Moment Precision | 4.9% | 12.9% | 1.3% | 22.7% |
| Mean IoU | 0.599 | 0.632 | 0.653 | 0.624 |
| Scaffolding Recall | 58.7% | 78.2% | 83.3% | 68.6% |
| Rapport Recall | 65.7% | 61.6% | 51.4% | 61.1% |

**Key observation**: Generous has lower recall (59.9%) despite more annotators, because generous annotators flag subtler moments the LLM misses. Demanding has very thin data (6 conversations).

### Round 0: Advisor Diagnosis

Ran 6 advisor calls (scaffolding + rapport x generous + balanced + demanding) against the baseline. All 6 identified nearly identical error patterns:

1. **Over-fragmentation** (~40-60% of FPs): LLM splits continuous teaching/rapport episodes into 5-10+ separate detections
2. **Over-extended boundaries** (~60-70% of near-misses): LLM includes too many context turns around the core event
3. **Missed inaction** (~25-35% of misses): tutor NOT acting is invisible to the action-focused prompt
4. **Missed session management** (~15-25% of misses): breaks, topic selection, time wasting
5. **Single-turn moments missed** (~20-25% of misses): prompt implies 2-3 turns is the floor
6. **Routine instruction flagged** (~25-30% of FPs): standard teaching moves aren't "notable"

**Style divergence was minimal.** All 3 archetypes showed the same patterns with similar proportions. The advisors' proposed changes were nearly identical across styles.

### Round 1: Structural Changes

Applied advisor-validated changes to all 3 style prompts (kept identical -- no pre-baked biases):
- Added inaction/missed-opportunity bullets
- Added session-management bullets
- Changed "even 2-3 turns" to "even a single turn"
- Added consolidation guidance section
- Added boundary precision section
- **Rejected**: count target reduction (v9 showed this kills recall), notability filters (evaluative language kills recall)

**Round 1 results:**

| Metric | Baseline | Round 1 | Delta |
|--------|:---:|:---:|:---:|
| **Overall Recall** | 64.9% | 61.3% | -3.6pp |
| Overall Precision | 22.7% | 24.5% | +1.7pp |
| Overall IoU | 0.624 | 0.641 | +0.017 |
| **Generous Recall** | 59.9% | 50.2% | **-9.7pp** |
| Balanced Recall | 68.8% | 68.2% | -0.6pp |
| Demanding Recall | 64.4% | 62.7% | -1.7pp |

Consolidation guidance hurt generous recall badly (-9.7pp) by merging moments that generous annotators flag separately. Balanced/demanding were within noise.

### Round 2: Refined Structural Changes

Ran 6 more advisor calls on r1 results. Advisors continued recommending the same patterns (stronger consolidation, notability filters, count reduction). Applied refined versions:
- Rewrote consolidation with concrete examples and explicit merge rules
- Added boundary example (turns 52-58, not 40-60)
- Expanded inaction guidance
- Added second example in output format showing single-turn inaction
- Expanded rapport-specific bullets (gamification, praise quality, scaffolding-as-rapport)
- **Still rejected**: count reduction, notability filters

**Round 2 results:**

| Metric | Baseline | Round 1 | Round 2 | Delta (base->r2) |
|--------|:---:|:---:|:---:|:---:|
| **Overall Recall** | 64.9% | 61.3% | 58.1% | **-6.9pp** |
| Overall Precision | 22.7% | 24.5% | 24.1% | +1.4pp |
| Overall IoU | 0.624 | 0.641 | 0.641 | +0.017 |
| **Generous Recall** | 59.9% | 50.2% | 45.9% | **-14.0pp** |
| **Balanced Recall** | 68.8% | 68.2% | 65.3% | **-3.4pp** |
| **Demanding Recall** | 64.4% | 62.7% | 57.6% | **-6.8pp** |

Clear trend: recall regresses with each round while precision stays flat. Every structural addition suppresses detections without meaningfully improving precision.

## Conclusions

### 1. Detection is model-limited, not prompt-limited

This confirms the finding from 12 prior Gemini iterations (v1-v11) and the prior Claude v9 attempt. The v4 detection prompt is near-optimal. Every content change -- consolidation guidance, boundary precision, expanded bullet lists, inaction detection, session-management detection -- either regresses recall or is within the +/-1-3pp variance band.

### 2. Per-style detection prompts are not viable

All 3 archetypes showed the same error patterns with the same proportions. There is no style-specific signal in detection to differentiate on. The advisors' proposed changes were nearly identical across styles, and the changes hurt all styles equally (or disproportionately hurt generous).

### 3. The v4 prompt's simplicity is its strength

The v4 detection prompt is 54 lines with 11 neutral bullets. Every attempt to add structure (consolidation rules, boundary guidance, expanded criteria) makes the model more selective and kills recall. The "cast a wide net" / "when in doubt, include it" philosophy is correct for detection -- selectivity is the annotator's job (Pass 2), not the detector's.

### 4. Generous recall is structurally harder

Generous annotators flag ~30% more moments per session than balanced annotators, including subtler micro-moments (single-turn decisions, missed opportunities). The LLM's detection ceiling is lower for generous because these subtle moments require human-level pedagogical judgment that current models lack.

### 5. Advisor diagnosis is good but prescriptions hurt

The advisor correctly identified real error patterns (inaction, fragmentation, boundary issues). But every proposed fix either added evaluative language or structural constraints that made detection more selective. Per lesson #11 from the iteration instructions: "The advisor is good at diagnosis, unreliable at prescription."

## Decision

- **Reverted** all `profiles/{style}/p1/` prompts to exact v4 copies
- **Profiles keep shared v4 detection** -- no per-style p1 differentiation
- **Profiles retain per-style p2 annotation** -- this is where style differentiation works
- **Detection iteration closed** for current model generation

## Artifacts

- Baseline advisor outputs: `results/annotator/iteration/anthropic/v8_baseline/advisor_detection_{scaffolding,rapport}_{generous,balanced,demanding}.json`
- Round 1 results + advisors: `history/profile_detection_iteration/r1/`
- Round 2 results: `history/profile_detection_iteration/r2/`
- Prior v9 attempt (regressed): `results/annotator/iteration/anthropic/v9_detection/`
- Prior v9 prompt: `prompts/archive/iteration/anthropic/attempts/v9_p1_detection/`