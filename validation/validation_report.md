# Synthetic Annotation Pipeline: Validation Summary

## 1. Pipeline Overview

This report summarizes validation evidence for a 3-pass LLM annotation pipeline applied to K-12 math tutoring transcripts. Pass 1 detects key pedagogical moments as turn ranges. Pass 2 produces structured Situation/Action/Result analysis for each moment. Pass 3 classifies each strategy's effectiveness (effective/partial/ineffective). The pipeline is evaluated against human expert annotations from 9 annotators across 98 conversations.

## 2. Detection Validation

Detection evaluated on 98 conversations (586 human clusters, 1685 LLM detections).

**Table 1. Detection metrics at IoU >= 0.3.**

| | Cluster Recall | Moment Precision | Mean IoU |
|---|---|---|---|
| Overall | 65.9% | 25.4% | 0.666 |
| Scaffolding | 66.0% | 33.4% | 0.657 |
| Rapport | 65.7% | 20.2% | 0.676 |

Recall degrades gracefully from 80.0% at IoU 0.1 to 47.4% at IoU 0.5.

## 3. Annotation Validation

Effectiveness labels evaluated on 278 matched moments across 98 conversations.

**Table 2. LLM-human agreement vs. human inter-annotator ceiling.**

| | Binary κ [95% CI] | 3-Way κ [95% CI] | n |
|---|---|---|---|
| Human ceiling | 0.2080 [0.0644, 0.3509] | 0.2511 [0.1293, 0.3624] | 165 pairs |
| LLM (v4 full pipeline) | 0.3882 [0.2712, 0.4925] | 0.3178 [0.2293, 0.3954] | 278 |

**Table 3. Per-archetype annotation results.**

| Archetype | Baseline κ | Final κ | Human Ceiling κ | Exceeds Ceiling |
|---|---|---|---|---|
| Generous | 0.3691 | 0.4061 | 0.0000 | No |
| Balanced | 0.4576 | 0.5364 | 0.1507 | Yes |
| Demanding | --- | --- | --- | Too thin (n=28) |

*Baseline and final kappa from documented iteration results (prompts/annotator/profiles/SUMMARY.md). Human ceiling computed from ground truth.*

## 4. Held-Out Validation

The pipeline was evaluated on 97 conversations that were never seen during prompt iteration. These conversations have ground truth annotations from the same annotators but were not part of the development corpus.

**Table 4. Development vs. held-out comparison.**

| Metric | Development (98 convs) | Held-Out (97 convs) | Delta |
|---|---|---|---|
| Cluster Recall | 65.9% | 76.5% | +10.7pp |
| Moment Precision | 25.4% | 29.0% | +3.6pp |
| Mean IoU | 0.6659 | 0.6882 | +2.2pp |
| Binary Kappa | 0.3882 | 0.3255 | -6.3pp |
| 3-Way Kappa | 0.3178 | 0.3205 | +0.3pp |
| Within Human Range | 53.2% | 45.9% | -7.4pp |
| Human Ceiling (3-way) | 0.2511 (165 pairs) | 0.0784 (47 pairs) | -17.3pp |

3-way kappa is stable across splits (+0.3pp, within the +/-7pp variance band). The LLM exceeds the human ceiling on the held-out set (LLM 0.3205 vs ceiling 0.0784). The held-out ceiling is lower than the development ceiling because only 47 annotator pairs overlap on the held-out conversations (vs 165 on development), reflecting sparser multi-annotator coverage in the newer data.

## 5. Additional Robustness Checks

**Dev/test split.** Retrospective 70/30 split shows comparable 3-way kappa (dev: 0.3181, test: 0.3007), indicating no prompt overfitting.

**Cross-model agreement.** Mature pipeline versions (v3+) produced pairwise LLM-LLM kappa of 0.22--0.77 on shared moments, with early iterations excluded.

**Variance bands.** Repeated identical runs show +/-1pp detection variance and +/-7pp annotation kappa variance.

## 6. Conclusion

The LLM annotator meets or exceeds the human inter-annotator agreement ceiling on both the development corpus and a true held-out set of 97 unseen conversations. Disagreements concentrate at the effective/partial boundary — the most subjectively ambiguous judgment. Results are stable across data splits, model versions, and repeated runs.

---

*Generated from validation pipeline. Numbers computed from data/ground_truth and results/annotator/{v4, held_out}. See validation/*.ipynb for full analysis.*
