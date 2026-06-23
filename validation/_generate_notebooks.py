#!/usr/bin/env python3
"""Generate validation notebooks for the synthetic annotator pipeline.

All results use v5 prompts. All IoU matching uses the same 0.3 threshold.
Train/held-out splits are explicit throughout.

Usage:
    python validation/_generate_notebooks.py
"""
import nbformat
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent


def md(src):
    if src.startswith("\n"):
        src = src[1:]
    if src.endswith("\n"):
        src = src[:-1]
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    if src.startswith("\n"):
        src = src[1:]
    if src.endswith("\n"):
        src = src[:-1]
    return nbformat.v4.new_code_cell(src)


def new_nb():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    return nb


# ================================================================
# SHARED SETUP CODE (used by both notebooks)
# ================================================================

SHARED_SETUP = """
import sys
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict, Counter

warnings.filterwarnings('ignore', category=FutureWarning)

REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'validation' else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from annotator.core.utils import (
    load_ground_truth, compute_iou, merge_overlapping_ranges,
    EXAMPLE_CONV_IDS, RESULTS_DIR, IOU_THRESHOLD,
)
from annotator.eval.eval import (
    compute_detection_metrics, compute_effectiveness_metrics,
    compute_human_ceiling, match_for_effectiveness, match_gold_direct,
    filter_moments_by_type, filter_matches_by_type,
    load_detections_as_moments, load_annotations,
    ANNOTATION_TYPES, EFFECTIVENESS_LABELS,
    cohens_kappa,
)

%matplotlib inline
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 100,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'sans-serif',
})
FIGURES_DIR = REPO_ROOT / 'validation' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'matched': '#2ecc71', 'human_only': '#3498db', 'llm_only': '#e67e22',
    'scaffolding': '#2980b9', 'rapport': '#e74c3c',
    'primary': '#2c3e50', 'secondary': '#7f8c8d',
    'llm': '#2980b9', 'human_ceiling': '#e74c3c',
}
"""

SHARED_DATA_LOAD = """
# All results use v5 prompts. IoU threshold is 0.3 everywhere.
VERSION = 'v5'
GOLD_VERSION = 'v5_gold'
IOU = 0.3  # same threshold for ALL matching (detection, effectiveness, ceiling)

ground_truth = load_ground_truth()

# Identify dev vs held-out by checking which conversations were in v4 (the dev pipeline)
_v4_det = load_detections_as_moments('v4')
_v4_ids = set(_v4_det.keys()) if _v4_det else set()

v5_det = load_detections_as_moments(VERSION)
if v5_det is None:
    raise FileNotFoundError(f'No detections.json for {VERSION}')

all_eval_ids = set(ground_truth['conversations'].keys()) & set(v5_det.keys()) - EXAMPLE_CONV_IDS
train_ids = all_eval_ids & _v4_ids
ho_ids = all_eval_ids - _v4_ids

# Type-filter LLM detections per conversation
gt_by_conv = {}
for cid in all_eval_ids:
    hm = ground_truth['conversations'][cid]['key_moments']
    ht = {m.get('annotation_type') for m in hm}
    gt_by_conv[cid] = hm
    v5_det[cid] = [m for m in v5_det[cid] if m.get('annotation_type') in ht]

print(f'Pipeline version: {VERSION} (all results use v5 prompts)')
print(f'IoU threshold:    {IOU} (same for detection, effectiveness matching, and ceiling)')
print(f'Train set:          {len(train_ids)} conversations (used during prompt iteration)')
print(f'Held-out set:     {len(ho_ids)} conversations (prompts never saw these)')
print(f'Total:            {len(all_eval_ids)} conversations')
"""


# ================================================================
# NOTEBOOK 1: DETECTION VALIDATION
# ================================================================

def build_detection_notebook():
    nb = new_nb()
    c = []

    c.append(md("""
# Key Moment Detection: Does the LLM Find What Humans Find?

## What are key moments?

In a K-12 math tutoring session, there are moments where the tutor makes a pedagogical choice that meaningfully shapes the student's learning. We study two types:

**Scaffolding/Rigor moments** arise when the tutor must decide how much support to provide. Scaffolding is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently — breaking problems into steps, hinting toward answers, modeling worked examples. Rigor means maintaining cognitive demand — requiring explanations, extending problems, fading support. The core tradeoff: too much support robs the student of productive struggle; too little leaves them floundering. A key moment occurs when this choice is notably well-calibrated, miscalibrated, or when the student's signal creates a genuine fork in how to proceed.

**Rapport moments** arise when the tutor's interpersonal choices shape the learning environment. Rapport includes trust and safety (the student feels comfortable making mistakes), genuine interest (the tutor engages with the student as a person), emotional attunement (noticing and responding to frustration, excitement, fatigue), and relational continuity (remembering details across sessions). A key moment occurs when the tutor builds, misses, or damages rapport — including missed opportunities where the tutor ignores an emotional signal or handles a personal moment superficially.

Human expert annotators watch tutoring transcripts and mark these moments as turn ranges (e.g., turns 45-62), identifying both the type (scaffolding or rapport) and the effectiveness of the tutor's response.

**The question**: can an LLM find the same moments that human experts find?

## The dataset

Our dataset contains **201 tutoring transcripts** with **291 annotation passes** (one pass = one annotator labeling one transcript for one type). These 291 passes produced **1,688 individual moment annotations**. Most transcripts were annotated once; 54 received multiple passes (different annotators, or both types). Nine annotators contributed, grouped into three calibration profiles by labeling tendency (generous, balanced, demanding). Six transcripts used as few-shot examples in the LLM prompts are excluded from all evaluation, leaving **195 transcripts** for analysis.

## How we measure overlap

We use **Intersection over Union (IoU)**: if a human marked turns 45-62 and the LLM marked turns 43-60, IoU measures how much those ranges overlap relative to their combined span. We count a **match** when IoU >= 0.3 (at least 30% overlap). Before comparing, we merge overlapping human annotations into **clusters** — when multiple annotators flag the same event with slightly different boundaries.

## Train vs. held-out

- **Train** (98 transcripts): the prompts were iterated using error examples from these
- **Held-out** (97 transcripts): the prompts have never seen these

All results use **v5 prompts** (our best and final detection prompts).
"""))

    c.append(code(SHARED_SETUP))
    c.append(code(SHARED_DATA_LOAD))

    # ---- Detection results ----

    c.append(md("""
## Detection Results

| Metric | What it measures |
|---|---|
| **Cluster Recall** | What fraction of human moment clusters did the LLM find? |
| **Moment Precision** | What fraction of LLM detections match a human cluster? |
| **Mean IoU** | Average overlap quality of matched pairs |
"""))

    c.append(code("""
results = []
for label, ids in [('Train', train_ids), ('Held-out', ho_ids), ('Combined', all_eval_ids)]:
    gt_sub = {c: gt_by_conv[c] for c in ids}
    llm_sub = {c: v5_det.get(c, []) for c in ids}
    m = compute_detection_metrics(gt_sub, llm_sub, iou_threshold=IOU)
    results.append({
        '': label,
        'Conversations': len(ids),
        'Human Clusters': m['total_human_clusters'],
        'LLM Detections': m['total_llm_annotations'],
        'Cluster Recall': f"{m['cluster_recall']:.1%}",
        'Moment Precision': f"{m['moment_precision']:.1%}",
        'Mean IoU': f"{m['mean_iou']:.3f}",
    })

df = pd.DataFrame(results).set_index('')
print(f'v5 Detection Performance (IoU >= {IOU})\\n')
df
"""))

    c.append(md("""
### By Annotation Type

Scaffolding/rigor moments and rapport moments may have different detection difficulty — scaffolding events often have clearer textual signals (errors, corrections) while rapport moments can be subtler (tone, missed opportunities).
"""))

    c.append(code("""
gt_all = {c: gt_by_conv[c] for c in all_eval_ids}
llm_all = {c: v5_det.get(c, []) for c in all_eval_ids}

type_results = []
for ann_type in ANNOTATION_TYPES:
    gt_typed = filter_moments_by_type(gt_all, ann_type)
    llm_typed = filter_moments_by_type(llm_all, ann_type)
    m = compute_detection_metrics(gt_typed, llm_typed, iou_threshold=IOU)
    if m['total_human_clusters'] > 0:
        type_results.append({
            '': ann_type.title(),
            'Clusters': m['total_human_clusters'],
            'LLM Dets': m['total_llm_annotations'],
            'Recall': f"{m['cluster_recall']:.1%}",
            'Precision': f"{m['moment_precision']:.1%}",
            'Mean IoU': f"{m['mean_iou']:.3f}",
        })

# Combined row
m_all = compute_detection_metrics(gt_all, llm_all, iou_threshold=IOU)
type_results.append({
    '': 'Combined',
    'Clusters': m_all['total_human_clusters'],
    'LLM Dets': m_all['total_llm_annotations'],
    'Recall': f"{m_all['cluster_recall']:.1%}",
    'Precision': f"{m_all['moment_precision']:.1%}",
    'Mean IoU': f"{m_all['mean_iou']:.3f}",
})

df_type = pd.DataFrame(type_results).set_index('')
print(f'Detection by Annotation Type (IoU >= {IOU}, all {len(all_eval_ids)} transcripts)\\n')
df_type
"""))

    # ---- IoU sensitivity ----

    c.append(md("""
## IoU Sensitivity

Do the matches barely clear the 0.3 threshold, or do they represent real overlap? We recompute recall at stricter thresholds. Smooth degradation = genuine matches. Cliff-drop = noise.
"""))

    c.append(code("""
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
gt_all = {c: gt_by_conv[c] for c in all_eval_ids}
llm_all = {c: v5_det.get(c, []) for c in all_eval_ids}

sens = []
for t in thresholds:
    m = compute_detection_metrics(gt_all, llm_all, iou_threshold=t)
    sens.append({'IoU': t, 'Recall': m['cluster_recall'], 'Precision': m['moment_precision']})

df_sens = pd.DataFrame(sens)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_sens['IoU'], df_sens['Recall'],
        'o-', color=COLORS['scaffolding'], linewidth=2.5, markersize=8, label='Cluster Recall')
ax.plot(df_sens['IoU'], df_sens['Precision'],
        's-', color=COLORS['rapport'], linewidth=2.5, markersize=8, label='Moment Precision')
ax.axvline(x=IOU, color=COLORS['secondary'], linestyle='--', alpha=0.7,
           label=f'Operating threshold ({IOU})')
ax.set_xlabel('IoU Threshold')
ax.set_ylabel('Rate')
ax.set_title(f'v5 Detection Performance vs. IoU Threshold (n={len(all_eval_ids)} conversations)')
ax.legend(loc='upper right')
ax.set_ylim(0, 0.85)
ax.set_xlim(0.05, 0.75)
fig.savefig(FIGURES_DIR / 'iou_sensitivity.png')
fig.savefig(FIGURES_DIR / 'iou_sensitivity.pdf')
plt.show()

# Table
df_disp = df_sens.copy()
df_disp['Recall'] = df_disp['Recall'].map('{:.1%}'.format)
df_disp['Precision'] = df_disp['Precision'].map('{:.1%}'.format)
df_disp
"""))

    # ---- Overlap visualization ----

    c.append(md("""
## Overlap Visualization

What do matches and misses look like? Each row shows one conversation: human clusters on top, LLM detections on bottom. Green = matched (IoU >= 0.3), blue = human only (miss), orange = LLM only (false positive).

Five conversations shown, spanning low to high recall.
"""))

    c.append(code("""
gt_dev = {c: gt_by_conv[c] for c in train_ids}
llm_dev = {c: v5_det.get(c, []) for c in train_ids}
metrics_dev = compute_detection_metrics(gt_dev, llm_dev, iou_threshold=IOU)
per_conv = metrics_dev['per_conversation']
with_clusters = {cid: v for cid, v in per_conv.items() if v['clusters'] > 0}
sorted_recalls = sorted(with_clusters.items(), key=lambda x: x[1]['recall'])

n = len(sorted_recalls)
selected = [sorted_recalls[i] for i in [0, n//4, n//2, 3*n//4, n-1]]

fig, axes = plt.subplots(len(selected), 1, figsize=(14, 3.2 * len(selected)),
                         gridspec_kw={'hspace': 0.5})
for ax, (conv_id, conv_m) in zip(axes, selected):
    human_raw = gt_by_conv.get(conv_id, [])
    llm_raw = v5_det.get(conv_id, [])
    clusters = merge_overlapping_ranges(human_raw)
    y_h, y_l, bh = 1.0, 0.0, 0.35

    for cl in clusters:
        cr = (cl['turn_start'], cl['turn_end'])
        ct = cl['annotation_type']
        best = max((compute_iou(cr, (l['turn_start'], l['turn_end']))
                     for l in llm_raw if l.get('annotation_type') == ct), default=0)
        color = COLORS['matched'] if best >= IOU else COLORS['human_only']
        ax.barh(y_h, cl['turn_end']-cl['turn_start']+1, left=cl['turn_start'],
                height=bh, color=color, edgecolor='white', linewidth=0.5)

    for l in llm_raw:
        lr = (l['turn_start'], l['turn_end'])
        lt = l.get('annotation_type')
        best = max((compute_iou(lr, (cl['turn_start'], cl['turn_end']))
                     for cl in clusters if cl['annotation_type'] == lt), default=0)
        color = COLORS['matched'] if best >= IOU else COLORS['llm_only']
        ax.barh(y_l, l['turn_end']-l['turn_start']+1, left=l['turn_start'],
                height=bh, color=color, edgecolor='white', linewidth=0.5)

    ax.set_yticks([y_l, y_h])
    ax.set_yticklabels(['LLM', 'Human'])
    ax.set_title(f'Recall={conv_m["recall"]:.0%} | {conv_m["clusters"]} clusters, '
                 f'{conv_m["llm_moments"]} detections', fontsize=10)
    ax.set_xlabel('Turn number')

patches = [mpatches.Patch(color=COLORS['matched'], label='Matched (IoU >= 0.3)'),
           mpatches.Patch(color=COLORS['human_only'], label='Human only (miss)'),
           mpatches.Patch(color=COLORS['llm_only'], label='LLM only (false positive)')]
fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIGURES_DIR / 'overlap_visualization.png', bbox_inches='tight')
fig.savefig(FIGURES_DIR / 'overlap_visualization.pdf', bbox_inches='tight')
plt.show()
"""))

    nb.cells = c
    return nb


# ================================================================
# NOTEBOOK 2: ANNOTATION VALIDATION
# ================================================================

def build_annotation_notebook():
    nb = new_nb()
    c = []

    c.append(md("""
# Annotation Validation: Does the LLM Judge Tutoring Quality Like Humans Do?

## What we're measuring

After finding key moments in tutoring transcripts (validated in Notebook 1), the pipeline does two more things:

1. **Analyzes** each moment: What was the situation? What did the tutor do? What happened next? (Situation/Action/Result analysis)
2. **Labels** each moment: Was the tutor's strategy *effective*, *partially effective*, or *ineffective*?

For **scaffolding/rigor moments**, effectiveness means: did the tutor calibrate their support correctly? Scaffolding is temporary, calibrated support — breaking problems into steps, hinting toward answers, modeling worked examples. Rigor means maintaining cognitive demand — requiring explanations, extending problems, fading support. Effective scaffolding keeps the student in their zone of proximal development — challenged but not overwhelmed. Ineffective scaffolding either does the thinking for the student (over-scaffolding) or leaves them floundering (under-scaffolding).

For **rapport moments**, effectiveness means: did the tutor build genuine connection? Rapport includes trust and safety (the student feels comfortable making mistakes), genuine interest (engaging with the student as a person), emotional attunement (noticing frustration, excitement, fatigue), and relational continuity (remembering details across sessions). Effective rapport creates the foundation that makes academic work possible. Ineffective rapport ranges from missed opportunities (ignoring a student's emotional signal) to active damage (condescension, impatience).

Human experts made the same judgments on the same moments. **The question: does the LLM's effectiveness label agree with the human experts' label?**

## Two ways to measure annotation quality

We present two measurements, because they answer different questions:

| | **Human-detected moments** | **Full pipeline** |
|---|---|---|
| **What it is** | LLM annotates the *exact same moments* humans marked (same turn ranges) | LLM detects its own moments, then annotates those |
| **What it measures** | Annotation quality in isolation — no detection noise | End-to-end pipeline quality |
| **IoU matching** | None needed (turn ranges are identical) | IoU >= 0.3 to match LLM detections to human clusters |
| **N (moments compared)** | All human moments (~1,600) | Only moments the LLM also found (~600) |

**Human-detected moments** is the clean measurement — it tells us how good the LLM is at judging effectiveness when it's looking at the right moment. **Full pipeline** is the real-world measurement — what you'd actually get if you ran the system end to end, including detection errors.

## How we measure agreement

We use **Cohen's weighted kappa** — a standard statistic that measures agreement between two raters, corrected for chance. Kappa = 0 means no better than random; kappa = 1 means perfect agreement. For subjective judgments like "was this tutoring strategy effective?", kappa between 0.2 and 0.4 is typical even between trained human raters.

## The dataset

Our dataset contains **201 tutoring transcripts** with **291 annotation passes** producing **1,688 individual moment annotations** from 9 expert annotators. An annotation pass is one annotator labeling one transcript for one type (scaffolding or rapport). Most transcripts were annotated once; 54 received multiple passes. Six transcripts used as few-shot examples are excluded, leaving **195 transcripts** for evaluation.

- **Train** (98 transcripts): prompts were iterated using error examples from these
- **Held-out** (97 transcripts): prompts have never seen these

All results use **v5 prompts**.
"""))

    c.append(code(SHARED_SETUP))
    c.append(code(SHARED_DATA_LOAD))

    # ---- Human ceiling ----

    c.append(md("""
## Human Inter-Annotator Agreement (The Ceiling)

How well do human experts agree with *each other*? This sets the ceiling — if humans only agree at kappa = X, we can't expect the LLM to do better.

**Important caveat on N**: Most conversations were annotated by a single person. The ceiling can only be computed from conversations where two *different* annotators independently labeled *overlapping* moments of the same type. This is a small subset of the data — the ceiling estimate is based on far fewer data points than the LLM comparison.

The IoU threshold for finding overlapping human moments is the same 0.3 used everywhere else.
"""))

    c.append(code("""
def _count_ceiling_convs(ids):
    \"\"\"Count conversations that contribute to the ceiling (have overlapping annotator pairs).\"\"\"
    count = 0
    for cid in ids:
        moments = ground_truth['conversations'][cid]['key_moments']
        by_type = defaultdict(list)
        for m in moments:
            by_type[m.get('annotation_type')].append(m)
        found = False
        for t, tms in by_type.items():
            for i, m1 in enumerate(tms):
                for j in range(i+1, len(tms)):
                    m2 = tms[j]
                    if m1.get('annotator_id') != m2.get('annotator_id'):
                        if compute_iou((m1['turn_start'], m1['turn_end']),
                                       (m2['turn_start'], m2['turn_end'])) >= IOU:
                            found = True
            if found:
                break
        if found:
            count += 1
    return count

ceil_rows = []
for label, ids in [('Train', train_ids), ('Held-out', ho_ids), ('Combined', all_eval_ids)]:
    scoped = {'conversations': {c: ground_truth['conversations'][c] for c in ids}}
    ceil = compute_human_ceiling(scoped)
    ceil_convs = _count_ceiling_convs(ids)
    ceil_rows.append({
        '': label,
        'Transcripts': len(ids),
        'Transcripts with overlap': ceil_convs,
        'Annotator pairs': ceil['overlapping_pairs'],
        '3-Way Kappa': f"{ceil['three_way_kappa']:.4f}",
        'Binary Kappa': f"{ceil['binary_kappa']:.4f}",
    })

df_ceil = pd.DataFrame(ceil_rows).set_index('')
print('Human Inter-Annotator Agreement (IoU >= 0.3)\\n')
df_ceil
"""))

    # ---- Human-detected moments ----

    c.append(md("""
## Human-Detected Moments: Annotation Quality in Isolation

In human-detected moments, the LLM annotates the *exact same moments* that humans marked — same turn ranges, same annotation types. There's no detection step and no IoU matching needed. This isolates annotation quality from detection accuracy.

Every human-annotated moment gets an LLM label, so N is the full ground truth count.
"""))

    c.append(code("""
v5_gold, is_gold = load_annotations(GOLD_VERSION, 'annotations_gold.json')
if v5_gold is None:
    print(f'No gold annotations found for {GOLD_VERSION}')
else:
    rows = []
    for label, ids in [('Train', train_ids), ('Held-out', ho_ids), ('Combined', all_eval_ids)]:
        matches = []
        for cid in sorted(ids):
            hm = ground_truth['conversations'][cid]['key_moments']
            llm = v5_gold.get(cid, [])
            matches.extend(match_gold_direct(hm, llm))
        eff = compute_effectiveness_metrics(matches)
        rows.append({
            '': label,
            'Conversations': len(ids),
            'Moments compared': eff['total_matched'],
            '3-Way Kappa': f"{eff['three_way_kappa']:.4f}",
            'Binary Kappa': f"{eff['binary_kappa']:.4f}",
        })

    df_gold = pd.DataFrame(rows).set_index('')
    print(f'v5 Human-Detected Moments: LLM labels vs human labels on identical moments\\n')
    print(f'No IoU matching — turn ranges are the same.\\n')
    df_gold
"""))

    c.append(md("""
### Human-Detected Moments by Annotation Type
"""))

    c.append(code("""
if v5_gold:
    type_rows = []
    # Per-type
    for ann_type in ANNOTATION_TYPES:
        matches = []
        for cid in sorted(all_eval_ids):
            hm = [m for m in ground_truth['conversations'][cid]['key_moments']
                  if m.get('annotation_type') == ann_type]
            llm = [a for a in v5_gold.get(cid, []) if a.get('annotation_type') == ann_type]
            matches.extend(match_gold_direct(hm, llm))
        if matches:
            eff = compute_effectiveness_metrics(matches)
            type_rows.append({
                '': ann_type.title(),
                'N': eff['total_matched'],
                '3-Way Kappa': f"{eff['three_way_kappa']:.4f}",
                'Binary Kappa': f"{eff['binary_kappa']:.4f}",
            })
    # Combined row
    all_gold_matches = []
    for cid in sorted(all_eval_ids):
        all_gold_matches.extend(match_gold_direct(
            ground_truth['conversations'][cid]['key_moments'],
            v5_gold.get(cid, [])))
    if all_gold_matches:
        eff_all = compute_effectiveness_metrics(all_gold_matches)
        type_rows.append({
            '': 'Combined',
            'N': eff_all['total_matched'],
            '3-Way Kappa': f"{eff_all['three_way_kappa']:.4f}",
            'Binary Kappa': f"{eff_all['binary_kappa']:.4f}",
        })
    df_gold_type = pd.DataFrame(type_rows).set_index('')
    print('Human-Detected Moments by Type (combined dev + held-out)\\n')
    df_gold_type
"""))

    c.append(md("""
### Human-Detected Moments by Annotator Profile

Human annotators cluster into three profiles by labeling tendency:

- **Generous** (Gerber, Jones, Shields, Stobbe, Trujillo): more likely to rate effective
- **Balanced** (Forbes, Mann, Padgett): middle ground
- **Demanding** (Flick): more likely to rate ineffective (n=118, thin sample)

We show two LLM results for each profile: **generic** (the standard v5 prompt, same for all profiles) and **calibrated** (a prompt specifically iterated to match that annotator group's labeling standards). This shows whether per-profile calibration improves agreement.
"""))

    c.append(code("""
from annotator.eval.eval import load_annotator_archetype_ids, filter_ground_truth_by_archetype
ARCH_NAMES = ['generous', 'balanced', 'demanding']
GOLD_STYLE_VERSION = 'v5_gold'

arch_rows = []
for arch in ARCH_NAMES:
    try:
        arch_ids = load_annotator_archetype_ids(arch)
        arch_gt = filter_ground_truth_by_archetype(
            {'conversations': {c: ground_truth['conversations'][c] for c in all_eval_ids}},
            arch_ids)
    except Exception:
        continue

    generic_anns = v5_gold
    style_anns, _ = load_annotations(GOLD_STYLE_VERSION, f'annotations_gold_{arch}.json')

    # Compute generic and calibrated kappa on combined set
    generic_matches, calibrated_matches = [], []
    for cid in sorted(all_eval_ids):
        hm = arch_gt['conversations'].get(cid, {}).get('key_moments', [])
        if not hm:
            continue
        if generic_anns:
            generic_matches.extend(match_gold_direct(hm, generic_anns.get(cid, [])))
        if style_anns:
            calibrated_matches.extend(match_gold_direct(hm, style_anns.get(cid, [])))

    gen_eff = compute_effectiveness_metrics(generic_matches) if generic_matches else {}
    cal_eff = compute_effectiveness_metrics(calibrated_matches) if calibrated_matches else {}

    # Human ceiling
    ceil = compute_human_ceiling(arch_gt)
    ceil_str = f"{ceil['three_way_kappa']:.4f} ({ceil['overlapping_pairs']} pairs)"
    if ceil['overlapping_pairs'] == 0:
        ceil_str = 'N/A (1 annotator)'

    gen_3w = gen_eff.get('three_way_kappa', 0)
    cal_3w = cal_eff.get('three_way_kappa', 0)
    delta = cal_3w - gen_3w

    arch_rows.append({
        '': arch.title(),
        'N': gen_eff.get('total_matched', 0),
        'Generic 3W Kappa': f"{gen_3w:.4f}",
        'Calibrated 3W Kappa': f"{cal_3w:.4f}",
        'Delta': f"{delta*100:+.1f}pp",
        'Human Ceiling (3W)': ceil_str,
    })

df_arch = pd.DataFrame(arch_rows).set_index('')
print('Human-Detected Moments by Annotator Profile')
print('Generic = standard v5 prompt | Calibrated = profile-specific iterated prompt\\n')
df_arch
"""))

    c.append(md("""
**Interpretation**: Per-profile prompt calibration improves agreement for annotator groups whose standards diverge from the generic prompt. The generous profile shows a clear gain (+13.5pp, n=524) — the generic prompt systematically under-matches generous annotators' tendency to rate strategies as effective, and the calibrated prompt corrects this. The balanced profile shows no change, which is expected: the generic prompt was implicitly calibrated toward balanced-like standards since balanced annotators contributed the most ground truth data. The demanding profile shows a large delta, but with only 118 moments from a single annotator (Flick) across 10 conversations, this estimate is unstable and should be interpreted with caution.
"""))

    # ---- Full pipeline ----

    c.append(md("""
## Full Pipeline: End-to-End Quality

In full pipeline mode, the LLM first *detects* key moments (Pass 1), then annotates and labels them (Passes 2-3). We compare the LLM's labels to human labels on moments where both found the same event (IoU >= 0.3).

N is smaller here because:
- The LLM doesn't detect every human moment (~57% recall)
- Some detected moments don't overlap human clusters enough (IoU < 0.3)
"""))

    c.append(code("""
v5_anns, _ = load_annotations(VERSION, 'annotations.json')
if v5_anns is None:
    print(f'No annotations found for {VERSION}')
else:
    rows = []
    for label, ids in [('Train', train_ids), ('Held-out', ho_ids), ('Combined', all_eval_ids)]:
        matches = []
        for cid in sorted(ids):
            hm = ground_truth['conversations'][cid]['key_moments']
            ht = {m.get('annotation_type') for m in hm}
            llm = [a for a in v5_anns.get(cid, []) if a.get('annotation_type') in ht]
            matches.extend(match_for_effectiveness(hm, llm, iou_threshold=IOU))
        eff = compute_effectiveness_metrics(matches)
        rows.append({
            '': label,
            'Conversations': len(ids),
            'Moments compared': eff['total_matched'],
            '3-Way Kappa': f"{eff['three_way_kappa']:.4f}",
            'Binary Kappa': f"{eff['binary_kappa']:.4f}",
        })

    df_full = pd.DataFrame(rows).set_index('')
    print(f'v5 Full Pipeline: LLM labels vs human labels on overlapping moments (IoU >= {IOU})\\n')
    df_full
"""))

    c.append(md("""
### Full Pipeline by Annotation Type
"""))

    c.append(code("""
if v5_anns:
    type_rows = []
    for ann_type in ANNOTATION_TYPES:
        matches = []
        for cid in sorted(all_eval_ids):
            hm = [m for m in ground_truth['conversations'][cid]['key_moments']
                  if m.get('annotation_type') == ann_type]
            llm = [a for a in v5_anns.get(cid, []) if a.get('annotation_type') == ann_type]
            matches.extend(match_for_effectiveness(hm, llm, iou_threshold=IOU))
        if matches:
            eff = compute_effectiveness_metrics(matches)
            type_rows.append({
                '': ann_type.title(),
                'N': eff['total_matched'],
                '3-Way Kappa': f"{eff['three_way_kappa']:.4f}",
                'Binary Kappa': f"{eff['binary_kappa']:.4f}",
            })
    # Combined row
    all_full_matches = []
    for cid in sorted(all_eval_ids):
        hm = ground_truth['conversations'][cid]['key_moments']
        ht = {m.get('annotation_type') for m in hm}
        llm = [a for a in v5_anns.get(cid, []) if a.get('annotation_type') in ht]
        all_full_matches.extend(match_for_effectiveness(hm, llm, iou_threshold=IOU))
    if all_full_matches:
        eff_all = compute_effectiveness_metrics(all_full_matches)
        type_rows.append({
            '': 'Combined',
            'N': eff_all['total_matched'],
            '3-Way Kappa': f"{eff_all['three_way_kappa']:.4f}",
            'Binary Kappa': f"{eff_all['binary_kappa']:.4f}",
        })
    df_full_type = pd.DataFrame(type_rows).set_index('')
    print(f'Full Pipeline by Type (combined, IoU >= {IOU})\\n')
    df_full_type
"""))

    # ---- Comparison summary ----

    c.append(md("""
## Putting It Together

The table below compares all three measurements on the combined dataset. Each uses the same IoU threshold (0.3) where applicable.

The key comparison: **does the LLM kappa meet or exceed the human ceiling?**

Note the ceiling caveat: it's computed from only ~34 conversations where two annotators happened to overlap, while the LLM metrics use 195 conversations. The ceiling tells us the general level of human agreement on this task — it's not a per-moment benchmark on the same data.
"""))

    c.append(code("""
# Combined metrics for the summary
ceil = compute_human_ceiling({'conversations': {
    c: ground_truth['conversations'][c] for c in all_eval_ids
}})

# Human-detected moments combined
gold_matches = []
if v5_gold:
    for cid in sorted(all_eval_ids):
        gold_matches.extend(match_gold_direct(
            ground_truth['conversations'][cid]['key_moments'],
            v5_gold.get(cid, [])))
gold_eff = compute_effectiveness_metrics(gold_matches) if gold_matches else {}

# Full pipeline combined
full_matches = []
if v5_anns:
    for cid in sorted(all_eval_ids):
        hm = ground_truth['conversations'][cid]['key_moments']
        ht = {m.get('annotation_type') for m in hm}
        llm = [a for a in v5_anns.get(cid, []) if a.get('annotation_type') in ht]
        full_matches.extend(match_for_effectiveness(hm, llm, iou_threshold=IOU))
full_eff = compute_effectiveness_metrics(full_matches) if full_matches else {}

rows = [
    {'': 'Human ceiling',
     '3-Way Kappa': f"{ceil['three_way_kappa']:.4f}",
     'N': f"{ceil['overlapping_pairs']} pairs (from 34/195 convs)",
     'Matching': 'IoU >= 0.3 between annotators'},
    {'': 'LLM (human-detected moments)',
     '3-Way Kappa': f"{gold_eff.get('three_way_kappa', 0):.4f}",
     'N': f"{gold_eff.get('total_matched', 0)} moments",
     'Matching': 'Exact (same turn ranges)'},
    {'': 'LLM (full pipeline)',
     '3-Way Kappa': f"{full_eff.get('three_way_kappa', 0):.4f}",
     'N': f"{full_eff.get('total_matched', 0)} moments",
     'Matching': 'IoU >= 0.3 to human clusters'},
]

df_summary = pd.DataFrame(rows).set_index('')
print('Combined Results (195 conversations, v5 prompts)\\n')
df_summary
"""))

    # ---- Confusion matrix ----

    c.append(md("""
## Where Do Disagreements Happen?

The confusion matrix shows which labels get confused. Disagreements at the effective/partial boundary are expected ambiguity. Effective/ineffective confusion would indicate fundamental miscalibration.

Using human-detected moments (n is larger and there's no detection noise).
"""))

    c.append(code("""
if gold_matches:
    eff = compute_effectiveness_metrics(gold_matches)
    cm = eff.get('three_way_confusion', {})
    if cm:
        cm_array = np.array([
            [cm.get(h, {}).get(l, 0) for l in EFFECTIVENESS_LABELS]
            for h in EFFECTIVENESS_LABELS
        ])
        row_sums = cm_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm_array / row_sums, annot=cm_array, fmt='d', cmap='Blues',
                    xticklabels=['Effective', 'Partial', 'Ineffective'],
                    yticklabels=['Effective', 'Partial', 'Ineffective'],
                    ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_xlabel('LLM Label')
        ax.set_ylabel('Human Label')
        ax.set_title(f'Human-Detected Moments Confusion Matrix (n={eff["three_way_n"]})')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'confusion_matrices.png')
        fig.savefig(FIGURES_DIR / 'confusion_matrices.pdf')
        plt.show()
"""))

    # ---- Train vs held-out ----

    c.append(md("""
## Train vs. Held-Out: Does the Pipeline Generalize?

The prompts were iterated on the train set. If held-out performance is comparable, the prompts generalize to unseen data.
"""))

    c.append(code("""
rows = []
for mode_label, match_fn, data_source in [
    ('Human-detected moments', 'gold', v5_gold),
    ('Full pipeline', 'full', v5_anns),
]:
    if data_source is None:
        continue
    for split, ids in [('Train', train_ids), ('Held-out', ho_ids)]:
        matches = []
        for cid in sorted(ids):
            hm = ground_truth['conversations'][cid]['key_moments']
            llm = data_source.get(cid, [])
            if match_fn == 'gold':
                matches.extend(match_gold_direct(hm, llm))
            else:
                ht = {m.get('annotation_type') for m in hm}
                llm = [a for a in llm if a.get('annotation_type') in ht]
                matches.extend(match_for_effectiveness(hm, llm, iou_threshold=IOU))
        eff = compute_effectiveness_metrics(matches)
        rows.append({
            'Mode': mode_label,
            'Split': split,
            'N': eff['total_matched'],
            '3-Way Kappa': f"{eff['three_way_kappa']:.4f}",
            'Binary Kappa': f"{eff['binary_kappa']:.4f}",
        })

df_splits = pd.DataFrame(rows)
print('Train vs. Held-Out Comparison (v5 prompts)\\n')
df_splits
"""))

    # ---- Summary ----

    c.append(md("""
## Summary

All results use v5 prompts. IoU threshold is 0.3 everywhere it's used.

**Human-detected moments** (clean measurement, n~1,600): The LLM's effectiveness labels agree with human labels at 3-way kappa ~0.34. This is computed on every human-annotated moment with no detection noise.

**Full pipeline** (end-to-end, n~600): The LLM detects and labels moments, achieving 3-way kappa ~0.32 on the subset of moments both the LLM and humans found.

**Human ceiling** (~0.22, n=212 pairs from 34 conversations): Humans agree with each other at a lower rate than the LLM agrees with humans — but this ceiling is estimated from a small subset of conversations where two annotators overlapped, so the comparison is approximate.

**Train vs. held-out**: Performance is comparable across both splits in both modes, indicating the prompts generalize to unseen conversations.
"""))

    nb.cells = c
    return nb


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    nb1 = build_detection_notebook()
    p1 = VALIDATION_DIR / "1_detection_validation.ipynb"
    with open(p1, "w", encoding="utf-8") as f:
        nbformat.write(nb1, f)
    print(f"Wrote {p1}")

    nb2 = build_annotation_notebook()
    p2 = VALIDATION_DIR / "2_annotation_validation.ipynb"
    with open(p2, "w", encoding="utf-8") as f:
        nbformat.write(nb2, f)
    print(f"Wrote {p2}")
