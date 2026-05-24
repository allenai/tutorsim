"""
8-metric evaluation scorecard.

Supports four modes:
  --mode detections      : evaluate only key moment detection (reads detections.json)
  --mode annotations_old : evaluate only labeling quality -- original approach (Cohen's kappa,
                           majority-vote consensus)
  --mode annotations     : evaluate labeling quality -- new approach (Krippendorff's alpha,
                           mean-score consensus)
  --mode full            : temporarily disabled; choose an explicit annotations mode

Annotation evaluation approach (annotations and annotations_old modes)
-----------------------------------------------------------------------
For each conversation, human moments and LM annotations are compared as follows:

  1. UNIQUENESS: moments are deduplicated by (conversation, turn_start, turn_end,
     annotation_type). Each unique span is one unit of evaluation.

  2. CONSENSUS: when multiple human annotators labeled the same span, their
     effectiveness labels are aggregated into a single consensus label.
     - annotations_old: majority vote with ordinal median tiebreak
     - annotations:     mean score (effective=1, partial=0, ineffective=-1);
                        threshold >=0.5 -> effective, <=-0.5 -> ineffective, else partial

  3. MATCHING: each unique human span is matched to a single LM annotation.
     - gold mode (--gold):     exact (turn_start, turn_end, annotation_type) lookup;
                               first LM annotation wins if duplicates exist
     - non-gold mode:          highest-IoU LM annotation (threshold 0.3);
                               each LM annotation can only be matched once

  4. AGREEMENT metric:
     - annotations_old: Cohen's kappa (binary and 3-way) between consensus and LM label
     - annotations:     Krippendorff's alpha (ordinal) between consensus and LM label

All agreement metrics are reported per annotation type (scaffolding / rapport) only —
no aggregated totals across types.

Usage:
    python -m annotator.eval.eval --version v1 --mode annotations --profile anthropic
    python -m annotator.eval.eval --version v1 --mode detections
    python -m annotator.eval.eval --version v1 --mode annotations_old --profile anthropic

    # Compare versions side-by-side
    python -m annotator.eval.eval --compare v1 v2 --mode detections

Ported from archive_per_annotator/eval.py with multi-mode support.
"""

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import krippendorff

from ..core.config import get_valid_styles, get_annotation_types
from ..core.utils import (
    compute_iou, merge_overlapping_ranges, load_ground_truth, load_split_ids,
    EXAMPLE_CONV_IDS,
)
from ..core.storage import (
    load_annotator_result, save_annotator_result, annotator_result_exists,
    list_annotator_result_files,
)

EFFECTIVENESS_LABELS = ["effective", "partial", "ineffective"]
BINARY_LABELS = ["right", "wrong"]
ANNOTATION_TYPES = ["scaffolding", "rapport"]


def compute_consensus_label(labels):
    """Majority vote with ordinal median tiebreak."""
    if not labels:
        return "unclear"
    counts = Counter(labels)
    max_count = max(counts.values())
    winners = [l for l, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return winners[0]
    ordinal = {"effective": 0, "partial": 1, "ineffective": 2}
    reverse = {0: "effective", 1: "partial", 2: "ineffective"}
    values = sorted([ordinal[l] for l in labels if l in ordinal])
    if not values:
        return "unclear"
    return reverse[values[len(values) // 2]]


def compute_mean_consensus_label(labels, threshold=0.5):
    """Mean score with thresholds: effective=1, partial=0, ineffective=-1.

    Score >= threshold -> effective, score <= -threshold -> ineffective, else partial.
    """
    _score = {"effective": 1, "partial": 0, "ineffective": -1}
    scores = [_score[l] for l in labels if l in _score]
    if not scores:
        return "unclear"
    mean = sum(scores) / len(scores)
    if mean >= threshold:
        return "effective"
    if mean <= -threshold:
        return "ineffective"
    return "partial"


_ORDINAL_CODE = {"effective": 0, "partial": 1, "ineffective": 2}


ALPHA_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
MIN_ANNOTATOR_MOMENTS = 50


def compute_per_annotator_alpha(matches, ground_truth, ann_type):
    """Compute Krippendorff's alpha between each prolific annotator and the LLM.

    Only includes annotators with more than MIN_ANNOTATOR_MOMENTS key moments
    in the ground truth for this annotation type. Alpha is computed over the
    subset of matched moments where that annotator has a label.
    """
    # Count unique spans per annotator for this type (an annotator may appear
    # multiple times for the same span in the raw ground truth).
    annotator_spans: dict[str, set] = defaultdict(set)
    for conv_id, conv_data in ground_truth.get("conversations", {}).items():
        for m in conv_data.get("key_moments", []):
            if (m.get("annotation_type") == ann_type
                    and m.get("strategy_label") in EFFECTIVENESS_LABELS):
                annotator_spans[m.get("annotator_id")].add(
                    (conv_id, m["turn_start"], m["turn_end"]))
    annotator_totals = {aid: len(spans) for aid, spans in annotator_spans.items()}

    results = {}
    for annotator_id, n_total in annotator_totals.items():
        if n_total <= MIN_ANNOTATOR_MOMENTS:
            continue

        pairs = [
            (_ORDINAL_CODE[m["per_annotator_labels"][annotator_id]],
             _ORDINAL_CODE[m["llm_label_3way"]])
            for m in matches
            if annotator_id in m["per_annotator_labels"]
            and m["per_annotator_labels"][annotator_id] in EFFECTIVENESS_LABELS
            and m["llm_label_3way"] in EFFECTIVENESS_LABELS
        ]
        if not pairs:
            continue

        matrix = np.full((2, len(pairs)), np.nan)
        for j, (ann_code, llm_code) in enumerate(pairs):
            matrix[0, j] = ann_code
            matrix[1, j] = llm_code

        try:
            alpha = round(
                krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"),
                4,
            )
        except ValueError:
            alpha = 1.0

        results[annotator_id] = {"alpha": alpha, "n_matched": len(pairs), "n_total": n_total}

    return results


def recompute_consensus(matches, threshold):
    """Return a copy of matches with consensus_3way recomputed at a given threshold."""
    result = []
    for m in matches:
        labels = [l for l in m["per_annotator_labels"].values() if l in EFFECTIVENESS_LABELS]
        consensus = compute_mean_consensus_label(labels, threshold=threshold) if labels else "unclear"
        result.append({**m, "consensus_3way": consensus,
                        "consensus_binary": map_to_binary(consensus)})
    return result


def compute_krippendorff_alpha(all_matches, consensus_label="unknown"):
    """Compute Krippendorff's alpha (ordinal) between human consensus and LLM.

    Builds a 2-row matrix: row 0 = human consensus label per unit,
    row 1 = LLM label per unit. consensus_3way on each match must already
    be set by the calling match function using the appropriate consensus_fn.
    """
    if not all_matches:
        return {"alpha": None, "n_units": 0, "consensus": consensus_label,
                "confusion": {}}

    matrix = np.full((2, len(all_matches)), np.nan)
    pairs = []
    for j, match in enumerate(all_matches):
        human_code = _ORDINAL_CODE.get(match["consensus_3way"])
        llm_code = _ORDINAL_CODE.get(match["llm_label_3way"])
        if human_code is not None:
            matrix[0, j] = human_code
        if llm_code is not None:
            matrix[1, j] = llm_code
        if match["consensus_3way"] in EFFECTIVENESS_LABELS and match["llm_label_3way"] in EFFECTIVENESS_LABELS:
            pairs.append((match["consensus_3way"], match["llm_label_3way"]))

    try:
        alpha = round(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"),
            4,
        )
    except ValueError:
        alpha = 1.0

    return {
        "alpha": alpha,
        "n_units": len(all_matches),
        "consensus": consensus_label,
        "confusion": build_confusion(pairs, EFFECTIVENESS_LABELS),
    }


def map_to_binary(label):
    """effective -> 'right', partial/ineffective -> 'wrong'."""
    if label == "effective":
        return "right"
    elif label in ("partial", "ineffective"):
        return "wrong"
    return None


def cohens_kappa(labels_a, labels_b, categories):
    """Cohen's Kappa with linear weights for ordinal scales."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            matrix[cat_idx[a]][cat_idx[b]] += 1
    weights = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)] if k > 1 else [[0]]
    po = sum(matrix[i][j] * weights[i][j] for i in range(k) for j in range(k)) / n
    row_totals = [sum(matrix[i]) for i in range(k)]
    col_totals = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_totals[i] * col_totals[j] * weights[i][j]
             for i in range(k) for j in range(k)) / (n * n)
    if pe == 0:
        return 1.0
    return round(1 - po / pe, 4)


def build_confusion(pairs, categories):
    """Build confusion matrix from (ground_truth, predicted) pairs."""
    matrix = {h: {l: 0 for l in categories} for h in categories}
    for h, l in pairs:
        if h in matrix and l in matrix[h]:
            matrix[h][l] += 1
    return matrix


def filter_moments_by_type(moments_by_conv, ann_type):
    filtered = {}
    for conv_id, moments in moments_by_conv.items():
        typed = [m for m in moments if m.get("annotation_type") == ann_type]
        if typed:
            filtered[conv_id] = typed
    return filtered


def filter_matches_by_type(matches, ann_type):
    return [m for m in matches if m["cluster"]["annotation_type"] == ann_type]


def filter_annotations_by_type(annotations_by_conv, ann_type):
    filtered = {}
    for conv_id, anns in annotations_by_conv.items():
        typed = [a for a in anns if a.get("annotation_type") == ann_type]
        if typed:
            filtered[conv_id] = typed
    return filtered


# ===================================================================
# 1. DETECTION -- Cluster Recall + Moment Precision (IoU >= 0.3)
# ===================================================================

def compute_detection_metrics(human_moments_by_conv, llm_moments_by_conv,
                              iou_threshold=0.3):
    """Compute cluster recall, moment precision, and mean IoU."""
    total_clusters = 0
    found_clusters = 0
    total_llm = 0
    matched_llm = 0
    matched_ious = []
    per_conv = {}

    for conv_id in set(human_moments_by_conv.keys()) | set(llm_moments_by_conv.keys()):
        human_moments = human_moments_by_conv.get(conv_id, [])
        llm_moments = llm_moments_by_conv.get(conv_id, [])
        clusters = merge_overlapping_ranges(human_moments)

        conv_found = 0
        conv_ious = []
        for cluster in clusters:
            c_range = (cluster["turn_start"], cluster["turn_end"])
            c_type = cluster["annotation_type"]
            best_iou = 0
            for l in llm_moments:
                if l.get("annotation_type") != c_type:
                    continue
                iou = compute_iou(c_range, (l["turn_start"], l["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                conv_found += 1
                conv_ious.append(best_iou)

        conv_matched_llm = 0
        for l in llm_moments:
            l_range = (l["turn_start"], l["turn_end"])
            l_type = l.get("annotation_type")
            best_iou = 0
            for cluster in clusters:
                if cluster["annotation_type"] != l_type:
                    continue
                iou = compute_iou(l_range, (cluster["turn_start"], cluster["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                conv_matched_llm += 1

        human_turns = set()
        for m in human_moments:
            for t in range(m["turn_start"], m["turn_end"] + 1):
                human_turns.add(t)
        novel = sum(1 for l in llm_moments
                    if not (set(range(l["turn_start"], l["turn_end"] + 1)) & human_turns))

        per_conv[conv_id] = {
            "clusters": len(clusters),
            "found": conv_found,
            "llm_moments": len(llm_moments),
            "matched_llm": conv_matched_llm,
            "novel": novel,
            "recall": round(conv_found / len(clusters), 4) if clusters else 0,
            "precision": round(conv_matched_llm / len(llm_moments), 4) if llm_moments else 0,
        }

        total_clusters += len(clusters)
        found_clusters += conv_found
        total_llm += len(llm_moments)
        matched_llm += conv_matched_llm
        matched_ious.extend(conv_ious)

    total_novel = sum(v["novel"] for v in per_conv.values())
    total_human_moments = sum(len(human_moments_by_conv.get(cid, []))
                              for cid in per_conv)

    return {
        "cluster_recall": round(found_clusters / total_clusters, 4) if total_clusters else 0,
        "moment_precision": round(matched_llm / total_llm, 4) if total_llm else 0,
        "mean_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0,
        "iou_threshold": iou_threshold,
        "total_human_clusters": total_clusters,
        "found_clusters": found_clusters,
        "total_llm_annotations": total_llm,
        "matched_llm_annotations": matched_llm,
        "total_human_annotations": total_human_moments,
        "novel_llm_annotations": total_novel,
        "per_conversation": per_conv,
    }


# ===================================================================
# 2. EFFECTIVENESS -- Binary Kappa + Within-Human-Range (IoU >= 0.5)
# ===================================================================

def match_for_effectiveness(human_moments, llm_moments, iou_threshold=0.3,
                            consensus_fn=None):
    """Match unique human spans to the best LLM moment by IoU.

    Groups human moments by (turn_start, turn_end, annotation_type) first,
    collecting all annotators' labels for each unique span into a consensus.
    Then finds the best-matching LLM annotation by IoU for each unique span.
    """
    fn = consensus_fn or compute_consensus_label

    # Group human moments by unique span, collecting all annotators' labels
    human_groups = defaultdict(list)
    for m in human_moments:
        key = (m["turn_start"], m["turn_end"], m.get("annotation_type", ""))
        human_groups[key].append(m)

    matches = []
    used_llm = set()

    for key, group_moments in human_groups.items():
        h_range = (key[0], key[1])
        h_type = key[2]
        best_iou = 0
        best_idx = None

        for i, l in enumerate(llm_moments):
            if i in used_llm:
                continue
            if l.get("annotation_type") != h_type:
                continue
            iou = compute_iou(h_range, (l["turn_start"], l["turn_end"]))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is not None and best_iou >= iou_threshold:
            llm_moment = llm_moments[best_idx]
            used_llm.add(best_idx)

            per_annotator = {}
            for m in group_moments:
                ann_id = m.get("annotator_id", "unknown")
                per_annotator[ann_id] = m.get("strategy_label", "unclear")  # last entry wins

            valid_labels = [l for l in per_annotator.values() if l in EFFECTIVENESS_LABELS]
            consensus_3way = fn(valid_labels) if valid_labels else "unclear"
            llm_label_3way = llm_moment.get("effectiveness", "unclear")

            matches.append({
                "cluster": {
                    "turn_start": key[0],
                    "turn_end": key[1],
                    "annotation_type": h_type,
                    "moments": group_moments,
                },
                "llm_moment": llm_moment,
                "iou": round(best_iou, 4),
                "consensus_3way": consensus_3way,
                "consensus_binary": map_to_binary(consensus_3way),
                "llm_label_3way": llm_label_3way,
                "llm_label_binary": map_to_binary(llm_label_3way),
                "per_annotator_labels": per_annotator,
            })

    return matches


def match_gold_direct(human_moments, llm_moments, consensus_fn=None):
    """Matching for gold moments, grouping by (turn_start, turn_end, annotation_type).

    Multiple human annotators may label the same turn range; their labels are
    collected and aggregated into a single consensus via consensus_fn.
    For the LLM side, the first annotation for each key is used; a warning is
    printed if duplicates are found (they should not exist after load_gold_moments
    deduplication).
    """
    fn = consensus_fn or compute_consensus_label

    # Index LLM annotations by turn range key; warn and take first if multiple
    llm_groups = {}
    for l in llm_moments:
        key = (l["turn_start"], l["turn_end"], l.get("annotation_type", ""))
        if key in llm_groups:
            print(f"WARNING: multiple LLM annotations for gold moment "
                  f"turns {key[0]}-{key[1]} ({key[2]}); using first")
        else:
            llm_groups[key] = l  # store full annotation to preserve action/result text

    # Group human moments by turn range key
    human_groups = defaultdict(list)
    for m in human_moments:
        key = (m["turn_start"], m["turn_end"], m.get("annotation_type", ""))
        human_groups[key].append(m)

    matches = []
    for key, group_moments in human_groups.items():
        if key not in llm_groups:
            continue

        per_annotator = {}
        for m in group_moments:
            ann_id = m.get("annotator_id", "unknown")
            if ann_id not in per_annotator:
                per_annotator[ann_id] = m.get("strategy_label", "unclear")

        valid_human = [l for l in per_annotator.values() if l in EFFECTIVENESS_LABELS]
        consensus_3way = fn(valid_human) if valid_human else "unclear"

        llm_ann = llm_groups[key]
        llm_label_3way = llm_ann.get("effectiveness", "unclear")
        if llm_label_3way not in EFFECTIVENESS_LABELS:
            llm_label_3way = "unclear"

        llm_moment = llm_ann

        matches.append({
            "cluster": {
                "turn_start": key[0],
                "turn_end": key[1],
                "annotation_type": key[2],
                "moments": group_moments,
            },
            "llm_moment": llm_moment,
            "iou": 1.0,
            "consensus_3way": consensus_3way,
            "consensus_binary": map_to_binary(consensus_3way),
            "llm_label_3way": llm_label_3way,
            "llm_label_binary": map_to_binary(llm_label_3way),
            "per_annotator_labels": per_annotator,
        })

    return matches


def compute_effectiveness_metrics(all_matches):
    """Compute binary kappa, binary accuracy, within-human-range, and confusion matrix."""
    pairs_binary = [(m["consensus_binary"], m["llm_label_binary"])
                    for m in all_matches
                    if m["consensus_binary"] is not None
                    and m["llm_label_binary"] is not None]

    result = {"total_matched": len(all_matches)}

    if pairs_binary:
        hb, lb = zip(*pairs_binary)
        agree = sum(1 for h, l in pairs_binary if h == l)
        result["binary_accuracy"] = round(agree / len(pairs_binary), 4)
        result["binary_kappa"] = cohens_kappa(list(hb), list(lb), BINARY_LABELS)
        result["binary_confusion"] = build_confusion(pairs_binary, BINARY_LABELS)
        result["binary_n"] = len(pairs_binary)
    else:
        result["binary_accuracy"] = 0
        result["binary_kappa"] = 0
        result["binary_confusion"] = {}
        result["binary_n"] = 0

    pairs_3way = [(m["consensus_3way"], m["llm_label_3way"])
                  for m in all_matches
                  if m["consensus_3way"] in EFFECTIVENESS_LABELS
                  and m["llm_label_3way"] in EFFECTIVENESS_LABELS]

    if pairs_3way:
        h3, l3 = zip(*pairs_3way)
        agree_3 = sum(1 for a, b in pairs_3way if a == b)
        result["three_way_accuracy"] = round(agree_3 / len(pairs_3way), 4)
        result["three_way_kappa"] = cohens_kappa(list(h3), list(l3), EFFECTIVENESS_LABELS)
        result["three_way_confusion"] = build_confusion(pairs_3way, EFFECTIVENESS_LABELS)
        result["three_way_n"] = len(pairs_3way)
    else:
        result["three_way_accuracy"] = 0
        result["three_way_kappa"] = 0
        result["three_way_confusion"] = {}
        result["three_way_n"] = 0

    within_range = 0
    for m in all_matches:
        llm_label = m["llm_label_3way"]
        annotator_labels = set(m["per_annotator_labels"].values())
        if llm_label in annotator_labels:
            within_range += 1
    result["within_human_range"] = within_range
    result["within_human_range_pct"] = round(
        within_range / len(all_matches), 4) if all_matches else 0

    return result


# ===================================================================
# 3. GUARDRAILS -- Distribution health checks
# ===================================================================

def compute_guardrails(annotations_by_conv):
    """Compute guardrail metrics: effective rate, zero-partial rate, invalid labels."""
    all_anns = []
    for anns in annotations_by_conv.values():
        all_anns.extend(anns)

    total = len(all_anns)
    if total == 0:
        return {"total_annotations": 0, "total_conversations": 0}

    eff_counts = Counter(a.get("effectiveness", "") for a in all_anns)
    effective_rate = eff_counts.get("effective", 0) / total
    partial_rate = eff_counts.get("partial", 0) / total
    ineffective_rate = eff_counts.get("ineffective", 0) / total

    zero_partial_convs = 0
    total_convs = len(annotations_by_conv)
    for anns in annotations_by_conv.values():
        effs = [a.get("effectiveness") for a in anns]
        if "partial" not in effs:
            zero_partial_convs += 1

    invalid_count = sum(1 for a in all_anns
                        if a.get("effectiveness", "") not in EFFECTIVENESS_LABELS)

    return {
        "total_annotations": total,
        "total_conversations": total_convs,
        "effective_rate": round(effective_rate, 4),
        "partial_rate": round(partial_rate, 4),
        "ineffective_rate": round(ineffective_rate, 4),
        "effectiveness_distribution": dict(eff_counts),
        "zero_partial_conv_rate": round(
            zero_partial_convs / total_convs, 4) if total_convs else 0,
        "zero_partial_convs": zero_partial_convs,
        "invalid_labels": invalid_count,
        "annotations_per_conversation": round(total / total_convs, 1) if total_convs else 0,
    }


# ===================================================================
# 4. HUMAN CEILING -- Inter-annotator agreement context
# ===================================================================

def _cluster_moments_for_iaa(moments):
    """Group moments from different annotators into IoU-based connected-component clusters.

    Uses union-find so that transitively overlapping moments end up in the same unit.
    Single-annotator moments are included but will be filtered by callers.
    """
    if not moments:
        return []
    n = len(moments)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if moments[i].get("annotator_id") == moments[j].get("annotator_id"):
                continue
            if compute_iou(
                (moments[i]["turn_start"], moments[i]["turn_end"]),
                (moments[j]["turn_start"], moments[j]["turn_end"]),
            ) >= 0.3:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters = defaultdict(list)
    for i in range(n):
        clusters[_find(i)].append(moments[i])
    return list(clusters.values())


def compute_human_ceiling(ground_truth, ann_type_filter=None):
    """Compute human-human Krippendorff's alpha (ordinal) as the agreement ceiling.

    Builds a raters x units reliability matrix from IoU-clustered human moments,
    where each unit is a cluster of overlapping moments from 2+ annotators.
    ann_type_filter must be specified (scaffolding or rapport) — do not call
    without it, as mixing annotation types is not meaningful.
    """
    units = []

    for conv_data in ground_truth.get("conversations", {}).values():
        moments = [
            m for m in conv_data.get("key_moments", [])
            if m.get("strategy_label") in EFFECTIVENESS_LABELS
            and (ann_type_filter is None or m.get("annotation_type") == ann_type_filter)
        ]
        for cluster in _cluster_moments_for_iaa(moments):
            if len({m["annotator_id"] for m in cluster}) < 2:
                continue
            unit = {}
            for m in cluster:
                if m["annotator_id"] not in unit:
                    unit[m["annotator_id"]] = _ORDINAL_CODE[m["strategy_label"]]
            units.append(unit)

    if not units:
        return {"alpha": None, "n_units": 0, "n_raters": 0}

    all_annotators = sorted({a for u in units for a in u})
    rater_idx = {a: i for i, a in enumerate(all_annotators)}

    matrix = np.full((len(all_annotators), len(units)), np.nan)
    for j, unit in enumerate(units):
        for ann_id, code in unit.items():
            matrix[rater_idx[ann_id], j] = code

    try:
        alpha = round(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"),
            4,
        )
    except ValueError:
        alpha = 1.0

    return {"alpha": alpha, "n_units": len(units), "n_raters": len(all_annotators)}


# ===================================================================
# Data loading helpers
# ===================================================================

def load_annotator_archetype_ids(archetype: str) -> set[str]:
    """Load the set of annotator IDs belonging to the given archetype.

    Reads from archetype_annotators in config.yaml.
    """
    from ..core.config import get_archetype_annotators
    result = get_archetype_annotators(archetype)
    if result is None:
        raise ValueError(
            f"Unknown archetype '{archetype}'. "
            f"Check archetype_annotators in config.yaml."
        )
    return result


def filter_ground_truth_by_archetype(ground_truth: dict, archetype_ids: set[str]) -> dict:
    """Return a copy of ground_truth with key_moments filtered to the given annotator IDs.

    Moments that have no annotators from the archetype are dropped.
    """
    filtered = {"conversations": {}}
    for conv_id, conv_data in ground_truth["conversations"].items():
        moments = [m for m in conv_data.get("key_moments", [])
                   if m.get("annotator_id") in archetype_ids]
        if moments:
            filtered["conversations"][conv_id] = {"key_moments": moments}
    return filtered


def load_detections_as_moments(version: str,
                               profile: str | None = None,
                               annotator_style: str | None = None,
                               split: str = "train") -> dict[str, list[dict]] | None:
    """Load detections file and return as {conv_id: [moment dicts]}.

    Tries suffixed filenames (matching detect.py output naming) before
    falling back to detections.json.
    """
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    # Only fall back to unsuffixed files for the default train split
    candidates = [f"detections{profile_suffix}{style_suffix}{split_suffix}.json"]
    if split == "train":
        candidates += [f"detections{profile_suffix}{style_suffix}.json", "detections.json"]
    # Deduplicate while preserving order
    seen: set = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    data = None
    for filename in candidates:
        data = load_annotator_result(version, filename)
        if data is not None:
            print(f"Loaded detections: {filename}")
            break

    # If no combined file found, try merging per-target files
    if data is None:
        merged_results: dict = {}
        for t in get_annotation_types():
            per_target_candidates = [f"detections{profile_suffix}{style_suffix}{split_suffix}_{t}.json"]
            if split == "train":
                per_target_candidates.append(f"detections{profile_suffix}{style_suffix}_{t}.json")
            for fname in per_target_candidates:
                tdata = load_annotator_result(version, fname)
                if tdata is not None:
                    print(f"Loaded detections: {fname}")
                    for conv_id, conv_data in tdata["results"].items():
                        if conv_id not in merged_results:
                            merged_results[conv_id] = {"detections": []}
                        merged_results[conv_id]["detections"].extend(conv_data.get("detections", []))
                    break
        if merged_results:
            data = {"results": merged_results}

    if data is None:
        return None

    moments_by_conv = {}
    for conv_id, conv_data in data["results"].items():
        transcript_id = conv_id.rsplit("_", 1)[-1]
        moments_by_conv[transcript_id] = conv_data.get("detections", [])
    return moments_by_conv


def resolve_annotations_filename(version: str, mode: str,
                                  annotator_style: str | None = None,
                                  profile: str | None = None,
                                  split: str = "train") -> str | None:
    """Resolve the correct annotations filename given mode, optional style, profile, and split.

    Preference order (annotations mode), split-suffixed candidates tried first:
      1. annotations_gold_{profile}_{style}_{split}.json
      2. annotations_gold_{profile}_{style}.json
      3. annotations_gold_{profile}_{split}.json
      4. annotations_gold_{profile}.json
      5. annotations_gold_{style}_{split}.json
      6. annotations_gold_{style}.json
      7. annotations_gold_{split}.json
      8. annotations_gold.json
      (then same pattern without _gold prefix)
    """
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""

    def _find_first(prefixes: list[str]) -> str | None:
        base_candidates = [
            f"{p}{profile_suffix}{style_suffix}{split_suffix}.json" for p in prefixes
        ] + [
            f"{p}{profile_suffix}{split_suffix}.json" for p in prefixes
        ] + [
            f"{p}{style_suffix}{split_suffix}.json" for p in prefixes
        ] + [
            f"{p}{split_suffix}.json" for p in prefixes
        ]
        # Only fall back to unsuffixed files for the default train split
        if split == "train":
            base_candidates += [
                f"{p}{profile_suffix}{style_suffix}.json" for p in prefixes
            ] + [
                f"{p}{profile_suffix}.json" for p in prefixes
            ] + [
                f"{p}{style_suffix}.json" for p in prefixes
            ] + [
                f"{p}.json" for p in prefixes
            ]
        # When no style specified, also try per-target suffixed files as fallback
        if not annotator_style:
            for t in get_annotation_types():
                t_suffix = f"_{t}"
                base_candidates += [f"{p}{profile_suffix}{split_suffix}{t_suffix}.json" for p in prefixes]
                if split == "train":
                    base_candidates += [f"{p}{profile_suffix}{t_suffix}.json" for p in prefixes]
        seen: set = set()
        for f in base_candidates:
            if f in seen:
                continue
            seen.add(f)
            if annotator_result_exists(version, f):
                return f
        return None

    if mode in ("annotations_old", "annotations"):
        result = _find_first(["annotations_gold"])
        if result:
            return result

    result = _find_first(["annotations"])
    return result or "annotations.json"


def load_annotations_for_eval(version: str, mode: str,
                               annotator_style: str | None = None,
                               profile: str | None = None,
                               split: str = "train") -> tuple[dict, bool, str] | tuple[None, None, None]:
    """Load and merge per-target annotation files, falling back to a single combined file.

    Returns (annotations_by_conv, is_gold, description) or (None, None, None).
    """
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    want_gold = mode in ("annotations_old", "annotations")

    merged: dict = {}
    loaded_files: list = []
    is_gold = False

    for target in get_annotation_types():
        candidates = []
        if want_gold:
            candidates.append(f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}_{target}.json")
            if split == "train":
                candidates += [
                    f"annotations_gold{profile_suffix}{style_suffix}_{target}.json",
                    f"annotations_gold{profile_suffix}_{target}.json",
                ]
        candidates.append(f"annotations{profile_suffix}{style_suffix}{split_suffix}_{target}.json")
        if split == "train":
            candidates += [
                f"annotations{profile_suffix}{style_suffix}_{target}.json",
                f"annotations{profile_suffix}_{target}.json",
            ]
        for fname in candidates:
            data = load_annotator_result(version, fname)
            if data is not None:
                loaded_files.append(fname)
                is_gold = data.get("source") == "gold_truth"
                for conv_id, conv_data in data["results"].items():
                    transcript_id = conv_id.rsplit("_", 1)[-1]
                    if transcript_id not in merged:
                        merged[transcript_id] = []
                    merged[transcript_id].extend(conv_data.get("annotations", []))
                break

    if merged:
        return merged, is_gold, ", ".join(loaded_files)

    # Fall back to single combined file
    filename = resolve_annotations_filename(version, mode, annotator_style, profile=profile, split=split)
    annotations_by_conv, is_gold = load_annotations(version, filename)
    if annotations_by_conv is None:
        return None, None, None
    return annotations_by_conv, is_gold, filename


def load_annotations(version: str, filename: str) -> tuple[dict[str, list[dict]], bool] | tuple[None, None]:
    """Load annotations and return ({conv_id: [annotation dicts]}, is_gold).

    is_gold is True when annotations were produced from gold truth moments
    (annotate.py --gold), which means we should use direct matching instead
    of IoU-based cluster matching.
    """
    data = load_annotator_result(version, filename)
    if data is None:
        return None, None

    is_gold = data.get("source") == "gold_truth"

    annotations_by_conv = {}
    for conv_id, conv_data in data["results"].items():
        # Annotation results use compound keys (tutor_student_<uuid>);
        # ground truth uses the bare transcript UUID as its key.
        transcript_id = conv_id.rsplit("_", 1)[-1]
        annotations_by_conv[transcript_id] = conv_data.get("annotations", [])
    return annotations_by_conv, is_gold


# ===================================================================
# Print scorecard
# ===================================================================

def print_scorecard(output):
    """Print the 8-metric scorecard + context."""
    version = output["version"]
    mode = output["mode"]

    print(f"\n{'=' * 68}")
    print(f"  EVALUATION SCORECARD -- {version} ({mode})")
    print(f"  {output['num_conversations']} conversations evaluated")
    print(f"{'=' * 68}")

    # --- Detection metrics ---
    det = output.get("detection")
    if det:
        print(f"\n  DETECTION (RQ1)")
        print(f"  {'-' * 40}")
        print(f"  1. Cluster Recall:     {det['cluster_recall']:.4f}  "
              f"({det['found_clusters']}/{det['total_human_clusters']} clusters found, "
              f"IoU >= {det['iou_threshold']})")
        print(f"  3. Moment Precision:   {det['moment_precision']:.4f}  "
              f"({det['matched_llm_annotations']}/{det['total_llm_annotations']} "
              f"LLM moments matched)")
        print(f"  4. Mean IoU:           {det['mean_iou']:.4f}  "
              f"(avg overlap of matched pairs)")
        print(f"     Annotations:        {det['total_llm_annotations']} LLM | "
              f"{det['total_human_annotations']} human | "
              f"{det['total_human_clusters']} clusters | "
              f"{det['novel_llm_annotations']} novel")

    # --- Effectiveness metrics ---
    eff = output.get("effectiveness")
    if eff and eff.get("binary_n", 0) > 0:
        print(f"\n  EFFECTIVENESS (RQ2)")
        print(f"  {'-' * 40}")
        print(f"  2. Binary Kappa:       {eff['binary_kappa']:.4f}  "
              f"(effective vs not-effective, n={eff['binary_n']})")
        print(f"     Binary Accuracy:    {eff['binary_accuracy']:.4f}")
        print(f"     3-Way Kappa:        {eff.get('three_way_kappa', 0):.4f}  "
              f"(n={eff.get('three_way_n', 0)})")
        print(f"     3-Way Accuracy:     {eff.get('three_way_accuracy', 0):.4f}")
        print(f"  5. Within Human Range: {eff['within_human_range_pct']:.4f}  "
              f"({eff['within_human_range']}/{eff['total_matched']} "
              f"match an annotator)")

        cm = eff.get("three_way_confusion", {})
        if cm:
            print(f"\n  Confusion Matrix (rows = human consensus, cols = LLM):")
            print(f"  {'':>16s}  {'effective':>10s}  {'partial':>10s}  {'ineffective':>12s}")
            for h in EFFECTIVENESS_LABELS:
                row = cm.get(h, {})
                print(f"  {h:>16s}  {row.get('effective', 0):>10d}  "
                      f"{row.get('partial', 0):>10d}  {row.get('ineffective', 0):>12d}")

    # --- Guardrails ---
    guard = output.get("guardrails")
    if guard and guard.get("total_annotations", 0) > 0:
        print(f"\n  GUARDRAILS (RQ2)")
        print(f"  {'-' * 40}")
        eff_flag = " << WARNING" if guard['effective_rate'] > 0.60 else ""
        zp_flag = " << WARNING" if guard['zero_partial_conv_rate'] > 0.30 else ""
        inv_flag = " << WARNING" if guard['invalid_labels'] > 0 else ""

        print(f"  6. Effective Rate:     {guard['effective_rate']:.1%}  "
              f"(flag if >60%){eff_flag}")
        print(f"  7. Zero-Partial Convs: {guard['zero_partial_conv_rate']:.1%}  "
              f"(flag if >30%){zp_flag}")
        print(f"  8. Invalid Labels:     {guard['invalid_labels']}  "
              f"(flag if >0){inv_flag}")

        dist = guard['effectiveness_distribution']
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        print(f"     Distribution:       {dist_str}")
        print(f"     Per conversation:   {guard['annotations_per_conversation']} annotations")

    # --- Per-type ---
    if output.get("by_type"):
        print(f"\n{'=' * 68}")
        print(f"  PER-TYPE BREAKDOWN")
        print(f"{'=' * 68}")

        for ann_type in ANNOTATION_TYPES:
            td = output["by_type"].get(ann_type, {})
            if not td:
                continue

            t_det = td.get("detection", {})
            t_eff = td.get("effectiveness", {})
            t_iaa = td.get("iaa", {})
            t_guard = td.get("guardrails", {})
            t_ceil = td.get("human_ceiling", {})

            n_convs = len(t_det.get("per_conversation", {})) if t_det else t_guard.get("total_conversations", 0)
            print(f"\n  --- {ann_type.upper()} ({n_convs} convs) ---")

            if t_det:
                print(f"  Cluster Recall:     {t_det.get('cluster_recall', 0):.4f}  "
                      f"({t_det.get('found_clusters', 0)}/{t_det.get('total_human_clusters', 0)})")
                print(f"  Moment Precision:   {t_det.get('moment_precision', 0):.4f}  "
                      f"({t_det.get('matched_llm_annotations', 0)}/{t_det.get('total_llm_annotations', 0)})")
                print(f"  Mean IoU:           {t_det.get('mean_iou', 0):.4f}")

            if t_eff.get("binary_n", 0) > 0:
                print(f"  Binary Kappa:       {t_eff['binary_kappa']:.4f}  (n={t_eff['binary_n']})")
                print(f"  Within Human Range: {t_eff.get('within_human_range_pct', 0):.4f}  "
                      f"({t_eff.get('within_human_range', 0)}/{t_eff.get('total_matched', 0)})")

            if t_iaa.get("alpha") is not None:
                n_all = t_iaa.get('n_units', 0)
                n_dense3 = td.get("iaa_dense", {}).get("n_units", 0)
                n_dense5 = td.get("iaa_dense5", {}).get("n_units", 0)
                print(f"  Model-Human α (Krippendorff ordinal):")
                print(f"    {'threshold':<14s}  {'all units':>10s}  {'>=3 annotators':>14s}  {'>=5 annotators':>14s}")
                by_thresh = td.get("iaa_by_threshold", {})
                by_thresh_dense3 = td.get("iaa_dense_by_threshold", {})
                by_thresh_dense5 = td.get("iaa_dense5_by_threshold", {})
                for t in ALPHA_THRESHOLDS:
                    a_all = by_thresh.get(t, {}).get("alpha")
                    a_d3 = by_thresh_dense3.get(t, {}).get("alpha")
                    a_d5 = by_thresh_dense5.get(t, {}).get("alpha")
                    marker = " *" if t == 0.5 else ""
                    print(f"    ±{t:<13}  "
                          f"{(f'{a_all:.4f}' if a_all is not None else 'n/a'):>10s}  "
                          f"{(f'{a_d3:.4f}' if a_d3 is not None else 'n/a'):>14s}  "
                          f"{(f'{a_d5:.4f}' if a_d5 is not None else 'n/a'):>14s}"
                          f"{marker}")
                print(f"    {'n units':<14s}  {n_all:>10d}  {n_dense3:>14d}  {n_dense5:>14d}")
                cm = t_iaa.get("confusion", {})
                if cm:
                    print(f"  Confusion at ±0.5 (rows=human consensus, cols=LLM):")
                    print(f"    {'':>12s}  {'effective':>10s}  {'partial':>10s}  {'ineffective':>12s}")
                    for h in EFFECTIVENESS_LABELS:
                        row = cm.get(h, {})
                        print(f"    {h:>12s}  {row.get('effective', 0):>10d}  "
                              f"{row.get('partial', 0):>10d}  {row.get('ineffective', 0):>12d}")

                kappa_thresh = td.get("kappa_by_threshold", {})
                if kappa_thresh:
                    print(f"  Cohen's κ 3-way (mean consensus):")
                    print(f"    {'threshold':<14s}  {'all units':>10s}  {'>=3 annotators':>14s}  {'>=5 annotators':>14s}")
                    kappa_d3 = td.get("kappa_dense_by_threshold", {})
                    kappa_d5 = td.get("kappa_dense5_by_threshold", {})
                    for t in ALPHA_THRESHOLDS:
                        k_all = kappa_thresh.get(t, {}).get("three_way_kappa")
                        k_d3  = kappa_d3.get(t, {}).get("three_way_kappa")
                        k_d5  = kappa_d5.get(t, {}).get("three_way_kappa")
                        marker = " *" if t == 0.5 else ""
                        print(f"    ±{t:<13}  "
                              f"{(f'{k_all:.4f}' if k_all is not None else 'n/a'):>10s}  "
                              f"{(f'{k_d3:.4f}'  if k_d3  is not None else 'n/a'):>14s}  "
                              f"{(f'{k_d5:.4f}'  if k_d5  is not None else 'n/a'):>14s}"
                              f"{marker}")

                per_ann = td.get("per_annotator_alpha", {})
                if per_ann:
                    print(f"  Per-annotator α vs LLM (>{MIN_ANNOTATOR_MOMENTS} moments in GT):")
                    print(f"    {'annotator':>10s}  {'α':>8s}  {'matched':>8s}  {'unique GT':>10s}")
                    for ann_id, info in sorted(per_ann.items(), key=lambda x: -x[1]["alpha"]):
                        print(f"    {ann_id[:10]:>10s}  {info['alpha']:>8.4f}  "
                              f"{info['n_matched']:>8d}  {info['n_total']:>10d}")

            if t_ceil.get("alpha") is not None:
                print(f"  Human-Human α:      {t_ceil['alpha']:.4f}  "
                      f"(ceiling, {t_ceil.get('n_units', 0)} units, "
                      f"{t_ceil.get('n_raters', 0)} raters)")

            if t_guard.get("total_annotations", 0) > 0:
                print(f"  Effective Rate:     {t_guard.get('effective_rate', 0):.1%}  |  "
                      f"Invalid: {t_guard.get('invalid_labels', 0)}")

    print(f"\n{'=' * 68}")


# ===================================================================
# Output helpers
# ===================================================================

def strip_per_conversation(output):
    """Return a copy of output with per_conversation data removed (for compact JSON)."""
    compact = copy.deepcopy(output)
    if "detection" in compact and "per_conversation" in compact.get("detection", {}):
        del compact["detection"]["per_conversation"]
    for ann_type_data in compact.get("by_type", {}).values():
        if "detection" in ann_type_data and "per_conversation" in ann_type_data.get("detection", {}):
            del ann_type_data["detection"]["per_conversation"]
    return compact


def load_eval_json(version, mode):
    """Load eval_{mode}.json, falling back to eval.json for legacy results."""
    data = load_annotator_result(version, f"eval_{mode}.json")
    if data is not None:
        return data
    legacy = load_annotator_result(version, "eval.json")
    if legacy is not None:
        if legacy.get("mode") == mode or mode == "full":
            return legacy
    return None


# ===================================================================
# Comparison
# ===================================================================

def fmt_pct(value, decimals=1):
    """Format a ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_delta(baseline, experiment, as_pct=True):
    """Format the delta between two values with direction indicator."""
    delta = experiment - baseline
    sign = "+" if delta >= 0 else ""
    if as_pct:
        return f"{sign}{delta * 100:.1f}pp"
    return f"{sign}{delta:.4f}"


def print_comparison(versions, evals, mode):
    """Print side-by-side comparison table for multiple versions."""
    n = len(versions)
    metric_w = 28
    val_w = 12
    delta_w = 10

    print(f"\n{'=' * 68}")
    print(f"  COMPARISON: {' vs '.join(versions)} ({mode})")
    print(f"{'=' * 68}")

    header = f"  {'Metric':<{metric_w}}"
    for v in versions:
        header += f" {v:>{val_w}}"
    if n >= 2:
        header += f" {'Delta':>{delta_w}}"
    print(header)
    print("  " + "-" * (metric_w + (val_w + 1) * n + (delta_w + 1 if n >= 2 else 0)))

    baseline = evals[0]
    latest = evals[-1]

    # --- RQ1: Detection ---
    if mode in ("detections", "full"):
        print(f"\n  DETECTION (RQ1)")
        for key, label in [
            ("cluster_recall", "Cluster Recall"),
            ("moment_precision", "Moment Precision"),
            ("mean_iou", "Mean IoU"),
        ]:
            row = f"  {label:<{metric_w}}"
            for e in evals:
                val = e.get("detection", {}).get(key, 0)
                row += f" {fmt_pct(val):>{val_w}}"
            if n >= 2:
                bv = baseline.get("detection", {}).get(key, 0)
                ev = latest.get("detection", {}).get(key, 0)
                row += f" {fmt_delta(bv, ev):>{delta_w}}"
            print(row)

        # Counts row
        row = f"  {'LLM annotations':<{metric_w}}"
        for e in evals:
            val = e.get("detection", {}).get("total_llm_annotations", 0)
            row += f" {val:>{val_w}}"
        if n >= 2:
            bv = baseline.get("detection", {}).get("total_llm_annotations", 0)
            ev = latest.get("detection", {}).get("total_llm_annotations", 0)
            row += f" {fmt_delta(bv, ev, as_pct=False):>{delta_w}}"
        print(row)

        row = f"  {'Novel (unmatched)':<{metric_w}}"
        for e in evals:
            val = e.get("detection", {}).get("novel_llm_annotations", 0)
            row += f" {val:>{val_w}}"
        if n >= 2:
            bv = baseline.get("detection", {}).get("novel_llm_annotations", 0)
            ev = latest.get("detection", {}).get("novel_llm_annotations", 0)
            row += f" {fmt_delta(bv, ev, as_pct=False):>{delta_w}}"
        print(row)

    # --- RQ2: Effectiveness (annotations_old) ---
    if mode in ("annotations_old", "full"):
        print(f"\n  EFFECTIVENESS (RQ2)")
        for key, label in [
            ("binary_kappa", "Binary Kappa"),
            ("binary_accuracy", "Binary Accuracy"),
            ("three_way_kappa", "3-Way Kappa"),
            ("within_human_range_pct", "Within Human Range"),
        ]:
            row = f"  {label:<{metric_w}}"
            for e in evals:
                val = e.get("effectiveness", {}).get(key, 0)
                row += f" {fmt_pct(val):>{val_w}}"
            if n >= 2:
                bv = baseline.get("effectiveness", {}).get(key, 0)
                ev = latest.get("effectiveness", {}).get(key, 0)
                row += f" {fmt_delta(bv, ev):>{delta_w}}"
            print(row)

        print(f"\n  GUARDRAILS (RQ2)")
        for key, label, is_count in [
            ("effective_rate", "Effective Rate", False),
            ("zero_partial_conv_rate", "Zero-Partial Rate", False),
            ("invalid_labels", "Invalid Labels", True),
        ]:
            row = f"  {label:<{metric_w}}"
            for e in evals:
                val = e.get("guardrails", {}).get(key, 0)
                if is_count:
                    row += f" {val:>{val_w}}"
                else:
                    row += f" {fmt_pct(val):>{val_w}}"
            if n >= 2:
                bv = baseline.get("guardrails", {}).get(key, 0)
                ev = latest.get("guardrails", {}).get(key, 0)
                row += f" {fmt_delta(bv, ev, as_pct=not is_count):>{delta_w}}"
            print(row)

    # --- RQ2: Agreement (annotations) ---
    if mode == "annotations":
        for ann_type in ANNOTATION_TYPES:
            print(f"\n  MODEL-HUMAN α ({ann_type.upper()})")
            row = f"  {'Krippendorff alpha':<{metric_w}}"
            for e in evals:
                val = e.get("by_type", {}).get(ann_type, {}).get("iaa", {}).get("alpha")
                row += f" {fmt_pct(val) if val is not None else 'n/a':>{val_w}}"
            if n >= 2:
                bv = baseline.get("by_type", {}).get(ann_type, {}).get("iaa", {}).get("alpha", 0)
                ev = latest.get("by_type", {}).get(ann_type, {}).get("iaa", {}).get("alpha", 0)
                row += f" {fmt_delta(bv, ev):>{delta_w}}"
            print(row)

    print(f"\n{'=' * 68}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="8-metric evaluation scorecard")
    parser.add_argument("--version", default=None,
                        help="Results version (e.g. v1)")
    parser.add_argument("--mode", choices=["full", "detections", "annotations_old", "annotations"],
                        default="full",
                        help="What to evaluate (default: full)")
    parser.add_argument("--compare", nargs="+", metavar="VERSION",
                        help="Compare eval results across versions (e.g. --compare v1 v2 v3)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Evaluate against only this annotator archetype's ground truth")
    parser.add_argument("--profile", default=None,
                        help="Config profile used when generating annotations (e.g. anthropic, gemini)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to evaluate against (default: train)")
    args = parser.parse_args()

    # --- Compare mode ---
    if args.compare:
        evals = []
        for v in args.compare:
            data = load_eval_json(v, args.mode)
            if data is None:
                print(f"ERROR: No eval results for {v} (mode: {args.mode})")
                return
            evals.append(data)
        print_comparison(args.compare, evals, args.mode)
        return

    # --- Normal eval mode ---
    if not args.version:
        from annotator.core.config import get_annotator_defaults
        defaults = get_annotator_defaults()
        version = defaults.get("version")
        if not version:
            parser.error("--version is required (unless using --compare, or set annotator.version in config.yaml)")
    else:
        version = args.version

    # Resolve style from config if not on CLI
    style = args.annotator_style
    if style is None:
        from annotator.core.config import get_annotator_defaults
        defaults = get_annotator_defaults()
        cfg_style = defaults.get("style")
        if cfg_style is not None:
            style = cfg_style

    # Load ground truth (with optional archetype filtering), restricted to the requested split
    ground_truth = load_ground_truth(annotator_style=style)
    split_ids = load_split_ids(args.split)
    ground_truth["conversations"] = {
        conv_id: conv_data
        for conv_id, conv_data in ground_truth["conversations"].items()
        if conv_id in split_ids
    }
    print(f"Restricted ground truth to {args.split} split: {len(ground_truth['conversations'])} conversations")

    if style:
        print(f"Filtered ground truth to '{style}' annotators")
        print(f"  Conversations with matching annotations: "
              f"{len(ground_truth['conversations'])}")

    if args.mode == "full":
        print("ERROR: --mode full is temporarily disabled.")
        print("  Choose an explicit annotations mode:")
        print("    --mode detections      (detection metrics only)")
        print("    --mode annotations_old (labeling quality, majority-vote consensus, Cohen's kappa)")
        print("    --mode annotations     (labeling quality, mean-score consensus, Krippendorff alpha)")
        return

    # --- Load LLM data based on mode ---
    llm_moments_by_conv = {}
    annotations_by_conv = {}
    is_gold = False

    if args.mode == "detections":
        llm_moments_by_conv = load_detections_as_moments(version,
                                                         profile=args.profile,
                                                         annotator_style=style,
                                                         split=args.split)
        if llm_moments_by_conv is None:
            print(f"ERROR: detections file not found for version {version}")
            return
    else:
        annotations_by_conv, is_gold, ann_desc = load_annotations_for_eval(
            version, args.mode, style, profile=args.profile, split=args.split)
        if annotations_by_conv is None:
            print(f"ERROR: No annotation files found for version {version}")
            return
        # Also use annotations as moments for detection metrics
        llm_moments_by_conv = annotations_by_conv
        source_str = "gold truth moments" if is_gold else "detected moments"
        print(f"Loaded annotations: {ann_desc} (source: {source_str})")

    # --- Build human moments ---
    if llm_moments_by_conv:
        eval_conv_ids = set(ground_truth["conversations"].keys()) & set(llm_moments_by_conv.keys())
    else:
        eval_conv_ids = set(ground_truth["conversations"].keys()) & set(annotations_by_conv.keys())

    # Exclude conversations used as few-shot examples in prompts (data leakage)
    excluded = eval_conv_ids & EXAMPLE_CONV_IDS
    if excluded:
        eval_conv_ids -= EXAMPLE_CONV_IDS
        print(f"Excluded {len(excluded)} example conversations from evaluation")

    print(f"Evaluating {len(eval_conv_ids)} conversations (mode: {args.mode})")

    consensus_fn = compute_mean_consensus_label if args.mode == "annotations" else compute_consensus_label

    human_moments_by_conv = {}
    all_matches = []

    for conv_id in eval_conv_ids:
        gt_conv = ground_truth["conversations"][conv_id]
        human_moments = gt_conv["key_moments"]
        human_types = {m.get("annotation_type") for m in human_moments}

        human_moments_by_conv[conv_id] = human_moments

        # Filter LLM moments to only types present in ground truth
        if conv_id in llm_moments_by_conv:
            llm_moments_by_conv[conv_id] = [
                m for m in llm_moments_by_conv[conv_id]
                if m.get("annotation_type") in human_types
            ]

        if conv_id in annotations_by_conv:
            annotations_by_conv[conv_id] = [
                a for a in annotations_by_conv[conv_id]
                if a.get("annotation_type") in human_types
            ]

        # Effectiveness matching
        if args.mode != "detections":
            llm_moments = annotations_by_conv.get(conv_id, [])
            if is_gold:
                matches = match_gold_direct(human_moments, llm_moments,
                                            consensus_fn=consensus_fn)
            else:
                matches = match_for_effectiveness(human_moments, llm_moments,
                                                  consensus_fn=consensus_fn)
            all_matches.extend(matches)

    # --- Compute metrics ---
    detection = None
    effectiveness = None
    guardrails = None

    if args.mode in ("full", "detections"):
        detection = compute_detection_metrics(human_moments_by_conv, llm_moments_by_conv)

    if args.mode in ("full", "annotations_old"):
        effectiveness = compute_effectiveness_metrics(all_matches)
        guardrails = compute_guardrails(annotations_by_conv)

    if args.mode == "annotations":
        guardrails = compute_guardrails(annotations_by_conv)

    # --- Per-type (all agreement metrics disaggregated by type) ---
    by_type = {}
    for ann_type in ANNOTATION_TYPES:
        type_result = {}
        h_filtered = filter_moments_by_type(human_moments_by_conv, ann_type)

        if detection:
            l_filtered = filter_moments_by_type(llm_moments_by_conv, ann_type)
            type_result["detection"] = compute_detection_metrics(h_filtered, l_filtered)

        if effectiveness:
            m_filtered = filter_matches_by_type(all_matches, ann_type)
            a_filtered = filter_annotations_by_type(annotations_by_conv, ann_type)
            type_result["effectiveness"] = compute_effectiveness_metrics(m_filtered)
            type_result["guardrails"] = compute_guardrails(a_filtered)

        if args.mode == "annotations":
            m_filtered = filter_matches_by_type(all_matches, ann_type)
            a_filtered = filter_annotations_by_type(annotations_by_conv, ann_type)
            m_dense3 = [m for m in m_filtered if len(m["per_annotator_labels"]) >= 3]
            m_dense5 = [m for m in m_filtered if len(m["per_annotator_labels"]) >= 5]
            type_result["iaa"] = compute_krippendorff_alpha(
                m_filtered, consensus_label="mean (±0.5 threshold)")
            type_result["iaa_by_threshold"] = {
                t: compute_krippendorff_alpha(
                    recompute_consensus(m_filtered, t),
                    consensus_label=f"mean (±{t} threshold)",
                )
                for t in ALPHA_THRESHOLDS
            }
            type_result["iaa_dense"] = compute_krippendorff_alpha(
                m_dense3, consensus_label="mean (±0.5 threshold), >=3 annotators")
            type_result["iaa_dense_by_threshold"] = {
                t: compute_krippendorff_alpha(
                    recompute_consensus(m_dense3, t),
                    consensus_label=f"mean (±{t} threshold), >=3 annotators",
                )
                for t in ALPHA_THRESHOLDS
            }
            type_result["iaa_dense5"] = compute_krippendorff_alpha(
                m_dense5, consensus_label="mean (±0.5 threshold), >=5 annotators")
            type_result["iaa_dense5_by_threshold"] = {
                t: compute_krippendorff_alpha(
                    recompute_consensus(m_dense5, t),
                    consensus_label=f"mean (±{t} threshold), >=5 annotators",
                )
                for t in ALPHA_THRESHOLDS
            }
            type_result["kappa_by_threshold"] = {
                t: compute_effectiveness_metrics(recompute_consensus(m_filtered, t))
                for t in ALPHA_THRESHOLDS
            }
            type_result["kappa_dense_by_threshold"] = {
                t: compute_effectiveness_metrics(recompute_consensus(m_dense3, t))
                for t in ALPHA_THRESHOLDS
            }
            type_result["kappa_dense5_by_threshold"] = {
                t: compute_effectiveness_metrics(recompute_consensus(m_dense5, t))
                for t in ALPHA_THRESHOLDS
            }
            type_result["per_annotator_alpha"] = compute_per_annotator_alpha(
                m_filtered, ground_truth, ann_type)
            type_result["guardrails"] = compute_guardrails(a_filtered)

        if args.mode != "detections":
            type_result["human_ceiling"] = compute_human_ceiling(
                ground_truth, ann_type_filter=ann_type)

        by_type[ann_type] = type_result

    # --- Assemble output ---
    output = {
        "version": version,
        "mode": args.mode,
        "num_conversations": len(eval_conv_ids),
    }
    if detection:
        output["detection"] = detection
    if effectiveness:
        output["effectiveness"] = effectiveness
    if guardrails:
        output["guardrails"] = guardrails
    output["by_type"] = by_type

    # Print and save
    print_scorecard(output)

    compact_output = strip_per_conversation(output)
    style_suffix = f"_{style}" if style else ""
    eval_filename = f"eval_{args.mode}{style_suffix}.json"
    save_annotator_result(version, eval_filename, compact_output)
    print(f"\nSaved to: {eval_filename} (version: {version})")


if __name__ == "__main__":
    main()
