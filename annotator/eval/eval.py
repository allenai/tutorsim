"""
8-metric evaluation scorecard.

Supports three modes:
  --mode detections   : evaluate only key moment detection (reads detections.json)
  --mode annotations  : evaluate only labeling quality (reads annotations.json)
  --mode full         : evaluate everything (reads annotations.json)

Metrics:
  PRIMARY (optimize):
    1. Cluster Recall (IoU >= 0.3)  -- fraction of human clusters found
    2. Binary Kappa                 -- effective vs not-effective agreement

  DIAGNOSTIC (understand quality):
    3. Moment Precision (IoU >= 0.3) -- fraction of LLM moments matching a human cluster
    4. Mean IoU                      -- average overlap quality of matched pairs
    5. Within-Human-Range            -- % of LLM labels matching at least one annotator

  GUARDRAILS (flag regressions):
    6. Effective Rate                -- flag if >60% (rubber-stamping)
    7. Zero-Partial Rate             -- flag if >30% (missing nuance)
    8. Invalid Labels                -- flag if >0 (hallucinated values)

  CONTEXT (not for hill-climbing):
    - Human ceiling, confusion matrix, counts, binary accuracy

Usage:
    python -m annotator.eval.eval --version v1
    python -m annotator.eval.eval --version v1 --mode detections
    python -m annotator.eval.eval --version v1 --mode annotations

    # Compare versions side-by-side
    python -m annotator.eval.eval --compare v1 v2 --mode detections
    python -m annotator.eval.eval --compare v1 v2 v3 --mode full

Ported from archive_per_annotator/eval.py with multi-mode support.
"""

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

from ..core.config import get_valid_styles
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

def match_for_effectiveness(human_moments, llm_moments, iou_threshold=0.5):
    """Match LLM moments to human clusters by IoU for effectiveness comparison."""
    clusters = merge_overlapping_ranges(human_moments)
    matches = []
    used_llm = set()

    for cluster in clusters:
        c_range = (cluster["turn_start"], cluster["turn_end"])
        c_type = cluster["annotation_type"]
        best_iou = 0
        best_idx = None

        for i, l in enumerate(llm_moments):
            if i in used_llm:
                continue
            if l.get("annotation_type") != c_type:
                continue
            iou = compute_iou(c_range, (l["turn_start"], l["turn_end"]))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is not None and best_iou >= iou_threshold:
            llm_moment = llm_moments[best_idx]
            used_llm.add(best_idx)

            labels_3way = [m.get("strategy_label", "unclear") for m in cluster["moments"]]
            valid_labels = [l for l in labels_3way if l in EFFECTIVENESS_LABELS]
            consensus_3way = compute_consensus_label(valid_labels) if valid_labels else "unclear"
            consensus_binary = map_to_binary(consensus_3way)

            llm_label_3way = llm_moment.get("effectiveness", "unclear")
            llm_label_binary = map_to_binary(llm_label_3way)

            per_annotator = {
                m.get("annotator_id", "unknown"): m.get("strategy_label", "unclear")
                for m in cluster["moments"]
            }

            matches.append({
                "cluster": cluster,
                "llm_moment": llm_moment,
                "iou": round(best_iou, 4),
                "consensus_3way": consensus_3way,
                "consensus_binary": consensus_binary,
                "llm_label_3way": llm_label_3way,
                "llm_label_binary": llm_label_binary,
                "per_annotator_labels": per_annotator,
            })

    return matches


def match_gold_direct(human_moments, llm_moments):
    """Direct 1-to-1 matching for annotations mode (gold moments).

    When LLM annotated gold truth moments, turn ranges are identical.
    Match by exact (turn_start, turn_end, annotation_type) -- no IoU needed.
    Each human moment gets compared to its corresponding LLM annotation.
    """
    # Index LLM moments by (turn_start, turn_end, annotation_type)
    llm_index = {}
    for l in llm_moments:
        key = (l["turn_start"], l["turn_end"], l.get("annotation_type", ""))
        llm_index[key] = l

    matches = []
    for m in human_moments:
        key = (m["turn_start"], m["turn_end"], m.get("annotation_type", ""))
        llm_moment = llm_index.get(key)
        if not llm_moment:
            continue

        human_label = m.get("strategy_label", "unclear")
        llm_label_3way = llm_moment.get("effectiveness", "unclear")

        matches.append({
            "cluster": {
                "turn_start": m["turn_start"],
                "turn_end": m["turn_end"],
                "annotation_type": m.get("annotation_type", ""),
                "moments": [m],
            },
            "llm_moment": llm_moment,
            "iou": 1.0,
            "consensus_3way": human_label,
            "consensus_binary": map_to_binary(human_label),
            "llm_label_3way": llm_label_3way,
            "llm_label_binary": map_to_binary(llm_label_3way),
            "per_annotator_labels": {
                m.get("annotator_id", "unknown"): human_label,
            },
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

def compute_human_ceiling(ground_truth, ann_type_filter=None):
    """Compute inter-annotator agreement (the ceiling we're trying to reach)."""
    pairs_3way = []
    pairs_binary = []

    for conv_data in ground_truth.get("conversations", {}).values():
        moments = conv_data["key_moments"]
        by_type = defaultdict(list)
        for m in moments:
            by_type[m.get("annotation_type")].append(m)

        for t, type_moments in by_type.items():
            if ann_type_filter and t != ann_type_filter:
                continue
            for i, m1 in enumerate(type_moments):
                for j in range(i + 1, len(type_moments)):
                    m2 = type_moments[j]
                    if m1.get("annotator_id") == m2.get("annotator_id"):
                        continue
                    iou = compute_iou(
                        (m1["turn_start"], m1["turn_end"]),
                        (m2["turn_start"], m2["turn_end"])
                    )
                    if iou >= 0.3:
                        l1 = m1.get("strategy_label", "unclear")
                        l2 = m2.get("strategy_label", "unclear")
                        if l1 in EFFECTIVENESS_LABELS and l2 in EFFECTIVENESS_LABELS:
                            pairs_3way.append((l1, l2))
                            b1, b2 = map_to_binary(l1), map_to_binary(l2)
                            if b1 and b2:
                                pairs_binary.append((b1, b2))

    result = {"overlapping_pairs": len(pairs_3way)}

    if pairs_3way:
        h3, l3 = zip(*pairs_3way)
        agree = sum(1 for a, b in pairs_3way if a == b)
        result["three_way_agreement"] = round(agree / len(pairs_3way), 4)
        result["three_way_kappa"] = cohens_kappa(list(h3), list(l3), EFFECTIVENESS_LABELS)
    else:
        result["three_way_agreement"] = 0
        result["three_way_kappa"] = 0

    if pairs_binary:
        hb, lb = zip(*pairs_binary)
        agree = sum(1 for a, b in pairs_binary if a == b)
        result["binary_agreement"] = round(agree / len(pairs_binary), 4)
        result["binary_kappa"] = cohens_kappa(list(hb), list(lb), BINARY_LABELS)
    else:
        result["binary_agreement"] = 0
        result["binary_kappa"] = 0

    return result


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


def load_detections_as_moments(version: str) -> dict[str, list[dict]] | None:
    """Load detections.json and return as {conv_id: [moment dicts]}."""
    data = load_annotator_result(version, "detections.json")
    if data is None:
        return None

    moments_by_conv = {}
    for conv_id, conv_data in data["results"].items():
        moments_by_conv[conv_id] = conv_data.get("detections", [])
    return moments_by_conv


def resolve_annotations_filename(version: str, mode: str,
                                  annotator_style: str | None = None) -> str | None:
    """Resolve the correct annotations filename given mode and optional style.

    Preference order (annotations mode):
      1. annotations_gold_{style}.json  (style-specific gold run)
      2. annotations_gold.json          (baseline gold run, any style eval)
      3. annotations_{style}.json       (style-specific detected-moments run)
      4. annotations.json               (baseline detected-moments run)
    """
    style_suffix = f"_{annotator_style}" if annotator_style else ""

    if mode == "annotations":
        f = f"annotations_gold{style_suffix}.json"
        if annotator_result_exists(version, f):
            return f
        if annotator_result_exists(version, "annotations_gold.json"):
            return "annotations_gold.json"

    f = f"annotations{style_suffix}.json"
    if annotator_result_exists(version, f):
        return f
    return "annotations.json"


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
        annotations_by_conv[conv_id] = conv_data.get("annotations", [])
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

    # --- Human ceiling ---
    ceiling = output.get("human_ceiling", {})
    if ceiling.get("overlapping_pairs", 0) > 0:
        print(f"\n  HUMAN CEILING")
        print(f"  {'-' * 40}")
        print(f"  Binary Kappa:          {ceiling.get('binary_kappa', 0):.4f}  "
              f"(agreement: {ceiling.get('binary_agreement', 0):.1%})")
        print(f"  3-Way Kappa:           {ceiling.get('three_way_kappa', 0):.4f}  "
              f"(agreement: {ceiling.get('three_way_agreement', 0):.1%})")
        print(f"  Overlapping pairs:     {ceiling['overlapping_pairs']}")

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

            if t_guard.get("total_annotations", 0) > 0:
                print(f"  Effective Rate:     {t_guard.get('effective_rate', 0):.1%}  |  "
                      f"Invalid: {t_guard.get('invalid_labels', 0)}")

            if t_ceil.get("overlapping_pairs", 0) > 0:
                print(f"  Human ceiling:      kappa={t_ceil.get('binary_kappa', 0):.4f}  "
                      f"({t_ceil['overlapping_pairs']} pairs)")

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

    # --- RQ2: Effectiveness ---
    if mode in ("annotations", "full"):
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

    # Human ceiling (context, from first version)
    ceiling = baseline.get("human_ceiling", {})
    if ceiling.get("overlapping_pairs", 0) > 0:
        print(f"\n  HUMAN CEILING (context)")
        print(f"  Binary Kappa:  {ceiling.get('binary_kappa', 0):.4f}  |  "
              f"3-Way Kappa:  {ceiling.get('three_way_kappa', 0):.4f}")

    print(f"\n{'=' * 68}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="8-metric evaluation scorecard")
    parser.add_argument("--version", default=None,
                        help="Results version (e.g. v1)")
    parser.add_argument("--mode", choices=["full", "detections", "annotations"],
                        default="full",
                        help="What to evaluate (default: full)")
    parser.add_argument("--compare", nargs="+", metavar="VERSION",
                        help="Compare eval results across versions (e.g. --compare v1 v2 v3)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Evaluate against only this annotator archetype's ground truth")
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

    # Load ground truth (with optional archetype filtering), restricted to train split
    ground_truth = load_ground_truth(annotator_style=style)
    train_ids = load_split_ids("train")
    ground_truth["conversations"] = {
        conv_id: conv_data
        for conv_id, conv_data in ground_truth["conversations"].items()
        if conv_id in train_ids
    }
    print(f"Restricted ground truth to train split: {len(ground_truth['conversations'])} conversations")

    if style:
        print(f"Filtered ground truth to '{style}' annotators")
        print(f"  Conversations with matching annotations: "
              f"{len(ground_truth['conversations'])}")

    # --- Load LLM data based on mode ---
    llm_moments_by_conv = {}
    annotations_by_conv = {}
    is_gold = False

    if args.mode == "detections":
        llm_moments_by_conv = load_detections_as_moments(version)
        if llm_moments_by_conv is None:
            print(f"ERROR: detections.json not found for version {version}")
            return
        print(f"Loaded detections for version {version}")
    else:
        ann_filename = resolve_annotations_filename(version, args.mode, style)
        annotations_by_conv, is_gold = load_annotations(version, ann_filename)
        if annotations_by_conv is None:
            print(f"ERROR: {ann_filename} not found for version {version}")
            return
        # Also use annotations as moments for detection metrics
        llm_moments_by_conv = annotations_by_conv
        source_str = "gold truth moments" if is_gold else "detected moments"
        print(f"Loaded annotations: {ann_filename} (source: {source_str})")

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
                # Direct 1-to-1 matching (gold moments have identical turn ranges)
                matches = match_gold_direct(human_moments, llm_moments)
            else:
                # IoU-based cluster matching (detected moments have different ranges)
                matches = match_for_effectiveness(human_moments, llm_moments)
            all_matches.extend(matches)

    # --- Compute metrics ---
    detection = None
    effectiveness = None
    guardrails = None

    if args.mode in ("full", "detections"):
        detection = compute_detection_metrics(human_moments_by_conv, llm_moments_by_conv)

    if args.mode in ("full", "annotations"):
        effectiveness = compute_effectiveness_metrics(all_matches)
        guardrails = compute_guardrails(annotations_by_conv)

    ceiling = compute_human_ceiling(ground_truth)

    # --- Per-type ---
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
    output["human_ceiling"] = ceiling
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
