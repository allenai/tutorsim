"""
Prompt iteration tools: Gemini advisor + detection disagreement analysis.

Two entry points:
  # Gemini advisor -- sends error examples + current prompt for analysis
  python -m annotator.iteration.advisor --pass detection --version v2 --type scaffolding
  python -m annotator.iteration.advisor --pass annotation --version v2 --type rapport
  python -m annotator.iteration.advisor --pass annotation_compare --version v2 --type rapport
  python -m annotator.iteration.advisor --pass annotation_excerpt --version v2 --type rapport
  python -m annotator.iteration.advisor --pass annotation_draft --version v2 --type rapport
  python -m annotator.iteration.advisor --pass detection_draft --version v2 --type scaffolding

  # annotation vs annotation_compare vs annotation_excerpt:
  #   annotation         -- includes tutoring transcript excerpts alongside SAR descriptions
  #   annotation_compare -- omits excerpts; shows full (untruncated) SAR descriptions only
  #   annotation_excerpt -- shows only transcript excerpts + labels; omits SAR descriptions entirely

  # Detection disagreement analysis -- detailed error breakdown
  python -m annotator.iteration.advisor analyze --version v1
  python -m annotator.iteration.advisor analyze --version v1 --type scaffolding --limit 10
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..core.client import ModelClient
from ..core.config import get_phase_config, load_config, get_valid_styles
from ..core.storage import load_annotator_result, save_annotator_result
from ..core.utils import (
    compute_iou, merge_overlapping_ranges, load_transcripts, get_excerpt,
    load_ground_truth, load_split_ids, REPO_ROOT, IOU_THRESHOLD,
    EXAMPLE_CONV_IDS,
)


def _load_ground_truth_train(annotator_style=None):
    """Load ground truth filtered to the train split (and optionally an annotator archetype)."""
    gt = load_ground_truth(annotator_style=annotator_style)
    train_ids = load_split_ids("train")
    gt["conversations"] = {k: v for k, v in gt["conversations"].items() if k in train_ids}
    return gt


# ===================================================================
# Detection error collection
# ===================================================================

def collect_detection_errors(version, ann_type, transcripts, limit=10, iou_threshold=0.5, profile=None,
                             show_human_annotations=True):
    """Collect detection errors and correct matches for comprehensive analysis."""
    gt = _load_ground_truth_train()

    profile_suffix = f"_{profile}" if profile else ""
    candidates = [
        f"detections{profile_suffix}_{ann_type}.json",
        f"detections{profile_suffix}.json",
        f"detections_{ann_type}.json",
        "detections.json",
    ]
    det_data = None
    for candidate in candidates:
        det_data = load_annotator_result(version, candidate)
        if det_data is not None:
            print(f"Loaded detections: {candidate} (version: {version})")
            break
    if det_data is None:
        raise FileNotFoundError(f"No detections found for version {version}. Tried: {', '.join(candidates)}")

    complete_misses = []
    near_misses = []
    false_positives = []
    good_matches = []

    # Detections use compound keys (tutor_student_uuid); ground truth uses bare UUIDs.
    uuid_to_det_conv = {}
    uuid_to_compound_key = {}
    for compound, conv_data in det_data.get("results", {}).items():
        uuid = compound.rsplit("_", 1)[-1]
        uuid_to_det_conv[uuid] = conv_data
        uuid_to_compound_key[uuid] = compound

    eval_ids = sorted(
        (set(gt["conversations"].keys()) & set(uuid_to_det_conv.keys()))
        - EXAMPLE_CONV_IDS
    )
    for conv_id in eval_ids:
        compound_key = uuid_to_compound_key[conv_id]
        human_moments = gt["conversations"][conv_id]["key_moments"]
        human_moments = [m for m in human_moments if m.get("annotation_type") == ann_type]
        llm_moments = [
            m for m in uuid_to_det_conv[conv_id].get("detections", [])
            if m.get("annotation_type") == ann_type
        ]

        clusters = merge_overlapping_ranges(human_moments)

        # Match clusters to detections
        for cluster in clusters:
            c_range = (cluster["turn_start"], cluster["turn_end"])
            best_iou = 0
            best_det = None
            for det in llm_moments:
                iou = compute_iou(c_range, (det["turn_start"], det["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            entry = {"conv_id": compound_key, "cluster": cluster, "nearest_det": best_det, "iou": best_iou}
            if best_iou >= iou_threshold:
                good_matches.append(entry)
            elif best_iou > 0:
                near_misses.append(entry)
            else:
                complete_misses.append(entry)

        # False positives
        for det in llm_moments:
            d_range = (det["turn_start"], det["turn_end"])
            best_iou = 0
            for cluster in clusters:
                iou = compute_iou(d_range, (cluster["turn_start"], cluster["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
            if best_iou < iou_threshold:
                false_positives.append({"conv_id": compound_key, "detection": det, "best_iou": best_iou})

    # Format examples as text blocks
    examples = []

    # Correct matches first -- so Gemini sees what works
    examples.append(f"=== CORRECT MATCHES ({len(good_matches)} total, showing {min(limit, len(good_matches))}) ===")
    examples.append("These are moments the LLM correctly detected (IoU >= 0.3). Study what makes these work.\n")
    for i, entry in enumerate(good_matches[:limit]):
        c = entry["cluster"]
        det = entry["nearest_det"]
        excerpt = get_excerpt(transcripts, entry["conv_id"], c["turn_start"], c["turn_end"])
        det_info = f"LLM: turns {det['turn_start']}-{det['turn_end']}, desc: {det.get('brief_description', '')[:200]}" if det else ""
        block = (
            f"Match {i+1}: Human turns {c['turn_start']}-{c['turn_end']} | {det_info} | IoU={entry['iou']:.2f}\n"
            f"  Transcript:\n{excerpt}\n"
        )
        if show_human_annotations:
            human_anns = "\n".join(
                f"    [{m.get('annotator_id', '?')}] S: {m.get('situation', '')[:200]} | A: {m.get('action', '')[:200]} | R: {m.get('result', '')[:200]}"
                for m in c["moments"]
            )
            block += f"  Human annotations:\n{human_anns}\n"
        examples.append(block)

    examples.append(f"\n=== COMPLETE MISSES ({len(complete_misses)} total, showing {min(limit, len(complete_misses))}) ===")
    examples.append("These are human-annotated moments the LLM completely missed.\n")
    for i, entry in enumerate(complete_misses[:limit]):
        c = entry["cluster"]
        excerpt = get_excerpt(transcripts, entry["conv_id"], c["turn_start"], c["turn_end"])
        block = (
            f"Miss {i+1}: {entry['conv_id'][:50]} turns {c['turn_start']}-{c['turn_end']}\n"
            f"  Transcript:\n{excerpt}\n"
        )
        if show_human_annotations:
            human_anns = "\n".join(
                f"    [{m.get('annotator_id', '?')}] S: {m.get('situation', '')[:200]} | A: {m.get('action', '')[:200]} | R: {m.get('result', '')[:200]}"
                for m in c["moments"]
            )
            block += f"  Human annotations:\n{human_anns}\n"
        examples.append(block)

    examples.append(f"\n=== NEAR-MISSES ({len(near_misses)} total, showing {min(limit, len(near_misses))}) ===")
    examples.append("LLM detected something nearby but IoU < 0.3 (wrong boundaries).\n")
    for i, entry in enumerate(near_misses[:limit]):
        c = entry["cluster"]
        det = entry["nearest_det"]
        excerpt = get_excerpt(transcripts, entry["conv_id"], c["turn_start"], c["turn_end"])
        det_info = f"LLM: turns {det['turn_start']}-{det['turn_end']}, desc: {det.get('brief_description', '')[:200]}" if det else "LLM: no nearby detection"
        block = (
            f"Near-miss {i+1}: Human turns {c['turn_start']}-{c['turn_end']} | {det_info} | IoU={entry['iou']:.2f}\n"
            f"  Transcript:\n{excerpt}\n"
        )
        if show_human_annotations:
            human_anns = "\n".join(
                f"    [{m.get('annotator_id', '?')}] S: {m.get('situation', '')[:200]} | A: {m.get('action', '')[:200]} | R: {m.get('result', '')[:200]}"
                for m in c["moments"]
            )
            block += f"  Human annotations:\n{human_anns}\n"
        examples.append(block)

    examples.append(f"\n=== FALSE POSITIVES ({len(false_positives)} total, showing {min(limit, len(false_positives))}) ===")
    examples.append("LLM detected a moment that doesn't match any human annotation.\n")
    for i, entry in enumerate(false_positives[:limit]):
        det = entry["detection"]
        excerpt = get_excerpt(transcripts, entry["conv_id"], det["turn_start"], det["turn_end"])
        examples.append(
            f"FP {i+1}: turns {det['turn_start']}-{det['turn_end']}, desc: {det.get('brief_description', '')[:200]}\n"
            f"  Transcript:\n{excerpt}\n"
        )

    stats = {
        "good_matches": len(good_matches),
        "total_clusters": len(complete_misses) + len(near_misses) + len(good_matches),
        "complete_misses": len(complete_misses),
        "near_misses": len(near_misses),
        "false_positives": len(false_positives),
    }

    return "\n".join(examples), stats


# ===================================================================
# Annotation error collection
# ===================================================================

ANNOTATOR_PROMPTS_DIR = REPO_ROOT / "prompts" / "annotator"


def collect_teacher_examples(ann_type: str, transcripts: dict,
                              batch_size: int = 100, batch_idx: int = 0,
                              ground_truth=None, annotator_style=None,
                              max_annotators: int = 5) -> tuple[str, str]:
    """Collect human SAR examples for annotation_draft mode.

    Groups ground truth moments by unique (conv_id, turn_start, turn_end),
    batches them, and formats up to 5 sampled annotators per moment.
    Returns (formatted_examples, batch_info).
    """
    if ground_truth is None:
        gt = _load_ground_truth_train(annotator_style=annotator_style)
    else:
        gt = ground_truth

    # Build UUID -> compound transcript key mapping
    uuid_to_compound = {k.rsplit("_", 1)[-1]: k for k in transcripts.keys()}

    # Collect all unique moments across conversations
    all_units = []
    for conv_id, conv_data in gt["conversations"].items():
        if conv_id in EXAMPLE_CONV_IDS:
            continue
        moments = [m for m in conv_data.get("key_moments", [])
                   if m.get("annotation_type") == ann_type]
        by_span = defaultdict(list)
        for m in moments:
            by_span[(m["turn_start"], m["turn_end"])].append(m)
        for (turn_start, turn_end), span_moments in by_span.items():
            all_units.append({
                "conv_id": conv_id,
                "compound_key": uuid_to_compound.get(conv_id, conv_id),
                "turn_start": turn_start,
                "turn_end": turn_end,
                "moments": span_moments,
            })

    random.shuffle(all_units)
    total = len(all_units)
    n_batches = max(1, (total + batch_size - 1) // batch_size)
    start = batch_idx * batch_size
    end = min(start + batch_size, total)
    batch = all_units[start:end]

    lines = []
    for i, unit in enumerate(batch):
        excerpt = get_excerpt(transcripts, unit["compound_key"],
                              unit["turn_start"], unit["turn_end"])
        sampled = random.sample(unit["moments"], min(max_annotators, len(unit["moments"])))
        lines.append(
            f"Example {i + 1}: turns {unit['turn_start']}-{unit['turn_end']} "
            f"({len(unit['moments'])} annotator(s), showing {len(sampled)})\n"
            f"  Transcript:\n{excerpt}\n"
        )
        for j, m in enumerate(sampled):
            label = m.get("strategy_label", "unclear")
            lines.append(
                f"  Annotator {j + 1} [{label}]:\n"
                f"    S: {m.get('situation', '')[:300]}\n"
                f"    A: {m.get('action', '')[:300]}\n"
                f"    R: {m.get('result', '')[:300]}\n"
            )

    batch_info = (f"Batch {batch_idx + 1} of {n_batches} "
                  f"(moments {start + 1}-{end} of {total} total for {ann_type})")
    return "".join(lines), batch_info, n_batches


def _format_human_annotators(annotators: list[dict], truncate: bool = True) -> str:
    """Format up to 5 sampled human annotators' SAR text for display."""
    lines = []
    for j, ann in enumerate(annotators):
        s = ann['situation'][:300] if truncate else ann['situation']
        a = ann['action'][:300] if truncate else ann['action']
        r = ann['result'][:300] if truncate else ann['result']
        lines.append(f"    Annotator {j+1} [{ann['label']}]:\n"
                     f"      S: {s}\n"
                     f"      A: {a}\n"
                     f"      R: {r}\n")
    return "".join(lines)


def collect_annotation_errors(version, ann_type, transcripts, limit=10,
                               ground_truth=None, annotator_style=None,
                               profile=None, show_excerpts=True, show_sar=True):
    """Collect annotation disagreements and agreements for comprehensive analysis.

    If ground_truth is provided, uses it directly (allows archetype filtering).
    Otherwise loads from data/ground_truth.json.

    When show_excerpts=False, transcript excerpts are omitted and SAR text is
    shown in full (untruncated) — useful for side-by-side annotation comparison.

    When show_sar=False, situation/action/result text is omitted entirely for
    both human annotators and the LLM -- only the human consensus label, the
    LLM label, and (if show_excerpts is True) transcript excerpts are shown
    (no per-annotator labels). Useful for judging moments from the transcript
    alone, without anchoring on existing SAR descriptions.
    """
    from ..eval.eval import (
        EFFECTIVENESS_LABELS, match_gold_direct, map_to_binary,
        compute_mean_consensus_label,
    )

    if ground_truth is None:
        gt = _load_ground_truth_train(annotator_style=annotator_style)
    else:
        gt = ground_truth

    # Try per-target first, then combined, then fallbacks
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    candidates = [
        f"annotations_gold{profile_suffix}{style_suffix}_{ann_type}.json",
        f"annotations_gold{profile_suffix}_{ann_type}.json",
        f"annotations_gold{style_suffix}_{ann_type}.json",
        f"annotations_gold_{ann_type}.json",
        f"annotations_gold{profile_suffix}{style_suffix}.json",
        f"annotations_gold{profile_suffix}.json",
        f"annotations_gold{style_suffix}.json",
        "annotations_gold.json",
        f"annotations{profile_suffix}{style_suffix}_{ann_type}.json",
        f"annotations{profile_suffix}_{ann_type}.json",
        f"annotations{style_suffix}_{ann_type}.json",
        f"annotations_{ann_type}.json",
        f"annotations{profile_suffix}{style_suffix}.json",
        f"annotations{profile_suffix}.json",
        f"annotations{style_suffix}.json",
        "annotations.json",
    ]
    llm_data = None
    loaded_file = None
    for candidate in candidates:
        llm_data = load_annotator_result(version, candidate)
        if llm_data is not None:
            loaded_file = candidate
            break
    if llm_data is None:
        raise FileNotFoundError(
            f"No annotations found for version {version}. Tried: {', '.join(candidates)}")
    print(f"Loaded annotations: {loaded_file} (version: {version})")

    # Results use compound keys (tutor_student_uuid); ground truth uses bare UUIDs.
    uuid_to_llm_conv = {}
    uuid_to_compound_key = {}
    for compound, conv_data in llm_data.get("results", {}).items():
        uuid = compound.rsplit("_", 1)[-1]
        uuid_to_llm_conv[uuid] = conv_data
        uuid_to_compound_key[uuid] = compound

    agreements = []
    disagreements = []
    confusion_counts = Counter()

    for conv_id, conv_data in gt["conversations"].items():
        if conv_id in EXAMPLE_CONV_IDS:
            continue
        llm_conv = uuid_to_llm_conv.get(conv_id)
        if not llm_conv:
            continue

        human_moments = [m for m in conv_data["key_moments"] if m.get("annotation_type") == ann_type]
        llm_moments = [m for m in llm_conv.get("annotations", []) if m.get("annotation_type") == ann_type]

        matches = match_gold_direct(human_moments, llm_moments,
                                    consensus_fn=compute_mean_consensus_label)

        for match in matches:
            human_label = match["consensus_3way"]
            llm_label = match["llm_label_3way"]
            if human_label not in EFFECTIVENESS_LABELS or llm_label not in EFFECTIVENESS_LABELS:
                continue

            confusion_counts[f"{human_label}_vs_{llm_label}" if human_label != llm_label else "agree"] += 1

            llm_m = match["llm_moment"]
            moments = match["cluster"]["moments"]
            sampled = random.sample(moments, min(5, len(moments)))
            human_annotators = [
                {
                    "annotator_id": m.get("annotator_id", "unknown"),
                    "label": m.get("strategy_label", "unclear"),
                    "situation": m.get("situation", ""),
                    "action": m.get("action", ""),
                    "result": m.get("result", ""),
                }
                for m in sampled
            ]
            entry = {
                "conv_id": uuid_to_compound_key.get(conv_id, conv_id),
                "turn_start": match["cluster"]["turn_start"],
                "turn_end": match["cluster"]["turn_end"],
                "human_label": human_label,
                "llm_label": llm_label,
                "human_annotators": human_annotators,
                "n_annotators": len(match["per_annotator_labels"]),
                "llm_situation": llm_m.get("situation", ""),
                "llm_action": llm_m.get("action", ""),
                "llm_result": llm_m.get("result", ""),
            }

            if human_label == llm_label:
                agreements.append(entry)
            else:
                disagreements.append(entry)

    # Format examples -- agreements first
    examples = []

    examples.append(f"=== AGREEMENTS ({len(agreements)} total, showing {min(limit, len(agreements))}) ===")
    examples.append("These are cases where human and LLM annotations produced the same label. Study what makes these work.\n")
    for i, d in enumerate(agreements[:limit]):
        block = f"Agreement {i+1}: turns {d['turn_start']}-{d['turn_end']} (both={d['human_label']})\n"
        if show_excerpts:
            excerpt = get_excerpt(transcripts, d["conv_id"], d["turn_start"], d["turn_end"])
            block += f"  Transcript:\n{excerpt}\n"
        if show_sar:
            human_block = _format_human_annotators(d["human_annotators"], truncate=show_excerpts)
            llm_s = d['llm_situation'][:300] if show_excerpts else d['llm_situation']
            llm_a = d['llm_action'][:300] if show_excerpts else d['llm_action']
            llm_r = d['llm_result'][:300] if show_excerpts else d['llm_result']
            block += (
                f"  HUMAN ({len(d['human_annotators'])} annotator(s) sampled):\n{human_block}"
                f"  LLM:\n"
                f"    S: {llm_s}\n"
                f"    A: {llm_a}\n"
                f"    R: {llm_r}\n"
            )
        examples.append(block)

    # Then disagreements grouped by confusion type
    by_confusion = defaultdict(list)
    for d in disagreements:
        by_confusion[f"{d['human_label']}_vs_{d['llm_label']}"].append(d)

    total_disagreements = sum(len(c) for c in by_confusion.values())
    total_budget = len(by_confusion) * limit
    total_shown = 0
    for confusion_type, cases in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        dense = [d for d in cases if d["n_annotators"] >= 3]
        pool = dense if dense else cases
        source_note = "" if dense else " (no >=3-annotator cases; using full pool)"
        type_limit = max(1, round(total_budget * len(cases) / total_disagreements)) if total_disagreements else limit
        examples.append(f"\n=== {confusion_type.upper()} ({len(cases)} total, showing {min(type_limit, len(pool))}{source_note}) ===\n")

        for i, d in enumerate(pool[:type_limit]):
            block = (
                f"Disagreement {total_shown + i + 1}: turns {d['turn_start']}-{d['turn_end']}\n"
            )
            if show_excerpts:
                excerpt = get_excerpt(transcripts, d["conv_id"], d["turn_start"], d["turn_end"])
                block += f"  Transcript:\n{excerpt}\n"
            if show_sar:
                human_block = _format_human_annotators(d["human_annotators"], truncate=show_excerpts)
                llm_s = d['llm_situation'][:300] if show_excerpts else d['llm_situation']
                llm_a = d['llm_action'][:300] if show_excerpts else d['llm_action']
                llm_r = d['llm_result'][:300] if show_excerpts else d['llm_result']
                block += (
                    f"  HUMAN consensus={d['human_label']} ({len(d['human_annotators'])} annotator(s) sampled):\n"
                    f"{human_block}"
                    f"  LLM ({d['llm_label']}):\n"
                    f"    S: {llm_s}\n"
                    f"    A: {llm_a}\n"
                    f"    R: {llm_r}\n"
                )
            else:
                block += (
                    f"  Human consensus: {d['human_label']}\n"
                    f"  LLM label: {d['llm_label']}\n"
                )
            examples.append(block)
        total_shown += min(type_limit, len(pool))

    stats = {
        "total_pairs": sum(confusion_counts.values()),
        "agreements": confusion_counts.get("agree", 0),
        "confusion_counts": {k: v for k, v in confusion_counts.items() if k != "agree"},
    }

    return "\n".join(examples), stats


# ===================================================================
# Eval metrics loader
# ===================================================================

def load_eval_metrics(version, mode, ann_type=None, profile=None, annotator_style=None):
    """Load evaluation metrics from eval JSON (must run eval.py first).

    Uses the shared eval.load_eval_json loader so the profile/style-suffixed
    scorecard is found (with fallback to the unsuffixed name), matching the
    naming eval.py writes. Returns a dict with key metrics, or empty dict if
    not found.
    """
    from ..eval.eval import load_eval_json
    data = load_eval_json(version, mode, profile=profile, annotator_style=annotator_style)
    if data is None:
        return {}

    # If requesting per-type metrics
    if ann_type and "by_type" in data:
        type_data = data["by_type"].get(ann_type, {})
        det = type_data.get("detection", {})
        eff = type_data.get("effectiveness", {})
    else:
        det = data.get("detection", {})
        eff = data.get("effectiveness", {})

    metrics = {}
    if det:
        metrics["cluster_recall"] = det.get("cluster_recall", 0)
        metrics["moment_precision"] = det.get("moment_precision", 0)
        metrics["mean_iou"] = det.get("mean_iou", 0)
        metrics["total_llm_annotations"] = det.get("total_llm_annotations", 0)
        metrics["total_human_clusters"] = det.get("total_human_clusters", 0)
    if eff:
        metrics["binary_kappa"] = eff.get("binary_kappa", 0)
        metrics["three_way_kappa"] = eff.get("three_way_kappa", 0)
        metrics["binary_accuracy"] = eff.get("binary_accuracy", 0)
        metrics["within_human_range_pct"] = eff.get("within_human_range_pct", 0)

    return metrics


def format_detection_metrics(metrics):
    """Format detection metrics as a readable string for the meta-prompt."""
    if not metrics:
        return "No evaluation metrics available (run eval.py first)."
    parts = []
    if "cluster_recall" in metrics:
        parts.append(f"Cluster Recall: {metrics['cluster_recall']:.1%} ({metrics.get('total_human_clusters', '?')} human clusters)")
    if "moment_precision" in metrics:
        parts.append(f"Moment Precision: {metrics['moment_precision']:.1%} ({metrics.get('total_llm_annotations', '?')} LLM detections)")
    if "mean_iou" in metrics:
        parts.append(f"Mean IoU: {metrics['mean_iou']:.3f}")
    return "\n".join(f"- {p}" for p in parts)


def format_annotation_metrics(metrics):
    """Format annotation metrics as a readable string for the meta-prompt."""
    if not metrics:
        return "No evaluation metrics available (run eval.py first)."
    parts = []
    if "binary_kappa" in metrics:
        parts.append(f"Binary Kappa: {metrics['binary_kappa']:.4f} (effective vs not-effective)")
    if "three_way_kappa" in metrics:
        parts.append(f"3-Way Kappa: {metrics['three_way_kappa']:.4f} (effective/partial/ineffective)")
    if "binary_accuracy" in metrics:
        parts.append(f"Binary Accuracy: {metrics['binary_accuracy']:.1%}")
    if "within_human_range_pct" in metrics:
        parts.append(f"Within Human Range: {metrics['within_human_range_pct']:.1%}")
    return "\n".join(f"- {p}" for p in parts)


# ===================================================================
# Meta-prompts for Gemini advisor
# ===================================================================

ITERATION_PROMPTS_DIR = REPO_ROOT / "prompts" / "archive" / "iteration"


def load_iteration_prompt(pass_type: str, profile: str = "gemini") -> str:
    """Load an iteration meta-prompt from pipeline/prompts/iteration/{profile}/{pass_type}.txt.

    Falls back to gemini/ if the requested profile directory doesn't exist.
    """
    filename = f"{pass_type}.txt"
    prompt_path = ITERATION_PROMPTS_DIR / profile / filename
    if not prompt_path.exists():
        fallback = ITERATION_PROMPTS_DIR / "gemini" / filename
        if fallback.exists():
            print(f"No iteration prompt for profile '{profile}', falling back to gemini/")
            prompt_path = fallback
        else:
            raise FileNotFoundError(
                f"Iteration prompt not found at {prompt_path} or {fallback}"
            )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ===================================================================
# Advisor main (from propose_iteration.py)
# ===================================================================

def main():
    phase_cfg = get_phase_config("advisor")

    parser = argparse.ArgumentParser(
        description="Single-shot Gemini advisor for prompt iteration"
    )
    parser.add_argument("--pass", dest="pass_type", required=True,
                        choices=["detection", "annotation", "annotation_compare",
                                 "annotation_excerpt", "annotation_draft", "detection_draft"],
                        help="Which pass to iterate on")
    parser.add_argument("--version", required=True,
                        help="Results version to analyze (e.g. v2)")
    parser.add_argument("--type", required=True,
                        choices=["scaffolding", "rapport"],
                        help="Annotation type to focus on")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max examples per error category (default: 10)")
    parser.add_argument("--prompt-version", default=None,
                        help="Prompt version to load (default: same as --version)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for advisor correct-match vs near-miss (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the meta-prompt without calling the API")
    parser.add_argument("--annotator-style", choices=get_valid_styles(),
                        default=None,
                        help="Filter ground truth to this annotator archetype and tune the "
                             "## Annotator Calibration section of the prompt")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of moments per batch for annotation_draft (default: 100)")
    parser.add_argument("--no-human-annotations", action="store_true",
                        help="Omit human annotation details from detection error examples")
    args = parser.parse_args()

    profile_name = args.profile or load_config().get("profile")
    if args.profile:
        phase_cfg = get_phase_config("advisor", args.profile)
    model = args.model or phase_cfg["model"]

    prompt_version = args.prompt_version or args.version

    # Determine which pass's prompt to load
    if args.pass_type in ("detection", "detection_draft"):
        pass_dir = "p1"
    else:
        pass_dir = "p2"  # annotation and annotation_draft use the p2 prompt

    prompt_path = None
    for ext in ("md", "txt"):
        candidate = REPO_ROOT / "prompts" / "annotator" / prompt_version / pass_dir / f"{args.type}.{ext}"
        if candidate.exists():
            prompt_path = candidate
            break
    if prompt_path is None:
        print(f"ERROR: Prompt not found at {REPO_ROOT / 'prompts' / 'annotator' / prompt_version / pass_dir / args.type}.{{md,txt}}")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        current_prompt = f.read()

    print(f"Loaded prompt: {prompt_path} ({len(current_prompt)} chars)")

    # Load transcripts
    print("Loading transcripts...")
    transcripts = load_transcripts()

    # Collect errors
    print(f"Collecting {args.pass_type} errors for {args.type} ({args.version})...")

    if args.pass_type == "detection":
        error_examples, stats = collect_detection_errors(
            args.version, args.type, transcripts, limit=args.limit,
            iou_threshold=args.iou_threshold, profile=profile_name,
            show_human_annotations=not args.no_human_annotations,
        )
        # Load evaluation metrics (eval.py must have been run first)
        eval_metrics = load_eval_metrics(args.version, "detections", ann_type=args.type,
                                         profile=profile_name)
        current_metrics = format_detection_metrics(eval_metrics)
        if eval_metrics:
            print(f"Eval metrics: recall={eval_metrics.get('cluster_recall', 0):.1%}, "
                  f"precision={eval_metrics.get('moment_precision', 0):.1%}, "
                  f"IoU={eval_metrics.get('mean_iou', 0):.3f}")

        advisor_prompt_path = ANNOTATOR_PROMPTS_DIR / "advisor_detection.md"
        with open(advisor_prompt_path, "r", encoding="utf-8") as f:
            meta_template = f.read()
        meta_prompt = meta_template.format(
            ann_type=args.type,
            current_prompt=current_prompt,
            current_metrics=current_metrics,
            good_matches=stats["good_matches"],
            complete_misses=stats["complete_misses"],
            near_misses=stats["near_misses"],
            false_positives=stats["false_positives"],
            error_examples=error_examples,
        )
    elif args.pass_type in ("annotation_draft", "detection_draft"):
        if args.pass_type == "annotation_draft":
            meta_prompt_filename = "advisor_drafting_annotation.md"
            output_prefix = "advisor_annotation_draft"
            mode_label = "annotation_draft"
        else:
            meta_prompt_filename = "advisor_drafting_detection.md"
            output_prefix = "advisor_detection_draft"
            mode_label = "detection_draft"

        meta_prompt_path = ANNOTATOR_PROMPTS_DIR / meta_prompt_filename
        if not meta_prompt_path.exists():
            print(f"ERROR: Meta-prompt not found at {meta_prompt_path}")
            return
        with open(meta_prompt_path, "r", encoding="utf-8") as f:
            meta_template = f.read()

        filtered_gt = None
        if args.annotator_style:
            filtered_gt = _load_ground_truth_train(annotator_style=args.annotator_style)
            print(f"Ground truth filtered to '{args.annotator_style}' annotators")

        current_prompt = current_prompt.replace("{annotator_style}", "")

        profile_suffix = f"_{profile_name}" if profile_name else ""
        style_suffix = f"_{args.annotator_style}" if args.annotator_style else ""
        client = ModelClient(model)

        max_annotators = 1 if args.pass_type == "detection_draft" else 5

        # Determine total batch count from batch 0
        _, _, n_batches = collect_teacher_examples(
            ann_type=args.type, transcripts=transcripts,
            batch_size=args.batch_size, batch_idx=0,
            ground_truth=filtered_gt, annotator_style=args.annotator_style,
            max_annotators=max_annotators,
        )
        max_batches = min(5, n_batches)
        print(f"{mode_label}: {n_batches} batch(es) of {args.batch_size} moments each (capped at {max_batches})")

        for batch_idx in range(max_batches):
            teacher_examples, batch_info, _ = collect_teacher_examples(
                ann_type=args.type, transcripts=transcripts,
                batch_size=args.batch_size, batch_idx=batch_idx,
                ground_truth=filtered_gt, annotator_style=args.annotator_style,
                max_annotators=max_annotators,
            )
            meta_prompt = meta_template.format(
                ann_type=args.type,
                current_prompt=current_prompt,
                teacher_examples=teacher_examples,
                batch_info=batch_info,
            )
            print(f"\n[{batch_info}] {len(meta_prompt)} chars (~{len(meta_prompt)//4} tokens)")

            if args.dry_run:
                print(f"--- DRY RUN (batch 0 of {n_batches} shown) ---\n")
                print(meta_prompt.encode("ascii", errors="replace").decode("ascii"))
                return

            response = client.generate(
                meta_prompt,
                json_mode=True,
                max_tokens=phase_cfg.get("max_tokens", 16384),
                timeout=phase_cfg.get("timeout", 300),
                thinking=phase_cfg.get("thinking", False),
                thinking_budget=phase_cfg.get("thinking_budget", 0),
            )
            print(f"  Tokens: {response.usage}")

            try:
                advice = json.loads(response.text)
            except json.JSONDecodeError:
                print("  WARNING: Response is not valid JSON. Saving raw text.")
                advice = {"raw_text": response.text}

            filename = (f"{output_prefix}_{args.type}"
                        f"{profile_suffix}{style_suffix}_batch{batch_idx}.json")
            save_annotator_result(args.version, filename, advice)
            print(f"  Saved: {filename}")

        return

    else:
        # --- Archetype filtering ---
        # Filter ground truth to the archetype's annotators only.
        # The prompt itself is NOT modified -- no style text injection.
        # The advisor iterates the whole prompt to naturally match the archetype.
        filtered_gt = None
        if args.annotator_style:
            filtered_gt = _load_ground_truth_train(annotator_style=args.annotator_style)
            print(f"Ground truth filtered to '{args.annotator_style}' annotators")

        # Strip the {annotator_style} placeholder (no style injection)
        current_prompt = current_prompt.replace("{annotator_style}", "")

        # annotation_compare: omit transcript excerpts, show full SAR text
        # annotation_excerpt: show transcript excerpts only, omit SAR text entirely
        show_excerpts = args.pass_type != "annotation_compare"
        show_sar = args.pass_type != "annotation_excerpt"

        error_examples, stats = collect_annotation_errors(
            args.version, args.type, transcripts, limit=args.limit,
            ground_truth=filtered_gt,
            annotator_style=args.annotator_style,
            profile=profile_name,
            show_excerpts=show_excerpts,
            show_sar=show_sar,
        )
        confusion_summary = ", ".join(f"{k}: {v}" for k, v in stats["confusion_counts"].items())

        # Load evaluation metrics (profile/archetype-specific if given; the
        # shared loader falls back to the unsuffixed scorecard automatically)
        eval_metrics = load_eval_metrics(args.version, "annotations", ann_type=args.type,
                                         profile=profile_name,
                                         annotator_style=args.annotator_style)
        current_metrics = format_annotation_metrics(eval_metrics)
        if eval_metrics:
            print(f"Eval metrics: binary_kappa={eval_metrics.get('binary_kappa', 0):.4f}, "
                  f"3way_kappa={eval_metrics.get('three_way_kappa', 0):.4f}")

        # annotation_excerpt uses its own meta-prompt (tuned for excerpt-only,
        # label-only examples); annotation and annotation_compare share the
        # regular annotation meta-prompt. When --annotator-style is set, the
        # only further difference is that error examples come from the
        # archetype-filtered ground truth.
        meta_prompt_filename = ("advisor_annotation_excerpt.md" if args.pass_type == "annotation_excerpt"
                                else "advisor_annotation.md")
        advisor_prompt_path = ANNOTATOR_PROMPTS_DIR / meta_prompt_filename
        with open(advisor_prompt_path, "r", encoding="utf-8") as f:
            meta_template = f.read()
        meta_prompt = meta_template.format(
            ann_type=args.type,
            current_prompt=current_prompt,
            current_metrics=current_metrics,
            total_pairs=stats["total_pairs"],
            agreements=stats["agreements"],
            confusion_summary=confusion_summary,
            error_examples=error_examples,
        )

    print(f"Meta-prompt: {len(meta_prompt)} chars (~{len(meta_prompt)//4} tokens)")
    print(f"Error stats: {json.dumps(stats, indent=2)}")

    if args.dry_run:
        print("\n--- DRY RUN: Meta-prompt below ---\n")
        print(meta_prompt.encode("ascii", errors="replace").decode("ascii"))
        return

    # Call advisor model
    print(f"\nCalling {model}...")
    client = ModelClient(model)
    response = client.generate(
        meta_prompt,
        json_mode=True,
        max_tokens=phase_cfg.get("max_tokens", 16384),
        timeout=phase_cfg.get("timeout", 300),
        thinking=phase_cfg.get("thinking", False),
        thinking_budget=phase_cfg.get("thinking_budget", 0),
    )

    print(f"Tokens: {response.usage}")

    # Parse and save
    try:
        advice = json.loads(response.text)
    except json.JSONDecodeError:
        print("WARNING: Response is not valid JSON. Saving raw text.")
        advice = {"raw_text": response.text}

    # Save to results directory
    # annotation_compare and annotation_excerpt use the same advisor logic as
    # annotation; name them accordingly with a variant suffix.
    variant_suffixes = {"annotation_compare": "_compare", "annotation_excerpt": "_excerpt"}
    variant_suffix = variant_suffixes.get(args.pass_type, "")
    pass_label = "annotation" if variant_suffix else args.pass_type
    profile_suffix = f"_{profile_name}" if profile_name else ""
    style_suffix = f"_{args.annotator_style}" if args.annotator_style else ""
    filename = f"advisor_{pass_label}_{args.type}{variant_suffix}{profile_suffix}{style_suffix}.json"
    save_annotator_result(args.version, filename, advice)
    print(f"\nSaved: {filename} (version: {args.version})")

    # Pretty-print summary
    print(f"\n{'=' * 70}")
    print(f"  ADVISOR ANALYSIS: {args.pass_type} / {args.type}")
    print(f"{'=' * 70}")

    if "patterns" in advice:
        print(f"\n  PATTERNS IDENTIFIED ({len(advice['patterns'])}):")
        for p in advice["patterns"]:
            direction = f" [{p['direction']}]" if "direction" in p else ""
            print(f"    - {p['name']}{direction} (~{p.get('share_of_errors', '?')} of errors)")
            print(f"      {p['description'][:200]}".encode('ascii', 'replace').decode())

    if "proposed_changes" in advice:
        print(f"\n  PROPOSED CHANGES ({len(advice['proposed_changes'])}):")
        for c in advice["proposed_changes"]:
            risk = c.get("regression_risk", c.get("directional_effect", "?"))
            change_type = c.get("change_type", "")
            prefix = f"[{change_type}] " if change_type else ""
            print(f"    - {prefix}{c['target_pattern']} (risk: {risk})".encode('ascii', 'replace').decode())
            if c.get("current_text"):
                print(f"      FROM: {c['current_text'][:100]}".encode('ascii', 'replace').decode())
            print(f"      TO:   {c['proposed_text'][:100]}".encode('ascii', 'replace').decode())
            print(f"      Why:  {c['rationale'][:150]}".encode('ascii', 'replace').decode())

    if "overall_assessment" in advice:
        print(f"\n  OVERALL: {advice['overall_assessment']}".encode('ascii', 'replace').decode())

    print(f"\n  Full output: {filename} (version: {args.version})")


# ===================================================================
# Disagreement analysis (from analyze_disagreements.py)
# ===================================================================

CONTEXT_TURNS = 10


def classify_errors(gt_convs, det_results, transcripts, type_filter=None):
    """Classify all detection errors across conversations."""
    complete_misses = []   # human cluster, no LLM detection nearby
    near_misses = []       # human cluster, LLM detected nearby but IoU < threshold
    false_positives = []   # LLM detection, no human cluster nearby
    good_matches = []      # matched pairs

    eval_conv_ids = sorted(
        (set(gt_convs.keys()) & set(det_results.keys())) - EXAMPLE_CONV_IDS
    )

    for conv_id in eval_conv_ids:
        human_moments = gt_convs[conv_id]["key_moments"]
        num_turns = gt_convs[conv_id].get("num_turns")

        # Filter by annotation type
        human_types = {m.get("annotation_type") for m in human_moments}
        llm_moments = [
            m for m in det_results.get(conv_id, {}).get("detections", [])
            if m.get("annotation_type") in human_types
        ]

        if type_filter:
            human_moments = [m for m in human_moments if m.get("annotation_type") == type_filter]
            llm_moments = [m for m in llm_moments if m.get("annotation_type") == type_filter]

        clusters = merge_overlapping_ranges(human_moments)

        # Match each human cluster to best LLM detection
        cluster_best_iou = []
        cluster_best_det = []
        for cluster in clusters:
            c_range = (cluster["turn_start"], cluster["turn_end"])
            c_type = cluster["annotation_type"]
            best_iou = 0
            best_det = None
            for det in llm_moments:
                if det.get("annotation_type") != c_type:
                    continue
                iou = compute_iou(c_range, (det["turn_start"], det["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            cluster_best_iou.append(best_iou)
            cluster_best_det.append(best_det)

        # Match each LLM detection to best human cluster
        det_best_iou = []
        for det in llm_moments:
            d_range = (det["turn_start"], det["turn_end"])
            d_type = det.get("annotation_type")
            best_iou = 0
            for cluster in clusters:
                if cluster["annotation_type"] != d_type:
                    continue
                iou = compute_iou(d_range, (cluster["turn_start"], cluster["turn_end"]))
                if iou > best_iou:
                    best_iou = iou
            det_best_iou.append(best_iou)

        # Classify clusters
        for ci, cluster in enumerate(clusters):
            iou = cluster_best_iou[ci]
            det = cluster_best_det[ci]
            entry = {
                "conv_id": conv_id,
                "cluster": cluster,
                "nearest_det": det,
                "iou": iou,
                "num_turns": num_turns,
            }
            if iou >= IOU_THRESHOLD:
                good_matches.append(entry)
            elif iou > 0:
                near_misses.append(entry)
            else:
                complete_misses.append(entry)

        # Classify LLM detections
        for di, det in enumerate(llm_moments):
            if det_best_iou[di] < IOU_THRESHOLD:
                false_positives.append({
                    "conv_id": conv_id,
                    "detection": det,
                    "best_iou": det_best_iou[di],
                    "num_turns": num_turns,
                })

    return complete_misses, near_misses, false_positives, good_matches


def print_miss(entry, transcripts, idx):
    """Print a missed cluster with transcript excerpt."""
    cluster = entry["cluster"]
    conv_id = entry["conv_id"]
    ann_type = cluster["annotation_type"]
    span = cluster["turn_end"] - cluster["turn_start"] + 1

    print(f"\n  --- Miss {idx}: {conv_id[:50]} turns {cluster['turn_start']}-{cluster['turn_end']} ({ann_type}, span={span}) ---")

    if entry["nearest_det"]:
        det = entry["nearest_det"]
        print(f"  Nearest LLM detection: turns {det['turn_start']}-{det['turn_end']} (IoU={entry['iou']:.3f})")
        print(f"  LLM description: {det.get('brief_description', 'N/A')[:200]}")
    else:
        print(f"  Nearest LLM detection: NONE (complete miss)")

    print(f"\n  TRANSCRIPT EXCERPT:")
    excerpt = get_excerpt(transcripts, conv_id, cluster["turn_start"], cluster["turn_end"],
                          context=CONTEXT_TURNS, bold_range=True)
    for line in excerpt.split("\n"):
        print(f"    {line}")

    print(f"\n  HUMAN ANNOTATIONS ({len(cluster['moments'])} annotator(s)):")
    for m in cluster["moments"]:
        print(f"    [{m.get('annotator_id', '?')}] label={m.get('strategy_label', 'N/A')}")
        print(f"      Situation: {m.get('situation', 'N/A')[:300]}")
        print(f"      Action:    {m.get('action', 'N/A')[:300]}")
        print(f"      Result:    {m.get('result', 'N/A')[:300]}")
    print()


def print_false_positive(entry, transcripts, idx):
    """Print a false positive detection with transcript excerpt."""
    det = entry["detection"]
    conv_id = entry["conv_id"]
    ann_type = det.get("annotation_type", "?")
    span = det["turn_end"] - det["turn_start"] + 1

    print(f"\n  --- FP {idx}: {conv_id[:50]} turns {det['turn_start']}-{det['turn_end']} ({ann_type}, span={span}) ---")
    print(f"  LLM description: {det.get('brief_description', 'N/A')[:200]}")
    print(f"  Best IoU with any human cluster: {entry['best_iou']:.3f}")

    print(f"\n  TRANSCRIPT EXCERPT:")
    excerpt = get_excerpt(transcripts, conv_id, det["turn_start"], det["turn_end"],
                          context=CONTEXT_TURNS, bold_range=True)
    for line in excerpt.split("\n"):
        print(f"    {line}")
    print()


def analyze_main():
    """Entry point for detection disagreement analysis."""
    parser = argparse.ArgumentParser(description="Analyze detection disagreements")
    parser.add_argument("--version", required=True, help="Results version (e.g. v1)")
    parser.add_argument("--type", choices=["scaffolding", "rapport"], default=None,
                        help="Filter to annotation type")
    parser.add_argument("--error-type", choices=["miss", "near_miss", "false_positive", "all"],
                        default="all", help="Filter to error type")
    parser.add_argument("--limit", type=int, default=10, help="Max examples per category")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary stats")
    args = parser.parse_args()

    # Load data
    gt = _load_ground_truth_train()

    det_data = load_annotator_result(args.version, "detections.json")
    if det_data is None:
        print(f"ERROR: No detections found for version {args.version}")
        return

    transcripts = load_transcripts()

    # Classify errors
    complete_misses, near_misses, false_positives, good_matches = classify_errors(
        gt["conversations"], det_data.get("results", {}), transcripts, args.type
    )

    total_clusters = len(complete_misses) + len(near_misses) + len(good_matches)
    total_llm = len(false_positives) + len(good_matches)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DETECTION DISAGREEMENT ANALYSIS -- {args.version}")
    if args.type:
        print(f"  Filtered to: {args.type}")
    print(f"{'=' * 70}")
    print(f"  Human clusters:    {total_clusters}")
    print(f"    Matched:         {len(good_matches)} ({len(good_matches)/total_clusters*100:.1f}%)" if total_clusters else "")
    print(f"    Near-misses:     {len(near_misses)} ({len(near_misses)/total_clusters*100:.1f}%)" if total_clusters else "")
    print(f"    Complete misses: {len(complete_misses)} ({len(complete_misses)/total_clusters*100:.1f}%)" if total_clusters else "")
    print(f"  LLM detections:    {total_llm}")
    print(f"    Matched:         {len(good_matches)} ({len(good_matches)/total_llm*100:.1f}%)" if total_llm else "")
    print(f"    False positives: {len(false_positives)} ({len(false_positives)/total_llm*100:.1f}%)" if total_llm else "")

    # Breakdown by type
    if not args.type:
        for ann_type in ["scaffolding", "rapport"]:
            t_miss = sum(1 for e in complete_misses if e["cluster"]["annotation_type"] == ann_type)
            t_near = sum(1 for e in near_misses if e["cluster"]["annotation_type"] == ann_type)
            t_match = sum(1 for e in good_matches if e["cluster"]["annotation_type"] == ann_type)
            t_fp = sum(1 for e in false_positives if e["detection"].get("annotation_type") == ann_type)
            t_total = t_miss + t_near + t_match
            recall = t_match / t_total * 100 if t_total else 0
            print(f"\n  {ann_type.upper()}: {t_total} clusters, {recall:.1f}% recall")
            print(f"    Matched: {t_match}  Near-miss: {t_near}  Complete miss: {t_miss}  FP: {t_fp}")

    # Breakdown by cluster span
    print(f"\n  MISSES BY CLUSTER SPAN:")
    for label, lo, hi in [("tiny 1-2", 1, 2), ("small 3-5", 3, 5), ("medium 6-10", 6, 10), ("large 11-20", 11, 20), ("very large 21+", 21, 9999)]:
        all_in_range = [e for e in complete_misses + near_misses + good_matches
                        if lo <= (e["cluster"]["turn_end"] - e["cluster"]["turn_start"] + 1) <= hi]
        missed_in_range = [e for e in complete_misses + near_misses
                           if lo <= (e["cluster"]["turn_end"] - e["cluster"]["turn_start"] + 1) <= hi]
        total = len(all_in_range)
        missed = len(missed_in_range)
        rate = missed / total * 100 if total else 0
        print(f"    {label:>15s}: {missed}/{total} missed ({rate:.1f}%)")

    if args.summary_only:
        return

    # Detailed examples
    print(f"\n{'=' * 70}")
    print(f"  DETAILED EXAMPLES")
    print(f"{'=' * 70}")

    if args.error_type in ("miss", "all"):
        print(f"\n{'~' * 70}")
        print(f"  COMPLETE MISSES ({len(complete_misses)} total)")
        print(f"{'~' * 70}")
        for i, entry in enumerate(complete_misses[:args.limit]):
            print_miss(entry, transcripts, i + 1)

    if args.error_type in ("near_miss", "all"):
        print(f"\n{'~' * 70}")
        print(f"  NEAR-MISSES ({len(near_misses)} total)")
        print(f"{'~' * 70}")
        for i, entry in enumerate(near_misses[:args.limit]):
            print_miss(entry, transcripts, i + 1)

    if args.error_type in ("false_positive", "all"):
        print(f"\n{'~' * 70}")
        print(f"  FALSE POSITIVES ({len(false_positives)} total)")
        print(f"{'~' * 70}")
        for i, entry in enumerate(false_positives[:args.limit]):
            print_false_positive(entry, transcripts, i + 1)


# ===================================================================
# Entry point dispatch
# ===================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        analyze_main()
    else:
        main()
