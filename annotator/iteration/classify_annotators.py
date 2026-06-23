"""
Step 1: Classify human annotators by harshness (generous/balanced/demanding)
based on effectiveness rating distributions in ground truth files.

Also computes per-transcript disagreement: moments where 2+ annotators
annotated the same turn range and gave different labels.

Usage:
    python -m annotator.iteration.classify_annotators

Output:
    results/annotator_profiles.json
"""

import json
from collections import defaultdict
from pathlib import Path

from ..core.config import get_valid_styles
from ..core.storage import get_annotator_result_path
from ..core.utils import load_ground_truth

VALID_LABELS = {"effective", "partial", "ineffective"}


def classify_archetype(effective_rate: float, ineffective_rate: float, total: int) -> str:
    """Bucket annotator into generous / balanced / demanding."""
    if total < 3:
        return "insufficient_data"
    if effective_rate >= 0.5:
        return "generous"
    if ineffective_rate >= 0.5:
        return "demanding"
    return "balanced"


def main():
    gold_raw = load_ground_truth()

    # Per-annotator label counts + example annotations
    counts = defaultdict(lambda: {"effective": 0, "partial": 0, "ineffective": 0, "total": 0})
    examples = defaultdict(list)

    # Same-moment tracking: key = (conv_id, turn_start, turn_end, ann_type)
    same_moment = defaultdict(list)

    for conv_id, conv_data in gold_raw["conversations"].items():
        for moment in conv_data.get("key_moments", []):
            ann_id = moment.get("annotator_id", "unknown")
            label = moment.get("strategy_label", "unclear")
            ann_type = moment.get("annotation_type", "unknown")

            if label in VALID_LABELS:
                counts[ann_id][label] += 1
                counts[ann_id]["total"] += 1
                examples[ann_id].append({
                    "conv_id": conv_id,
                    "turn_start": moment["turn_start"],
                    "turn_end": moment["turn_end"],
                    "annotation_type": ann_type,
                    "situation": moment.get("situation", ""),
                    "action": moment.get("action", ""),
                    "result": moment.get("result", ""),
                    "strategy_label": label,
                })

            key = (conv_id, moment["turn_start"], moment["turn_end"], ann_type)
            same_moment[key].append({
                "annotator_id": ann_id,
                "strategy_label": label,
            })

    # Build profiles
    profiles = {}
    for ann_id, c in counts.items():
        total = c["total"]
        if total == 0:
            continue
        eff_rate = c["effective"] / total
        partial_rate = c["partial"] / total
        ineff_rate = c["ineffective"] / total
        archetype = classify_archetype(eff_rate, ineff_rate, total)

        profiles[ann_id] = {
            "annotator_id": ann_id,
            "total_annotations": total,
            "effective": c["effective"],
            "partial": c["partial"],
            "ineffective": c["ineffective"],
            "effective_rate": round(eff_rate, 3),
            "partial_rate": round(partial_rate, 3),
            "ineffective_rate": round(ineff_rate, 3),
            "archetype": archetype,
        }

    # Group by archetype
    archetypes = defaultdict(list)
    for ann_id, p in profiles.items():
        archetypes[p["archetype"]].append(ann_id)

    # Same-transcript disagreements (2+ annotators, different labels)
    disagreements = []
    multi_annotator_moments = 0
    for key, moments in same_moment.items():
        if len(moments) < 2:
            continue
        multi_annotator_moments += 1
        labels = [m["strategy_label"] for m in moments if m["strategy_label"] in VALID_LABELS]
        if len(set(labels)) > 1:
            conv_id, ts, te, at = key
            disagreements.append({
                "conv_id": conv_id,
                "turn_start": ts,
                "turn_end": te,
                "annotation_type": at,
                "annotations": moments,
            })

    # Disagreement rate by annotator-pair
    pair_agreement = defaultdict(lambda: {"agree": 0, "total": 0})
    for key, moments in same_moment.items():
        valid = [(m["annotator_id"], m["strategy_label"])
                 for m in moments if m["strategy_label"] in VALID_LABELS]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a1, l1 = valid[i]
                a2, l2 = valid[j]
                pair_key = tuple(sorted([a1, a2]))
                pair_agreement[pair_key]["total"] += 1
                if l1 == l2:
                    pair_agreement[pair_key]["agree"] += 1

    pair_stats = {
        f"{k[0]} vs {k[1]}": {
            "agree": v["agree"],
            "total": v["total"],
            "agreement_rate": round(v["agree"] / v["total"], 3) if v["total"] else 0,
        }
        for k, v in pair_agreement.items()
    }

    output = {
        "profiles": profiles,
        "archetypes": {k: sorted(v) for k, v in archetypes.items()},
        "pair_agreement": pair_stats,
        "same_transcript_disagreements": disagreements,
        "stats": {
            "total_annotators": len(profiles),
            "multi_annotator_moments": multi_annotator_moments,
            "disagreements": len(disagreements),
            "archetype_counts": {k: len(v) for k, v in archetypes.items()},
        },
    }

    results_dir = get_annotator_result_path("")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "annotator_profiles.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\nAnnotator Classification")
    print("=" * 65)
    print(f"  {'Annotator':<20} {'Archetype':<12} {'Effective':>10} {'Partial':>8} {'Ineffective':>12} {'N':>5}")
    print("  " + "-" * 63)
    for ann_id, p in sorted(profiles.items(), key=lambda x: -x[1]["effective_rate"]):
        print(f"  {ann_id:<20} {p['archetype']:<12} "
              f"{p['effective_rate']:>9.0%} "
              f"{p['partial_rate']:>7.0%} "
              f"{p['ineffective_rate']:>11.0%} "
              f"{p['total_annotations']:>5}")

    print(f"\nArchetype buckets:")
    for arch in list(get_valid_styles()) + ["insufficient_data"]:
        members = archetypes.get(arch, [])
        if members:
            print(f"  {arch}: {', '.join(sorted(members))}")

    print(f"\nSame-transcript moments (2+ annotators): {multi_annotator_moments}")
    print(f"  Disagreements: {len(disagreements)}")

    if pair_stats:
        print(f"\nAnnotator pair agreement:")
        for pair, stats in sorted(pair_stats.items()):
            print(f"  {pair}: {stats['agreement_rate']:.0%} ({stats['agree']}/{stats['total']})")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
