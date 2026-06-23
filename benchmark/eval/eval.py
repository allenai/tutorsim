"""
Benchmark evaluation: summarize AI tutor performance across scenarios.

For detected scenarios, the detection tells us where to cut; the benchmark
annotations tell us how the AI tutor performed from that cut point.
This module produces per-profile summary statistics.

Also produces per-profile summary statistics: label distributions, style agreement,
scaffolding vs rapport breakdown.

Usage:
    python -m benchmark.eval.eval --version v1 --profile gemini
    python -m benchmark.eval.eval --version v1 --compare gemini openai anthropic
"""

import argparse
import json
from collections import Counter, defaultdict

from annotator.core.config import get_valid_styles, get_annotation_types
from annotator.core.storage import (
    load_benchmark_result, save_benchmark_result, list_benchmark_result_files,
)


def load_benchmark_data(version: str, profile: str) -> dict:
    """Load scenarios, annotations for a profile."""
    scenarios_data = load_benchmark_result(version, "scenarios.json")
    scenarios = {s["scenario_id"]: s for s in scenarios_data} if scenarios_data else {}

    # Load annotations per style
    style_annotations = {}
    for style in get_valid_styles():
        ann_files = list_benchmark_result_files(version, "annotations", profile, style)
        if ann_files:
            style_annotations[style] = {}
            for fname in ann_files:
                data = load_benchmark_result(version, "annotations", profile, style, fname)
                if data:
                    style_annotations[style][fname.replace(".json", "")] = data

    return {"scenarios": scenarios, "style_annotations": style_annotations}


def extract_labels(annotation_data: dict, scenario_id: str) -> list[dict]:
    """Extract annotation labels from benchmark annotation output.

    Uses scenario_id as key since annotator_bridge remaps conv_id -> scenario_id.
    """
    if not annotation_data:
        return []
    results = annotation_data.get("results", {})
    conv_results = results.get(scenario_id, {})
    anns = conv_results.get("annotations", [])
    return [
        {
            "annotation_type": a.get("annotation_type", "unknown"),
            "effectiveness": a.get("effectiveness", "unclear"),
            "turn_start": a.get("turn_start", 0),
            "turn_end": a.get("turn_end", 0),
        }
        for a in anns
    ]


def eval_profile(version: str, profile: str) -> dict:
    """Evaluate a single tutor profile's benchmark results.

    Returns a summary dict with:
    - Label distributions per style and annotation type
    - Style agreement metrics
    """
    data = load_benchmark_data(version, profile)
    scenarios = data["scenarios"]
    style_annotations = data["style_annotations"]

    # Per-style, per-type label counts
    style_type_labels = defaultdict(lambda: defaultdict(Counter))
    # Style agreement tracking
    style_agreement = defaultdict(lambda: {"agree": 0, "total": 0})

    for scenario_id, scenario in scenarios.items():
        conv_id = scenario["conv_id"]
        mode = scenario["mode"]
        detection = scenario.get("detection")

        # Collect AI labels per style
        ai_labels_by_style = {}
        for style, style_data in style_annotations.items():
            ann_data = style_data.get(scenario_id)
            labels = extract_labels(ann_data, scenario_id)
            ai_labels_by_style[style] = labels

            for label_info in labels:
                ann_type = label_info["annotation_type"]
                eff = label_info["effectiveness"]
                style_type_labels[style][ann_type][eff] += 1

        # Style agreement (pairwise)
        styles = sorted(ai_labels_by_style.keys())
        for i in range(len(styles)):
            for j in range(i + 1, len(styles)):
                s1, s2 = styles[i], styles[j]
                labels1 = ai_labels_by_style[s1]
                labels2 = ai_labels_by_style[s2]
                # Match by annotation_type
                for l1 in labels1:
                    for l2 in labels2:
                        if l1["annotation_type"] == l2["annotation_type"]:
                            pair_key = f"{s1} vs {s2}"
                            style_agreement[pair_key]["total"] += 1
                            if l1["effectiveness"] == l2["effectiveness"]:
                                style_agreement[pair_key]["agree"] += 1

    # Format label distributions
    label_distributions = {}
    for style, type_labels in style_type_labels.items():
        label_distributions[style] = {}
        for ann_type, counts in type_labels.items():
            total = sum(counts.values())
            label_distributions[style][ann_type] = {
                label: {"count": count, "rate": round(count / total, 3) if total else 0}
                for label, count in counts.items()
            }
            label_distributions[style][ann_type]["_total"] = total

    # Format style agreement
    agreement_stats = {}
    for pair, stats in style_agreement.items():
        total = stats["total"]
        agreement_stats[pair] = {
            "agree": stats["agree"],
            "total": total,
            "agreement_rate": round(stats["agree"] / total, 3) if total else 0,
        }

    return {
        "profile": profile,
        "n_scenarios": len(scenarios),
        "n_detected": sum(1 for s in scenarios.values() if s["mode"] == "detected"),
        "n_random": sum(1 for s in scenarios.values() if s["mode"] == "random"),
        "label_distributions": label_distributions,
        "style_agreement": agreement_stats,
    }


def print_eval(result: dict):
    """Print evaluation results."""
    profile = result["profile"]
    print(f"\n{'=' * 70}")
    print(f"  Benchmark Evaluation: {profile}")
    print(f"{'=' * 70}")
    print(f"  Scenarios: {result['n_scenarios']} "
          f"(detected: {result.get('n_detected', 0)}, random: {result['n_random']})")

    # Label distributions
    print(f"\n  Label Distributions:")
    for style in get_valid_styles():
        if style not in result["label_distributions"]:
            continue
        style_data = result["label_distributions"][style]
        print(f"\n    {style.upper()}:")
        for ann_type in get_annotation_types():
            if ann_type not in style_data:
                continue
            type_data = style_data[ann_type]
            total = type_data.get("_total", 0)
            parts = []
            for label in ("effective", "partial", "ineffective", "unclear"):
                if label in type_data:
                    parts.append(f"{label}={type_data[label]['rate']:.0%}")
            print(f"      {ann_type:<14} (n={total}): {', '.join(parts)}")

    # Style agreement
    if result["style_agreement"]:
        print(f"\n  Style Agreement:")
        for pair, stats in sorted(result["style_agreement"].items()):
            print(f"    {pair}: {stats['agreement_rate']:.0%} "
                  f"({stats['agree']}/{stats['total']})")

    print()


def compare_profiles(version: str, profiles: list[str]):
    """Compare multiple tutor profiles side by side."""
    results = {}
    for profile in profiles:
        results[profile] = eval_profile(version, profile)

    print(f"\n{'=' * 80}")
    print(f"  Cross-Model Comparison (version: {version})")
    print(f"{'=' * 80}")

    # Per-style label distribution comparison (balanced only for readability)
    print(f"\n  Balanced Style - Label Rates:")
    print(f"  {'Profile':<20} {'Type':<14} {'Effective':<12} {'Partial':<12} {'Ineffective':<12}")
    print(f"  {'-' * 70}")
    for profile in profiles:
        # Uses balanced as representative style for cross-model comparison
        dist = results[profile]["label_distributions"].get("balanced", {})
        for ann_type in ("scaffolding", "rapport"):
            if ann_type not in dist:
                continue
            td = dist[ann_type]
            print(f"  {profile:<20} {ann_type:<14} "
                  f"{td.get('effective', {}).get('rate', 0):<12.0%} "
                  f"{td.get('partial', {}).get('rate', 0):<12.0%} "
                  f"{td.get('ineffective', {}).get('rate', 0):<12.0%}")

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--version", required=True, help="Benchmark version")
    parser.add_argument("--profile", help="Single profile to evaluate")
    parser.add_argument("--compare", nargs="+", help="Compare multiple profiles")
    args = parser.parse_args()

    if args.compare:
        results = compare_profiles(args.version, args.compare)
        save_benchmark_result(args.version, "eval_comparison.json", data=results)
        print(f"Saved: eval_comparison.json (version: {args.version})")
    elif args.profile:
        result = eval_profile(args.version, args.profile)
        print_eval(result)
        save_benchmark_result(args.version, f"eval_{args.profile}.json", data=result)
        print(f"Saved: eval_{args.profile}.json (version: {args.version})")
    else:
        parser.error("Specify --profile or --compare")


if __name__ == "__main__":
    main()
