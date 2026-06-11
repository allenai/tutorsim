"""One-off: rescore the cached dyn_smoke exchanges with Lucy's v12 annotator.

Reads cached scenarios + exchanges from `dyn_smoke_2026_06_08`, reruns Phase 2
(annotate) and Phase 3 (label + score) only -- no new tutor/student turns are
generated. Output is saved under a new benchmark version so the original smoke
stays intact for side-by-side comparison.

Throwaway. Delete after rescore is verified.
"""
import argparse
import json
import logging
import sys

from annotator.core.config import get_phase_config
from annotator.core.storage import (
    load_benchmark_result, save_benchmark_result, list_benchmark_result_files,
)
from annotator.core.utils import load_transcripts
from benchmark.core.scenarios import Scenario, _format_prefix
from benchmark.core.exchange import Exchange
from benchmark.core.annotator_bridge import (
    prepare_bulk_entries, execute_and_parse_bulk, label_bulk,
)
from benchmark.core.aggregate import (
    label_to_score, extract_effectiveness_by_type, DEFAULT_LABEL_WEIGHTS,
)
from annotator.core.config import get_annotation_types

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _rebuild_scenario(s_dict: dict, transcripts: dict) -> Scenario | None:
    """Recompute transcript_prefix and return a Scenario object."""
    conv = transcripts.get(s_dict["conv_id"])
    if conv is None:
        logger.warning("missing transcript for %s", s_dict["conv_id"])
        return None
    prefix = _format_prefix(conv, s_dict["cut_turn"])
    return Scenario(
        scenario_id=s_dict["scenario_id"],
        conv_id=s_dict["conv_id"],
        cut_turn=s_dict["cut_turn"],
        transcript_prefix=prefix,
        student_context=s_dict.get("student_context", ""),
        last_student_message=s_dict.get("last_student_message", ""),
        mode=s_dict.get("mode", "human"),
        detection=s_dict.get("detection"),
    )


def _load_exchange(version: str, profile: str, sid: str) -> Exchange | None:
    data = load_benchmark_result(version, "exchanges", profile, f"{sid}.json")
    if data is None or not data.get("completed", False):
        return None
    return Exchange(
        scenario_id=data["scenario_id"],
        tutor_model=data["tutor_model"],
        generated_turns=data["generated_turns"],
        tutor_usage=data.get("tutor_usage", {}),
        student_usage=data.get("student_usage", {}),
        completed=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-version", default="dyn_smoke_2026_06_08")
    p.add_argument("--target-version", default="dyn_smoke_v12_2026_06_09")
    p.add_argument("--profile", default="anthropic")
    p.add_argument("--annotator-profile", default="anthropic")
    p.add_argument("--prompt-version", default="v12")
    p.add_argument("--style-label", default="v12",
                   help="Output directory label under annotations/ (cosmetic).")
    p.add_argument("--context-window", type=int, default=50)
    p.add_argument("--mode", default="sync", choices=["sync", "batch"])
    args = p.parse_args()

    # --- Load cached scenarios + exchanges ---
    src_scenarios_raw = load_benchmark_result(args.source_version, "scenarios.json")
    if not src_scenarios_raw:
        sys.exit(f"No scenarios.json under {args.source_version}")

    transcripts = load_transcripts()
    scenarios = [s for s in (_rebuild_scenario(d, transcripts) for d in src_scenarios_raw) if s]
    logger.info("Rebuilt %d scenarios from %s", len(scenarios), args.source_version)

    exchanges = {}
    for s in scenarios:
        ex = _load_exchange(args.source_version, args.profile, s.scenario_id)
        if ex is None:
            logger.warning("no cached exchange for %s", s.scenario_id)
            continue
        exchanges[s.scenario_id] = ex
    logger.info("Loaded %d cached exchanges", len(exchanges))

    if not exchanges:
        sys.exit("No usable cached exchanges found.")

    # Save a copy of scenarios.json under the target version for the viewer.
    save_benchmark_result(args.target_version, "scenarios.json", data=src_scenarios_raw)

    # Also copy exchanges so the viewer / future tooling can find them.
    for sid, ex in exchanges.items():
        save_benchmark_result(args.target_version, "exchanges", args.profile,
                              f"{sid}.json", data=ex.to_dict())

    # --- Phase 2: annotate with v12 ---
    scenarios_with_exchanges = [s for s in scenarios if s.scenario_id in exchanges]
    entries, all_detections, _ = prepare_bulk_entries(
        scenarios=scenarios_with_exchanges,
        exchanges=exchanges,
        annotator_style=args.style_label,   # cosmetic; v12 prompt itself drives behavior
        prompt_version=args.prompt_version,
        context_window=args.context_window,
        with_screenshots=False,
    )
    logger.info("Prepared %d annotation entries across %d scenarios",
                len(entries), len(all_detections))

    per_scenario_results = execute_and_parse_bulk(
        entries=entries,
        all_detections=all_detections,
        annotator_profile=args.annotator_profile,
        mode=args.mode,
        existing_batch_id=None,
        on_batch_created=lambda *_a, **_k: None,
    )
    logger.info("Parsed %d scenario results", len(per_scenario_results))

    # --- Label ---
    annotate_cfg = get_phase_config("annotate", args.annotator_profile)
    per_scenario_labeled = label_bulk(
        per_scenario_results=per_scenario_results,
        annotator_style=args.style_label,
        annotator_profile=args.annotator_profile,
        annotator_model=annotate_cfg["model"],
        mode=args.mode,
    )
    logger.info("Labeled %d scenarios", len(per_scenario_labeled))

    for scenario_id, labeled_data in per_scenario_labeled.items():
        save_benchmark_result(args.target_version, "annotations", args.profile,
                              args.style_label, f"{scenario_id}.json",
                              data=labeled_data)

    # --- Phase 3: scoring ---
    label_weights = DEFAULT_LABEL_WEIGHTS
    style_scenario_scores = []
    for s in scenarios_with_exchanges:
        sid = s.scenario_id
        if sid not in per_scenario_labeled:
            continue
        type_labels = extract_effectiveness_by_type(per_scenario_labeled[sid])
        if not type_labels:
            continue
        type_scores = {t: label_to_score(l, label_weights) for t, l in type_labels.items()}
        mean_score = sum(type_scores.values()) / len(type_scores) if type_scores else 0.0
        style_scenario_scores.append({
            "scenario_id": sid,
            "tutor_model": args.profile,
            "mode": s.mode,
            "labels": type_labels,
            "scores": type_scores,
            "mean_score": mean_score,
        })

    n = len(style_scenario_scores)
    overall_mean = sum(s["mean_score"] for s in style_scenario_scores) / n if n else 0.0
    type_means = {}
    for ann_type in get_annotation_types():
        vals = [s["scores"][ann_type] for s in style_scenario_scores if ann_type in s["scores"]]
        type_means[ann_type] = sum(vals) / len(vals) if vals else 0.0

    summary = {
        "profile": args.profile,
        "style": args.style_label,
        "n_scenarios": n,
        "mean_score": round(overall_mean, 4),
        "by_type": {k: round(v, 4) for k, v in type_means.items()},
        "scenario_scores": style_scenario_scores,
    }
    save_benchmark_result(args.target_version, "scores",
                          f"{args.profile}_{args.style_label}.json", data=summary)
    logger.info("[%s] mean=%.3f n=%d scaffolding=%.3f rapport=%.3f",
                args.style_label, overall_mean, n,
                type_means.get("scaffolding", 0), type_means.get("rapport", 0))

    # Minimal config.json so the viewer has something to read.
    save_benchmark_result(args.target_version, "config.json", data={
        "source_version": args.source_version,
        "annotator_prompt_version": args.prompt_version,
        "context_window": args.context_window,
        "rescore_only": True,
    })

    logger.info("Done. Target version: %s", args.target_version)


if __name__ == "__main__":
    main()
