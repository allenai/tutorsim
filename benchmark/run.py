"""Benchmark pipeline orchestrator.

All-batch architecture:
  Phase 1: Generate exchanges (batch per round -- all scenarios in parallel)
  Phase 2: Annotate all exchanges (batch per style)
  Phase 3: Aggregate scores + build leaderboard

Usage:
    python -m benchmark.run --version v1
    python -m benchmark.run --version v1 --tutor-profile anthropic
    python -m benchmark.run --version v1 --max-scenarios 10
    python -m benchmark.run --version v1 --mode sync  # use sync instead of batch
"""

import argparse
import json
import yaml
from pathlib import Path

from annotator.core.client import ModelClient
from annotator.core.config import get_phase_config
from annotator.core.detect import run_detect
from annotator.core.storage import (
    save_benchmark_result, load_benchmark_result, list_benchmark_result_files,
)

from .core.scenarios import load_scenarios, Scenario
from .core.exchange import run_exchange, run_exchanges_batch, Exchange
from .core.annotator_bridge import (
    annotate_exchange,
    prepare_bulk_entries, execute_and_parse_bulk, label_bulk,
)
from .core.aggregate import (
    label_to_score, extract_effectiveness_by_type, DEFAULT_LABEL_WEIGHTS,
)


BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "benchmark"


def load_config(overrides: dict | None = None) -> dict:
    """Load benchmark config with optional CLI overrides."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    config = full_config.get("benchmark", {})

    if overrides:
        if overrides.get("scenario_mode"):
            config["scenarios"]["mode"] = overrides["scenario_mode"]
        if overrides.get("max_scenarios"):
            config["scenarios"]["max_scenarios"] = overrides["max_scenarios"]
        if overrides.get("max_per_conv"):
            config["scenarios"]["max_per_conv"] = overrides["max_per_conv"]
        if overrides.get("tutor_profile"):
            config["tutor_profiles"] = [overrides["tutor_profile"]]
        if overrides.get("mode"):
            config["annotator"]["mode"] = overrides["mode"]
        if overrides.get("test_transcripts"):
            config["scenarios"]["test_transcripts"] = overrides["test_transcripts"]

    return config


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def run_benchmark(version: str, config: dict):
    """Run the full benchmark pipeline."""
    # Resolve and record actual model names for traceability
    resolved_models = {}
    for profile_name in config.get("tutor_profiles", []):
        resolved_models[f"tutor_{profile_name}"] = get_phase_config("tutor", profile_name)["model"]
    student_profile = config.get("student", {}).get("profile", "gemini")
    resolved_models["student"] = get_phase_config("tutor", student_profile)["model"]
    ann_profile = config.get("annotator", {}).get("profile", "gemini")
    resolved_models["annotator"] = get_phase_config("annotate", ann_profile)["model"]
    resolved_models["labeler"] = get_phase_config("label", ann_profile)["model"]
    config["resolved_models"] = resolved_models
    config["run_version"] = version

    # Resolve detect model
    detect_cfg_section = config.get("detect", {})
    detect_profile = detect_cfg_section.get("profile", "anthropic")
    detect_prompt_version = detect_cfg_section.get("prompt_version", "v5")
    resolved_models["detector"] = get_phase_config("detect", detect_profile)["model"]
    config["resolved_models"] = resolved_models

    save_benchmark_result(version, "config.json", data=config)

    # --- Step 0: Run detection on all transcripts ---
    scenario_mode = config.get("scenarios", {}).get("mode", "detected")
    detections_by_conv = None

    if scenario_mode in ("detected", "both"):
        print("\n=== Step 0: Key Moment Detection ===")
        detect_phase_cfg = get_phase_config("detect", detect_profile)
        detect_model = detect_phase_cfg["model"]
        detect_mode = config.get("annotator", {}).get("mode", "batch")

        detect_output = run_detect(
            version=f"benchmark_{version}",
            model=detect_model,
            mode=detect_mode,
            prompt_version=detect_prompt_version,
            targets=["scaffolding", "rapport"],
            phase_cfg=detect_phase_cfg,
            test=config.get("scenarios", {}).get("test_transcripts", 0),
        )
        detections_by_conv = detect_output["results"]
        save_benchmark_result(version, "detections.json", data=detect_output)

    # --- Step 1: Extract scenarios from detections ---
    print("\n=== Step 1: Extract Scenarios ===")
    scenarios = load_scenarios(config["scenarios"], detections_by_conv=detections_by_conv)
    save_benchmark_result(version, "scenarios.json", data=[s.to_dict() for s in scenarios])

    tutor_profiles = config.get("tutor_profiles", ["gemini"])
    exchange_cfg = config.get("exchange", {})
    annotator_cfg = config.get("annotator", {})
    agg_cfg = config.get("aggregation", {})
    ann_mode = annotator_cfg.get("mode", "batch")
    annotator_profile = annotator_cfg.get("profile", "gemini")
    prompt_version_base = annotator_cfg["prompt_version"]
    context_window = annotator_cfg.get("context_window", 20)
    styles = annotator_cfg.get("styles", ["generous", "balanced", "demanding"])

    student_profile = config.get("student", {}).get("profile", "gemini")
    student_cfg = get_phase_config("tutor", student_profile)
    student_client = ModelClient(student_cfg["model"])

    for profile in tutor_profiles:
        tutor_cfg = get_phase_config("tutor", profile)
        tutor_model = tutor_cfg["model"]
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {profile} ({tutor_model})")
        print(f"{'=' * 60}")

        tutor_client = ModelClient(tutor_model)

        # ---------------------------------------------------------------
        # Phase 1: Generate all exchanges (skip if already on disk)
        # ---------------------------------------------------------------
        print(f"\n--- Phase 1: Generate Exchanges ({len(scenarios)} scenarios) ---")

        existing_files = list_benchmark_result_files(version, "exchanges", profile)
        existing = {f.replace(".json", "") for f in existing_files}
        missing = [s for s in scenarios if s.scenario_id not in existing]

        def _load_exchange(sid):
            data = load_benchmark_result(version, "exchanges", profile, f"{sid}.json")
            if data is None:
                return None
            return Exchange(
                scenario_id=data["scenario_id"],
                tutor_model=data["tutor_model"],
                generated_turns=data["generated_turns"],
                tutor_usage=data.get("tutor_usage", {}),
                student_usage=data.get("student_usage", {}),
            )

        if not missing:
            print(f"  All {len(scenarios)} exchanges already cached -- loading...")
            exchanges = {}
            for scenario in scenarios:
                ex = _load_exchange(scenario.scenario_id)
                if ex:
                    exchanges[scenario.scenario_id] = ex
        else:
            print(f"  {len(existing)} cached, {len(missing)} to generate...")
            def _save_exchange(sid, exchange):
                save_benchmark_result(version, "exchanges", profile, f"{sid}.json",
                                      data=exchange.to_dict())

            if ann_mode == "batch":
                new_exchanges = run_exchanges_batch(
                    scenarios=missing,
                    tutor_client=tutor_client,
                    student_client=student_client,
                    num_turns=exchange_cfg.get("num_turns", 4),
                    tutor_max_tokens=tutor_cfg["max_tokens"],
                    student_max_tokens=student_cfg["max_tokens"],
                    poll_interval=exchange_cfg.get("poll_interval", 60),
                    save_callback=_save_exchange,
                )
            else:
                new_exchanges = {}
                for i, scenario in enumerate(missing):
                    print(f"  [{i+1}/{len(missing)}] {scenario.scenario_id}...",
                          end=" ", flush=True)
                    try:
                        exchange = run_exchange(
                            scenario=scenario,
                            tutor_client=tutor_client,
                            student_client=student_client,
                            num_turns=exchange_cfg.get("num_turns", 4),
                            tutor_max_tokens=tutor_cfg["max_tokens"],
                            student_max_tokens=student_cfg["max_tokens"],
                        )
                        new_exchanges[scenario.scenario_id] = exchange
                        print(f"{len(exchange.generated_turns)} turns")
                    except Exception as e:
                        print(f"ERROR: {e}")

            # Save new exchanges
            for sid, exchange in new_exchanges.items():
                save_benchmark_result(version, "exchanges", profile, f"{sid}.json",
                                      data=exchange.to_dict())

            # Load all exchanges (cached + new)
            exchanges = {}
            for scenario in scenarios:
                ex = _load_exchange(scenario.scenario_id)
                if ex:
                    exchanges[scenario.scenario_id] = ex

        print(f"\n  Exchanges ready: {len(exchanges)}/{len(scenarios)}")

        # ---------------------------------------------------------------
        # Phase 2: Annotate all exchanges (batch per style)
        # ---------------------------------------------------------------
        print(f"\n--- Phase 2: Annotate ({ann_mode} mode, {len(styles)} styles) ---")

        all_style_results = {}

        for style in styles:
            # Resolve per-style prompt version (e.g. annotator_profiles/generous)
            if prompt_version_base in ("annotator_profiles", "profiles"):
                prompt_version = f"profiles/{style}"
            else:
                prompt_version = prompt_version_base

            print(f"\n  [{style.upper()}] Preparing entries (prompts: {prompt_version})...")
            entries, all_detections, all_conversations = prepare_bulk_entries(
                scenarios=scenarios,
                exchanges=exchanges,
                annotator_style=style,
                prompt_version=prompt_version,
                context_window=context_window,
            )
            print(f"    {len(entries)} annotation entries across {len(all_detections)} scenarios")

            if not entries:
                all_style_results[style] = {}
                continue

            # Execute annotation
            print(f"    Running annotation ({ann_mode})...")
            per_scenario_results = execute_and_parse_bulk(
                entries=entries,
                all_detections=all_detections,
                annotator_profile=annotator_profile,
                mode=ann_mode,
            )
            print(f"    Parsed {len(per_scenario_results)} scenario results")

            # Label
            annotate_cfg_full = get_phase_config("annotate", annotator_profile)
            print(f"    Running labeling ({ann_mode})...")
            per_scenario_labeled = label_bulk(
                per_scenario_results=per_scenario_results,
                annotator_style=style,
                annotator_profile=annotator_profile,
                annotator_model=annotate_cfg_full["model"],
                mode=ann_mode,
            )
            print(f"    Labeled {len(per_scenario_labeled)} scenarios")

            all_style_results[style] = per_scenario_labeled

            # Save per-scenario annotations
            for scenario_id, labeled_data in per_scenario_labeled.items():
                save_benchmark_result(version, "annotations", profile, style,
                                      f"{scenario_id}.json", data=labeled_data)

        # ---------------------------------------------------------------
        # Phase 3: Per-style scores (no composite aggregation)
        # ---------------------------------------------------------------
        print(f"\n--- Phase 3: Per-Style Scores ---")
        label_weights = agg_cfg.get("label_weights", DEFAULT_LABEL_WEIGHTS)

        for style in styles:
            style_results = all_style_results.get(style, {})
            style_scenario_scores = []

            for scenario in scenarios:
                sid = scenario.scenario_id
                if sid not in exchanges or sid not in style_results:
                    continue

                type_labels = extract_effectiveness_by_type(style_results[sid])
                if not type_labels:
                    continue

                type_scores = {
                    ann_type: label_to_score(label, label_weights)
                    for ann_type, label in type_labels.items()
                }
                mean_score = sum(type_scores.values()) / len(type_scores) if type_scores else 0.0

                style_scenario_scores.append({
                    "scenario_id": sid,
                    "tutor_model": profile,
                    "mode": scenario.mode,
                    "labels": type_labels,
                    "scores": type_scores,
                    "mean_score": mean_score,
                })

            n = len(style_scenario_scores)
            overall_mean = sum(s["mean_score"] for s in style_scenario_scores) / n if n else 0.0

            # Per-type means
            type_means = {}
            for ann_type in ("scaffolding", "rapport"):
                vals = [s["scores"][ann_type] for s in style_scenario_scores if ann_type in s["scores"]]
                type_means[ann_type] = sum(vals) / len(vals) if vals else 0.0

            style_summary = {
                "profile": profile,
                "style": style,
                "n_scenarios": n,
                "mean_score": round(overall_mean, 4),
                "by_type": {k: round(v, 4) for k, v in type_means.items()},
                "scenario_scores": style_scenario_scores,
            }

            save_benchmark_result(version, "scores", f"{profile}_{style}.json",
                                  data=style_summary)
            print(f"  {style}: {overall_mean:.3f} (n={n}) | "
                  f"scaffolding={type_means.get('scaffolding', 0):.3f}, "
                  f"rapport={type_means.get('rapport', 0):.3f}")

    print(f"\nResults saved (version: {version})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI tutor models")
    parser.add_argument("--version", required=True, help="Benchmark version (e.g., v1)")
    parser.add_argument("--tutor-profile",
                        help="Single tutor profile (e.g. gemini, openai, anthropic)")
    parser.add_argument("--scenario-mode", choices=["key_moment", "random", "both"],
                        help="Override scenario extraction mode")
    parser.add_argument("--max-scenarios", type=int, help="Limit number of scenarios")
    parser.add_argument("--max-per-conv", type=int,
                        help="Max scenarios per conversation (randomly sampled)")
    parser.add_argument("--test", type=int, default=0,
                        help="Limit detection to N transcripts (0 = all)")
    parser.add_argument("--mode", choices=["batch", "sync"],
                        help="Override execution mode (batch or sync)")
    args = parser.parse_args()

    overrides = {
        "scenario_mode": args.scenario_mode,
        "max_scenarios": args.max_scenarios,
        "max_per_conv": args.max_per_conv,
        "tutor_profile": args.tutor_profile,
        "mode": args.mode,
        "test_transcripts": args.test,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config = load_config(overrides)
    run_benchmark(args.version, config)


if __name__ == "__main__":
    main()
