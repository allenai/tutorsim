"""Benchmark pipeline orchestrator.

All-batch architecture:
  Phase 1: Generate exchanges (batch per round -- all scenarios in parallel)
  Phase 2: Annotate all exchanges (batch per style)
  Phase 3: Score per style (no composite aggregation)

Usage:
    python -m benchmark.run                              # auto-generated version
    python -m benchmark.run --version v1
    python -m benchmark.run --version v1 --tutor-profile anthropic
    python -m benchmark.run --version v1 --max-scenarios 10
    python -m benchmark.run --version v1 --mode sync  # use sync instead of batch
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path

from common.logging_setup import setup_logging
from annotator.core.client import ModelClient
from annotator.core.config import get_phase_config, get_annotation_types, get_benchmark_config
from annotator.core.detect import run_detect
from annotator.core.storage import (
    save_benchmark_result, load_benchmark_result, list_benchmark_result_files,
    save_benchmark_inflight_batch, load_benchmark_inflight_batch,
    clear_benchmark_inflight_batch,
)

from .core.scenarios import load_scenarios, Scenario
from .core.exchange import run_exchange, run_exchanges_batch, Exchange
from .core.annotator_bridge import (
    prepare_bulk_entries, execute_and_parse_bulk, label_bulk,
)
from .core.aggregate import (
    label_to_score, extract_effectiveness_by_type, DEFAULT_LABEL_WEIGHTS,
)


BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent

logger = logging.getLogger(__name__)


def _entries_keys_hash(entries: list[dict]) -> str:
    """Stable short hash of an entries list, keyed on entry order + keys.
    Mirrors annotator/core/annotate.py:_entries_keys_hash so the resume
    guard catches a changed scenario set between runs."""
    joined = "\n".join(e["key"] for e in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


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
    student_profile = config["student"]["profile"]
    # Student uses the base model from its profile (no separate "student" phase in config)
    resolved_models["student"] = get_phase_config("tutor", student_profile)["model"]
    student_mode = config["student"].get("mode")
    if student_mode:
        resolved_models["student_mode"] = student_mode
    ann_profile = config["annotator"]["profile"]
    resolved_models["annotator"] = get_phase_config("annotate", ann_profile)["model"]
    resolved_models["labeller"] = get_phase_config("label", ann_profile)["model"]
    config["run_version"] = version

    # Resolve detect model
    detect_cfg_section = config["detect"]
    detect_profile = detect_cfg_section["profile"]
    detect_prompt_version = detect_cfg_section["prompt_version"]
    resolved_models["detector"] = get_phase_config("detect", detect_profile)["model"]
    config["resolved_models"] = resolved_models

    with_screenshots = config.get("with_screenshots", False)
    if with_screenshots:
        from annotator.core.client import validate_vision_support
        for role, model in resolved_models.items():
            validate_vision_support(model)
        logger.info("Screenshots: enabled -- validated vision support on all models (%s)",
                    ", ".join(sorted(set(resolved_models.values()))))

    transcripts_for_screenshots = None
    if with_screenshots:
        from annotator.core.storage import load_all_transcripts
        transcripts_for_screenshots = load_all_transcripts()

    save_benchmark_result(version, "config.json", data=config)

    # --- Step 0: Run detection on all transcripts ---
    scenario_mode = config["scenarios"]["mode"]
    detections_by_conv = None

    if scenario_mode in ("detected", "both"):
        logger.info("=== Step 0: Key Moment Detection ===")
        detect_phase_cfg = get_phase_config("detect", detect_profile)
        detect_model = detect_phase_cfg["model"]
        detect_mode = detect_phase_cfg.get("mode", "batch")

        detect_output = run_detect(
            version=f"benchmark_{version}",
            model=detect_model,
            mode=detect_mode,
            prompt_version=detect_prompt_version,
            targets=["scaffolding", "rapport"],
            phase_cfg=detect_phase_cfg,
            test=config.get("scenarios", {}).get("test_transcripts", 0),
            with_screenshots=with_screenshots,
        )
        detections_by_conv = detect_output["results"]
        save_benchmark_result(version, "detections.json", data=detect_output)

    # --- Step 1: Extract scenarios from detections ---
    logger.info("=== Step 1: Extract Scenarios ===")
    scenarios = load_scenarios(config["scenarios"], detections_by_conv=detections_by_conv)
    save_benchmark_result(version, "scenarios.json", data=[s.to_dict() for s in scenarios])

    tutor_profiles = config["tutor_profiles"]
    exchange_cfg = config["exchange"]
    annotator_cfg = config["annotator"]
    agg_cfg = config.get("aggregation", {})
    ann_mode = annotator_cfg["mode"]
    annotator_profile = annotator_cfg["profile"]
    prompt_version_base = annotator_cfg["prompt_version"]
    context_window = annotator_cfg["context_window"]
    styles = annotator_cfg["styles"]

    student_profile = config["student"]["profile"]
    student_mode = config["student"].get("mode")
    student_cfg = get_phase_config("tutor", student_profile)
    student_client = ModelClient(student_cfg["model"])

    for profile in tutor_profiles:
        tutor_cfg = get_phase_config("tutor", profile)
        tutor_model = tutor_cfg["model"]
        logger.info("=== Evaluating: %s (%s) ===", profile, tutor_model)

        tutor_client = ModelClient(tutor_model)

        # ---------------------------------------------------------------
        # Phase 1: Generate all exchanges (skip if already on disk)
        # ---------------------------------------------------------------
        logger.info("--- Phase 1: Generate Exchanges (%d scenarios) ---", len(scenarios))

        images_by_scenario = None
        if with_screenshots:
            from annotator.core.screenshots import load_anchored_screenshots
            images_by_scenario = {}
            for scenario in scenarios:
                conv = transcripts_for_screenshots.get(scenario.conv_id)
                if not conv:
                    images_by_scenario[scenario.scenario_id] = []
                    continue
                anchored = load_anchored_screenshots(scenario.conv_id, conv["turns"])
                visible = [s for s in anchored if s["anchor_turn"] <= scenario.cut_turn]
                images_by_scenario[scenario.scenario_id] = [s["storage_path"] for s in visible]
            total_images = sum(len(v) for v in images_by_scenario.values())
            logger.info("Screenshots: loaded for %d scenarios (%d images total, filtered by cut_turn)",
                        len(images_by_scenario), total_images)

        existing_files = list_benchmark_result_files(version, "exchanges", profile)
        existing = set()
        for f in existing_files:
            sid = f.replace(".json", "")
            data = load_benchmark_result(version, "exchanges", profile, f)
            if data and data.get("completed", False):
                existing.add(sid)
        missing = [s for s in scenarios if s.scenario_id not in existing]

        def _load_exchange(sid):
            data = load_benchmark_result(version, "exchanges", profile, f"{sid}.json")
            if data is None:
                return None
            if not data.get("completed", False):
                return None
            return Exchange(
                scenario_id=data["scenario_id"],
                tutor_model=data["tutor_model"],
                generated_turns=data["generated_turns"],
                tutor_usage=data.get("tutor_usage", {}),
                student_usage=data.get("student_usage", {}),
                completed=True,
            )

        if not missing:
            logger.info("All %d exchanges already cached -- loading", len(scenarios))
            exchanges = {}
            for scenario in scenarios:
                ex = _load_exchange(scenario.scenario_id)
                if ex:
                    exchanges[scenario.scenario_id] = ex
        else:
            logger.info("%d cached, %d to generate", len(existing), len(missing))
            def _save_exchange(sid, exchange):
                save_benchmark_result(version, "exchanges", profile, f"{sid}.json",
                                      data=exchange.to_dict())

            exchange_prompt_version = exchange_cfg["prompt_version"]

            if ann_mode == "batch":
                new_exchanges = run_exchanges_batch(
                    scenarios=missing,
                    tutor_client=tutor_client,
                    student_client=student_client,
                    num_turns=exchange_cfg["num_turns"],
                    tutor_max_tokens=tutor_cfg["max_tokens"],
                    student_max_tokens=student_cfg["max_tokens"],
                    poll_interval=exchange_cfg["poll_interval"],
                    save_callback=_save_exchange,
                    prompt_version=exchange_prompt_version,
                    images_by_scenario=images_by_scenario,
                    student_mode=student_mode,
                )
            else:
                new_exchanges = {}
                for i, scenario in enumerate(missing):
                    try:
                        exchange = run_exchange(
                            scenario=scenario,
                            tutor_client=tutor_client,
                            student_client=student_client,
                            num_turns=exchange_cfg["num_turns"],
                            tutor_max_tokens=tutor_cfg["max_tokens"],
                            student_max_tokens=student_cfg["max_tokens"],
                            prompt_version=exchange_prompt_version,
                            images=(images_by_scenario or {}).get(scenario.scenario_id),
                            student_mode=student_mode,
                        )
                        new_exchanges[scenario.scenario_id] = exchange
                        logger.debug("[%d/%d] %s -> %d turns",
                                     i + 1, len(missing), scenario.scenario_id,
                                     len(exchange.generated_turns))
                    except Exception as e:
                        logger.warning("scenario %s failed: %s", scenario.scenario_id, e)

            # Save completed exchanges (skip partial failures)
            for sid, exchange in new_exchanges.items():
                if exchange.completed:
                    save_benchmark_result(version, "exchanges", profile, f"{sid}.json",
                                          data=exchange.to_dict())

            # Load all exchanges (cached + new)
            exchanges = {}
            for scenario in scenarios:
                ex = _load_exchange(scenario.scenario_id)
                if ex:
                    exchanges[scenario.scenario_id] = ex

        logger.info("Exchanges ready: %d/%d", len(exchanges), len(scenarios))

        # ---------------------------------------------------------------
        # Phase 2: Annotate all exchanges (batch per style)
        # ---------------------------------------------------------------
        logger.info("=== Phase 2: Annotate (%s mode, %d styles) ===", ann_mode, len(styles))

        all_style_results = {}

        def _load_cached_annotations(style):
            """Return {scenario_id: labeled_data} for shards already on disk."""
            cached = {}
            for fname in list_benchmark_result_files(version, "annotations", profile, style):
                if not fname.endswith(".json"):
                    continue
                sid = fname[:-5]
                data = load_benchmark_result(version, "annotations", profile, style, fname)
                if data is not None:
                    cached[sid] = data
            return cached

        for style in styles:
            if prompt_version_base == "profiles":
                prompt_version = f"profiles/{style}"
            else:
                prompt_version = prompt_version_base

            cached = _load_cached_annotations(style)
            missing = [s for s in scenarios if s.scenario_id not in cached]
            logger.info("[%s] %d cached, %d to annotate (prompts: %s)",
                        style, len(cached), len(missing), prompt_version)

            if not missing:
                all_style_results[style] = cached
                continue

            entries, all_detections, _ = prepare_bulk_entries(
                scenarios=missing,
                exchanges=exchanges,
                annotator_style=style,
                prompt_version=prompt_version,
                context_window=context_window,
                with_screenshots=with_screenshots,
            )
            logger.info("[%s] %d annotation entries across %d scenarios",
                        style, len(entries), len(all_detections))

            if not entries:
                all_style_results[style] = cached
                continue

            sidecar = load_benchmark_inflight_batch(version, profile, style)
            existing_batch_id = None
            if sidecar:
                expected = sidecar.get("entry_keys_hash")
                actual = _entries_keys_hash(entries)
                if expected == actual:
                    existing_batch_id = sidecar["batch_id"]
                    logger.info("[%s] resuming in-flight batch %s (submitted %s)",
                                style, existing_batch_id, sidecar.get("submitted_at", "?"))
                else:
                    logger.error(
                        "[%s] in-flight sidecar exists but entry-keys hash differs "
                        "(sidecar=%s, current=%s). Scenario set changed between runs. "
                        "Delete results/benchmark/%s/in_flight/%s_%s.json to start fresh.",
                        style, expected, actual, version, profile, style,
                    )
                    raise RuntimeError("entry-keys mismatch on benchmark in-flight resume")

            def _record(batch_id, _profile=profile, _style=style):
                save_benchmark_inflight_batch(version, _profile, _style, {
                    "provider": "unknown",
                    "model": annotator_cfg.get("model", ""),
                    "batch_id": batch_id,
                    "n_entries": len(entries),
                    "entry_keys_hash": _entries_keys_hash(entries),
                    "display_name": "benchmark_annotate",
                    "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })

            per_scenario_results = execute_and_parse_bulk(
                entries=entries,
                all_detections=all_detections,
                annotator_profile=annotator_profile,
                mode=ann_mode,
                existing_batch_id=existing_batch_id,
                on_batch_created=_record,
            )
            logger.info("[%s] parsed %d scenario results", style, len(per_scenario_results))

            annotate_cfg_full = get_phase_config("annotate", annotator_profile)
            per_scenario_labeled = label_bulk(
                per_scenario_results=per_scenario_results,
                annotator_style=style,
                annotator_profile=annotator_profile,
                annotator_model=annotate_cfg_full["model"],
                mode=ann_mode,
            )
            logger.info("[%s] labeled %d scenarios", style, len(per_scenario_labeled))

            for scenario_id, labeled_data in per_scenario_labeled.items():
                save_benchmark_result(version, "annotations", profile, style,
                                      f"{scenario_id}.json", data=labeled_data)

            clear_benchmark_inflight_batch(version, profile, style)

            merged = dict(cached)
            merged.update(per_scenario_labeled)
            all_style_results[style] = merged

        # ---------------------------------------------------------------
        # Phase 3: Per-style scores (no composite aggregation)
        # ---------------------------------------------------------------
        logger.info("--- Phase 3: Per-Style Scores ---")
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
            for ann_type in get_annotation_types():
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
            logger.info("[%s] mean=%.3f n=%d scaffolding=%.3f rapport=%.3f",
                        style, overall_mean, n,
                        type_means.get('scaffolding', 0), type_means.get('rapport', 0))

    logger.info("Results saved (version: %s)", version)
    save_benchmark_result(version, "_complete.json", data={
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })


def _resolve_or_create_version(config: dict) -> str:
    """Return the version for this run, reusing an in-progress run if one exists.

    Reads results/benchmark/_active_runs/{tutor_profile}.json. If that pointer
    names a version whose _complete.json marker doesn't exist, reuse it; this
    is what makes a ctrl-C'd run resumable across midnight. Otherwise generate
    f'{profile}_{date}' and write a fresh pointer.
    """
    tutor_profile = config.get("tutor_profiles", ["anthropic"])[0]
    pointer = load_benchmark_result("_active_runs", f"{tutor_profile}.json")
    if pointer:
        candidate = pointer.get("version")
        if candidate:
            complete = load_benchmark_result(candidate, "_complete.json")
            if complete is None:
                logger.info("Resuming in-progress version: %s", candidate)
                return candidate

    new_version = f"{tutor_profile}_{datetime.date.today().strftime('%Y-%m-%d')}"
    save_benchmark_result("_active_runs", f"{tutor_profile}.json", data={
        "version": new_version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    logger.info("Auto-generated version: %s", new_version)
    return new_version


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI tutor models")
    parser.add_argument("--version", default=None,
                        help="Benchmark version (default: auto-generated from tutor profile + date)")
    parser.add_argument("--tutor-profile",
                        help="Single tutor profile (e.g. gemini, openai, anthropic)")
    parser.add_argument("--scenario-mode", choices=["detected", "random", "both"],
                        help="Override scenario extraction mode")
    parser.add_argument("--max-scenarios", type=int, help="Limit number of scenarios")
    parser.add_argument("--max-per-conv", type=int,
                        help="Max scenarios per conversation (randomly sampled)")
    parser.add_argument("--test", type=int, default=0,
                        help="Limit detection to N transcripts (0 = all)")
    parser.add_argument("--mode", choices=["batch", "sync"],
                        help="Override execution mode (batch or sync)")
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Attach anchored screenshots to detection, exchange, and annotation prompts. Requires vision-capable models.")
    args = parser.parse_args()

    overrides = {
        "scenario_mode": args.scenario_mode,
        "max_scenarios": args.max_scenarios,
        "max_per_conv": args.max_per_conv,
        "tutor_profile": args.tutor_profile,
        "mode": args.mode,
        "test_transcripts": args.test,
        "with_screenshots": args.with_screenshots if args.with_screenshots else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config = get_benchmark_config(overrides)

    if args.version:
        version = args.version
    else:
        version = config.get("version") or _resolve_or_create_version(config)

    setup_logging(version=version)

    run_benchmark(version, config)


if __name__ == "__main__":
    main()
