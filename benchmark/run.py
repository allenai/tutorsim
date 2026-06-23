"""Benchmark pipeline orchestrator.

All-batch architecture:
  Phase 1: Generate exchanges (batch per round -- all scenarios in parallel)
  Phase 2: Annotate all exchanges (annotate -> decompose -> structure)
  Phase 3: Score (action F1 + outcome rate against situation_label_agg)

Usage:
    python -m benchmark.run                              # auto-generated version
    python -m benchmark.run --version v1
    python -m benchmark.run --version v1 --tutor-profile anthropic
    python -m benchmark.run --version v1 --max-scenarios 10
    python -m benchmark.run --version v1 --mode sync  # use sync instead of batch
"""

import argparse
import datetime
import logging
import time
from pathlib import Path

from common.logging_setup import setup_logging
from annotator.core.client import ModelClient
from annotator.core.config import get_phase_config, get_benchmark_config
from annotator.core.detect import run_detect
from annotator.core.storage import (
    save_benchmark_result, load_benchmark_result, list_benchmark_result_files,
)

from .core.scenarios import load_scenarios, Scenario
from .core.exchange import run_exchange, run_exchanges_batch, Exchange


BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent

logger = logging.getLogger(__name__)



def _sum_usage(*usages: dict) -> dict:
    """Sum input/output/total tokens across N usage dicts."""
    out = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not isinstance(u, dict):
            continue
        for k in out:
            out[k] += int(u.get(k, 0) or 0)
    return out


def _collect_exchange_tokens(exchanges: dict) -> tuple[dict, dict]:
    """Aggregate tutor + student tokens across all exchanges in this run."""
    tutor = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    student = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for ex in exchanges.values():
        tutor = _sum_usage(tutor, getattr(ex, "tutor_usage", {}) or {})
        student = _sum_usage(student, getattr(ex, "student_usage", {}) or {})
    return tutor, student


def _collect_annotation_tokens(per_scenario_results: dict) -> dict:
    """Aggregate annotator tokens (Pass 2 / decompose / structure share the
    same `usage` accumulator inside each scenario's result block)."""
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for outer in per_scenario_results.values():
        # outer = {scenario_id: {annotations: [...], usage: {...}}}
        for inner in outer.values() if isinstance(outer, dict) else []:
            if isinstance(inner, dict):
                total = _sum_usage(total, inner.get("usage", {}))
    return total


def _latency_stats(samples: list[float]) -> dict | None:
    """Mean / p50 / p95 over per-call latency samples. None on empty."""
    if not samples:
        return None
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2]
    p95_idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
    p95 = s[p95_idx]
    return {
        "n": n,
        "total_seconds": round(sum(samples), 3),
        "mean_seconds": round(sum(samples) / n, 3),
        "p50_seconds": round(p50, 3),
        "p95_seconds": round(p95, 3),
    }


def _collect_exchange_latencies(exchanges: dict) -> dict:
    """Per-role latency stats aggregated across all scenarios.

    Sync mode populates Exchange.tutor_latencies / student_latencies; batch
    mode leaves them empty (batch wall-clock = queue + processing, not
    a meaningful model-comparison latency)."""
    tutor_samples: list[float] = []
    student_samples: list[float] = []
    for ex in exchanges.values():
        tutor_samples.extend(getattr(ex, "tutor_latencies", []) or [])
        student_samples.extend(getattr(ex, "student_latencies", []) or [])
    return {
        "tutor": _latency_stats(tutor_samples),
        "student": _latency_stats(student_samples),
    }


def run_phase2_and_score(
    version: str,
    profile: str,
    annotator_profile: str,
    annotator_mode: str,
    prompt_version: str,
    context_window: int,
    scenarios: list,
    exchanges: dict,
    with_screenshots: bool = False,
    phase1_seconds: float | None = None,
) -> dict:
    """Phase 2 (annotate -> decompose -> structure) + Phase 3 (score).

    Single annotator pass per scenario, then in-memory decompose + structure,
    then action-F1 + outcome-rate against situation_label_agg.

    Returns the summary dict saved to scores/{profile}.json.
    """
    from .core.annotator_bridge import (
        prepare_bulk_entries, execute_and_parse_bulk,
        decompose_bulk, structure_bulk,
    )
    from .core.score import score_scenarios

    # --- Annotate ---
    # Force context_window=0 for benchmark scoring. The annotator must see
    # ONLY the AI replay -- the surrounding excerpt window otherwise leaks
    # pre-cut human-tutor turns into the prompt, and the LM may attribute
    # them to "the tutor" despite the >>> DETECTED MOMENT <<< markers.
    # This is a benchmark-only override; the annotator pipeline still uses
    # the configured context_window for its own (non-cut) runs.
    phase2_t0 = time.monotonic()
    entries, all_detections, _ = prepare_bulk_entries(
        scenarios=scenarios,
        exchanges=exchanges,
        annotator_style=None,
        prompt_version=prompt_version,
        context_window=0,
        with_screenshots=with_screenshots,
    )
    logger.info("Phase 2: %d annotation entries across %d scenarios",
                len(entries), len(all_detections))

    if not entries:
        empty = {"scaffolding": {"tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,"f1":0.0},
                 "rigor":       {"tp":0,"fp":0,"fn":0,"precision":0.0,"recall":0.0,"f1":0.0},
                 "outcome_pos_rate": 0.0,
                 "n_scenarios": 0, "n_scored_for_f1": 0,
                 "profile": profile}
        save_benchmark_result(version, "scores", f"{profile}.json", data=empty)
        return empty

    per_scenario_results = execute_and_parse_bulk(
        entries=entries,
        all_detections=all_detections,
        annotator_profile=annotator_profile,
        mode=annotator_mode,
        existing_batch_id=None,
        on_batch_created=lambda *_a, **_k: None,
    )
    logger.info("Phase 2: parsed %d scenario results", len(per_scenario_results))

    per_scenario_results = decompose_bulk(per_scenario_results, annotator_profile, mode=annotator_mode)
    logger.info("Phase 2: decomposed")
    per_scenario_results = structure_bulk(per_scenario_results, annotator_profile, mode=annotator_mode)
    logger.info("Phase 2: structured")
    phase2_seconds = time.monotonic() - phase2_t0

    # Save per-scenario annotations (flat, no styles subdir).
    for scenario_id, results in per_scenario_results.items():
        save_benchmark_result(version, "annotations", profile,
                              f"{scenario_id}.json", data=results)

    # --- Phase 3: score ---
    scenario_dicts = [s.to_dict() for s in scenarios]
    annotation_dicts = []
    for s in scenarios:
        results = per_scenario_results.get(s.scenario_id, {})
        ann = results.get(s.scenario_id, {})
        annotation_dicts.append(ann)

    summary = score_scenarios(scenario_dicts, annotation_dicts)
    summary["profile"] = profile

    # Latency + token roll-up.
    tutor_tokens, student_tokens = _collect_exchange_tokens(exchanges)
    annotation_tokens = _collect_annotation_tokens(per_scenario_results)
    total_tokens = _sum_usage(tutor_tokens, student_tokens, annotation_tokens)
    summary["timings"] = {
        "phase1_exchange_seconds": phase1_seconds,
        "phase2_annotate_seconds": phase2_seconds,
        "total_seconds": (
            (phase1_seconds or 0.0) + phase2_seconds
            if phase1_seconds is not None else None
        ),
    }
    summary["tokens"] = {
        "tutor": tutor_tokens,
        "student": student_tokens,
        "annotation": annotation_tokens,
        "total": total_tokens,
    }
    summary["latency"] = _collect_exchange_latencies(exchanges)
    save_benchmark_result(version, "scores", f"{profile}.json", data=summary)
    def _fmt(rate):
        return f"{rate:.3f}" if isinstance(rate, (int, float)) else "—"
    logger.info(
        "[%s] scaffolding_did=%s (%d/%d)  rigor_did=%s (%d/%d)  overscaffold=%s (%d/%d, avail=%s)  outcome+=%.3f  n=%d",
        profile,
        _fmt(summary["scaffolding_did"]["rate"]),
        summary["scaffolding_did"]["n_yes"], summary["scaffolding_did"]["n_total"],
        _fmt(summary["rigor_did"]["rate"]),
        summary["rigor_did"]["n_yes"], summary["rigor_did"]["n_total"],
        _fmt(summary["overscaffold"]["rate"]),
        summary["overscaffold"]["n_yes"], summary["overscaffold"]["n_total"],
        summary["overscaffold"]["available"],
        summary["outcome_pos_rate"],
        summary["n_scenarios"],
    )
    logger.info(
        "[%s] scaffold_calibrated=%s (%d clean/%d, %d over-scaffolded)  rigor_calibrated=%s (%d clean/%d)",
        profile,
        _fmt(summary["scaffold_calibrated"]["score"]),
        summary["scaffold_calibrated"]["n_clean_yes"],
        summary["scaffold_calibrated"]["n_total"],
        summary["scaffold_calibrated"]["n_overscaffold"],
        _fmt(summary["rigor_calibrated"]["score"]),
        summary["rigor_calibrated"]["n_clean_yes"],
        summary["rigor_calibrated"]["n_total"],
    )
    p1 = phase1_seconds or 0.0
    p2 = phase2_seconds
    logger.info(
        "[%s] phase1=%.0fs phase2=%.0fs total=%.0fs | tokens tutor=%d student=%d annotation=%d total=%d",
        profile, p1, p2, p1 + p2,
        tutor_tokens["total_tokens"], student_tokens["total_tokens"],
        annotation_tokens["total_tokens"], total_tokens["total_tokens"],
    )
    tutor_lat = summary["latency"]["tutor"]
    if tutor_lat:
        logger.info(
            "[%s] tutor per-call latency: n=%d mean=%.2fs p50=%.2fs p95=%.2fs",
            profile, tutor_lat["n"], tutor_lat["mean_seconds"],
            tutor_lat["p50_seconds"], tutor_lat["p95_seconds"],
        )
    return summary


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
    if with_screenshots or config.get("tutor", {}).get("mode"):
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
    ann_mode = annotator_cfg["mode"]
    annotator_profile = annotator_cfg["profile"]
    prompt_version = annotator_cfg["prompt_version"]
    context_window = annotator_cfg["context_window"]

    student_profile = config["student"]["profile"]
    student_mode = config["student"].get("mode")
    student_cfg = get_phase_config("tutor", student_profile)
    student_client = ModelClient(student_cfg["model"])

    trait_client = None
    trait_model = None
    if student_mode == "trait":
        trait_client = student_client
        trait_model = student_cfg["model"]

    tutor_mode = config.get("tutor", {}).get("mode")

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
                    max_turns=exchange_cfg["max_turns"],
                    tutor_max_tokens=tutor_cfg["max_tokens"],
                    student_max_tokens=student_cfg["max_tokens"],
                    poll_interval=exchange_cfg["poll_interval"],
                    save_callback=_save_exchange,
                    prompt_version=exchange_prompt_version,
                    images_by_scenario=images_by_scenario,
                    student_mode=student_mode,
                    trait_client=trait_client,
                    trait_model=trait_model,
                    tutor_mode=tutor_mode,
                    transcripts=transcripts_for_screenshots,
                )
            else:
                new_exchanges = {}
                for i, scenario in enumerate(missing):
                    try:
                        exchange = run_exchange(
                            scenario=scenario,
                            tutor_client=tutor_client,
                            student_client=student_client,
                            max_turns=exchange_cfg["max_turns"],
                            tutor_max_tokens=tutor_cfg["max_tokens"],
                            student_max_tokens=student_cfg["max_tokens"],
                            prompt_version=exchange_prompt_version,
                            images=(images_by_scenario or {}).get(scenario.scenario_id),
                            student_mode=student_mode,
                            trait_client=trait_client,
                            trait_model=trait_model,
                            tutor_mode=tutor_mode,
                            transcripts=transcripts_for_screenshots,
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
        # Phase 2 + 3: Annotate (annotate -> decompose -> structure) + score
        # ---------------------------------------------------------------
        annotator_mode = annotator_cfg["mode"]
        run_phase2_and_score(
            version=version,
            profile=profile,
            annotator_profile=annotator_profile,
            annotator_mode=annotator_mode,
            prompt_version=prompt_version,
            context_window=context_window,
            scenarios=scenarios,
            exchanges=exchanges,
            with_screenshots=with_screenshots,
        )

    logger.info("Results saved (version: %s)", version)
    save_benchmark_result(version, "_complete.json", data={
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })


def _resolve_or_create_version(config: dict) -> str:
    """Return the version for this run, reusing an in-progress run if one exists.

    Reads results/benchmark/_active_runs/{tutor_profile}.json. If that pointer
    names a version whose _complete.json marker doesn't exist, reuse it; this
    is what makes a ctrl-C'd run resumable across midnight. Otherwise generate
    f'{tutor_model}_{prompt_version}_{date}' and write a fresh pointer.

    Naming convention: the tutor MODEL + prompt_version go in the version
    string so replays against a different LM OR a different prompt iteration
    are visually distinct on disk (and don't silently overwrite each other).
    """
    from annotator.core.config import get_phase_config

    tutor_profile = config.get("tutor_profiles", ["anthropic"])[0]
    tutor_model = get_phase_config("tutor", tutor_profile)["model"]
    tutor_slug = tutor_model.replace("/", "_")
    prompt_version = (config.get("exchange", {}) or {}).get("prompt_version", "v1")

    pointer = load_benchmark_result("_active_runs", f"{tutor_profile}.json")
    if pointer:
        candidate = pointer.get("version")
        if candidate:
            complete = load_benchmark_result(candidate, "_complete.json")
            if complete is None:
                logger.info("Resuming in-progress version: %s", candidate)
                return candidate

    new_version = f"{tutor_slug}_{prompt_version}_{datetime.date.today().strftime('%Y-%m-%d')}"
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
    parser.add_argument("--scenario-mode", choices=["detected", "random", "both", "human"],
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
