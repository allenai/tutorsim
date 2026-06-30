"""CLI entry point for tutorsim benchmark runs.

Provides:
  run_cell(tutor, mode, run_cfg, *, date, results_root, trait_cache_dir)
    -- single-cell pipeline: load -> conversation -> score -> store -> summary.
    Resumable: scenarios already on disk are skipped.
    Skip-on-error: one bad scenario logs + skips, never crashes the cell.
    trials>1: runs conversation+score N times per scenario; summary has mean+spread.

  expand_cells(tutors, modes) -> list[dict]
    -- expand (tutors x modes) into cell dicts with lane assignment by provider.

  run_sweep(cells, run_cfg, *, date, results_root, trait_cache_dir, _run_cell_fn)
    -- schedule cells: lanes parallel (ThreadPoolExecutor), within-lane sequential.
    Returns list of all run_ids.

  main() / argparse "run" subcommand:
    tutorsim run --tutors X [--modes ...] [--dataset ...] [--sample N]
                 [--trials N] [--seed N] [--max-turns N]
                 [--trait-cache-dir DIR]

No module-level SDK imports.
ASCII console output only (no Unicode / em-dash / box-drawing).
All file I/O is UTF-8.
"""
import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell expansion + lane scheduling (ported from _archive/benchmark/sweep.py)
# ---------------------------------------------------------------------------

def expand_cells(tutors: list[str], modes: list[str]) -> list[dict]:
    """Expand (tutors x modes) into one cell dict per combination.

    Each cell: {"tutor": str, "mode": str, "lane": str}
    Lane is inferred from provider (e.g. "anthropic", "gemini", "openai").
    Registered tutors (not in the API roster) get lane "default".

    Args:
        tutors: List of tutor model IDs.
        modes:  List of prompt mode strings.

    Returns:
        List of cell dicts in sweep order (tutor-major, mode-minor).
    """
    # Lazy import to avoid module-level SDK import
    from tutorsim.client import infer_provider
    from tutorsim.config import get_registered_tutor

    cells = []
    for tutor in tutors:
        # Registered (non-API) tutors -> "default" lane
        if get_registered_tutor(tutor) is not None:
            lane = "default"
        else:
            try:
                lane = infer_provider(tutor)
            except ValueError:
                lane = "default"
        for mode in modes:
            cells.append({"tutor": tutor, "mode": mode, "lane": lane})
    return cells


def run_sweep(
    cells: list[dict],
    run_cfg,
    *,
    date: str,
    results_root: str = "results",
    trait_cache_dir: str = "results/benchmark/_trait_cache",
    _run_cell_fn=None,
) -> list[str]:
    """Schedule cells: lanes run in parallel; cells within a lane run sequentially.

    Ported from _archive/benchmark/sweep.py run_lane + ThreadPoolExecutor pattern.

    Args:
        cells:           Output of expand_cells().
        run_cfg:         Pre-built RunConfig passed through to run_cell.
        date:            Date string for run_id (e.g. "20260626").
        results_root:    Root directory for results.
        trait_cache_dir: Directory for student trait cache.
        _run_cell_fn:    Injectable for testing (default: run_cell from this module).

    Returns:
        List of run_ids (one per cell), in lane-then-within-lane order.
    """
    if _run_cell_fn is None:
        _run_cell_fn = run_cell

    # Group cells by lane, preserving expansion order
    by_lane: OrderedDict = OrderedDict()
    for cell in cells:
        by_lane.setdefault(cell["lane"], []).append(cell)

    n_lanes = len(by_lane)
    all_run_ids: list[str] = []

    def _run_lane(lane: str, lane_cells: list[dict]) -> list[str]:
        """Run one lane's cells sequentially. Returns run_ids in order."""
        lane_run_ids = []
        for cell in lane_cells:
            logger.info("[lane %s] START %s/%s", lane, cell["tutor"], cell["mode"])
            rid = _run_cell_fn(
                cell["tutor"],
                cell["mode"],
                run_cfg,
                date=date,
                results_root=results_root,
                trait_cache_dir=trait_cache_dir,
            )
            logger.info("[lane %s] DONE  %s/%s -> %s", lane, cell["tutor"], cell["mode"], rid)
            lane_run_ids.append(rid)
        return lane_run_ids

    # Lanes in parallel
    with ThreadPoolExecutor(max_workers=n_lanes) as pool:
        futs = {
            pool.submit(_run_lane, lane, lane_cells): lane
            for lane, lane_cells in by_lane.items()
        }
        for fut in as_completed(futs):
            all_run_ids.extend(fut.result())

    return all_run_ids


# ---------------------------------------------------------------------------
# Lazy module imports (no module-level SDK imports)
# ---------------------------------------------------------------------------

def _import_modules():
    """Import heavy modules lazily to avoid SDK import at module level."""
    from tutorsim import conversation, scoring, results, report
    from tutorsim.config import build_run_config
    from tutorsim.scenarios import load_scenarios
    return conversation, scoring, results, report, build_run_config, load_scenarios


# Expose at module level for monkeypatching in tests.
# These are imported at function call time, not at module load time,
# but we re-export references so patch targets resolve correctly.
import tutorsim.conversation as conversation
import tutorsim.scoring as scoring
import tutorsim.results as results
import tutorsim.report as report
from tutorsim.config import build_run_config
from tutorsim.scenarios import (
    DatasetNotFoundError,
    load_manifest,
    load_scenarios,
    validate_dataset,
)


def _json_sha256(data) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version() -> str | None:
    try:
        return version("tutorsim")
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _resource_hashes(root: str) -> dict[str, str]:
    """Hash packaged resource files under a package-relative directory."""
    base = files("tutorsim") / root
    out: dict[str, str] = {}

    def walk(node, rel: str) -> None:
        if node.is_file():
            out[rel] = hashlib.sha256(node.read_bytes()).hexdigest()
            return
        for child in sorted(node.iterdir(), key=lambda p: p.name):
            child_rel = f"{rel}/{child.name}" if rel else child.name
            walk(child, child_rel)

    walk(base, root)
    return out


# ---------------------------------------------------------------------------
# Trials aggregation helpers
# ---------------------------------------------------------------------------

def _mean_spread(values: list) -> tuple[float | None, float | None]:
    """Return (mean, std) for a list of numeric values; None for empty/all-None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None
    n = len(nums)
    mean = sum(nums) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in nums) / n
    return mean, math.sqrt(variance)


def _aggregate_trials(trial_metrics: list[dict], n_trials: int) -> dict:
    """Compute mean+spread summary across N trial metric dicts.

    Convention:
      - "trials": N
      - "mean": aggregate dict (same shape as single-trial output)
      - "spread": matching dict with std values (None where metric was None everywhere)

    Numeric leaf fields from report.aggregate are:
      n_scenarios, outcome_pos_rate,
      scaffolding_did.{n_yes, n_total, rate},
      rigor_did.{n_yes, n_total, rate},
      overscaffold.{n_yes, n_total, rate},
      scaffold_calibrated.{n_clean_yes, n_overscaffold, n_total, score},
      rigor_calibrated.{n_clean_yes, n_total, score}.
    """
    def _field(dicts, *keys):
        """Extract a nested field from each dict in dicts."""
        vals = []
        for d in dicts:
            v = d
            for k in keys:
                if v is None:
                    break
                v = v.get(k) if isinstance(v, dict) else None
            vals.append(v)
        return vals

    def _sub_mean_spread(dicts, subkey, fields):
        mean_sub = {}
        spread_sub = {}
        for f in fields:
            m, s = _mean_spread(_field(dicts, subkey, f))
            mean_sub[f] = m
            spread_sub[f] = s
        return mean_sub, spread_sub

    # top-level scalar
    m_nscen, s_nscen = _mean_spread(_field(trial_metrics, "n_scenarios"))
    m_opr, s_opr = _mean_spread(_field(trial_metrics, "outcome_pos_rate"))

    # sub-dicts
    m_scaf, s_scaf = _sub_mean_spread(trial_metrics, "scaffolding_did", ["n_yes", "n_total", "rate"])
    m_rigor, s_rigor = _sub_mean_spread(trial_metrics, "rigor_did", ["n_yes", "n_total", "rate"])
    m_over, s_over = _sub_mean_spread(trial_metrics, "overscaffold", ["n_yes", "n_total", "rate"])

    # scaffold_calibrated has an extra field
    m_sc, s_sc = _sub_mean_spread(
        trial_metrics, "scaffold_calibrated",
        ["n_clean_yes", "n_overscaffold", "n_total", "score"],
    )
    m_rc, s_rc = _sub_mean_spread(
        trial_metrics, "rigor_calibrated",
        ["n_clean_yes", "n_total", "score"],
    )

    # "available" is boolean -- take first trial's value (invariant across trials)
    over_available = (trial_metrics[0].get("overscaffold") or {}).get("available", False)

    mean_dict = {
        "n_scenarios": m_nscen,
        "outcome_pos_rate": m_opr,
        "scaffolding_did": m_scaf,
        "rigor_did": m_rigor,
        "overscaffold": {**m_over, "available": over_available},
        "scaffold_calibrated": m_sc,
        "rigor_calibrated": m_rc,
    }
    spread_dict = {
        "n_scenarios": s_nscen,
        "outcome_pos_rate": s_opr,
        "scaffolding_did": s_scaf,
        "rigor_did": s_rigor,
        "overscaffold": {**s_over},
        "scaffold_calibrated": s_sc,
        "rigor_calibrated": s_rc,
    }

    return {
        "trials": n_trials,
        "mean": mean_dict,
        "spread": spread_dict,
    }


# ---------------------------------------------------------------------------
# Public API: run_cell
# ---------------------------------------------------------------------------

def run_cell(
    tutor: str,
    mode: str,
    run_cfg,
    *,
    date: str,
    results_root: str = "results",
    trait_cache_dir: str = "results/benchmark/_trait_cache",
) -> str:
    """Run a single (tutor, mode) cell end-to-end.

    Sequence (mirrors _archive/benchmark/replay.py single-cell path):
      1. build_run_config -> load_scenarios -> (slice sample)
      2. make_run_id + write_config
      3. For each scenario (skip if is_done):
           run_conversation -> score -> write_transcript + write_score
           On any exception: log SKIP <id>: <err> and continue
      4. aggregate over completed annotations -> write_summary
      5. Return run_id

    Args:
        tutor: Tutor model id (e.g. "claude-opus-4-8").
        mode: Prompt mode (e.g. "plain").
        run_cfg: Pre-built RunConfig or None (if None, built from config defaults).
        date: Date string for run_id (e.g. "20260626"). Caller supplies; not auto.
        results_root: Root directory for results (default "results").
        trait_cache_dir: Directory for student trait persona cache.

    Returns:
        run_id string (e.g. "claude-opus-4-8_plain_balanced_520_20260626").
    """
    # Step 1: build config
    if run_cfg is None:
        cfg = build_run_config(tutors=[tutor], modes=[mode])
    else:
        cfg = run_cfg

    # Load scenarios
    all_scenarios = load_scenarios(cfg.dataset)
    if cfg.sample is not None:
        all_scenarios = all_scenarios[: cfg.sample]

    n_total = len(all_scenarios)
    if n_total == 0:
        raise RuntimeError(
            f"Dataset '{cfg.dataset}' yielded zero scenarios after sampling; "
            "refusing to write an empty benchmark run."
        )
    logger.info("Loaded %d scenarios from dataset '%s'", n_total, cfg.dataset)
    dataset_manifest = load_manifest(cfg.dataset)

    # Step 2: run_id + config on disk
    run_id = results.make_run_id(tutor, mode, cfg.dataset, date)

    config_dict = {
        "tutor": tutor,
        "mode": mode,
        "dataset": cfg.dataset,
        "sample": cfg.sample,
        "max_turns": cfg.max_turns,
        "trials": cfg.trials,
        "seed": cfg.seed,
        "student": cfg.student,
        "scorer": cfg.scorer,
        "resolved_tutors": cfg.resolved_tutors,
        "config_source": getattr(cfg, "config_source", None),
        "trait_cache_dir": trait_cache_dir,
        "reproducibility": {
            "tutorsim_version": _package_version(),
            "git_commit": _git_commit(),
            "config_hash": _json_sha256({
                "student": cfg.student,
                "scorer": cfg.scorer,
                "resolved_tutors": cfg.resolved_tutors,
                "defaults": {
                    "sample": cfg.sample,
                    "max_turns": cfg.max_turns,
                    "trials": cfg.trials,
                    "seed": cfg.seed,
                },
            }),
            "prompt_hashes": _resource_hashes("prompts"),
            "dataset_manifest": dataset_manifest,
        },
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    results.write_config(run_id, config_dict, results_root=results_root)
    logger.info("Run id: %s", run_id)

    n_trials = getattr(cfg, "trials", 1) or 1

    # Step 3: per-scenario loop
    # For trials>1: collect per-trial aggregate metrics; for trials=1: single pass.
    # Each trial runs all scenarios; we aggregate per-trial, then compute mean+spread.

    _EMPTY_METRICS = {
        "n_scenarios": 0,
        "scaffolding_did": {"n_yes": 0, "n_total": 0, "rate": None},
        "rigor_did": {"n_yes": 0, "n_total": 0, "rate": None},
        "overscaffold": {"n_yes": 0, "n_total": 0, "rate": None, "available": False},
        "outcome_pos_rate": 0.0,
        "scaffold_calibrated": {"n_clean_yes": 0, "n_overscaffold": 0, "n_total": 0, "score": None},
        "rigor_calibrated": {"n_clean_yes": 0, "n_total": 0, "score": None},
    }

    # ---------------------------------------------------------------------------
    # Latency / token helpers (ported from _archive/benchmark/scoring.py)
    # ---------------------------------------------------------------------------

    def _latency_stats(samples: list) -> dict | None:
        """Mean / p50 / p95 over per-call latency samples. None on empty.

        Verbatim port of _archive/benchmark/scoring.py::_latency_stats.
        Sorted-index percentile: p50 = s[n//2], p95_idx = max(0, min(n-1, round(0.95*n)-1)).
        """
        if not samples:
            return None
        s = sorted(samples)
        n = len(s)
        p50 = s[n // 2]
        p95_idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
        p95 = s[p95_idx]
        total = sum(samples)
        return {
            "n": n,
            "total_seconds": round(total, 3),
            "mean_seconds": round(total / n, 3),
            "p50_seconds": round(p50, 3),
            "p95_seconds": round(p95, 3),
        }

    def _run_trial(trial_idx: int) -> tuple:
        """Run all scenarios once (one trial).

        Returns:
            (metrics_dict, transcripts_list)
            where transcripts_list are the Transcript objects for this trial.
        """
        completed_scenarios = []
        completed_annotations = []
        completed_transcripts = []
        counts = {
            "attempted": n_total,
            "succeeded": 0,
            "failed": 0,
            "resumed": 0,
        }
        failed_scenarios = []

        for i, scenario in enumerate(all_scenarios, 1):
            sid = scenario.id

            # Resume key includes trial index for trials>1
            if n_trials > 1:
                resume_sid = f"{sid}_t{trial_idx}"
            else:
                resume_sid = sid

            # Resume: skip if both transcript and score already exist
            if results.is_done(run_id, resume_sid, results_root=results_root):
                logger.info("[trial %d][%d/%d] SKIP (already done): %s", trial_idx, i, n_total, sid)
                score_dict = results.read_score(run_id, resume_sid, results_root=results_root)
                if score_dict is not None:
                    from tutorsim.scoring import Annotation
                    try:
                        ann = Annotation(**score_dict)
                        completed_scenarios.append(scenario)
                        completed_annotations.append(ann)
                        counts["succeeded"] += 1
                        counts["resumed"] += 1
                        # No transcript object on resume -- latencies not available
                    except Exception as e:
                        counts["failed"] += 1
                        failed_scenarios.append({"id": sid, "error": str(e), "phase": "resume"})
                        logger.warning(
                            "[trial %d][%d/%d] Could not reload score for %s: %s",
                            trial_idx, i, n_total, sid, e,
                        )
                continue

            try:
                # Run conversation
                transcript = conversation.run_conversation(
                    scenario,
                    tutor_id=tutor,
                    tutor_mode=mode if mode else None,
                    student_id=(cfg.student or {}).get("model"),
                    student_mode=(cfg.student or {}).get("mode", "oracle"),
                    max_turns=cfg.max_turns,
                    trait_cache_dir=trait_cache_dir,
                )

                # Write transcript before scoring (so a score failure doesn't lose it)
                transcript_dict = (
                    transcript.to_dict() if hasattr(transcript, "to_dict") else dict(transcript)
                )
                results.write_transcript(run_id, resume_sid, transcript_dict, results_root=results_root)

                # Score
                annotation = scoring.score(scenario, transcript)

                # Write score
                annotation_dict = (
                    annotation.to_dict() if hasattr(annotation, "to_dict") else dict(annotation)
                )
                results.write_score(run_id, resume_sid, annotation_dict, results_root=results_root)

                completed_scenarios.append(scenario)
                completed_annotations.append(annotation)
                completed_transcripts.append(transcript)
                counts["succeeded"] += 1
                logger.info("[trial %d][%d/%d] OK: %s", trial_idx, i, n_total, sid)

            except Exception as e:
                counts["failed"] += 1
                failed_scenarios.append({"id": sid, "error": str(e), "phase": "run"})
                logger.error("[trial %d][%d/%d] SKIP %s: %s", trial_idx, i, n_total, sid, e)
                continue

        if completed_scenarios:
            metrics = report.aggregate(completed_scenarios, completed_annotations)
        else:
            raise RuntimeError(
                f"No scenarios completed for {tutor}/{mode} trial {trial_idx}; "
                f"{counts['failed']} of {counts['attempted']} attempted scenarios failed."
            )
        metrics["run_counts"] = counts
        if failed_scenarios:
            metrics["failed_scenarios"] = failed_scenarios
        return metrics, completed_transcripts, counts

    # Run all trials
    trial_results = [_run_trial(t) for t in range(1, n_trials + 1)]
    trial_metrics = [m for m, _, _ in trial_results]
    trial_counts = [c for _, _, c in trial_results]
    all_trial_transcripts = [t for _, ts, _ in trial_results for t in ts]

    # Build latency + token blocks from all completed transcripts
    # (ported from _archive/benchmark/scoring.py::_collect_exchange_latencies +
    # run_phase2_and_score token roll-up).
    tutor_lat_samples: list = []
    student_lat_samples: list = []
    tutor_input = tutor_output = 0
    student_input = student_output = 0
    for tx in all_trial_transcripts:
        tutor_lat_samples.extend(getattr(tx, "tutor_latencies", []) or [])
        student_lat_samples.extend(getattr(tx, "student_latencies", []) or [])
        tu = getattr(tx, "tutor_usage", {}) or {}
        su = getattr(tx, "student_usage", {}) or {}
        tutor_input += tu.get("input_tokens", 0) or 0
        tutor_output += tu.get("output_tokens", 0) or 0
        student_input += su.get("input_tokens", 0) or 0
        student_output += su.get("output_tokens", 0) or 0

    tutor_tokens = {
        "input_tokens": tutor_input,
        "output_tokens": tutor_output,
        "total_tokens": tutor_input + tutor_output,
    }
    student_tokens = {
        "input_tokens": student_input,
        "output_tokens": student_output,
        "total_tokens": student_input + student_output,
    }
    total_tokens = {
        "input_tokens": tutor_input + student_input,
        "output_tokens": tutor_output + student_output,
        "total_tokens": tutor_input + tutor_output + student_input + student_output,
    }
    latency_block = {
        "tutor": _latency_stats(tutor_lat_samples),
        "student": _latency_stats(student_lat_samples),
    }
    token_block = {
        "tutor": tutor_tokens,
        "student": student_tokens,
        "total": total_tokens,
    }

    # Step 4: aggregate + write summary
    if n_trials == 1:
        # trials=1: plain aggregate dict -- identical to Task-4 output
        metrics = trial_metrics[0]
    else:
        # trials>1: compute mean and spread (std) across trials for numeric leaf values
        metrics = _aggregate_trials(trial_metrics, n_trials)

    metrics["run_counts"] = {
        "attempted": sum(c["attempted"] for c in trial_counts),
        "succeeded": sum(c["succeeded"] for c in trial_counts),
        "failed": sum(c["failed"] for c in trial_counts),
        "resumed": sum(c["resumed"] for c in trial_counts),
    }

    # Merge latency + token blocks into summary (spec S7)
    metrics = dict(metrics)
    metrics["latency"] = latency_block
    metrics["tokens"] = token_block

    results.write_summary(run_id, metrics, results_root=results_root)
    logger.info(
        "Done. n_trials=%d  run_id=%s",
        n_trials,
        run_id,
    )

    return run_id


# ---------------------------------------------------------------------------
# CLI: argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tutorsim",
        description="Tutorsim benchmark runner",
    )
    subs = parser.add_subparsers(dest="command")

    # -- run subcommand -------------------------------------------------------
    run_p = subs.add_parser(
        "run",
        help="Run one or more (tutor x mode) cells",
    )
    run_p.add_argument(
        "--tutors",
        nargs="+",
        required=True,
        metavar="MODEL_ID",
        help="Tutor model id(s), e.g. claude-opus-4-8",
    )
    run_p.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Config file override (default: TUTORSIM_CONFIG, ./config.yaml, then packaged default)",
    )
    run_p.add_argument(
        "--modes",
        nargs="+",
        default=None,
        metavar="MODE",
        help="Prompt mode(s) (default: plain scaffolding_rigor)",
    )
    run_p.add_argument(
        "--dataset",
        default=None,
        metavar="NAME",
        help="Dataset name (default: balanced_520)",
    )
    run_p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use first N scenarios from the dataset (default: all)",
    )
    run_p.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help="Number of trials per cell (default: from config)",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed (default: from config)",
    )
    run_p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        dest="max_turns",
        metavar="N",
        help="Max turns per conversation (default: from config)",
    )
    run_p.add_argument(
        "--trait-cache-dir",
        default="results/benchmark/_trait_cache",
        dest="trait_cache_dir",
        metavar="DIR",
        help="Directory for student trait cache (default: results/benchmark/_trait_cache)",
    )

    # -- report subcommand ----------------------------------------------------
    report_p = subs.add_parser(
        "report",
        help="Aggregate all run summaries into a leaderboard (md + csv)",
    )
    report_p.add_argument(
        "--results-root",
        default="results",
        dest="results_root",
        metavar="DIR",
        help="Root directory containing run subdirectories (default: results)",
    )
    report_p.add_argument(
        "--out",
        default="leaderboard",
        metavar="PATH",
        help="Output file stem (default: leaderboard); writes .md and .csv",
    )

    # -- view subcommand ------------------------------------------------------
    view_p = subs.add_parser(
        "view",
        help="Build a self-contained HTML leaderboard viewer",
    )
    view_p.add_argument(
        "--results-root",
        default="results",
        dest="results_root",
        metavar="DIR",
        help="Root directory containing run subdirectories (default: results)",
    )
    view_p.add_argument(
        "--out",
        default="viewer.html",
        metavar="FILE",
        help="Output HTML file (default: viewer.html)",
    )

    # -- build-scenarios subcommand -------------------------------------------
    bs_p = subs.add_parser(
        "build-scenarios",
        help="(Dev-only) Hydrate and freeze a scenario set from ground-truth data",
    )
    bs_p.add_argument(
        "--set",
        required=True,
        metavar="NAME",
        help="Set name, e.g. balanced_520",
    )
    bs_p.add_argument(
        "--ids",
        required=True,
        metavar="FILE",
        help="Path to JSON list of scenario ids",
    )
    bs_p.add_argument(
        "--ground-truth",
        required=True,
        dest="ground_truth",
        metavar="DIR",
        help="Ground truth directory",
    )
    bs_p.add_argument(
        "--transcripts",
        required=True,
        metavar="DIR",
        help="Transcripts directory",
    )
    bs_p.add_argument(
        "--step-up-jsonl",
        default=None,
        dest="step_up_jsonl",
        metavar="FILE",
        help="Path to normalized step_up JSONL transcript file (optional)",
    )
    bs_p.add_argument(
        "--created",
        default="",
        metavar="DATE",
        help="ISO date string for manifest (default: empty)",
    )
    bs_p.add_argument(
        "--version",
        default="0",
        metavar="VERSION",
        help="Dataset version string for manifest (default: 0)",
    )
    bs_p.add_argument(
        "--out-root",
        default="scenarios",
        dest="out_root",
        metavar="DIR",
        help="Output root directory (default: scenarios/)",
    )

    # -- dataset subcommands ---------------------------------------------------
    dataset_p = subs.add_parser(
        "dataset",
        help="Build, validate, and eventually download scenario datasets",
    )
    dataset_subs = dataset_p.add_subparsers(dest="dataset_command")

    ds_build_p = dataset_subs.add_parser(
        "build",
        help="(Dev-only) Hydrate and freeze a scenario set from ground-truth data",
    )
    ds_build_p.add_argument("--set", required=True, metavar="NAME")
    ds_build_p.add_argument("--ids", required=True, metavar="FILE")
    ds_build_p.add_argument("--ground-truth", required=True, dest="ground_truth", metavar="DIR")
    ds_build_p.add_argument("--transcripts", required=True, metavar="DIR")
    ds_build_p.add_argument("--step-up-jsonl", default=None, dest="step_up_jsonl", metavar="FILE")
    ds_build_p.add_argument("--created", default="", metavar="DATE")
    ds_build_p.add_argument("--version", default="0", metavar="VERSION")
    ds_build_p.add_argument("--out-root", default="scenarios", dest="out_root", metavar="DIR")

    ds_validate_p = dataset_subs.add_parser(
        "validate",
        help="Validate an installed scenario set manifest and content hash",
    )
    ds_validate_p.add_argument("--set", required=True, metavar="NAME")
    ds_validate_p.add_argument("--root", default="scenarios", metavar="DIR")

    return parser


def _cmd_report(args) -> None:
    """Implement the 'report' subcommand: read all run summaries -> leaderboard md+csv."""
    import os
    from pathlib import Path

    run_ids = results.list_runs(args.results_root)
    summaries = []
    for run_id in run_ids:
        summary = results.read_summary(run_id, results_root=args.results_root)
        if summary is None:
            logger.warning("No summary.json for run %s -- skipping", run_id)
            continue
        # Inject run_id-derived fields if not already present
        if "tutor_model" not in summary or not summary.get("tutor_model"):
            # Best-effort: parse tutor from run_id prefix
            parts = run_id.split("_")
            summary = dict(summary)
            summary.setdefault("tutor_model", parts[0] if parts else run_id)
        if "mode" not in summary or not summary.get("mode"):
            parts = run_id.split("_")
            summary = dict(summary)
            summary.setdefault("mode", parts[1] if len(parts) > 1 else "")
        summaries.append(summary)

    if not summaries:
        print("No run summaries found in: " + args.results_root)
        return

    markdown, csv_str = report.leaderboard(summaries)

    out_stem = args.out
    md_path = Path(out_stem + ".md")
    csv_path = Path(out_stem + ".csv")

    md_path.write_text(markdown, encoding="utf-8")
    csv_path.write_text(csv_str, encoding="utf-8")

    print("Leaderboard written:")
    print("  md : " + str(md_path))
    print("  csv: " + str(csv_path))
    print("Rows: " + str(len(summaries)))


def _cmd_view(args) -> None:
    """Implement the 'view' subcommand: read all run summaries -> HTML viewer."""
    from pathlib import Path

    run_ids = results.list_runs(args.results_root)
    summaries = []
    for run_id in run_ids:
        summary = results.read_summary(run_id, results_root=args.results_root)
        if summary is None:
            logger.warning("No summary.json for run %s -- skipping", run_id)
            continue
        summary = dict(summary)
        summary.setdefault("tutor_model", run_id.split("_")[0] if run_id else "")
        summary.setdefault("mode", run_id.split("_")[1] if len(run_id.split("_")) > 1 else "")
        summaries.append(summary)

    html = report.view(summaries)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")

    print("Viewer written: " + str(out_path))
    print("Runs included: " + str(len(summaries)))


def _cmd_build_scenarios(args) -> None:
    """Implement the 'build-scenarios' subcommand: dispatch to scenarios._cli_build."""
    from tutorsim.scenarios import _cli_build
    _cli_build(args)


def _cmd_dataset(args) -> None:
    """Implement the 'dataset' command group."""
    if args.dataset_command == "build":
        _cmd_build_scenarios(args)
    elif args.dataset_command == "validate":
        report = validate_dataset(args.set, root=args.root)
        print("Dataset valid: " + report["name"])
        print("  records: " + str(report["record_count"]))
        print("  sha256 : " + report["content_hash"])
    else:
        raise SystemExit("Choose a dataset subcommand: build or validate")


def main(argv=None) -> None:
    """Main entry point: parse args and dispatch subcommand."""
    # `tutorsim taxonomy ...` forwards everything after "taxonomy" to its own
    # CLI dispatcher so we don't have to mirror its argument tree in two
    # places. Handled before the main parser to keep argument handling clean.
    full_argv = list(sys.argv[1:] if argv is None else argv)
    if full_argv and full_argv[0] == "taxonomy":
        from tutorsim.taxonomy import cli_dispatch
        sys.exit(cli_dispatch(full_argv[1:]))

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        if args.config:
            os.environ["TUTORSIM_CONFIG"] = args.config
        cfg = build_run_config(
            tutors=args.tutors,
            modes=args.modes,
            dataset=args.dataset,
            sample=args.sample,
            trials=args.trials,
            seed=args.seed,
            max_turns=args.max_turns,
            config_path=args.config,
        )
        date = datetime.date.today().strftime("%Y%m%d")
        cells = expand_cells(cfg.tutors, cfg.modes)
        try:
            run_ids = run_sweep(
                cells=cells,
                run_cfg=cfg,
                date=date,
                results_root="results",
                trait_cache_dir=args.trait_cache_dir,
            )
        except DatasetNotFoundError as e:
            print("Error: " + str(e), file=sys.stderr)
            sys.exit(2)
        for run_id in run_ids:
            print("Completed run: " + run_id)

    elif args.command == "report":
        _cmd_report(args)

    elif args.command == "view":
        _cmd_view(args)

    elif args.command == "build-scenarios":
        _cmd_build_scenarios(args)

    elif args.command == "dataset":
        _cmd_dataset(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
