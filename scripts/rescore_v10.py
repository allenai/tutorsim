"""Rescore a benchmark run using cached scenarios + exchanges.

Reads scenarios.json + exchanges/{profile}/*.json from --version, then runs
Phase 2 (annotate -> decompose -> structure) + Phase 3 (score) only.
No tutor/student turns are regenerated, so the comparison is apples to
apples vs the previous scoring of the same exchanges.

Use after a bridge / annotator change that requires re-annotation but not
new exchanges (e.g. the context_window=0 fix in benchmark/run.py).
"""
import argparse
import logging
import sys

from annotator.core.storage import load_benchmark_result
from annotator.core.utils import load_transcripts
from benchmark.core.scenarios import Scenario, _format_prefix
from benchmark.core.exchange import Exchange
from benchmark.run import run_phase2_and_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _rebuild_scenario(s_dict: dict, transcripts: dict):
    conv = transcripts.get(s_dict["conv_id"])
    if conv is None:
        return None
    return Scenario(
        scenario_id=s_dict["scenario_id"],
        conv_id=s_dict["conv_id"],
        cut_turn=s_dict["cut_turn"],
        transcript_prefix=_format_prefix(conv, s_dict["cut_turn"]),
        student_context=s_dict.get("student_context", ""),
        last_student_message=s_dict.get("last_student_message", ""),
        mode=s_dict.get("mode", "human"),
        detection=s_dict.get("detection"),
    )


def _load_exchange(version: str, profile: str, sid: str):
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
    p.add_argument("--version", required=True,
                   help="Run dir name under results/benchmark/")
    p.add_argument("--profile", default="anthropic")
    p.add_argument("--annotator-profile", default="anthropic")
    p.add_argument("--annotator-mode", default="batch", choices=["sync", "batch"])
    p.add_argument("--prompt-version", default="v13",
                   help="Annotator prompt version (Pass 2). Default v13.")
    p.add_argument("--context-window", type=int, default=0,
                   help="Ignored -- run.py forces 0 for the benchmark bridge.")
    args = p.parse_args()

    scenarios_raw = load_benchmark_result(args.version, "scenarios.json")
    if not scenarios_raw:
        sys.exit(f"No scenarios.json under {args.version}")

    transcripts = load_transcripts()
    scenarios = [s for s in (_rebuild_scenario(d, transcripts) for d in scenarios_raw) if s]
    logger.info("Rebuilt %d scenarios from %s", len(scenarios), args.version)

    exchanges = {}
    for s in scenarios:
        ex = _load_exchange(args.version, args.profile, s.scenario_id)
        if ex is not None:
            exchanges[s.scenario_id] = ex
    logger.info("Loaded %d cached exchanges", len(exchanges))
    if not exchanges:
        sys.exit("No usable cached exchanges; nothing to rescore.")

    scenarios = [s for s in scenarios if s.scenario_id in exchanges]

    summary = run_phase2_and_score(
        version=args.version,
        profile=args.profile,
        annotator_profile=args.annotator_profile,
        annotator_mode=args.annotator_mode,
        prompt_version=args.prompt_version,
        context_window=args.context_window,
        scenarios=scenarios,
        exchanges=exchanges,
        with_screenshots=False,
    )
    logger.info("Rescored %s. summary keys: %s", args.version, list(summary.keys()))


if __name__ == "__main__":
    main()
