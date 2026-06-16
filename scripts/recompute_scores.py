"""Recompute scores from existing annotations -- no API calls.

Loads scenarios.json + every annotation JSON under annotations/{profile}/,
calls score_scenarios(), and overwrites scores/{profile}.json. Use this
when the scoring formula changes but the annotations themselves are still
valid (no need to re-annotate or hit the model).

Usage:
    python scripts/recompute_scores.py --version <run-dir-name>
    python scripts/recompute_scores.py --version <name> --profile anthropic
"""
import argparse
import json
import logging
from pathlib import Path

from benchmark.core.score import score_scenarios

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True,
                   help="Run dir under results/benchmark/")
    p.add_argument("--profile", default="anthropic")
    p.add_argument("--benchmark-root", default="results/benchmark")
    args = p.parse_args()

    run_dir = Path(args.benchmark_root) / args.version
    if not run_dir.exists():
        raise SystemExit(f"No such run dir: {run_dir}")

    scenarios_path = run_dir / "scenarios.json"
    scenarios_raw = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios = scenarios_raw if isinstance(scenarios_raw, list) else scenarios_raw.get("scenarios", [])

    ann_dir = run_dir / "annotations" / args.profile
    if not ann_dir.exists():
        raise SystemExit(f"No annotations dir: {ann_dir}")

    annotations = []
    n_missing = 0
    for s in scenarios:
        f = ann_dir / f"{s['scenario_id']}.json"
        if not f.exists():
            annotations.append({})
            n_missing += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        # File format: {results: {scenario_id: {annotations: [...]}}} OR
        # {scenario_id: {annotations: [...]}} depending on how it was saved.
        # Find the inner annotation list.
        inner = d.get("results", d)
        if isinstance(inner, dict) and s["scenario_id"] in inner:
            annotations.append(inner[s["scenario_id"]])
        else:
            annotations.append(inner if isinstance(inner, dict) else {})

    if n_missing:
        logger.warning("%d/%d scenarios missing annotation files", n_missing, len(scenarios))

    summary = score_scenarios(scenarios, annotations)
    summary["profile"] = args.profile

    scores_path = run_dir / "scores" / f"{args.profile}.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", scores_path)

    cal_s = summary.get("scaffold_calibrated", {})
    cal_r = summary.get("rigor_calibrated", {})
    print(f"{args.version}")
    print(f"  scaffold_calibrated = {cal_s.get('score')}  "
          f"({cal_s.get('n_clean_yes')} - {cal_s.get('n_overscaffold')}) / {cal_s.get('n_total')}")
    print(f"  rigor_calibrated    = {cal_r.get('score')}  "
          f"{cal_r.get('n_clean_yes')} / {cal_r.get('n_total')}")
    print(f"  overscaffold_rate   = {summary.get('overscaffold', {}).get('rate')}")
    print(f"  outcome_pos_rate    = {summary.get('outcome_pos_rate')}")


if __name__ == "__main__":
    main()
