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
from benchmark.run import (
    _collect_exchange_tokens, _collect_annotation_tokens, _sum_usage,
    _collect_exchange_latencies,
)
from benchmark.core.exchange import Exchange

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_exchanges_from_disk(run_dir: Path, profile: str) -> dict:
    """Rebuild Exchange objects from saved exchange JSONs so latency + token
    aggregates can be recomputed without re-running the model."""
    ex_dir = run_dir / "exchanges" / profile
    exchanges = {}
    if not ex_dir.exists():
        return exchanges
    for f in ex_dir.iterdir():
        if not f.name.endswith(".json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        exchanges[d.get("scenario_id", f.stem)] = Exchange(
            scenario_id=d.get("scenario_id", f.stem),
            tutor_model=d.get("tutor_model", ""),
            generated_turns=d.get("generated_turns", []),
            tutor_usage=d.get("tutor_usage", {}),
            student_usage=d.get("student_usage", {}),
            tutor_latencies=d.get("tutor_latencies", []),
            student_latencies=d.get("student_latencies", []),
            completed=d.get("completed", False),
            ended_via=d.get("ended_via", ""),
        )
    return exchanges


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

    # Rebuild latency + tokens from the saved exchanges + annotations on disk
    # (no API calls). Falls back to whatever's already in the scores file for
    # timings, which we can't reconstruct (wall-clock isn't stored per phase
    # anywhere but the scores file).
    exchanges = _load_exchanges_from_disk(run_dir, args.profile)
    if exchanges:
        tutor_tokens, student_tokens = _collect_exchange_tokens(exchanges)
        # annotations list is [{annotations: [...]}]; rebuild the nested shape
        # _collect_annotation_tokens expects ({sid: {usage}}).
        ann_token_input = {
            s["scenario_id"]: {s["scenario_id"]: ann}
            for s, ann in zip(scenarios, annotations)
        }
        annotation_tokens = _collect_annotation_tokens(ann_token_input)
        summary["tokens"] = {
            "tutor": tutor_tokens,
            "student": student_tokens,
            "annotation": annotation_tokens,
            "total": _sum_usage(tutor_tokens, student_tokens, annotation_tokens),
        }
        summary["latency"] = _collect_exchange_latencies(exchanges)

    if scores_path.exists():
        try:
            prior = json.loads(scores_path.read_text(encoding="utf-8"))
            if "timings" in prior:
                summary["timings"] = prior["timings"]
            # If we couldn't rebuild latency/tokens (no exchanges), keep prior.
            for key in ("latency", "tokens"):
                if key not in summary and key in prior:
                    summary[key] = prior[key]
        except Exception as e:
            logger.warning("could not read prior scores to preserve timings: %s", e)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", scores_path)

    cal_s = summary.get("scaffold_calibrated", {})
    cal_r = summary.get("rigor_calibrated", {})
    print(f"{args.version}")
    print(f"  scaffold_calibrated = {cal_s.get('score')}  "
          f"{cal_s.get('n_clean_yes')} clean / {cal_s.get('n_total')} "
          f"({cal_s.get('n_overscaffold')} over-scaffolded, reported not scored)")
    print(f"  rigor_calibrated    = {cal_r.get('score')}  "
          f"{cal_r.get('n_clean_yes')} clean / {cal_r.get('n_total')}")
    print(f"  overscaffold_rate   = {summary.get('overscaffold', {}).get('rate')}")
    print(f"  outcome_pos_rate    = {summary.get('outcome_pos_rate')}")


if __name__ == "__main__":
    main()
