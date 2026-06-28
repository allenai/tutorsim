"""Score the Human baseline row: the real human tutor's first 5 speaking turns
(TSTST) after the cut point, on the balanced_520 set, with the same scorer the
AI tutors use. Resumable; aggregates the Human leaderboard row.

  python scripts/score_human_baseline.py --sample 2 --workers 1     # smoke
  python scripts/score_human_baseline.py --workers 8                # full 520
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

from tutor_bench.benchmark import results
from tutor_bench.benchmark.human import build_human_transcript
from tutor_bench.benchmark.report import aggregate
from tutor_bench.benchmark.scenarios import load_scenarios
from tutor_bench.benchmark.scoring import score

RESULTS_ROOT = "results"
RUN_ID = "human_5turn_tstst"
# Scorer $ per 1M tokens (input, output).
PRICE_IN, PRICE_OUT = 15.0, 75.0


def _resume_id(scenario_id: str) -> str:
    """Filesystem-safe scenario id for the results store (``:`` -> ``__``)."""
    return scenario_id.replace(":", "__")


def _score_one(scenario):
    sid = _resume_id(scenario.id)
    cached_rec = results.read_score(RUN_ID, sid, results_root=RESULTS_ROOT)
    if cached_rec is not None:
        return cached_rec, True  # cached
    transcript = build_human_transcript(scenario, max_turns=5)
    if not transcript.generated_turns:
        return None, False  # no human continuation (empty/all-student reference)
    try:
        judgment = score(scenario, transcript)
    except Exception as exc:  # transient API/batch timeout: skip, don't kill run
        print(f"  ERROR scoring {scenario.id}: {type(exc).__name__}: {exc}")
        return None, False  # no cache written -> retried on next resume
    rec = judgment.to_dict() if hasattr(judgment, "to_dict") else dict(judgment)
    results.write_score(RUN_ID, sid, rec, results_root=RESULTS_ROOT)
    return rec, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = full set")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    scenarios = load_scenarios("balanced_520")
    if args.sample:
        scenarios = scenarios[: args.sample]
    print(f"scoring Human baseline on {len(scenarios)} scenarios (workers={args.workers})")

    results_list = [None] * len(scenarios)
    done = 0
    cached = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_score_one, s): i for i, s in enumerate(scenarios)}
        for fut, i in futs.items():
            rec, was_cached = fut.result()
            results_list[i] = rec
            done += 1
            cached += int(was_cached)
            if done % 25 == 0:
                print(f"  ...{done}/{len(scenarios)} scored ({cached} cached)")

    # Rehydrate minimal Judgment-like objects for aggregate (needs attributes).
    class _A:
        def __init__(self, d):
            self.__dict__.update(d)

    # Drop scenarios with no human continuation (empty/all-student references) so
    # the Human row is over the scenarios that actually have a baseline.
    kept = [(s, r) for s, r in zip(scenarios, results_list, strict=False) if r is not None]
    skipped = len(scenarios) - len(kept)
    kept_scenarios = [s for s, _ in kept]
    judgments = [_A(r) for _, r in kept]
    metrics = aggregate(kept_scenarios, judgments)
    print(f"scored {len(kept)} scenarios; skipped {skipped} with no human continuation")

    tin = sum((r.get("usage") or {}).get("input_tokens", 0) for _, r in kept)
    tout = sum((r.get("usage") or {}).get("output_tokens", 0) for _, r in kept)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6

    scaf = (metrics.get("scaffold_calibrated") or {}).get("score")
    rig = (metrics.get("rigor_calibrated") or {}).get("score")
    over = (metrics.get("overscaffold") or {}).get("rate")
    avoid = (1 - over) if isinstance(over, (int, float)) else None

    summary = {
        "row": "Human",
        "n": metrics.get("n_scenarios"),
        "scaffolding": scaf,
        "rigor": rig,
        "avoids_overscaffolding": avoid,
        "outcome_pos": metrics.get("outcome_pos_rate"),
        "scorer_input_tokens": tin,
        "scorer_output_tokens": tout,
        "scorer_cost_usd": round(cost, 2),
    }
    results.write_summary(RUN_ID, summary, results_root=RESULTS_ROOT)

    print("\n" + "=" * 50)
    print("  HUMAN BASELINE (5-turn TSTST)")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
