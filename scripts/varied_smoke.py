"""One-off varied benchmark smoke: 10 scenarios (5 scaffolding + 5 rigor),
each from a distinct conversation, fresh exchange + annotation + scoring.

Throwaway helper. Sidesteps `python -m benchmark` so we can hand-pick a
balanced sample without adding CLI flags. Saves under a new version dir.
"""
import argparse
import datetime
import logging
import random
import sys

from annotator.core.config import get_phase_config
from annotator.core.storage import save_benchmark_result
from annotator.core.utils import load_transcripts
from annotator.core.client import ModelClient
from benchmark.core.scenarios import extract_human_scenarios
from benchmark.core.exchange import run_exchange, run_exchanges_batch
from benchmark.run import run_phase2_and_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def pick_balanced(scenarios, per_label: int, seed: int = 42):
    """Pick `per_label` scaffolding + `per_label` rigor scenarios from distinct convs.

    Uses a fixed seed; within each agg label, picks one scenario per conv
    deterministically (smallest scenario_id first), then samples convs.
    """
    by_conv: dict = {}
    for s in scenarios:
        agg = (s.detection or {}).get("situation_label_agg")
        if agg not in ("scaffolding", "rigor"):
            continue
        # First scenario per (conv_id, agg) by sorted scenario_id.
        key = (s.conv_id, agg)
        if key not in by_conv or s.scenario_id < by_conv[key].scenario_id:
            by_conv[key] = s

    scaff_pool = [s for (cid, agg), s in by_conv.items() if agg == "scaffolding"]
    rigor_pool = [s for (cid, agg), s in by_conv.items() if agg == "rigor"]

    # Sort by conv_id for determinism, then shuffle with seed.
    scaff_pool.sort(key=lambda s: s.conv_id)
    rigor_pool.sort(key=lambda s: s.conv_id)
    rng = random.Random(seed)
    rng.shuffle(scaff_pool)
    rng.shuffle(rigor_pool)

    # Ensure distinct conv_ids across both groups.
    chosen, used_convs = [], set()
    for s in scaff_pool:
        if s.conv_id in used_convs: continue
        chosen.append(s); used_convs.add(s.conv_id)
        if sum(1 for x in chosen if x.detection["situation_label_agg"] == "scaffolding") == per_label:
            break
    for s in rigor_pool:
        if s.conv_id in used_convs: continue
        chosen.append(s); used_convs.add(s.conv_id)
        if sum(1 for x in chosen if x.detection["situation_label_agg"] == "rigor") == per_label:
            break

    return chosen


def _default_version(profile: str, tutor_mode: str | None, student_mode: str) -> str:
    """Build a version name that surfaces the tutor MODEL, so replays under
    a different LM are visually distinct on disk."""
    tutor_model = get_phase_config("tutor", profile)["model"].replace("/", "_")
    tm = tutor_mode or "default"
    return f"{tutor_model}_{tm}_tutor_{student_mode}_student_{datetime.date.today().strftime('%Y%m%d')}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None,
                   help="Run directory name. Default: {tutor_model}_{tutor_mode}_tutor_{student_mode}_student_{date}")
    p.add_argument("--per-label", type=int, default=5,
                   help="scenarios per agg label (scaffolding, rigor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", default="anthropic")
    p.add_argument("--style", default="balanced",
                   help="single annotator style to run (one of generous|balanced|demanding); profiles prompt is used")
    p.add_argument("--mode", default="batch", choices=["sync", "batch"])
    p.add_argument("--poll-interval", type=int, default=60,
                   help="seconds between batch-job poll checks (batch mode only)")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--prompt-version", default="v6")
    p.add_argument("--student-mode", default="imitate_example")
    p.add_argument("--tutor-mode", default=None,
                   help="None=default tutor; 'oracle'=tutor sees post-cut real transcript")
    args = p.parse_args()

    if args.version is None:
        args.version = _default_version(args.profile, args.tutor_mode, args.student_mode)
        logger.info("Auto-generated version: %s", args.version)

    # --- Phase 0: pick scenarios ---
    transcripts = load_transcripts()
    all_scenarios = extract_human_scenarios(transcripts)
    logger.info("Total human scenarios available: %d", len(all_scenarios))

    chosen = pick_balanced(all_scenarios, per_label=args.per_label, seed=args.seed)
    logger.info("Picked %d scenarios (%d scaffolding + %d rigor) across %d convs",
                len(chosen),
                sum(1 for s in chosen if s.detection["situation_label_agg"] == "scaffolding"),
                sum(1 for s in chosen if s.detection["situation_label_agg"] == "rigor"),
                len({s.conv_id for s in chosen}))
    if len(chosen) < 2 * args.per_label:
        sys.exit(f"Could not find {args.per_label} distinct-conv scenarios per label.")

    save_benchmark_result(args.version, "scenarios.json", data=[s.to_dict() for s in chosen])
    save_benchmark_result(args.version, "config.json", data={
        "smoke_script": "varied_smoke.py",
        "per_label": args.per_label,
        "seed": args.seed,
        "profile": args.profile,
        "prompt_version": args.prompt_version,
        "student_mode": args.student_mode,
        "tutor_mode": args.tutor_mode,
        "mode": args.mode,
        "max_turns": args.max_turns,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })

    # --- Phase 1: exchange (sync or batch) ---
    from benchmark.core.students import is_trait_mode

    tutor_cfg = get_phase_config("tutor", args.profile)
    student_cfg = get_phase_config("tutor", args.profile)  # student uses same profile
    tutor_client = ModelClient(tutor_cfg["model"])
    student_client = ModelClient(student_cfg["model"])

    trait_client = None
    trait_model = None
    if is_trait_mode(args.student_mode):
        trait_client = student_client
        trait_model = student_cfg["model"]

    def _save_exchange(scenario_id: str, ex):
        save_benchmark_result(args.version, "exchanges", args.profile,
                              f"{scenario_id}.json", data=ex.to_dict())

    if args.mode == "batch":
        logger.info("Running %d scenarios in batch mode (poll=%ds)",
                    len(chosen), args.poll_interval)
        exchanges = run_exchanges_batch(
            scenarios=chosen,
            tutor_client=tutor_client,
            student_client=student_client,
            max_turns=args.max_turns,
            tutor_max_tokens=tutor_cfg["max_tokens"],
            student_max_tokens=student_cfg["max_tokens"],
            poll_interval=args.poll_interval,
            save_callback=_save_exchange,
            prompt_version=args.prompt_version,
            student_mode=args.student_mode,
            trait_client=trait_client,
            trait_model=trait_model,
            tutor_mode=args.tutor_mode,
            transcripts=transcripts if args.tutor_mode else None,
        )
        for sid, ex in exchanges.items():
            _save_exchange(sid, ex)
            logger.info("  %s: turns=%d ended_via=%s",
                        sid[-30:], len(ex.generated_turns), ex.ended_via)
    else:
        exchanges = {}
        for i, s in enumerate(chosen, 1):
            logger.info("[%d/%d] %s (cut %d, agg %s)",
                        i, len(chosen), s.scenario_id[-30:], s.cut_turn,
                        s.detection["situation_label_agg"])
            ex = run_exchange(
                scenario=s,
                tutor_client=tutor_client,
                student_client=student_client,
                max_turns=args.max_turns,
                tutor_max_tokens=tutor_cfg["max_tokens"],
                student_max_tokens=student_cfg["max_tokens"],
                prompt_version=args.prompt_version,
                student_mode=args.student_mode,
                trait_client=trait_client,
                trait_model=trait_model,
                tutor_mode=args.tutor_mode,
                transcripts=transcripts if args.tutor_mode else None,
            )
            exchanges[s.scenario_id] = ex
            _save_exchange(s.scenario_id, ex)
            logger.info("  turns=%d ended_via=%s", len(ex.generated_turns), ex.ended_via)

    # --- Phase 2 + 3: annotate -> decompose -> structure -> score ---
    summary = run_phase2_and_score(
        version=args.version,
        profile=args.profile,
        annotator_profile=args.profile,
        annotator_mode=args.mode,
        prompt_version="v13",
        context_window=20,
        scenarios=chosen,
        exchanges=exchanges,
        with_screenshots=False,
    )
    logger.info("Done. Version: %s | scaffolding F1=%.3f rigor F1=%.3f outcome_pos_rate=%.3f n=%d",
                args.version,
                summary["scaffolding"]["f1"],
                summary["rigor"]["f1"],
                summary["outcome_pos_rate"],
                summary["n_scenarios"])


if __name__ == "__main__":
    main()
