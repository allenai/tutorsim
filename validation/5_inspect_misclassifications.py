"""Inspect labeller misclassifications on a split.

Joins predictions JSONL with the source SAR text and prints rows where
human_rating != predicted_label. Supports filtering by direction
(e.g. only partial -> effective/ineffective polarization).

Usage:
  PYTHONPATH=. python validation/5_inspect_misclassifications.py \
      --split train_v2 \
      --model-profile anthropic \
      --filter partial_polarized \
      --limit 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
SAR_TYPES = ("scaffolding", "rapport")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotation_key(transcript_id, source_annotator_id, annotation_type, ts, te):
    return f"{transcript_id}|{source_annotator_id}|{annotation_type}|{ts}|{te}"


def build_sar_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_jsonl(path):
        if row.get("annotation_type") not in SAR_TYPES:
            continue
        for ta in row.get("turn_annotations", []):
            key = annotation_key(
                row["transcript_id"], row["source_annotator_id"],
                row["annotation_type"], ta["turn_number_start"], ta["turn_number_end"],
            )
            lookup[key] = {
                "situation": ta.get("situation", ""),
                "action": ta.get("action", ""),
                "result": ta.get("result", ""),
            }
    return lookup


HUMAN_TO_LLM = {
    "effective": "effective",
    "partially_effective": "partial",
    "ineffective": "ineffective",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--model-profile", required=True)
    p.add_argument("--filter", default="all",
                   choices=["all", "partial_polarized", "polar_swap",
                            "missed_partial", "false_partial"])
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--annotation-type", default=None,
                   choices=["scaffolding", "rapport"])
    args = p.parse_args()

    preds_path = EVAL / f"labeller_predictions_{args.split}_{args.model_profile}.jsonl"
    sar = build_sar_lookup(ROOT / "step_up_annotations.jsonl")

    rows = list(load_jsonl(preds_path))
    print(f"Loaded {len(rows)} predictions from {preds_path.name}")

    # Filter
    def matches(r):
        human = HUMAN_TO_LLM[r["human_rating"]]
        pred = r["predicted_label"]
        if args.annotation_type and r["annotation_type"] != args.annotation_type:
            return False
        if args.filter == "all":
            return human != pred
        if args.filter == "partial_polarized":
            return human == "partial" and pred in ("effective", "ineffective")
        if args.filter == "polar_swap":
            return (human == "effective" and pred == "ineffective") or \
                   (human == "ineffective" and pred == "effective")
        if args.filter == "missed_partial":
            return human == "partial" and pred != "partial"
        if args.filter == "false_partial":
            return human != "partial" and pred == "partial"
        return False

    matched = [r for r in rows if matches(r)]
    print(f"Filter '{args.filter}' matched {len(matched)} rows "
          f"(showing first {min(args.limit, len(matched))})\n")

    # Direction breakdown for partial_polarized
    if args.filter == "partial_polarized":
        dist = Counter((r["annotation_type"], r["predicted_label"]) for r in matched)
        print(f"Direction breakdown: {dict(dist)}\n")

    for i, r in enumerate(matched[: args.limit]):
        s = sar.get(r["annotation_key"], {})
        print(f"{'=' * 78}")
        print(f"[{i+1}] {r['annotation_key']}")
        print(f"    type={r['annotation_type']:11s}  "
              f"human={r['human_rating']:20s}  pred={r['predicted_label']}")
        print(f"    raw labeller output: {r['raw_text'][:120]!r}")
        print(f"    SITUATION: {s.get('situation', '')}")
        print(f"    ACTION   : {s.get('action', '')}")
        print(f"    RESULT   : {s.get('result', '')}")


if __name__ == "__main__":
    main()
