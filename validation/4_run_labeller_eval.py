"""Run classify_v2 over the v2 labeller-validation test set and compare to
human ratings.

Inputs:
  data/labeller_validation/eval/labeller_test_v2.jsonl  (147 rows: annotation_key + human_rating)
  data/labeller_validation/step_up_annotations.jsonl     (source SAR text)
  prompts/annotator/labeller/classify_v2.txt             (labeller prompt)

Outputs:
  data/labeller_validation/eval/labeller_predictions_v2_{model_slug}.jsonl
  data/labeller_validation/eval/labeller_metrics_v2_{model_slug}.json

Model selection is mandatory via --model-profile (anthropic|openai|gemini).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from annotator.core.client import (
    ModelClient,
    build_batch_entry,
    run_batch,
)
from annotator.core.config import get_phase_config
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
PROMPTS_DIR = Path("prompts/annotator/labeller")
SAR_TYPES = ("scaffolding", "rapport")

VALID_LLM_LABELS = {"effective", "partial", "ineffective"}
HUMAN_TO_LLM = {
    "effective": "effective",
    "partially_effective": "partial",
    "ineffective": "ineffective",
}
LLM_TO_HUMAN = {v: k for k, v in HUMAN_TO_LLM.items()}


def annotation_key(transcript_id, source_annotator_id, annotation_type,
                   turn_number_start, turn_number_end) -> str:
    return (
        f"{transcript_id}|{source_annotator_id}|{annotation_type}|"
        f"{turn_number_start}|{turn_number_end}"
    )


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_sar_lookup(path: Path) -> dict[str, dict]:
    """Walk step_up_annotations.jsonl, flatten turn_annotations, key by
    annotation_key. Returns {key: {situation, action, result}}."""
    lookup: dict[str, dict] = {}
    for row in load_jsonl(path):
        if row.get("annotation_type") not in SAR_TYPES:
            continue
        for ta in row.get("turn_annotations", []):
            key = annotation_key(
                row["transcript_id"],
                row["source_annotator_id"],
                row["annotation_type"],
                ta["turn_number_start"],
                ta["turn_number_end"],
            )
            lookup[key] = {
                "annotation_type": row["annotation_type"],
                "situation": ta.get("situation", ""),
                "action": ta.get("action", ""),
                "result": ta.get("result", ""),
            }
    return lookup


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, int]:
    n = len(pairs)
    if n == 0:
        return None, 0
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0, n
    return (po - pe) / (1 - pe), n


def binary_collapse(label: str) -> str:
    """Collapse partial -> effective for binary kappa."""
    return "effective" if label in ("effective", "partial", "partially_effective") else "ineffective"


def build_entries(test_rows: list[dict], sar_lookup: dict[str, dict],
                  template: str) -> tuple[list, list, list]:
    entries, locations, missing = [], [], []
    for row in test_rows:
        key = row["annotation_key"]
        sar = sar_lookup.get(key)
        if not sar:
            missing.append(key)
            continue
        prompt = (template
                  .replace("{annotation_type}", sar["annotation_type"])
                  .replace("{situation}", sar["situation"])
                  .replace("{action}", sar["action"])
                  .replace("{result_text}", sar["result"]))
        entries.append(build_batch_entry(key, prompt, json_mode=False))
        locations.append(key)
    return entries, locations, missing


def parse_label(raw_text: str) -> str:
    """Robust parse: strip, lowercase, drop trailing punctuation."""
    if not raw_text:
        return "unclear"
    txt = raw_text.strip().lower()
    txt = txt.rstrip(".!?,;: \n\t")
    for tok in ("effective", "partial", "ineffective"):
        if txt == tok:
            return tok
    # Fallback: first matching word
    for tok in ("ineffective", "partial", "effective"):
        if tok in txt:
            return tok
    return "unclear"


def compute_metrics(rows: list[dict]) -> dict:
    """rows: list of {annotation_key, annotation_type, human_rating, predicted_label}"""
    pairs_3way = []
    pairs_3way_by_type: dict[str, list] = defaultdict(list)
    pairs_binary = []
    confusion: dict[tuple[str, str], int] = Counter()
    errors = 0
    for r in rows:
        pred = r["predicted_label"]
        if pred == "unclear":
            errors += 1
            continue
        human_3way = HUMAN_TO_LLM[r["human_rating"]]
        pairs_3way.append((human_3way, pred))
        pairs_3way_by_type[r["annotation_type"]].append((human_3way, pred))
        pairs_binary.append((binary_collapse(human_3way), binary_collapse(pred)))
        confusion[(human_3way, pred)] += 1

    k3, n3 = cohens_kappa(pairs_3way)
    kb, nb = cohens_kappa(pairs_binary)
    accuracy_3way = sum(1 for a, b in pairs_3way if a == b) / n3 if n3 else None
    accuracy_binary = sum(1 for a, b in pairs_binary if a == b) / nb if nb else None

    by_type = {}
    for t, pairs in pairs_3way_by_type.items():
        k, n = cohens_kappa(pairs)
        acc = sum(1 for a, b in pairs if a == b) / n if n else None
        by_type[t] = {"n": n, "kappa_3way": k, "accuracy_3way": acc}

    return {
        "n_evaluated": n3,
        "n_errors": errors,
        "accuracy_3way": accuracy_3way,
        "kappa_3way": k3,
        "accuracy_binary": accuracy_binary,
        "kappa_binary": kb,
        "by_type": by_type,
        "confusion_matrix": {f"{h}->{p}": c for (h, p), c in sorted(confusion.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", choices=["anthropic", "openai", "gemini"],
                        required=True)
    parser.add_argument("--prompt-version", default="v2",
                        help="Labeller prompt version (loads classify_{version}.txt)")
    parser.add_argument("--split", default="test_v2",
                        help="Which split to run (test_v2|train_v2|test_v1|train_v1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N rows (smoke test only)")
    args = parser.parse_args()

    setup_logging(version=f"labeller_eval_{args.split}_{args.model_profile}_{args.prompt_version}")

    prompt_path = PROMPTS_DIR / f"classify_{args.prompt_version}.txt"
    template = prompt_path.read_text(encoding="utf-8")
    logger.info("Loaded prompt from %s", prompt_path)
    test_rows = list(load_jsonl(EVAL / f"labeller_{args.split}.jsonl"))
    if args.limit:
        test_rows = test_rows[: args.limit]
    logger.info("Loaded %d test rows from %s", len(test_rows), args.split)

    sar_lookup = build_sar_lookup(ROOT / "step_up_annotations.jsonl")
    logger.info("Built SAR lookup with %d keys", len(sar_lookup))

    entries, locations, missing = build_entries(test_rows, sar_lookup, template)
    if missing:
        logger.warning("Missing SAR text for %d keys (will skip): %s",
                       len(missing), missing[:3])
    logger.info("Built %d entries to send to labeller", len(entries))

    phase_cfg = get_phase_config("label", args.model_profile)
    model = phase_cfg["model"]
    logger.info("Model: %s | profile: %s | mode: batch", model, args.model_profile)

    client = ModelClient(model)
    raw = run_batch(
        client, entries,
        json_mode=False,
        display_name=f"labeller_eval_{args.split}",
        poll_interval=phase_cfg.get("poll_interval", 60),
        thinking=phase_cfg.get("thinking", False),
        thinking_budget=phase_cfg.get("thinking_budget", 0),
        reasoning_effort=phase_cfg.get("reasoning_effort", ""),
    )

    # Build predictions
    test_by_key = {r["annotation_key"]: r for r in test_rows}
    pred_rows = []
    for key in locations:
        entry = raw.get(key, {})
        raw_text = entry.get("text", "")
        pred = parse_label(raw_text)
        src = test_by_key[key]
        pred_rows.append({
            "annotation_key": key,
            "annotation_type": src["annotation_type"],
            "human_rating": src["human_rating"],
            "predicted_label": pred,
            "raw_text": raw_text,
            "model": model,
        })

    EVAL.mkdir(parents=True, exist_ok=True)
    slug = f"{args.model_profile}_{args.prompt_version}"
    preds_path = EVAL / f"labeller_predictions_{args.split}_{slug}.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d predictions to %s", len(pred_rows), preds_path)

    metrics = compute_metrics(pred_rows)
    metrics["model"] = model
    metrics["prompt_version"] = args.prompt_version
    metrics["split"] = args.split
    metrics["n_test_rows"] = len(test_rows)
    metrics["n_missing_sar"] = len(missing)
    metrics_path = EVAL / f"labeller_metrics_{args.split}_{slug}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"LABELLER EVAL -- {args.split} -- {model}")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
