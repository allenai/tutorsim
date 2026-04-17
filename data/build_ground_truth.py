#!/usr/bin/env python3
"""Build ground truth files from consolidated annotations.

For each conversation in data/raw/consolidated/:
  - Reuse strategy_label for moments unchanged from the existing ground_truth file
    (matched by annotator_id + turn_start + turn_end + annotation_type + result text)
  - Classify new/changed moments via Anthropic batch API
  - Write the merged result to data/ground_truth_<labeller>/<conv_id>.json

Also writes transcript files for any conversations in consolidated that are missing one.

Usage:
    python data/build_ground_truth.py
    python data/build_ground_truth.py --dry-run
    python data/build_ground_truth.py --labeller v2
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from annotator.core.label import JUNK_TEXTS

CONSOLIDATED_DIR = DATA_DIR / "raw" / "consolidated"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "annotator" / "labeller"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

VALID_LABELS = {"effective", "partial", "ineffective"}


def moment_key(m):
    """Stable key identifying an annotation moment across runs."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("result", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def load_existing_labels():
    """Return {conv_id: {moment_key: strategy_label}} from current ground truth."""
    existing = {}
    if not GROUND_TRUTH_DIR.exists():
        return existing
    for f in GROUND_TRUTH_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        existing[f.stem] = {
            moment_key(m): m.get("strategy_label")
            for m in d.get("key_moments", [])
            if m.get("strategy_label")
        }
    return existing


def classify_batch(items, labeller="v2"):
    """Run batch classification. `items` is list of dicts with keys:
    key, annotation_type, situation, action, result_text.
    Returns {key: label}."""
    if not items:
        return {}
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("label")
    client = ModelClient(cfg["model"])

    template = _load_prompt(f"classify_{labeller}")

    entries = []
    labels = {}
    for it in items:
        text = it["result_text"]
        stripped = (text or "").strip().lower()
        if stripped in JUNK_TEXTS:
            labels[it["key"]] = "unclear"
            continue
        prompt = (template
                  .replace("{annotation_type}", it.get("annotation_type", "unknown"))
                  .replace("{situation}", it.get("situation", ""))
                  .replace("{action}", it.get("action", ""))
                  .replace("{result_text}", text))
        entries.append(build_batch_entry(
            key=it["key"],
            prompt_text=prompt,
            json_mode=False,
            max_tokens=32,
        ))

    if not entries:
        return labels

    print(f"  Submitting {len(entries)} classifications to Anthropic batch API "
          f"(model={cfg['model']})...")
    results = run_batch(
        client, entries,
        json_mode=False,
        display_name="effectiveness_classification_refresh",
        poll_interval=cfg.get("poll_interval", 60),
    )

    for key, result in results.items():
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            labels[key] = "unclear"
            continue
        label = result["text"].strip().lower().rstrip(".")
        labels[key] = label if label in VALID_LABELS else "unclear"

    return labels


def build_moment(ann, label):
    return {
        "turn_start": ann.get("turn_start"),
        "turn_end": ann.get("turn_end"),
        "annotation_type": ann.get("annotation_type", ""),
        "annotator_id": ann.get("annotator_id", ""),
        "situation": ann.get("situation", ""),
        "action": ann.get("action", ""),
        "result": ann.get("result", ""),
        "strategy_label": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without submitting batch or writing files")
    parser.add_argument("--labeller", default="v2",
                        help="Labeller version (determines output dir and prompt, default: v2)")
    args = parser.parse_args()

    global GROUND_TRUTH_DIR
    if args.labeller != "v1":
        GROUND_TRUTH_DIR = DATA_DIR / f"ground_truth_{args.labeller}"

    if not CONSOLIDATED_DIR.exists():
        print(f"ERROR: consolidated dir not found: {CONSOLIDATED_DIR}")
        return

    existing_labels = load_existing_labels()
    print(f"Loaded existing labels for {len(existing_labels)} conversations")

    # First pass: build per-conv plan (reuse vs classify)
    conv_plans = []
    to_classify = []  # list of {key, result_text}
    for consolidated_file in sorted(CONSOLIDATED_DIR.glob("*.json")):
        conv_id = consolidated_file.stem
        with open(consolidated_file, "r", encoding="utf-8") as fp:
            conv_data = json.load(fp)
        annotations = conv_data.get("annotations", [])
        known = existing_labels.get(conv_id, {})

        plan = []
        for idx, ann in enumerate(annotations):
            k = moment_key(ann)
            if k in known:
                plan.append(("reuse", ann, known[k]))
            else:
                ckey = f"{conv_id}__{idx}"
                to_classify.append({
                    "key": ckey,
                    "annotation_type": ann.get("annotation_type", "unknown"),
                    "situation": ann.get("situation", ""),
                    "action": ann.get("action", ""),
                    "result_text": ann.get("result", ""),
                })
                plan.append(("classify", ann, ckey))
        conv_plans.append((conv_id, conv_data, plan))

    total_moments = sum(len(p) for _, _, p in conv_plans)
    reused = sum(1 for _, _, p in conv_plans for kind, *_ in p if kind == "reuse")
    to_class = total_moments - reused
    new_convs = sum(1 for cid, _, _ in conv_plans if cid not in existing_labels)

    print(f"Plan: {len(conv_plans)} conversations, {total_moments} moments")
    print(f"  Reuse existing labels: {reused}")
    print(f"  Classify new moments: {to_class}")
    print(f"  Brand new conversations: {new_convs}")

    if args.dry_run:
        print("\nDry run — exiting without classifying or writing.")
        return

    # Second pass: batch classify
    new_labels = classify_batch(to_classify, labeller=args.labeller)

    # Third pass: write ground truth files
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    gt_written = 0
    tx_written = 0
    for conv_id, conv_data, plan in conv_plans:
        moments = []
        for kind, ann, val in plan:
            if kind == "reuse":
                moments.append(build_moment(ann, val))
            else:
                label = new_labels.get(val, "unclear")
                moments.append(build_moment(ann, label))
        out = {
            "conversation_id": conv_id,
            "num_turns": conv_data.get("num_turns", 0),
            "key_moments": moments,
        }
        gt_path = GROUND_TRUTH_DIR / f"{conv_id}.json"
        with open(gt_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        gt_written += 1

        tx_path = TRANSCRIPTS_DIR / f"{conv_id}.json"
        if not tx_path.exists():
            with open(tx_path, "w", encoding="utf-8") as fp:
                json.dump(conv_data, fp, indent=2, ensure_ascii=False)
            tx_written += 1

    print(f"\nDone!")
    print(f"  Ground truth files written: {gt_written}")
    print(f"  Transcript files written (new only): {tx_written}")
    print(f"  Total ground truth: {len(list(GROUND_TRUTH_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
