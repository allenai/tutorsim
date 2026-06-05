"""
Sample 200 scaffolding moments from ground_truth_hybrid for transcripts in the train split.
Saves to data/scaffolding_sample_200.csv.

Also evaluates ground_truth_hybrid situation_label against human annotations in
scaffolding_sample_200_done.csv. Done CSV encoding: y=yes, empty=no_mention, n=no.
"""

import json
import os
import random
import csv
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GT_DIR = os.path.join(DATA_DIR, "ground_truth_hybrid")
SPLIT_FILE = os.path.join(DATA_DIR, "split.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "scaffolding_sample_200.csv")

SAMPLE_SIZE = 200
RANDOM_SEED = 42


def create_val_set():
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = set(split["train"])

    rows = []
    for fname in os.listdir(GT_DIR):
        if not fname.endswith(".json"):
            continue
        cid = fname.replace(".json", "")
        if cid not in train_ids:
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        for moment in data.get("key_moments", []):
            if moment.get("annotation_type") == "scaffolding":
                rows.append({
                    "conversation_id": cid,
                    "turn_start": moment["turn_start"],
                    "turn_end": moment["turn_end"],
                    "annotator_ID": moment["annotator_id"],
                    "situation": moment["situation"],
                })

    print(f"Total scaffolding moments in train split: {len(rows)}")

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(rows, SAMPLE_SIZE)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "turn_start", "turn_end", "annotator_ID", "situation"])
        writer.writeheader()
        writer.writerows(sample)

    print(f"Saved {SAMPLE_SIZE} rows to {OUTPUT_FILE}")


def _parse_csv_label(value):
    """Map done-CSV annotation encoding to ground_truth_hybrid encoding."""
    v = value.strip().lower()
    if v == "y":
        return "yes"
    elif v == "n":
        return "no"
    else:
        return "no_mention"


def evaluate_situation_labels(done_csv_path=None, examples_per_error=2):
    """
    Compare situation_label in ground_truth_hybrid against human annotations in
    scaffolding_sample_200_done.csv.

    Done CSV columns: conversation_id, turn_start, turn_end, annotator_ID, situation,
                      scaffolding, rigor
    Annotation encoding: y=yes, empty=no_mention, n=no

    Prints per-label agreement statistics and examples of each error type.
    Returns a dict with the full results.
    """
    if done_csv_path is None:
        done_csv_path = os.path.join(DATA_DIR, "scaffolding_sample_200_done.csv")

    # Load done CSV
    with open(done_csv_path, newline="") as f:
        done_rows = list(csv.DictReader(f))

    # Build lookup: (conversation_id, turn_start, turn_end, annotator_ID) -> moment
    gt_index = {}
    for fname in os.listdir(GT_DIR):
        if not fname.endswith(".json"):
            continue
        cid = fname.replace(".json", "")
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        for moment in data.get("key_moments", []):
            if moment.get("annotation_type") != "scaffolding":
                continue
            key = (cid, str(moment["turn_start"]), str(moment["turn_end"]), moment["annotator_id"])
            gt_index[key] = moment

    labels = ["scaffolding", "rigor"]
    # per-label: counts and examples of (gt_value, annotated_value) pairs
    confusion = {label: defaultdict(int) for label in labels}
    error_examples = {label: defaultdict(list) for label in labels}
    unmatched = 0

    for row in done_rows:
        key = (row["conversation_id"], row["turn_start"], row["turn_end"], row["annotator_ID"])
        if key not in gt_index:
            unmatched += 1
            continue
        moment = gt_index[key]
        situation_label = moment.get("situation_label", {})
        for label in labels:
            gt_val = situation_label.get(label, "no_mention")
            if gt_val == "unclear":
                gt_val = "no_mention"
            annotated_val = _parse_csv_label(row.get(label, ""))
            confusion[label][(gt_val, annotated_val)] += 1
            if gt_val != annotated_val:
                error_examples[label][(gt_val, annotated_val)].append({
                    "conversation_id": row["conversation_id"],
                    "turns": f"{row['turn_start']}-{row['turn_end']}",
                    "situation": row["situation"],
                })

    # Compute stats
    results = {"unmatched": unmatched, "labels": {}}
    total_agree = 0
    total_pairs = 0

    for label in labels:
        counts = confusion[label]
        n = sum(counts.values())
        agree = sum(v for (gt, ann), v in counts.items() if gt == ann)
        accuracy = agree / n if n else 0.0

        # breakdown by gt value
        by_gt = defaultdict(lambda: {"agree": 0, "total": 0, "errors": defaultdict(int)})
        for (gt_val, ann_val), cnt in counts.items():
            by_gt[gt_val]["total"] += cnt
            if gt_val == ann_val:
                by_gt[gt_val]["agree"] += cnt
            else:
                by_gt[gt_val]["errors"][ann_val] += cnt

        results["labels"][label] = {
            "n": n,
            "agree": agree,
            "accuracy": accuracy,
            "by_gt_value": {k: dict(v) for k, v in by_gt.items()},
            "error_examples": {str(k): v for k, v in error_examples[label].items()},
        }
        total_agree += agree
        total_pairs += n

    results["overall"] = {
        "n": total_pairs,
        "agree": total_agree,
        "accuracy": total_agree / total_pairs if total_pairs else 0.0,
    }

    # Print report
    print(f"=== Situation Label Evaluation (n={len(done_rows)} rows) ===\n")
    if unmatched:
        print(f"WARNING: {unmatched} rows could not be matched to ground_truth_hybrid\n")

    for label in labels:
        r = results["labels"][label]
        print(f"--- {label} (n={r['n']}) ---")
        print(f"  Agreement: {r['agree']}/{r['n']} = {r['accuracy']:.1%}")
        for gt_val, stats in sorted(r["by_gt_value"].items()):
            pct = stats["agree"] / stats["total"] if stats["total"] else 0
            print(f"  GT={gt_val:10s}  agree={stats['agree']}/{stats['total']} ({pct:.1%})", end="")
            if stats["errors"]:
                err_str = ", ".join(f"annotated={k}: {v}" for k, v in sorted(stats["errors"].items()))
                print(f"  | errors: {err_str}", end="")
            print()

            for ann_val in sorted(stats["errors"]):
                examples = error_examples[label][(gt_val, ann_val)][:examples_per_error]
                for ex in examples:
                    print(f"    Example (GT={gt_val} → annotated={ann_val}) [{ex['conversation_id']} turns {ex['turns']}]")
                    print(f"      {ex['situation'][:200]}{'...' if len(ex['situation']) > 200 else ''}")
        print()

    ov = results["overall"]
    print(f"--- Overall (both labels, n={ov['n']}) ---")
    print(f"  Agreement: {ov['agree']}/{ov['n']} = {ov['accuracy']:.1%}\n")

    return results


if __name__ == "__main__":
    evaluate_situation_labels()
