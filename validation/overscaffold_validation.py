"""
Build data/overscaffolding_sample_200.csv for human over-scaffolding validation,
and evaluate decompose_overscaffold.md against the human labels.

create_val_set() takes the EXACT 200 moments already in
data/scaffolding_sample_200.csv (so this set is directly comparable to the
situation-validation set) and augments each row with the action and result
columns -- the over-scaffold decomposition reads situation + action + result, so
all three are needed for annotation.

action/result are looked up per row by (conversation_id, turn_start, turn_end,
annotator_ID, situation):
  - primary source: ground_truth_hybrid (where the situation column came from), and
  - fallback: the raw teacher annotations JSONL, for the handful of rows whose
    moments have since drifted out of the current ground truth.
Every row in scaffolding_sample_200.csv is recoverable from one of these, so no
action/result is left blank.

evaluate_overscaffold_labels() runs decompose_overscaffold.md (via the same
decompose_batch code path the ground-truth pipeline uses) over the human-labelled
overscaffolding_sample_200_done.csv and reports it as a binary classifier:
predict over-scaffolding when the prompt returns >=1 span, and compare to the
human overscaffolding? column (y=positive, n=negative).

Usage:
    python validation/overscaffold_validation.py            # build the sample CSV
    python validation/overscaffold_validation.py --evaluate # run the prompt eval
"""

import argparse
import csv
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(DATA_DIR, "ground_truth_hybrid")
SOURCE_CSV = os.path.join(DATA_DIR, "scaffolding_sample_200.csv")
ANNOTATIONS_JSONL = os.path.join(DATA_DIR, "teacher_annotations", "step_up_annotations.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "overscaffolding_sample_200.csv")
DONE_CSV = os.path.join(DATA_DIR, "overscaffolding_sample_200_done.csv")
EVAL_OUTPUT_CSV = os.path.join(DATA_DIR, "overscaffolding_sample_200_eval.csv")

FIELDNAMES = [
    "conversation_id", "turn_start", "turn_end", "annotator_ID",
    "situation", "action", "result",
]


def _row_key(conversation_id, turn_start, turn_end, annotator_id, situation):
    """Identity of a moment, as strings, for matching across sources."""
    return (
        conversation_id,
        str(turn_start),
        str(turn_end),
        annotator_id,
        situation or "",
    )


def _build_gt_lookup():
    """{row_key: (action, result)} from scaffolding moments in ground_truth_hybrid."""
    lookup = {}
    for fname in os.listdir(GT_DIR):
        if not fname.endswith(".json"):
            continue
        cid = fname[:-len(".json")]
        with open(os.path.join(GT_DIR, fname)) as f:
            data = json.load(f)
        for m in data.get("key_moments", []):
            if m.get("annotation_type") != "scaffolding":
                continue
            key = _row_key(cid, m.get("turn_start"), m.get("turn_end"),
                           m.get("annotator_id"), m.get("situation"))
            lookup[key] = (m.get("action", ""), m.get("result", ""))
    return lookup


def _build_jsonl_lookup():
    """{row_key: (action, result)} from raw scaffolding teacher annotations."""
    lookup = {}
    with open(ANNOTATIONS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("annotation_type") != "scaffolding":
                continue
            cid = rec["transcript_id"]
            annotator_id = rec.get("annotator_id", "")
            for ta in rec.get("turn_annotations", []):
                key = _row_key(cid, ta.get("turn_number_start"), ta.get("turn_number_end"),
                               annotator_id, ta.get("situation"))
                lookup[key] = (ta.get("action", ""), ta.get("result", ""))
    return lookup


def create_val_set():
    with open(SOURCE_CSV) as f:
        source_rows = list(csv.DictReader(f))

    gt_lookup = _build_gt_lookup()
    jsonl_lookup = _build_jsonl_lookup()

    out_rows = []
    from_gt = from_jsonl = blank = 0
    for r in source_rows:
        key = _row_key(r["conversation_id"], r["turn_start"], r["turn_end"],
                       r["annotator_ID"], r["situation"])
        if key in gt_lookup:
            action, result = gt_lookup[key]
            from_gt += 1
        elif key in jsonl_lookup:
            action, result = jsonl_lookup[key]
            from_jsonl += 1
        else:
            action, result = "", ""
            blank += 1
        out_rows.append({**r, "action": action, "result": result})

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Source rows: {len(source_rows)}")
    print(f"  action/result from ground_truth_hybrid: {from_gt}")
    print(f"  action/result from JSONL fallback:      {from_jsonl}")
    print(f"  left blank (no source found):           {blank}")
    print(f"Saved {len(out_rows)} rows to {OUTPUT_FILE}")


def _binary_metrics(pairs):
    """Confusion matrix + precision/recall/F1/accuracy for (gold, pred) booleans.

    Positive class = over-scaffolding present. Division is guarded: when a
    denominator is 0 (e.g. the model predicted no positives) the rate is 0.0.
    """
    tp = sum(1 for gold, pred in pairs if gold and pred)
    fp = sum(1 for gold, pred in pairs if not gold and pred)
    fn = sum(1 for gold, pred in pairs if gold and not pred)
    tn = sum(1 for gold, pred in pairs if not gold and not pred)
    n = len(pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def evaluate_overscaffold_labels(done_csv_path=None, examples_per_error=3):
    """Evaluate decompose_overscaffold.md against the human overscaffolding? labels.

    Runs the prompt over every row of overscaffolding_sample_200_done.csv via the
    same decompose_batch path the ground-truth pipeline uses (production label
    config: claude-opus-4-8, batch, thinking), then scores it as a binary
    classifier: pred=positive when the prompt returns >=1 span, gold=positive when
    the human label is 'y'. Prints the confusion matrix + metrics and example
    false positives / false negatives (with the spans the model produced), writes a
    per-row predictions CSV, and returns the results dict.
    """
    if done_csv_path is None:
        done_csv_path = DONE_CSV
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from data.build_ground_truth import decompose_batch

    with open(done_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Drop rows without a usable human label (y/n); report how many.
    labelled, skipped = [], 0
    for i, r in enumerate(rows):
        if (r.get("overscaffolding?") or "").strip().lower() in ("y", "n"):
            labelled.append((i, r))
        else:
            skipped += 1

    items = [
        {"key": f"row_{i}", "field": "overscaffold",
         "situation": r.get("situation", ""), "action": r.get("action", ""),
         "result": r.get("result", "")}
        for i, r in labelled
    ]
    print(f"Running decompose_overscaffold.md over {len(items)} labelled rows "
          f"({skipped} unlabelled rows skipped)...")
    facets_by_key = decompose_batch(items)

    pairs = []
    eval_rows = []
    for i, r in labelled:
        spans = facets_by_key.get(f"row_{i}", [])
        pred = len(spans) > 0
        gold = (r.get("overscaffolding?") or "").strip().lower() == "y"
        pairs.append((gold, pred))
        eval_rows.append({
            "conversation_id": r["conversation_id"],
            "turn_start": r["turn_start"],
            "turn_end": r["turn_end"],
            "gold": "y" if gold else "n",
            "pred": "y" if pred else "n",
            "n_spans": len(spans),
            "spans": json.dumps(spans, ensure_ascii=False),
        })

    metrics = _binary_metrics(pairs)

    # Persist per-row predictions for inspection / prompt iteration.
    with open(EVAL_OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "conversation_id", "turn_start", "turn_end", "gold", "pred", "n_spans", "spans"])
        writer.writeheader()
        writer.writerows(eval_rows)

    # Report.
    n_pos = sum(1 for gold, _ in pairs if gold)
    n_neg = len(pairs) - n_pos
    print(f"\n=== Over-Scaffolding Prompt Evaluation (n={len(pairs)} rows) ===")
    print("Gold: human overscaffolding? label (y=positive, n=negative)")
    print("Pred: decompose_overscaffold.md returned >=1 span\n")
    print("Confusion matrix:")
    print(f"                 pred=y   pred=n")
    print(f"  gold=y ({n_pos:3d})     {metrics['tp']:4d}     {metrics['fn']:4d}")
    print(f"  gold=n ({n_neg:3d})     {metrics['fp']:4d}     {metrics['tn']:4d}\n")
    print(f"  Precision: {metrics['precision']:.1%}  (of predicted over-scaffolding, how many humans agreed)")
    print(f"  Recall:    {metrics['recall']:.1%}  (of human over-scaffolding, how many the prompt caught)")
    print(f"  F1:        {metrics['f1']:.1%}")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}\n")

    def _examples(want_gold, want_pred, header):
        picks = [(i, r, facets_by_key.get(f"row_{i}", []))
                 for (i, r), (gold, pred) in zip(labelled, pairs)
                 if gold == want_gold and pred == want_pred]
        print(f"--- {header}: {len(picks)} ---")
        for i, r, spans in picks[:examples_per_error]:
            print(f"  [{r['conversation_id']} turns {r['turn_start']}-{r['turn_end']}]")
            print(f"    action: {r['action'][:160]}{'...' if len(r['action']) > 160 else ''}")
            print(f"    result: {r['result'][:160]}{'...' if len(r['result']) > 160 else ''}")
            print(f"    spans:  {spans}")
        print()

    _examples(False, True, "False positives (gold=n, pred=y)")
    _examples(True, False, "False negatives (gold=y, pred=n)")

    print(f"Per-row predictions written to {EVAL_OUTPUT_CSV}")

    return {"metrics": metrics, "skipped": skipped, "eval_rows": eval_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate decompose_overscaffold.md against the human labels in "
                             "overscaffolding_sample_200_done.csv (instead of building the sample CSV).")
    args = parser.parse_args()
    if args.evaluate:
        evaluate_overscaffold_labels()
    else:
        create_val_set()
