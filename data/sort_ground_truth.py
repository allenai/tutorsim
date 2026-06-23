#!/usr/bin/env python3
"""Re-sort key_moments in ground truth files by annotation_timestamp.

One-off script to apply chronological ordering to existing ground truth files
without re-running the Anthropic batch API. Reads timestamps from the source
annotations JSONL and rewrites each ground truth file in place.

Usage:
    python data/sort_ground_truth.py
    python data/sort_ground_truth.py --ground-truth-dir data/ground_truth_v2
"""
import argparse
import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
DEFAULT_GT_DIR = DATA_DIR / "ground_truth_v2"
DEFAULT_ANNOTATIONS = DATA_DIR / "teacher_annotations" / "step_up_annotations.jsonl"


def moment_key(conv_id, annotator_id, annotation_type, turn_start, turn_end, result):
    """Matches the key used in build_ground_truth.py."""
    return (
        conv_id,
        annotator_id,
        turn_start,
        turn_end,
        annotation_type,
        hashlib.md5((result or "").encode("utf-8")).hexdigest()[:12],
    )


def build_timestamp_map(annotations_path):
    """Build a mapping from moment_key → annotation_timestamp from the source JSONL.

    Where the same key appears multiple times (revisions), keeps the latest timestamp.
    """
    ts_map = {}
    with open(annotations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("annotation_type") not in ("scaffolding", "rapport"):
                continue
            conv_id = record["transcript_id"]
            annotator_id = record.get("annotator_id", "")
            annotation_type = record["annotation_type"]
            for ta in record.get("turn_annotations", []):
                if ta.get("turn_number_start") is None or ta.get("turn_number_end") is None:
                    continue
                key = moment_key(
                    conv_id,
                    annotator_id,
                    annotation_type,
                    ta["turn_number_start"],
                    ta["turn_number_end"],
                    ta.get("result", ""),
                )
                ts = ta.get("annotation_timestamp", "")
                # Keep the latest timestamp for this key
                if key not in ts_map or ts > ts_map[key]:
                    ts_map[key] = ts
    return ts_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth_dir)
    annotations_path = Path(args.annotations)

    print(f"Building timestamp map from {annotations_path} ...")
    ts_map = build_timestamp_map(annotations_path)
    print(f"  {len(ts_map)} entries")

    files = sorted(gt_dir.glob("*.json"))
    unmatched_total = 0

    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        conv_id = data["conversation_id"]
        moments = data["key_moments"]

        unmatched = []
        def sort_key(m):
            key = moment_key(
                conv_id,
                m.get("annotator_id", ""),
                m.get("annotation_type", ""),
                m.get("turn_start"),
                m.get("turn_end"),
                m.get("result", ""),
            )
            ts = ts_map.get(key, "")
            if not ts:
                unmatched.append(m)
            return ts

        data["key_moments"] = sorted(moments, key=sort_key)
        unmatched_total += len(unmatched)

        with open(f, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)

    print(f"Rewrote {len(files)} ground truth files")
    if unmatched_total:
        print(f"  WARNING: {unmatched_total} moments had no timestamp match (sorted to front)")
    else:
        print(f"  All moments matched to a timestamp")


if __name__ == "__main__":
    main()
