#!/usr/bin/env python3
"""Split ground truth conversation IDs into train and test sets.

Reads all conversation IDs from a ground truth directory, shuffles them with a
fixed random seed, and writes a 50/50 train/test split to a JSON file.

Output format:
  {
    "seed": <int>,
    "ground_truth_dir": "<str>",
    "train": ["<conv_id>", ...],
    "test":  ["<conv_id>", ...]
  }

Usage:
    python data/split_ground_truth.py
    python data/split_ground_truth.py --seed 123
    python data/split_ground_truth.py --ground-truth-dir data/ground_truth_v2
    python data/split_ground_truth.py --output data/my_split.json
"""
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
DEFAULT_GT_DIR = DATA_DIR / "ground_truth_v2"
DEFAULT_OUTPUT = DATA_DIR / "split.json"
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", default=str(DEFAULT_GT_DIR),
                        help=f"Ground truth directory (default: {DEFAULT_GT_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth_dir)
    if not gt_dir.exists():
        print(f"ERROR: ground truth directory not found: {gt_dir}")
        return

    conv_ids = sorted(f.stem for f in gt_dir.glob("*.json"))
    if not conv_ids:
        print(f"ERROR: no JSON files found in {gt_dir}")
        return

    rng = random.Random(args.seed)
    rng.shuffle(conv_ids)

    mid = len(conv_ids) // 2
    # odd N: test gets the extra conversation
    train_ids = conv_ids[:mid]
    test_ids = conv_ids[mid:]

    out = {
        "seed": args.seed,
        "ground_truth_dir": str(gt_dir),
        "train": train_ids,
        "test": test_ids,
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Split {len(conv_ids)} conversations (seed={args.seed})")
    print(f"  train: {len(train_ids)}")
    print(f"  test:  {len(test_ids)}")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
