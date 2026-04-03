#!/usr/bin/env python3
"""Process new consolidated conversations into ground truth + transcript files.

Only processes conversations not already in data/ground_truth/.
Uses Anthropic batch API for effectiveness classification.

Usage:
    python data/process_new_conversations.py
    python data/process_new_conversations.py --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

CONSOLIDATED_DIR = DATA_DIR / "raw" / "consolidated"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

CLASSIFICATION_PROMPT = """Classify this tutoring strategy evaluation into exactly one category.

Categories:
- "effective": The strategy worked well, had positive impact on the student
- "partial": Mixed results — some benefits but notable limitations
- "ineffective": The strategy did not work, missed the mark, or was counterproductive

Annotator's evaluation:
"{result_text}"

Respond with ONLY one word: effective, partial, or ineffective"""

VALID_LABELS = {"effective", "partial", "ineffective"}
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


def find_new_conversations():
    """Return list of (conv_id, consolidated_path) for conversations not yet in ground truth."""
    existing_gt = {f.stem for f in GROUND_TRUTH_DIR.glob("*.json")}
    consolidated = sorted(CONSOLIDATED_DIR.glob("*.json"))
    new = [(f.stem, f) for f in consolidated if f.stem not in existing_gt]
    return new


def load_annotations_from_consolidated(conv_id, path):
    """Load a consolidated file and return (conversation_data, annotations_list)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    return data, annotations


def classify_batch(result_texts, keys):
    """Classify effectiveness labels using Anthropic batch API."""
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("label", "anthropic")
    client = ModelClient(cfg["model"])

    entries = []
    key_to_idx = {}
    labels = ["unclear"] * len(result_texts)

    for i, (text, key) in enumerate(zip(result_texts, keys)):
        stripped = text.strip().lower()
        if stripped in JUNK_TEXTS:
            labels[i] = "unclear"
            continue
        prompt = CLASSIFICATION_PROMPT.replace("{result_text}", text)
        entries.append(build_batch_entry(
            key=key,
            prompt_text=prompt,
            json_mode=False,
            max_tokens=32,
        ))
        key_to_idx[key] = i

    if not entries:
        return labels

    print(f"  Submitting {len(entries)} classifications to Anthropic batch API...")
    results = run_batch(
        client, entries,
        json_mode=False,
        display_name="effectiveness_classification",
        poll_interval=cfg.get("poll_interval", 60),
    )

    for key, result in results.items():
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            continue
        label = result["text"].strip().lower().rstrip(".")
        if label in VALID_LABELS:
            labels[idx] = label
        else:
            labels[idx] = "unclear"

    return labels


def write_ground_truth(conv_id, num_turns, annotations_with_labels):
    """Write a per-conversation ground truth file."""
    moments = []
    for ann, label in annotations_with_labels:
        moments.append({
            "turn_start": ann.get("turn_start"),
            "turn_end": ann.get("turn_end"),
            "annotation_type": ann.get("annotation_type", ""),
            "annotator_id": ann.get("annotator_id", ""),
            "situation": ann.get("situation", ""),
            "action": ann.get("action", ""),
            "result": ann.get("result", ""),
            "strategy_label": label,
        })

    output = {
        "conversation_id": conv_id,
        "num_turns": num_turns,
        "key_moments": moments,
    }
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GROUND_TRUTH_DIR / f"{conv_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return out_path


def write_transcript(conv_id, consolidated_data):
    """Write a transcript file from consolidated data (same format, just copied)."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRANSCRIPTS_DIR / f"{conv_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Process new conversations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running")
    args = parser.parse_args()

    new_convs = find_new_conversations()
    print(f"Found {len(new_convs)} new conversations to process")

    if not new_convs:
        print("Nothing to do.")
        return

    # Also find conversations with GT but missing transcript
    existing_gt = {f.stem for f in GROUND_TRUTH_DIR.glob("*.json")}
    existing_tx = {f.stem for f in TRANSCRIPTS_DIR.glob("*.json")}
    gt_missing_tx = existing_gt - existing_tx
    if gt_missing_tx:
        print(f"Also found {len(gt_missing_tx)} conversations with GT but missing transcript")

    if args.dry_run:
        for conv_id, path in new_convs[:10]:
            data, anns = load_annotations_from_consolidated(conv_id, path)
            print(f"  {conv_id}: {len(anns)} annotations, {data.get('num_turns', '?')} turns")
        if len(new_convs) > 10:
            print(f"  ... and {len(new_convs) - 10} more")
        return

    # Collect all result texts for batch classification
    all_texts = []
    all_keys = []
    conv_annotation_map = []  # (conv_idx, ann_idx) for each text

    conv_data_list = []
    for conv_idx, (conv_id, path) in enumerate(new_convs):
        data, anns = load_annotations_from_consolidated(conv_id, path)
        conv_data_list.append((conv_id, data, anns))
        for ann_idx, ann in enumerate(anns):
            result_text = ann.get("result", "")
            key = f"{conv_id}__{ann_idx}"
            all_texts.append(result_text)
            all_keys.append(key)
            conv_annotation_map.append((conv_idx, ann_idx))

    print(f"Total annotations to classify: {len(all_texts)}")

    # Classify
    labels = classify_batch(all_texts, all_keys)

    # Write outputs
    gt_written = 0
    tx_written = 0
    for conv_idx, (conv_id, data, anns) in enumerate(conv_data_list):
        # Pair annotations with their labels
        ann_labels = []
        for ann_idx, ann in enumerate(anns):
            # Find the label for this annotation
            label_idx = next(
                i for i, (ci, ai) in enumerate(conv_annotation_map)
                if ci == conv_idx and ai == ann_idx
            )
            ann_labels.append((ann, labels[label_idx]))

        write_ground_truth(conv_id, data.get("num_turns", 0), ann_labels)
        gt_written += 1

        write_transcript(conv_id, data)
        tx_written += 1

    # Fix GT-missing-transcript conversations
    tx_fixed = 0
    for conv_id in gt_missing_tx:
        consolidated_path = CONSOLIDATED_DIR / f"{conv_id}.json"
        if consolidated_path.exists():
            with open(consolidated_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            write_transcript(conv_id, data)
            tx_fixed += 1

    print(f"\nDone!")
    print(f"  Ground truth written: {gt_written}")
    print(f"  Transcripts written:  {tx_written}")
    print(f"  Missing transcripts fixed: {tx_fixed}")
    print(f"  Total ground truth files: {len(list(GROUND_TRUTH_DIR.glob('*.json')))}")
    print(f"  Total transcript files:   {len(list(TRANSCRIPTS_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
