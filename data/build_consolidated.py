"""
Build per-transcript JSON files that merge full transcripts with all associated annotations.
Only creates files for transcripts that have at least one annotation.

Usage:
    python data/build_consolidated.py
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
OUTPUT_DIR = RAW_DIR / "consolidated"

RAPPORT_DIR = ANNOTATIONS_DIR / "rapport_annotations" / "stepup"
SCAFFOLDING_DIR = ANNOTATIONS_DIR / "scaffolding_annotations" / "stepup"

STEPUP_CSV = TRANSCRIPTS_DIR / "stepup.csv"
EEDI_CSV = TRANSCRIPTS_DIR / "eedi.csv"

# Regex to strip leading [TUTOR] or [STUDENT] from text since role is a separate field
SPEAKER_PREFIX_RE = re.compile(r"^\[(?:TUTOR|STUDENT)\]\s*")

# Regex to extract tutor_id and student_id from conversation_id
CONV_ID_RE = re.compile(r"(\d{4}-t\d+)_(\d{4}-s\d+)_(.+)")


def normalize_type(raw_type: str) -> str:
    """Remove brackets from type field, e.g. '[SCREEN INTERACTION]' -> 'SCREEN_INTERACTION'."""
    t = raw_type.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    return t.replace(" ", "_").upper()


def clean_text(raw_text: str) -> str:
    """Strip redundant [TUTOR]/[STUDENT] prefix from text."""
    return SPEAKER_PREFIX_RE.sub("", raw_text.strip())


def parse_conv_id(conversation_id: str):
    """Extract tutor_id, student_id from conversation_id."""
    m = CONV_ID_RE.match(conversation_id)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ---------------------------------------------------------------------------
# Step 1: Collect all annotated conversation IDs from snapshot files
# ---------------------------------------------------------------------------

def collect_annotated_conversations():
    """Walk annotation dirs, find snapshot JSONL files, return dict mapping
    conversation_id -> list of (annotation_type, snapshot_path)."""
    result = defaultdict(list)

    for ann_type, ann_dir in [("rapport", RAPPORT_DIR), ("scaffolding", SCAFFOLDING_DIR)]:
        if not ann_dir.exists():
            print(f"WARNING: annotation dir not found: {ann_dir}")
            continue
        for snapshot_path in ann_dir.rglob("*_snapshot.jsonl"):
            # The conversation_id is the parent directory name
            conv_id = snapshot_path.parent.name
            result[conv_id].append((ann_type, snapshot_path))

    return result


# ---------------------------------------------------------------------------
# Step 2: Parse transcript CSVs, group by conversation_id
# ---------------------------------------------------------------------------

def parse_transcript_csv(csv_path: Path, platform: str):
    """Parse a transcript CSV file, return dict mapping filename -> list of turn dicts."""
    transcripts = defaultdict(list)

    if not csv_path.exists():
        print(f"WARNING: transcript CSV not found: {csv_path}")
        return transcripts

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            transcripts[filename].append({
                "timestamp": row.get("timestamp", "").strip(),
                "role": row.get("role", "").strip().upper(),
                "text": clean_text(row.get("text", "")),
                "type": normalize_type(row.get("type", "dialogue")),
            })

    # Store context (same for all rows in a conversation) and platform
    context_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if filename and filename not in context_map:
                context_map[filename] = row.get("context", "").strip()

    return transcripts, context_map


# ---------------------------------------------------------------------------
# Step 3: Read annotations from snapshot files
# ---------------------------------------------------------------------------

def read_annotations(conv_id: str, snapshot_entries):
    """Read all snapshot files for a conversation, return flat list of annotation dicts."""
    annotations = []

    for ann_type, snapshot_path in snapshot_entries:
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    # Snapshot files have an "annotations" array with the individual annotations
                    inner_annotations = record.get("annotations", [])
                    annotator_id = record.get("annotator_id", "")

                    for ann in inner_annotations:
                        data_key = f"{ann_type}_data"
                        data = ann.get(data_key, {})

                        # Annotation tool uses 0-based turn indices;
                        # transcripts use 1-based turn_number. Convert here.
                        raw_start = ann.get("turn_start")
                        raw_end = ann.get("turn_end")
                        # Normalize annotator ID casing (EC2 data has
                        # lowercase dirs like 'gerber', 'stobbe')
                        raw_aid = ann.get("annotator_id", annotator_id)
                        normalized_aid = raw_aid[0].upper() + raw_aid[1:] if raw_aid else raw_aid

                        annotations.append({
                            "annotation_type": ann_type,
                            "annotator_id": normalized_aid,
                            "turn_start": raw_start + 1 if raw_start is not None else None,
                            "turn_end": raw_end + 1 if raw_end is not None else None,
                            "situation": data.get("situation", ""),
                            "action": data.get("action", ""),
                            "result": data.get("result", ""),
                            "timestamp": ann.get("timestamp", ""),
                        })
        except Exception as e:
            print(f"WARNING: error reading {snapshot_path}: {e}")

    return annotations


# ---------------------------------------------------------------------------
# Step 4: Merge and write output
# ---------------------------------------------------------------------------

def build_conversation_json(conv_id, turns, context, platform, annotations):
    """Build the final JSON structure for one conversation."""
    tutor_id, student_id = parse_conv_id(conv_id)

    numbered_turns = []
    for i, turn in enumerate(turns, start=1):
        numbered_turns.append({
            "turn_number": i,
            "timestamp": turn["timestamp"],
            "role": turn["role"],
            "text": turn["text"],
            "type": turn["type"],
        })

    return {
        "conversation_id": conv_id,
        "tutor_id": tutor_id,
        "student_id": student_id,
        "context": context,
        "platform": platform,
        "num_turns": len(numbered_turns),
        "turns": numbered_turns,
        "annotations": annotations,
    }


def main():
    print("Collecting annotated conversation IDs...")
    annotated = collect_annotated_conversations()
    print(f"  Found {len(annotated)} annotated conversations")

    print("Parsing stepup.csv...")
    stepup_transcripts, stepup_contexts = parse_transcript_csv(STEPUP_CSV, "stepup")
    print(f"  Found {len(stepup_transcripts)} conversations in stepup.csv")

    print("Parsing eedi.csv...")
    eedi_transcripts, eedi_contexts = parse_transcript_csv(EEDI_CSV, "eedi")
    print(f"  Found {len(eedi_transcripts)} conversations in eedi.csv")

    # Merge all transcripts into one lookup
    all_transcripts = {}
    all_contexts = {}
    all_platforms = {}

    for conv_id, turns in stepup_transcripts.items():
        all_transcripts[conv_id] = turns
        all_contexts[conv_id] = stepup_contexts.get(conv_id, "")
        all_platforms[conv_id] = "stepup"

    for conv_id, turns in eedi_transcripts.items():
        if conv_id not in all_transcripts:
            all_transcripts[conv_id] = turns
            all_contexts[conv_id] = eedi_contexts.get(conv_id, "")
            all_platforms[conv_id] = "eedi"

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_transcript = 0

    for conv_id, snapshot_entries in sorted(annotated.items()):
        if conv_id not in all_transcripts:
            print(f"  WARNING: no transcript found for annotated conversation {conv_id}")
            missing_transcript += 1
            continue

        turns = all_transcripts[conv_id]
        context = all_contexts.get(conv_id, "")
        platform = all_platforms.get(conv_id, "unknown")
        annotations = read_annotations(conv_id, snapshot_entries)

        result = build_conversation_json(conv_id, turns, context, platform, annotations)

        output_path = OUTPUT_DIR / f"{conv_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        written += 1

    print(f"\nDone!")
    print(f"  Written: {written} files to {OUTPUT_DIR}")
    print(f"  Missing transcripts: {missing_transcript}")
    print(f"  Total annotated conversations: {len(annotated)}")


if __name__ == "__main__":
    main()
