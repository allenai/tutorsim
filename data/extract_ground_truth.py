"""
Extract structured ground truth labels from consolidated JSON files.

Reads consolidated JSON files from data/consolidated/ and produces
a single data/raw/ground_truth.json with structured labels for evaluation.

Note: The pipeline reads per-conversation files from data/ground_truth*/
produced by refresh_ground_truth.py. This script produces a legacy single-file
format. Both now use classify_v2.txt for consistent labelling.

Usage:
    python data/extract_ground_truth.py
"""

import json
import os
import time
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
CONSOLIDATED_DIR = RAW_DIR / "consolidated"
OUTPUT_PATH = RAW_DIR / "ground_truth.json"


PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "annotator" / "labeller"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def classify_effectiveness(annotations: list[dict]) -> list[str]:
    """Classify annotations using Gemini LLM.

    Each annotation dict must have keys: annotation_type, situation, action, result.
    Returns a list of labels corresponding to each input annotation.
    Returns "unclear" for empty/garbage texts.
    """
    from dotenv import load_dotenv
    from google import genai

    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-pro-preview"
    valid_labels = {"effective", "partial", "ineffective"}
    template = _load_prompt("classify_v2")

    labels = []
    total = len(annotations)

    for i, ann in enumerate(annotations):
        result_text = ann.get("result", "")
        # Skip empty/garbage texts
        stripped = result_text.strip().lower()
        if not stripped or stripped in ("n/a", "test", "sdf", "this is a test annotation"):
            labels.append("unclear")
            continue

        prompt = (template
                  .replace("{annotation_type}", ann.get("annotation_type", "unknown"))
                  .replace("{situation}", ann.get("situation", ""))
                  .replace("{action}", ann.get("action", ""))
                  .replace("{result_text}", result_text))

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                label = response.text.strip().lower().rstrip(".")
                if label in valid_labels:
                    labels.append(label)
                else:
                    labels.append("unclear")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  WARNING: Failed after 3 attempts for item {i}: {e}")
                    labels.append("unclear")

        if (i + 1) % 100 == 0:
            print(f"  Classified {i + 1}/{total}...")

    return labels


def main():
    if not CONSOLIDATED_DIR.exists():
        print(f"ERROR: consolidated dir not found: {CONSOLIDATED_DIR}")
        return

    files = sorted(CONSOLIDATED_DIR.glob("*.json"))
    print(f"Found {len(files)} consolidated files")

    # First pass: collect all annotations and result texts
    all_data = []  # list of (conv_id, num_turns, annotation_dict)
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv_id = data["conversation_id"]
        num_turns = data["num_turns"]
        for ann in data.get("annotations", []):
            all_data.append((conv_id, num_turns, ann))

    print(f"Total annotations to classify: {len(all_data)}")

    # Classify all annotations
    ann_dicts = [ann for _, _, ann in all_data]
    print("Classifying with Gemini LLM...")
    labels = classify_effectiveness(ann_dicts)

    # Build output structures
    conversations = {}
    total_annotations = 0
    by_type = {"rapport": 0, "scaffolding": 0}
    by_label = {"effective": 0, "partial": 0, "ineffective": 0, "unclear": 0}
    unclear_examples = []

    for (conv_id, num_turns, ann), strategy_label in zip(all_data, labels):
        result_text = ann.get("result", "")

        moment = {
            "turn_start": ann.get("turn_start"),
            "turn_end": ann.get("turn_end"),
            "annotation_type": ann.get("annotation_type", ""),
            "annotator_id": ann.get("annotator_id", ""),
            "situation": ann.get("situation", ""),
            "action": ann.get("action", ""),
            "result": result_text,
            "strategy_label": strategy_label,
        }

        if conv_id not in conversations:
            conversations[conv_id] = {"num_turns": num_turns, "key_moments": []}
        conversations[conv_id]["key_moments"].append(moment)

        ann_type = ann.get("annotation_type", "")
        if ann_type in by_type:
            by_type[ann_type] += 1

        by_label[strategy_label] += 1
        total_annotations += 1

        if strategy_label == "unclear":
            unclear_examples.append({
                "conversation_id": conv_id,
                "annotator_id": ann.get("annotator_id", ""),
                "annotation_type": ann_type,
                "result_text": result_text[:300],
            })

    output = {
        "conversations": conversations,
        "stats": {
            "total_conversations": len(conversations),
            "total_annotations": total_annotations,
            "annotations_by_type": by_type,
            "effective_count": by_label["effective"],
            "partial_count": by_label["partial"],
            "ineffective_count": by_label["ineffective"],
            "unclear_count": by_label["unclear"],
        },
        "unclear_examples": unclear_examples,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten: {OUTPUT_PATH}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  By type: {by_type}")
    print(f"  Effective: {by_label['effective']}")
    print(f"  Partial: {by_label['partial']}")
    print(f"  Ineffective: {by_label['ineffective']}")
    print(f"  Unclear: {by_label['unclear']}")

    if unclear_examples:
        print(f"\n  {len(unclear_examples)} annotations classified as 'unclear':")
        for ex in unclear_examples[:5]:
            print(f"    [{ex['annotation_type']}] {ex['annotator_id']}: {ex['result_text'][:100]}...")


if __name__ == "__main__":
    main()
