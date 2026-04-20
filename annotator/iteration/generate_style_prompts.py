"""
Step 2: Generate annotator-style prompt fragments for each archetype.

Uses the LLM to synthesize a calibration paragraph from each archetype's
actual annotations. The output is injected into the p2 annotation prompt
via the {annotator_style} template slot.

Usage:
    python -m pipeline.iteration.generate_style_prompts --profile gemini

Requires:
    results/annotator_profiles.json  (from classify_annotators.py)
    data/gold_raw.json

Output:
    pipeline/prompts/annotator_styles/{generous,balanced,demanding}.txt
"""

import argparse
import json
import random
from pathlib import Path

from ..core.client import ModelClient
from ..core.config import get_phase_config

from ..core.storage import get_annotator_result_path

REPO_ROOT = Path(__file__).parent.parent.parent
GOLD_RAW_PATH = REPO_ROOT / "data" / "raw" / "gold_raw.json"  # legacy one-time script
PROFILES_PATH = get_annotator_result_path("", "annotator_profiles.json")
STYLES_DIR = Path(__file__).parent.parent.parent / "prompts" / "archive" / "annotator_styles_old"

EXAMPLES_PER_ARCHETYPE = 12

VALID_LABELS = {"effective", "partial", "ineffective"}

ARCHETYPE_CONTEXT = {
    "generous": (
        "relatively generous and encouraging — they tend to rate tutoring strategies highly "
        "when genuine effort is made to meet the student's need, even if execution was imperfect"
    ),
    "balanced": (
        "balanced and nuanced — they weigh both strengths and weaknesses carefully, "
        "using 'partial' freely when there is a meaningful gap between what the tutor did "
        "and what would have been ideal"
    ),
    "demanding": (
        "rigorous and demanding — they hold tutors to a high standard, "
        "reserving 'effective' for well-executed strategies and using 'ineffective' "
        "whenever a tutor misses an opportunity to push deeper thinking"
    ),
}

META_PROMPT = """You are calibrating an AI tutoring coach evaluator.

Below are {n} real annotation examples from human raters classified as "{archetype}" raters.
These annotators are {archetype_context}.

Your task: Write a SHORT paragraph (4-6 sentences) that will be injected into an
annotation prompt to make an AI annotator adopt this type of rater's perspective.

The paragraph should:
1. Describe this annotator type's rating philosophy in concrete terms
2. Give specific guidance on the threshold for each label (effective / partial / ineffective)
3. Be written as direct instructions to the AI (e.g. "When rating, consider...")
4. NOT reference that these are example annotations or that you're mimicking humans

Examples (situation summary / label):
{examples}

Write ONLY the calibration paragraph. No heading, no preamble, no quotes."""


def collect_examples(profiles_data: dict, gold_raw: dict, archetype: str, n: int) -> list[dict]:
    """Collect n example annotations from annotators of the given archetype."""
    archetype_annotators = set(profiles_data["archetypes"].get(archetype, []))
    all_examples = []

    for conv_id, conv_data in gold_raw["conversations"].items():
        for moment in conv_data.get("key_moments", []):
            ann_id = moment.get("annotator_id", "unknown")
            if ann_id not in archetype_annotators:
                continue
            label = moment.get("strategy_label", "unclear")
            if label not in VALID_LABELS:
                continue
            all_examples.append({
                "situation": moment.get("situation", ""),
                "action": moment.get("action", ""),
                "result": moment.get("result", ""),
                "strategy_label": label,
            })

    random.shuffle(all_examples)
    return all_examples[:n]


def format_examples(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        sit = ex["situation"][:180].replace("\n", " ")
        act = ex["action"][:120].replace("\n", " ")
        lines.append(f"{i}. [{ex['strategy_label'].upper()}]")
        lines.append(f"   Situation: {sit}")
        lines.append(f"   Action: {act}")
        lines.append("")
    return "\n".join(lines)


def generate_style_prompt(client: ModelClient, archetype: str, examples: list[dict]) -> str:
    examples_str = format_examples(examples)
    prompt = (META_PROMPT
              .replace("{archetype}", archetype)
              .replace("{archetype_context}", ARCHETYPE_CONTEXT[archetype])
              .replace("{n}", str(len(examples)))
              .replace("{examples}", examples_str))

    result = client.generate(prompt, json_mode=False, max_tokens=1024, thinking=False)
    return result.text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate annotator style prompts")
    parser.add_argument("--profile", default=None,
                        help="Config profile for LLM (default: config.yaml default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for example selection")
    args = parser.parse_args()

    random.seed(args.seed)

    if not PROFILES_PATH.exists():
        print(f"ERROR: {PROFILES_PATH} not found.")
        print("Run: python -m pipeline.iteration.classify_annotators first.")
        return

    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    with open(GOLD_RAW_PATH, "r", encoding="utf-8") as f:
        gold_raw = json.load(f)

    phase_cfg = get_phase_config("annotate", args.profile)
    client = ModelClient(phase_cfg["model"])
    print(f"Using model: {phase_cfg['model']}")

    STYLES_DIR.mkdir(parents=True, exist_ok=True)

    for archetype in ("generous", "balanced", "demanding"):
        annotators = profiles_data["archetypes"].get(archetype, [])
        if not annotators:
            print(f"\n[{archetype}] No annotators in this bucket, skipping.")
            continue

        print(f"\n[{archetype}] Collecting examples from: {', '.join(sorted(annotators))}")
        examples = collect_examples(profiles_data, gold_raw, archetype, EXAMPLES_PER_ARCHETYPE)

        if not examples:
            print(f"  No examples found, skipping.")
            continue

        print(f"  {len(examples)} examples collected. Generating style prompt...")
        style_text = generate_style_prompt(client, archetype, examples)

        output_path = STYLES_DIR / f"{archetype}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(style_text)

        print(f"  Saved: {output_path}")
        print(f"  Preview: {style_text[:250]}...")

    print(f"\nDone. Style prompts saved to: {STYLES_DIR}")
    print("Next: python -m pipeline.core.annotate --version v4_gold --gold --annotator-style generous")


if __name__ == "__main__":
    main()
