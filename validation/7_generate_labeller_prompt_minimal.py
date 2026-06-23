"""Minimal meta-prompt generator. Calls a designer model to produce a labeller
classifier prompt from the 343 train examples, with no diagnostic priming.

Usage:
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model claude-opus-4-7 \
    --output-version v6        # (original behavior)
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model gemini-3.1-pro-preview --output-version v7
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model gpt-5.4 --output-version v8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from annotator.core.client import ModelClient
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
PROMPTS_DIR = Path("prompts/annotator/labeller")
SAR_TYPES = ("scaffolding", "rapport")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotation_key(transcript_id, source_annotator_id, annotation_type, ts, te):
    return f"{transcript_id}|{source_annotator_id}|{annotation_type}|{ts}|{te}"


def build_sar_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_jsonl(path):
        if row.get("annotation_type") not in SAR_TYPES:
            continue
        for ta in row.get("turn_annotations", []):
            key = annotation_key(
                row["transcript_id"], row["source_annotator_id"],
                row["annotation_type"], ta["turn_number_start"], ta["turn_number_end"],
            )
            lookup[key] = ta
    return lookup


HUMAN_TO_LLM = {
    "effective": "effective",
    "partially_effective": "partial",
    "ineffective": "ineffective",
}


def format_examples(train_rows, sar_lookup) -> str:
    parts = []
    for i, row in enumerate(train_rows, 1):
        sar = sar_lookup.get(row["annotation_key"])
        if not sar:
            continue
        parts.append(
            f"---\n"
            f"Example {i}\n"
            f"annotation_type: {row['annotation_type']}\n"
            f"situation: {sar.get('situation','').strip()}\n"
            f"action: {sar.get('action','').strip()}\n"
            f"result_text: {sar.get('result','').strip()}\n"
            f"label: {HUMAN_TO_LLM[row['human_rating']]}\n"
        )
    return "\n".join(parts)


META_PROMPT_TEMPLATE = """Design a classifier prompt.

Below are __N_EXAMPLES__ labeled examples. Each has four input fields (annotation_type, situation, action, result_text) and one output label (effective, partial, or ineffective).

Design a prompt that, when given the four input fields, causes an LLM classifier (Claude Opus 4.6 with extended thinking) to output the correct label.

Requirements for your output:
- Use {annotation_type}, {situation}, {action}, {result_text} as placeholders.
- The classifier must output exactly one of: effective, partial, ineffective (lowercase, no punctuation, no other text).
- Output ONLY the prompt text. No explanation, no preamble, no postamble.

Examples:

__EXAMPLES__
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-model", required=True,
                        help="Model to design the prompt (any provider: anthropic|gemini|openai prefix-recognized)")
    parser.add_argument("--output-version", required=True,
                        help="Output prompt version, e.g. 'v6', 'v7', 'v8'. Writes to prompts/annotator/labeller/classify_{version}.txt")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max tokens for the designer response")
    args = parser.parse_args()

    setup_logging(version=f"labeller_meta_{args.output_version}")
    load_dotenv()

    train_rows = list(load_jsonl(EVAL / "labeller_train_v2.jsonl"))
    sar_lookup = build_sar_lookup(ROOT / "step_up_annotations.jsonl")
    logger.info("Loaded %d train rows", len(train_rows))

    examples_block = format_examples(train_rows, sar_lookup)
    meta_prompt = (
        META_PROMPT_TEMPLATE
        .replace("__N_EXAMPLES__", str(len(train_rows)))
        .replace("__EXAMPLES__", examples_block)
    )
    logger.info("Meta-prompt size: %d chars", len(meta_prompt))

    logger.info("Calling %s ...", args.meta_model)
    client = ModelClient(args.meta_model)
    response = client.generate(
        meta_prompt,
        json_mode=False,
        max_tokens=args.max_tokens,
        timeout=300,
    )
    prompt_text = response.text.strip()

    logger.info("Generated prompt: %d chars, %d lines",
                len(prompt_text), len(prompt_text.splitlines()))
    logger.info("Usage: input=%d output=%d tokens",
                response.usage.get("input_tokens", 0),
                response.usage.get("output_tokens", 0))

    output_path = PROMPTS_DIR / f"classify_{args.output_version}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt_text + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
