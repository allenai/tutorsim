"""Minimal meta-prompt: only task + I/O + data. No diagnostic priming.

Same data as 6_generate_labeller_prompt.py, but the meta-prompt itself is
stripped down so the model has to infer all patterns from the examples
themselves -- not from our priors about what's hard.

Output: prompts/annotator/labeller/classify_v6.txt
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
OUTPUT_PROMPT = Path("prompts/annotator/labeller/classify_v6.txt")
META_MODEL = "claude-opus-4-7"
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
    setup_logging(version="labeller_v6_meta")
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

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    logger.info("Calling %s ...", META_MODEL)
    response = client.messages.create(
        model=META_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": meta_prompt}],
    )
    prompt_text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()

    logger.info("Generated prompt: %d chars, %d lines",
                len(prompt_text), len(prompt_text.splitlines()))
    logger.info("Usage: input=%d output=%d tokens",
                response.usage.input_tokens, response.usage.output_tokens)

    OUTPUT_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROMPT.write_text(prompt_text + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT_PROMPT)


if __name__ == "__main__":
    main()
