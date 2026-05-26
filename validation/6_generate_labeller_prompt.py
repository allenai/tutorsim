"""Meta-prompt: send the 343 train_v2 ratings to Claude Opus 4.7 (1M context)
and have it design a labeller prompt from scratch.

Test set (test_v2) is NEVER sent. Output is saved to
prompts/annotator/labeller/classify_v5.txt for downstream evaluation.

Usage:
  PYTHONPATH=. python validation/6_generate_labeller_prompt.py
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
OUTPUT_PROMPT = Path("prompts/annotator/labeller/classify_v5.txt")
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
    lookup: dict[str, dict] = {}
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
    """Render all 343 train rows in a compact, parseable format."""
    parts = []
    for i, row in enumerate(train_rows, 1):
        sar = sar_lookup.get(row["annotation_key"])
        if not sar:
            continue
        label = HUMAN_TO_LLM[row["human_rating"]]
        parts.append(
            f"---\n"
            f"Example {i}\n"
            f"Type: {row['annotation_type']}\n"
            f"Situation: {sar.get('situation','').strip()}\n"
            f"Action: {sar.get('action','').strip()}\n"
            f"Assessment: {sar.get('result','').strip()}\n"
            f"Label: {label}\n"
        )
    return "\n".join(parts)


META_PROMPT_TEMPLATE = """You are designing a prompt for an LLM classifier.

# The classification task
The classifier reads a teaching coach's written analysis of a tutoring moment and assigns one of three labels:
- "effective" -- the strategy worked overall
- "partial" -- mixed results
- "ineffective" -- the strategy failed overall

The classifier receives the analysis as four template fields:
- <ANN_TYPE>: either "scaffolding" or "rapport"
- <SIT>: context for the moment
- <ACT>: what the tutor did
- <RES>: the teaching coach's assessment of the outcome

In your output prompt, use literal Python-style single-curly placeholders:
{annotation_type}, {situation}, {action}, {result_text}

The classifier model is Claude Opus 4.6 with extended thinking enabled.

# Output constraint (strict)
The classifier must output EXACTLY one word, all lowercase, no punctuation:
  effective
  partial
  ineffective

Your prompt must instruct the classifier to produce this format. No other tokens.

# What we know about the data
- 490 ratings from 4 human reviewers (dani, nathan, query, rebecca).
- 343 are in this training set; 147 are held out as a test set you will NOT see.
- Zero cross-reviewer overlap -- there is no consensus ground truth on any item.
  Each item carries one reviewer's stance. Different reviewers can disagree on
  ambiguous cases, and we cannot measure that disagreement directly.
- Two annotation types ("scaffolding", "rapport") empirically have different
  partial-vs-polar boundaries. A pattern that humans call partial on rapport may
  not be called partial on scaffolding, and vice versa.
- Polar agreement (effective vs ineffective) is already easy. The disagreement
  concentrates in the partial cell.
- The classifier is evaluated by 3-way Cohen's kappa vs the held-out human
  ratings. Optimize for that.

# Prior attempts (high-level diagnostic only -- do not replicate)
- Rule-based prompts plateau around kappa 0.72-0.77. Rules cannot reliably
  separate "substantive" from "stylistic" improvement notes.
- Adding a small number of hand-picked examples improved rapport substantially
  but regressed scaffolding by a similar amount -- the same example set does
  not generalize symmetrically across types. Picking individual examples
  appears to encode individual reviewers' idiosyncrasies on ambiguous patterns.

# Constraints on your prompt
- Use {annotation_type}, {situation}, {action}, {result_text} as template
  placeholders in the OUTPUT (Python str.format-style single curlies).
- The output classifier prompt must be self-contained text. No JSON wrapper,
  no markdown commentary about your design choices -- just the prompt itself.
- Output ONLY the prompt. No preamble, no postamble, no explanation.

# Your task
Design the best possible classifier prompt that will maximize 3-way Cohen's
kappa against held-out reviewer ratings. You have full freedom over structure:
rules, examples, structured reasoning, type-conditional logic, decision trees,
multi-step decomposition, anything. The 343 training examples follow.

# Training examples (__N_EXAMPLES__ total)

__EXAMPLES__
"""


def main():
    setup_logging(version="labeller_v5_meta")
    load_dotenv()

    train_rows = list(load_jsonl(EVAL / "labeller_train_v2.jsonl"))
    sar_lookup = build_sar_lookup(ROOT / "step_up_annotations.jsonl")
    logger.info("Loaded %d train rows + %d SAR entries", len(train_rows), len(sar_lookup))

    examples_block = format_examples(train_rows, sar_lookup)
    meta_prompt = (
        META_PROMPT_TEMPLATE
        .replace("__N_EXAMPLES__", str(len(train_rows)))
        .replace("__EXAMPLES__", examples_block)
    )
    logger.info("Meta-prompt total chars: %d (approx %d tokens)",
                len(meta_prompt), len(meta_prompt) // 4)

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

    logger.info("Generated prompt: %d chars", len(prompt_text))
    logger.info("Usage: input=%d output=%d tokens",
                response.usage.input_tokens, response.usage.output_tokens)

    OUTPUT_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROMPT.write_text(prompt_text + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUTPUT_PROMPT)

    # Print first 30 lines as a preview
    preview = "\n".join(prompt_text.splitlines()[:30])
    print("\n=== v5 prompt preview (first 30 lines) ===")
    print(preview)
    print(f"\n... ({len(prompt_text.splitlines())} total lines)")


if __name__ == "__main__":
    main()
