"""
Situation Labeller -- Classify scaffolding key moment situations.

Reads scaffolding annotations (from annotate.py output or gold truth) and
classifies each situation as appropriate for scaffolding / rigor using
prompts/annotator/situation_labeller/classify_scaffolding.md.

The prompt takes only the {situation} field (not action/result) and returns
JSON: {"scaffolding": "yes|no|unclear|no_mention", "rigor": "yes|no|unclear|no_mention"}.

Usage:
    python -m annotator.core.situate --version v1
    python -m annotator.core.situate --version v1 --gold
    python -m annotator.core.situate --version v1 --split test
    python -m annotator.core.situate --version v1 --style balanced
"""

import argparse
import json
import logging
import re
from pathlib import Path

from common.logging_setup import setup_logging
from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_valid_styles
from .storage import (
    load_annotator_result, save_annotator_result,
    get_annotator_result_path,
)
from .utils import load_split_ids

logger = logging.getLogger(__name__)

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "annotator" / "situation_labeller" / "classify_scaffolding.md"
)

VALID_SITUATION_LABELS = {"yes", "no", "unclear", "no_mention"}
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}

def _parse_situation_label(text: str) -> tuple[dict, bool]:
    """Parse a situation label from model output text.

    Returns (situation_label dict, had_error).
    Tries json.loads first; falls back to regex extraction field-by-field.
    A list-wrapped response (e.g. [{...}]) is unwrapped automatically.
    """
    def _coerce(val: str) -> str:
        v = val.strip().lower()
        return v if v in VALID_SITUATION_LABELS else "unclear"

    # --- attempt 1: standard JSON parse ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        return {
            "scaffolding": _coerce(str(parsed.get("scaffolding", "unclear"))),
            "rigor": _coerce(str(parsed.get("rigor", "unclear"))),
        }, False
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        pass

    # --- attempt 2: regex field extraction ---
    # Handles unquoted keys, unquoted values, and extra surrounding text.
    result = {}
    for field in ("scaffolding", "rigor"):
        m = re.search(rf'["\']?{field}["\']?\s*:\s*["\']?([a-z_]+)["\']?', text)
        result[field] = _coerce(m.group(1)) if m else "unclear"

    had_error = result["scaffolding"] == "unclear" and result["rigor"] == "unclear"
    return result, had_error


def _load_prompt() -> str:
    logger.info("Loading situation labeller prompt: %s", PROMPT_PATH)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def run_situation_label(version: str, model: str, mode: str, phase_cfg: dict,
                        gold: bool = False,
                        annotator_style: str | None = None,
                        annotations_data: dict | None = None,
                        profile: str | None = None,
                        split: str = "train") -> dict | None:
    """Run situation labelling pass. Returns labeled annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_annotate().

    Only processes scaffolding annotations. Writes situation_label:
    {"scaffolding": "yes|no|unclear|no_mention", "rigor": "yes|no|unclear|no_mention"}
    onto each annotation, then saves to situation_labels_scaffolding.json.
    """
    in_memory = annotations_data is not None
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    gold_prefix = "annotations_gold" if gold else "annotations"
    input_filename = f"{gold_prefix}{profile_suffix}{style_suffix}{split_suffix}_scaffolding.json"

    if in_memory:
        data = annotations_data
    else:
        data = load_annotator_result(version, input_filename)
        if data is None:
            logger.error("%s not found for version %s. Run annotate first.", input_filename, version)
            return None
        logger.info("Loaded: %s", input_filename)

    split_ids = load_split_ids(split)
    results = {
        conv_id: conv_data
        for conv_id, conv_data in data["results"].items()
        if conv_id.rsplit("_", 1)[-1] in split_ids
    }

    prompt_template = _load_prompt()
    entries = []
    skipped = []
    locations = []

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data["annotations"]):
            if ann.get("annotation_type", "scaffolding") != "scaffolding":
                continue
            situation = ann.get("situation", "")
            if situation.strip().lower() in JUNK_TEXTS:
                skipped.append((conv_id, idx))
                continue
            prompt = prompt_template.replace("{situation}", situation)
            key = f"{conv_id}__{idx}"
            entries.append(build_batch_entry(key, prompt, json_mode=True))
            locations.append((conv_id, idx))

    logger.info("Situations to classify: %d (%d skipped as junk)", len(entries), len(skipped))
    logger.info("Model: %s | Mode: %s", model, mode)

    for conv_id, idx in skipped:
        results[conv_id]["annotations"][idx]["situation_label"] = {
            "scaffolding": "unclear", "rigor": "unclear",
        }

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        jsonl_path = str(output_dir / f"situation_label_requests{profile_suffix}.jsonl")
        write_jsonl(entries, jsonl_path)

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, json_mode=True, display_name="situation_label",
                        poll_interval=poll_interval,
                        thinking=phase_cfg.get("thinking", False),
                        thinking_budget=phase_cfg.get("thinking_budget", 0),
                        reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        logger.info("Running %d entries in sync mode...", len(entries))
        raw = run_sync_entries(client, entries, json_mode=True)

    _ZERO_COUNTS = lambda: {"yes": 0, "no": 0, "unclear": 0, "no_mention": 0}
    scaffolding_counts = _ZERO_COUNTS()
    rigor_counts = _ZERO_COUNTS()
    errors = 0
    total_input = 0
    total_output = 0

    for conv_id, idx in locations:
        key = f"{conv_id}__{idx}"
        entry = raw.get(key, {})

        if "error" in entry or not entry.get("text"):
            situation_label = {"scaffolding": "unclear", "rigor": "unclear"}
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            situation_label, had_error = _parse_situation_label(entry["text"])
            if had_error:
                logger.warning("Could not parse situation label for %s: %r", key, entry["text"][:200])
                errors += 1

        results[conv_id]["annotations"][idx]["situation_label"] = situation_label
        scaffolding_counts[situation_label["scaffolding"]] += 1
        rigor_counts[situation_label["rigor"]] += 1

    output = {
        **data,
        "results": results,
        "situation_labeled": True,
        "situation_label_stats": {
            "scaffolding": scaffolding_counts,
            "rigor": rigor_counts,
        },
        "token_summary": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": errors,
        },
    }

    output_filename = f"situation_labels{profile_suffix}{style_suffix}{split_suffix}_scaffolding.json"
    save_annotator_result(version, output_filename, output)
    n_classified = len(locations) + len(skipped)
    logger.info("Saved: %s | %d situations classified", output_filename, n_classified)

    logger.info("  Scaffolding appropriateness:")
    for label, count in scaffolding_counts.items():
        logger.info("    %-12s %d", label + ":", count)
    logger.info("  Rigor appropriateness:")
    for label, count in rigor_counts.items():
        logger.info("    %-12s %d", label + ":", count)
    logger.info("  Tokens: %s", f"{total_input + total_output:,}")
    if errors:
        logger.warning("  Errors: %d", errors)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Situation Labeller: classify scaffolding moment situations"
    )
    parser.add_argument("--version", default=None,
                        help="Version to label (reads annotations_scaffolding.json). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--gold", action="store_true",
                        help="Label gold truth annotations (annotations_gold_scaffolding.json)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Match the annotations_{style}_scaffolding.json file from annotate --style")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    args = parser.parse_args()

    setup_logging()

    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.annotator_style,
        cli_prompt_version=None,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]

    setup_logging(version=version)

    phase_cfg = get_phase_config("label", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    run_situation_label(version=version, model=model, mode=mode,
                        phase_cfg=phase_cfg, gold=args.gold,
                        annotator_style=style, profile=profile,
                        split=args.split)


if __name__ == "__main__":
    main()
