"""
Decompose -- Break action and result annotations into atomic facets.

Reads annotations (from annotate.py output or gold truth) and decomposes each
annotation's action and result fields into lists of short, standalone, atomic
statements, using:
  - prompts/annotator/decomposer/decompose_action.md  (for action field)
  - prompts/annotator/decomposer/decompose_result.md  (for result field)

Adds action_decomposed and result_decomposed (list[str]) to each annotation,
then saves to decomposed_{target}.json.

Usage:
    python -m annotator.core.decompose --version v1
    python -m annotator.core.decompose --version v1 --gold
    python -m annotator.core.decompose --version v1 --split test
    python -m annotator.core.decompose --version v1 --target rapport
    python -m annotator.core.decompose --version v1 --style balanced
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
from .config import get_phase_config, get_valid_styles, get_annotation_types
from .storage import (
    load_annotator_result, save_annotator_result,
    get_annotator_result_path,
)
from .utils import load_split_ids

logger = logging.getLogger(__name__)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "annotator" / "decomposer"
)

JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    logger.info("Loading decomposer prompt: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_decomposed(text: str) -> tuple[list[str], bool]:
    """Parse a JSON array of strings from model output.

    Returns (facets list, had_error). Falls back to regex extraction if
    json.loads fails.
    """
    # Attempt 1: standard JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s) for s in parsed], False
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: extract a bracketed array from surrounding text
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(s) for s in parsed], False
        except (json.JSONDecodeError, TypeError):
            pass

    return [], True


def run_decompose(version: str, model: str, mode: str, phase_cfg: dict,
                  gold: bool = False,
                  annotator_style: str | None = None,
                  annotations_data: dict | None = None,
                  profile: str | None = None,
                  target: str = "scaffolding",
                  split: str = "train",
                  dry_run: bool = False) -> dict | None:
    """Run decomposition pass. Returns enriched annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_annotate().

    Adds action_decomposed and result_decomposed (list[str]) to each
    annotation of the target type, then saves to decomposed_{target}.json.
    """
    in_memory = annotations_data is not None
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    gold_prefix = "annotations_gold" if gold else "annotations"
    input_filename = f"{gold_prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"

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

    action_template = _load_prompt("decompose_action.md")
    result_template = _load_prompt("decompose_result.md")

    action_entries = []
    result_entries = []
    locations_action = []
    locations_result = []
    skipped_action = 0
    skipped_result = 0

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data["annotations"]):
            if ann.get("annotation_type", target) != target:
                continue

            situation = ann.get("situation", "")
            action = ann.get("action", "")
            result_text = ann.get("result", "")

            if action.strip().lower() in JUNK_TEXTS:
                results[conv_id]["annotations"][idx]["action_decomposed"] = []
                skipped_action += 1
            else:
                key = f"action__{conv_id}__{idx}"
                prompt = action_template.replace("{action}", action)
                action_entries.append(build_batch_entry(key, prompt, json_mode=True))
                locations_action.append((conv_id, idx))

            if result_text.strip().lower() in JUNK_TEXTS:
                results[conv_id]["annotations"][idx]["result_decomposed"] = []
                skipped_result += 1
            else:
                key = f"result__{conv_id}__{idx}"
                prompt = (result_template
                          .replace("{situation}", situation)
                          .replace("{action}", action)
                          .replace("{result}", result_text))
                result_entries.append(build_batch_entry(key, prompt, json_mode=True))
                locations_result.append((conv_id, idx))

    entries = action_entries + result_entries
    logger.info(
        "Action entries: %d (%d skipped) | Result entries: %d (%d skipped)",
        len(action_entries), skipped_action, len(result_entries), skipped_result,
    )
    logger.info("Model: %s | Mode: %s", model, mode)

    if dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Input:          {input_filename}")
        print(f"Model:          {model}  [{mode}]")
        print(f"Action entries: {len(action_entries)}  ({skipped_action} skipped as junk)")
        print(f"Result entries: {len(result_entries)}  ({skipped_result} skipped as junk)")
        print(f"Total API calls: {len(entries)}")
        for label, sample_entries in [("action", action_entries[:2]), ("result", result_entries[:2])]:
            for e in sample_entries:
                body = e.get("body", e)
                messages = body.get("messages", [])
                text = messages[0].get("content", "") if messages else ""
                print(f"\n  [{label}] {e.get('custom_id', '')}")
                print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        return None

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        jsonl_path = str(output_dir / f"decompose_requests{profile_suffix}.jsonl")
        write_jsonl(entries, jsonl_path)

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, json_mode=True, display_name="decompose",
                        poll_interval=poll_interval,
                        thinking=phase_cfg.get("thinking", False),
                        thinking_budget=phase_cfg.get("thinking_budget", 0),
                        reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        logger.info("Running %d entries in sync mode...", len(entries))
        raw = run_sync_entries(client, entries, json_mode=True)

    total_input = 0
    total_output = 0
    errors = 0
    total_action_facets = 0
    total_result_facets = 0

    for conv_id, idx in locations_action:
        key = f"action__{conv_id}__{idx}"
        entry = raw.get(key, {})
        if "error" in entry or not entry.get("text"):
            facets = []
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            facets, had_error = _parse_decomposed(entry["text"])
            if had_error:
                logger.warning("Could not parse action decomposition for %s: %r", key, entry["text"][:200])
                errors += 1
        results[conv_id]["annotations"][idx]["action_decomposed"] = facets
        total_action_facets += len(facets)

    for conv_id, idx in locations_result:
        key = f"result__{conv_id}__{idx}"
        entry = raw.get(key, {})
        if "error" in entry or not entry.get("text"):
            facets = []
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            facets, had_error = _parse_decomposed(entry["text"])
            if had_error:
                logger.warning("Could not parse result decomposition for %s: %r", key, entry["text"][:200])
                errors += 1
        results[conv_id]["annotations"][idx]["result_decomposed"] = facets
        total_result_facets += len(facets)

    output = {
        **data,
        "results": results,
        "decomposed": True,
        "decompose_stats": {
            "action_entries": len(action_entries),
            "result_entries": len(result_entries),
            "skipped_action": skipped_action,
            "skipped_result": skipped_result,
            "total_action_facets": total_action_facets,
            "total_result_facets": total_result_facets,
        },
        "token_summary": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": errors,
        },
    }

    output_filename = f"decomposed{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
    save_annotator_result(version, output_filename, output)
    logger.info("Saved: %s", output_filename)

    logger.info("  Action facets extracted: %d", total_action_facets)
    logger.info("  Result facets extracted: %d", total_result_facets)
    logger.info("  Tokens: %s", f"{total_input + total_output:,}")
    if errors:
        logger.warning("  Errors: %d", errors)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Decompose action and result annotations into atomic facets"
    )
    parser.add_argument("--version", default=None,
                        help="Version to process. Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--gold", action="store_true",
                        help="Decompose gold truth annotations")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Match the annotations_{style}_{target}.json file from annotate --style")
    parser.add_argument("--target", choices=get_annotation_types(), default="scaffolding",
                        help="Annotation type to decompose (default: scaffolding)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print entries that would be submitted without calling the API")
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

    run_decompose(version=version, model=model, mode=mode,
                  phase_cfg=phase_cfg, gold=args.gold,
                  annotator_style=style, profile=profile,
                  target=args.target, split=args.split,
                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()
