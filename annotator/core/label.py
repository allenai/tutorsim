"""
Pass 3 -- Label annotations for effectiveness.

Uses classify_v2.txt -- the same prompt used by data/build_ground_truth.py
for ground truth labelling. This ensures labels are on a consistent scale
regardless of which model produced the annotations.

Reads annotations (from annotate.py output) and classifies each result
text as effective / partial / ineffective.

Usage:
    python -m annotator.core.label --version v1
    python -m annotator.core.label --version v1 --gold
"""

import argparse
import datetime
import json
from pathlib import Path

from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_annotator_defaults, get_valid_styles
from .storage import (
    load_annotator_result, save_annotator_result, annotator_result_exists,
    get_annotator_result_path,
)
from .utils import load_split_ids
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "annotator" / "labeller"


def _load_prompt(name: str) -> str:
    """Load a labeller prompt from the prompts/labeller/ directory."""
    path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

VALID_LABELS = {"effective", "partial", "ineffective"}
VALID_LABELS_BINARY = {"effective", "ineffective"}
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


def run_label(version: str, model: str, mode: str, phase_cfg: dict,
              gold: bool = False, binary: bool = False,
              annotator_style: str | None = None,
              annotations_data: dict | None = None) -> dict:
    """Run labeling pass. Returns the labeled annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_annotate().
    """
    in_memory = annotations_data is not None
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    if gold:
        filename = f"annotations_gold{style_suffix}.json"
    else:
        filename = f"annotations{style_suffix}.json"

    if in_memory:
        data = annotations_data
    else:
        data = load_annotator_result(version, filename)
        if data is None:
            print(f"ERROR: {filename} not found for version {version}. Run annotate first.")
            return None

    # The transcript UUID is the last _-delimited segment of the compound conv_id.
    # Filter to train split as a safety net (detect/annotate already filter upstream).
    train_ids = load_split_ids("train")
    results = {
        conv_id: conv_data
        for conv_id, conv_data in data["results"].items()
        if conv_id.rsplit("_", 1)[-1] in train_ids
    }

    # Load template once
    from .config import get_annotator_defaults
    if binary:
        template = _load_prompt("classify_binary")
    else:
        labeller = get_annotator_defaults()["labeller"]
        template = _load_prompt(labeller)

    entries = []
    skipped = []
    locations = []

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data["annotations"]):
            result_text = ann.get("result", "")
            if result_text.strip().lower() in JUNK_TEXTS:
                skipped.append((conv_id, idx))
                continue

            situation = ann.get("situation", "")
            action = ann.get("action", "")
            if binary:
                prompt = template.replace("{result_text}", result_text)
            else:
                annotation_type = ann.get("annotation_type", "unknown")
                prompt = (template
                          .replace("{annotation_type}", annotation_type)
                          .replace("{situation}", situation)
                          .replace("{action}", action)
                          .replace("{result_text}", result_text))
            key = f"{conv_id}__{idx}"
            entries.append(build_batch_entry(key, prompt, json_mode=False))
            locations.append((conv_id, idx))

    print(f"Annotations to label: {len(entries)} ({len(skipped)} skipped as junk)")
    print(f"Model: {model} | Mode: {mode}")

    for conv_id, idx in skipped:
        results[conv_id]["annotations"][idx]["effectiveness"] = "unclear"

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        jsonl_path = str(output_dir / "label_requests.jsonl")
        write_jsonl(entries, jsonl_path)

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, json_mode=False, display_name="label", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0),
                       reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        print(f"Running {len(entries)} entries in sync mode...")
        raw = run_sync_entries(client, entries, json_mode=False)

    valid = VALID_LABELS_BINARY if binary else VALID_LABELS
    by_label = {"effective": 0, "ineffective": 0, "unclear": len(skipped)}
    if not binary:
        by_label["partial"] = 0
    errors = 0

    for conv_id, idx in locations:
        key = f"{conv_id}__{idx}"
        entry = raw.get(key, {})

        if "error" in entry or not entry.get("text"):
            label = "unclear"
            errors += 1
        else:
            label = entry["text"].strip().lower().rstrip(".")
            if label not in valid:
                label = "unclear"

        results[conv_id]["annotations"][idx]["effectiveness"] = label
        by_label[label] += 1

    data["labeled"] = True
    data["label_stats"] = by_label

    save_annotator_result(version, filename, data)
    print(f"\nSaved: {filename} (version: {version})")
    print(f"  Effective:   {by_label['effective']}")
    if not binary:
        print(f"  Partial:     {by_label['partial']}")
    print(f"  Ineffective: {by_label['ineffective']}")
    print(f"  Unclear:     {by_label['unclear']}")
    if errors:
        print(f"  Batch errors: {errors}")

    return data


def main():
    parser = argparse.ArgumentParser(description="Pass 3: Label annotation effectiveness")
    parser.add_argument("--version", default=None,
                        help="Version to label (reads annotations.json from results/{version}/). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--gold", action="store_true",
                        help="Label gold truth annotations (annotations_gold.json)")
    parser.add_argument("--binary", action="store_true",
                        help="Binary labeling only (effective/ineffective, no partial)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Match the annotations_{style}.json file from annotate --style")
    args = parser.parse_args()

    from common.logging_setup import setup_logging
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

    phase_cfg = get_phase_config("label", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    output = run_label(version=version, model=model, mode=mode,
                       phase_cfg=phase_cfg, gold=args.gold,
                       binary=args.binary, annotator_style=style)
    if output:
        mode_hint = " --mode annotations" if args.gold else ""
        style_flag = f" --annotator-style {style}" if style else ""
        print(f"\nNext: python -m annotator.eval.eval --version {version}{mode_hint}{style_flag}")


if __name__ == "__main__":
    main()
