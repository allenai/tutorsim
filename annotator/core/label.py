"""
Pass 3 -- Label annotations for effectiveness.

Always uses Gemini -- same classifier used on gold truth human annotations
in data/extract_ground_truth.py. This ensures labels are on a consistent
scale regardless of which model produced the annotations.

Reads annotations (from annotate.py output) and classifies each result
text as effective / partial / ineffective.

Usage:
    python -m pipeline.core.label --version v1
    python -m pipeline.core.label --version v1 --gold
"""

import argparse
import json
from pathlib import Path

from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config
from .storage import (
    load_annotator_result, save_annotator_result, annotator_result_exists,
    get_annotator_result_path,
)
from .utils import RESULTS_DIR

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

    results = data["results"]

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
                template = _load_prompt("classify_binary")
                prompt = template.replace("{result_text}", result_text)
            else:
                template = _load_prompt("classify")
                prompt = (template
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
        poll_interval = phase_cfg.get("poll_interval", 30)
        raw = run_batch(client, entries, json_mode=False, display_name="label", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0))
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
    phase_cfg = get_phase_config("label")

    parser = argparse.ArgumentParser(description="Pass 3: Label annotation effectiveness")
    parser.add_argument("--version", required=True,
                        help="Version to label (reads annotations.json from results/{version}/)")
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
    parser.add_argument("--annotator-style", "--style", choices=["generous", "balanced", "demanding"],
                        default=None, dest="annotator_style",
                        help="Match the annotations_{style}.json file from annotate --style")
    args = parser.parse_args()

    if args.profile:
        phase_cfg = get_phase_config("label", args.profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    output = run_label(version=args.version, model=model, mode=mode,
                       phase_cfg=phase_cfg, gold=args.gold,
                       binary=args.binary, annotator_style=args.annotator_style)
    if output:
        mode_hint = " --mode annotations" if args.gold else ""
        style_flag = f" --annotator-style {args.annotator_style}" if args.annotator_style else ""
        print(f"\nNext: python -m pipeline.eval.eval --version {args.version}{mode_hint}{style_flag}")


if __name__ == "__main__":
    main()
