"""
Pass 3 -- Label annotations for effectiveness.

The labeller routes by annotation_type when `annotator.labeller` is a dict
(e.g. {scaffolding: classify_scaffolding, rapport: classify_rapport}), and
loads a single template when it's a string. Same shared prompts are used by
data/build_ground_truth.py so labels stay on a consistent scale.

Reads annotations (from annotate.py output) and classifies each as
effective / partial / ineffective. In 3-way mode, the labeller prompt
receives situation, action, and result; in binary mode, result only.

Usage:
    python -m annotator.core.label --version v1
    python -m annotator.core.label --version v1 --gold
    python -m annotator.core.label --version v1 --split test
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

from common.logging_setup import setup_logging
from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_annotator_defaults, get_valid_styles, get_annotation_types
from .storage import (
    load_annotator_result, save_annotator_result,
    get_annotator_result_path,
)
from .utils import load_split_ids

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "annotator" / "labeller"


def _load_prompt(name: str) -> str:
    """Load a labeller prompt from the prompts/annotator/labeller/ directory."""
    path = PROMPTS_DIR / f"{name}.txt"
    logger.info("Loading labeller prompt: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

VALID_LABELS = {"effective", "partial", "ineffective"}
VALID_LABELS_BINARY = {"effective", "ineffective"}
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


def load_labeller_templates(labeller_cfg: str | dict) -> dict[str | None, str]:
    """Resolve the `annotator.labeller` config value to {annotation_type: template}.

    If `labeller_cfg` is a string (e.g. "classify_v2"), returns
    {None: <template>} -- the None key is the fallback used for every type.

    If it's a dict (e.g. {"scaffolding": "classify_scaffolding", ...}), returns
    one entry per type.
    """
    if isinstance(labeller_cfg, dict):
        return {ann_type: _load_prompt(name) for ann_type, name in labeller_cfg.items()}
    return {None: _load_prompt(labeller_cfg)}


def pick_template(templates: dict[str | None, str], annotation_type: str) -> str:
    """Pick the per-type template; fall back to None key if type is unmapped."""
    if annotation_type in templates:
        return templates[annotation_type]
    if None in templates:
        return templates[None]
    raise KeyError(
        f"No labeller template for annotation_type={annotation_type!r}. "
        f"Available keys: {list(templates.keys())}"
    )


def run_label(version: str, model: str, mode: str, phase_cfg: dict,
              gold: bool = False, binary: bool = False,
              annotator_style: str | None = None,
              annotations_data: dict | None = None,
              profile: str | None = None,
              targets: list[str] | None = None,
              split: str = "train") -> dict:
    """Run labeling pass. Returns the labeled annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_annotate().
    """
    in_memory = annotations_data is not None
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    all_types = set(get_annotation_types())
    effective_targets = set(targets) if targets else all_types
    target_suffix = "" if effective_targets == all_types else "_" + "_".join(sorted(effective_targets))
    if gold:
        filename = f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}{target_suffix}.json"
    else:
        filename = f"annotations{profile_suffix}{style_suffix}{split_suffix}{target_suffix}.json"

    if in_memory:
        data = annotations_data
    else:
        # Prefer per-target files; merge what's found
        merged_results: dict = {}
        loaded_files: list = []
        for target in sorted(effective_targets):
            t_fname = (
                f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
                if gold else
                f"annotations{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
            )
            tdata = load_annotator_result(version, t_fname)
            if tdata is not None:
                loaded_files.append(t_fname)
                for conv_id, conv_data in tdata.get("results", {}).items():
                    if conv_id not in merged_results:
                        merged_results[conv_id] = {**conv_data, "annotations": list(conv_data.get("annotations", []))}
                    else:
                        merged_results[conv_id]["annotations"].extend(conv_data.get("annotations", []))

        if merged_results:
            base = load_annotator_result(version, loaded_files[0])
            data = {**base, "results": merged_results}
            print(f"Loaded: {', '.join(loaded_files)}")
        else:
            # Fall back to combined file
            data = load_annotator_result(version, filename)

        if data is None:
            logger.error("%s not found for version %s. Run annotate first.", filename, version)
            return None

    # The transcript UUID is the last _-delimited segment of the compound conv_id.
    # Filter to the requested split as a safety net (detect/annotate already filter upstream).
    split_ids = load_split_ids(split)
    results = {
        conv_id: conv_data
        for conv_id, conv_data in data["results"].items()
        if conv_id.rsplit("_", 1)[-1] in split_ids
    }

    # Load templates once. Binary mode is a single shared template; the
    # 3-way labeller may route per annotation_type.
    if binary:
        templates = {None: _load_prompt("classify_binary")}
    else:
        templates = load_labeller_templates(get_annotator_defaults()["labeller"])

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
                prompt = templates[None].replace("{result_text}", result_text)
            else:
                annotation_type = ann.get("annotation_type", "unknown")
                template = pick_template(templates, annotation_type)
                prompt = (template
                          .replace("{annotation_type}", annotation_type)
                          .replace("{situation}", situation)
                          .replace("{action}", action)
                          .replace("{result_text}", result_text))
            key = f"{conv_id}__{idx}"
            entries.append(build_batch_entry(key, prompt, json_mode=False))
            locations.append((conv_id, idx))

    logger.info("Annotations to label: %d (%d skipped as junk)", len(entries), len(skipped))
    logger.info("Model: %s | Mode: %s", model, mode)

    for conv_id, idx in skipped:
        results[conv_id]["annotations"][idx]["effectiveness"] = "unclear"

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        jsonl_path = str(output_dir / f"label_requests{profile_suffix}.jsonl")
        write_jsonl(entries, jsonl_path)

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, json_mode=False, display_name="label", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0),
                       reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        logger.info("Running %d entries in sync mode...", len(entries))
        raw = run_sync_entries(client, entries, json_mode=False)

    valid = VALID_LABELS_BINARY if binary else VALID_LABELS
    by_label = {"effective": 0, "ineffective": 0, "unclear": len(skipped)}
    if not binary:
        by_label["partial"] = 0
    errors = 0
    total_input = 0
    total_output = 0

    for conv_id, idx in locations:
        key = f"{conv_id}__{idx}"
        entry = raw.get(key, {})

        if "error" in entry or not entry.get("text"):
            label = "unclear"
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            label = entry["text"].strip().lower().rstrip(".")
            if label not in valid:
                label = "unclear"

        results[conv_id]["annotations"][idx]["effectiveness"] = label
        by_label[label] += 1

    data["labeled"] = True
    data["label_stats"] = by_label
    data["token_summary"] = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "errors": errors,
    }

    for target in sorted(effective_targets):
        t_fname = (
            f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
            if gold else
            f"annotations{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
        )
        target_results = {
            conv_id: {
                **conv_data,
                "annotations": [a for a in conv_data.get("annotations", []) if a.get("annotation_type") == target],
            }
            for conv_id, conv_data in data["results"].items()
        }
        if not any(r["annotations"] for r in target_results.values()):
            continue
        save_annotator_result(version, t_fname, {**data, "results": target_results})
        n = sum(len(r["annotations"]) for r in target_results.values())
        logger.info(f"\nSaved: {t_fname} | {n} annotations")

    logger.info(f"\n  Effective:   {by_label['effective']}")

    if not binary:
        logger.info("  Partial:     %d", by_label['partial'])
    logger.info("  Ineffective: %d", by_label['ineffective'])
    logger.info("  Unclear:     %d", by_label['unclear'])
    logger.info("  Tokens: %s", f"{total_input + total_output:,}")
    if errors:
        logger.warning("  Errors: %d", errors)

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
    parser.add_argument("--target", nargs="+", choices=get_annotation_types(),
                        default=None,
                        help="Annotation targets (must match what was passed to annotate)")
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

    output = run_label(version=version, model=model, mode=mode,
                       phase_cfg=phase_cfg, gold=args.gold,
                       binary=args.binary, annotator_style=style,
                       profile=profile, targets=args.target,
                       split=args.split)
    if output:
        mode_hint = " --mode annotations" if args.gold else ""
        style_flag = f" --annotator-style {style}" if style else ""
        profile_flag = f" --profile {profile}" if profile else ""
        split_flag = f" --split {args.split}" if args.split != "train" else ""
        logger.info(f"\nNext: python -m annotator.eval.eval --version {version}{profile_flag}{mode_hint}{style_flag}{split_flag}")


if __name__ == "__main__":
    main()
