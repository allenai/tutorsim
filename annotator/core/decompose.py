"""
Decompose -- Break action and result annotations into atomic facets.

Reads annotations (from annotate.py output or gold truth) and decomposes each
annotation's action and result fields into lists of short, standalone, atomic
statements, using:
  - prompts/annotator/decomposer/decompose_action.md  (for action field)
  - prompts/annotator/decomposer/decompose_result.md  (for result field)

For the scaffolding target it additionally extracts spans suggesting the tutor
over-scaffolded, using:
  - prompts/annotator/decomposer/decompose_overscaffold.md  (situation+action+result)

Adds action_decomposed and result_decomposed (and, for scaffolding,
overscaffold_decomposed) (list[str]) to each annotation, then saves to
decomposed_{target}.json.

Usage:
    python -m annotator.core.decompose --version v13
    python -m annotator.core.decompose --version v13 --gold
    python -m annotator.core.decompose --version v13 --split test
    python -m annotator.core.decompose --version v13 --target rapport
    python -m annotator.core.decompose --version v13 --gold --profile anthropic --target scaffolding --split train
    # backfill only overscaffold_decomposed, reusing existing action/result facets:
    python -m annotator.core.decompose --version v13 --gold --profile anthropic --target scaffolding --only-overscaffold
"""

import argparse
import json
import logging
import re
import sys
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
from .utils import load_split_ids, JUNK_TEXTS  # JUNK_TEXTS re-exported for data/build_ground_truth.py

logger = logging.getLogger(__name__)

PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "annotator" / "decomposer"
)


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    logger.info("Loading decomposer prompt: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _coerce_facets(parsed: object) -> list[str] | None:
    """Coerce a parsed JSON value into a list of facet strings.

    Returns the facet list, or None if the value isn't a recognizable facet
    container (signals a parse failure to the caller).

    Handles two shapes:
      - a bare array (Gemini/Anthropic honor the prompt's requested format), and
      - an object, because OpenAI's response_format={"type": "json_object"}
        cannot emit a top-level array. The model wraps the facets either under a
        key whose value is the list (e.g. {"facets": [...]}) or, when it has no
        list to hand, crams them across the object's keys and values
        (e.g. {"facet a": "facet b", ...}).
    """
    if isinstance(parsed, list):
        return [str(s) for s in parsed]

    if isinstance(parsed, dict):
        # Prefer the wrapper shape: a list-valued key holds the facets (e.g.
        # {"facets": [...]} or {"spans": [...]}). An empty list value
        # ({"spans": []}) is a real empty result -- return [], do NOT fall through
        # to the cram path, or the key and "[]" come back as two bogus facets.
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if list_values:
            return [str(s) for v in list_values for s in v]
        # No list value anywhere: facets were crammed across keys and values.
        # Interleave to preserve each pair's order.
        facets: list[str] = []
        for k, v in parsed.items():
            facets.append(str(k))
            facets.append(str(v))
        return facets

    return None


def _input_prefix(only_overscaffold: bool, gold: bool) -> str:
    """Filename prefix for the decompose input.

    A normal run reads raw annotations. An only-overscaffold run reads the
    already-decomposed file instead, so the existing action_decomposed /
    result_decomposed facets are carried through to the output untouched.
    """
    if only_overscaffold:
        return "decomposed_gold" if gold else "decomposed"
    return "annotations_gold" if gold else "annotations"


def _result_filename(prefix: str, target: str, profile: str | None,
                     style: str | None, split: str) -> str:
    """Assemble a decompose input/output filename from its suffix parts."""
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{style}" if style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    return f"{prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"


def _build_overscaffold_prompt(situation: str, action: str, result_text: str,
                               template: str) -> str | None:
    """Build the over-scaffolding decomposition prompt for one annotation.

    The prompt asks for spans (in situation/action/result) that suggest the
    tutor over-scaffolded. Returns None when both action and result are junk --
    there is no described tutor behavior or outcome to analyze, so we skip the
    API call (the caller writes an empty facet list instead).
    """
    if (action.strip().lower() in JUNK_TEXTS
            and result_text.strip().lower() in JUNK_TEXTS):
        return None
    return (template
            .replace("{situation}", situation)
            .replace("{action}", action)
            .replace("{result}", result_text))


def _parse_decomposed(text: str) -> tuple[list[str], bool]:
    """Parse facet strings from model output.

    Returns (facets list, had_error). Accepts a bare JSON array or an object
    wrapper (see _coerce_facets), and falls back to regex array extraction if
    json.loads fails.
    """
    # Attempt 1: standard JSON parse
    try:
        facets = _coerce_facets(json.loads(text))
        if facets is not None:
            return facets, False
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


def _assign_facets(raw: dict, locations: list[tuple[str, int]], prefix: str,
                   field: str, results: dict) -> tuple[int, int, int, int]:
    """Parse model output for each location and write `field` onto the matching
    annotation in `results`.

    `prefix` is the request-key namespace (e.g. "action") and also labels parse
    warnings. Returns (input_tokens, output_tokens, errors, total_facets).
    """
    in_tok = out_tok = errors = total_facets = 0
    for conv_id, idx in locations:
        key = f"{prefix}__{conv_id}__{idx}"
        entry = raw.get(key, {})
        if "error" in entry or not entry.get("text"):
            facets = []
            errors += 1
        else:
            usage = entry.get("usage", {})
            in_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
            facets, had_error = _parse_decomposed(entry["text"])
            if had_error:
                logger.warning("Could not parse %s decomposition for %s: %r",
                               prefix, key, entry["text"][:200])
                errors += 1
        results[conv_id]["annotations"][idx][field] = facets
        total_facets += len(facets)
    return in_tok, out_tok, errors, total_facets


def run_decompose(version: str, model: str, mode: str, phase_cfg: dict,
                  gold: bool = False,
                  annotator_style: str | None = None,
                  annotations_data: dict | None = None,
                  profile: str | None = None,
                  target: str = "scaffolding",
                  split: str = "train",
                  only_overscaffold: bool = False,
                  dry_run: bool = False) -> dict | None:
    """Run decomposition pass. Returns enriched annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_annotate().

    Adds action_decomposed and result_decomposed (list[str]) to each
    annotation of the target type, then saves to decomposed_{target}.json.

    only_overscaffold runs just the over-scaffold pass: it reads the existing
    decomposed_{target}.json, leaves action_decomposed/result_decomposed
    untouched, (re)computes overscaffold_decomposed, and writes back. Use it to
    backfill the over-scaffold field without paying to re-decompose action and
    result. Only valid for the scaffolding target.
    """
    if only_overscaffold and target != "scaffolding":
        logger.error("--only-overscaffold is only valid for --target scaffolding "
                     "(got %s); over-scaffolding is scaffolding-specific.", target)
        return None

    in_memory = annotations_data is not None
    input_filename = _result_filename(
        _input_prefix(only_overscaffold, gold), target, profile,
        annotator_style, split)

    if in_memory:
        data = annotations_data
    else:
        data = load_annotator_result(version, input_filename)
        if data is None:
            logger.error("%s not found for version %s. Run annotate first.", input_filename, version)
            return None
        logger.info("Loaded: %s", input_filename)

    if in_memory:
        # Bridge-style invocations (e.g. benchmark.core.annotator_bridge.decompose_bulk)
        # pass scenario-keyed data that is NOT in data/split.json (scenario IDs
        # like "{conv_id}__hum_{ts}_{te}", not raw conv UUIDs). The split filter
        # below would discard all of them. Skip it for in-memory callers --
        # they're already passing exactly the subset they want decomposed.
        results = data["results"]
    else:
        split_ids = load_split_ids(split)
        results = {
            conv_id: conv_data
            for conv_id, conv_data in data["results"].items()
            if conv_id.rsplit("_", 1)[-1] in split_ids
        }

    action_template = _load_prompt("decompose_action.md")
    result_template = _load_prompt("decompose_result.md")
    # Over-scaffolding is a scaffolding-specific concept; only decompose it for
    # the scaffolding target.
    overscaffold_enabled = target == "scaffolding"
    overscaffold_template = (
        _load_prompt("decompose_overscaffold.md") if overscaffold_enabled else None
    )

    action_entries = []
    result_entries = []
    overscaffold_entries = []
    locations_action = []
    locations_result = []
    locations_overscaffold = []
    skipped_action = 0
    skipped_result = 0
    skipped_overscaffold = 0

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data["annotations"]):
            if ann.get("annotation_type", target) != target:
                continue

            situation = ann.get("situation", "")
            action = ann.get("action", "")
            result_text = ann.get("result", "")

            # only_overscaffold skips these passes entirely, leaving the
            # annotation's existing action_decomposed/result_decomposed intact.
            if not only_overscaffold:
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

            if overscaffold_enabled:
                prompt = _build_overscaffold_prompt(
                    situation, action, result_text, overscaffold_template)
                if prompt is None:
                    results[conv_id]["annotations"][idx]["overscaffold_decomposed"] = []
                    skipped_overscaffold += 1
                else:
                    key = f"overscaffold__{conv_id}__{idx}"
                    overscaffold_entries.append(
                        build_batch_entry(key, prompt, json_mode=True))
                    locations_overscaffold.append((conv_id, idx))

    entries = action_entries + result_entries + overscaffold_entries
    logger.info(
        "Action entries: %d (%d skipped) | Result entries: %d (%d skipped) | "
        "Over-scaffold entries: %d (%d skipped)",
        len(action_entries), skipped_action, len(result_entries), skipped_result,
        len(overscaffold_entries), skipped_overscaffold,
    )
    logger.info("Model: %s | Mode: %s", model, mode)

    if dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Input:          {input_filename}")
        print(f"Model:          {model}  [{mode}]")
        print(f"Action entries: {len(action_entries)}  ({skipped_action} skipped as junk)")
        print(f"Result entries: {len(result_entries)}  ({skipped_result} skipped as junk)")
        print(f"Over-scaffold entries: {len(overscaffold_entries)}  ({skipped_overscaffold} skipped as junk)")
        print(f"Total API calls: {len(entries)}")
        for label, sample_entries in [("action", action_entries[:2]), ("result", result_entries[:2]), ("overscaffold", overscaffold_entries[:2])]:
            for e in sample_entries:
                contents = e.get("request", {}).get("contents", [])
                parts = contents[0].get("parts", []) if contents else []
                text = parts[0].get("text", "") if parts else ""
                print(f"\n  [{label}] {e.get('key', '')}")
                print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        return None

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        profile_suffix = f"_{profile}" if profile else ""
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

    in_a, out_a, err_a, total_action_facets = _assign_facets(
        raw, locations_action, "action", "action_decomposed", results)
    in_r, out_r, err_r, total_result_facets = _assign_facets(
        raw, locations_result, "result", "result_decomposed", results)
    in_o, out_o, err_o, total_overscaffold_facets = _assign_facets(
        raw, locations_overscaffold, "overscaffold", "overscaffold_decomposed", results)

    total_input = in_a + in_r + in_o
    total_output = out_a + out_r + out_o
    errors = err_a + err_r + err_o

    if only_overscaffold:
        # Preserve the action/result stats and token cost from the prior run;
        # refresh only the over-scaffold pass and add its token cost.
        prev_stats = data.get("decompose_stats", {})
        decompose_stats = {
            **prev_stats,
            "overscaffold_entries": len(overscaffold_entries),
            "skipped_overscaffold": skipped_overscaffold,
            "total_overscaffold_facets": total_overscaffold_facets,
        }
        prev_tokens = data.get("token_summary", {})
        token_summary = {
            "total_input_tokens": prev_tokens.get("total_input_tokens", 0) + total_input,
            "total_output_tokens": prev_tokens.get("total_output_tokens", 0) + total_output,
            "total_tokens": prev_tokens.get("total_tokens", 0) + total_input + total_output,
            "errors": errors,
        }
    else:
        decompose_stats = {
            "action_entries": len(action_entries),
            "result_entries": len(result_entries),
            "overscaffold_entries": len(overscaffold_entries),
            "skipped_action": skipped_action,
            "skipped_result": skipped_result,
            "skipped_overscaffold": skipped_overscaffold,
            "total_action_facets": total_action_facets,
            "total_result_facets": total_result_facets,
            "total_overscaffold_facets": total_overscaffold_facets,
        }
        token_summary = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": errors,
        }

    output = {
        **data,
        "results": results,
        "decomposed": True,
        "decompose_stats": decompose_stats,
        "token_summary": token_summary,
    }

    decomposed_prefix = "decomposed_gold" if gold else "decomposed"
    output_filename = _result_filename(
        decomposed_prefix, target, profile, annotator_style, split)
    save_annotator_result(version, output_filename, output)
    logger.info("Saved: %s", output_filename)

    logger.info("  Action facets extracted: %d", total_action_facets)
    logger.info("  Result facets extracted: %d", total_result_facets)
    if overscaffold_enabled:
        logger.info("  Over-scaffold facets extracted: %d", total_overscaffold_facets)
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
    parser.add_argument("--only-overscaffold", action="store_true",
                        dest="only_overscaffold",
                        help="Only run the over-scaffold pass: read the existing "
                             "decomposed_{target}.json, keep action/result facets, "
                             "and (re)compute overscaffold_decomposed. "
                             "Scaffolding target only.")
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

    result = run_decompose(version=version, model=model, mode=mode,
                           phase_cfg=phase_cfg, gold=args.gold,
                           annotator_style=style, profile=profile,
                           target=args.target, split=args.split,
                           only_overscaffold=args.only_overscaffold,
                           dry_run=args.dry_run)
    # run_decompose returns None on missing input (a real failure) and also on
    # --dry-run (expected). Only the former should be a non-zero exit.
    if result is None and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
