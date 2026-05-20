"""
Pass 2 -- Annotate key moments with situation/action/result.

Reads detected moments (from detect.py output OR gold truth) and sends
focused excerpts to Gemini batch API for detailed analysis.

Usage:
    # Annotate moments found by detect.py
    python -m annotator.core.annotate --version v1

    # Annotate gold truth moments (explanations mode)
    python -m annotator.core.annotate --version v1 --gold

    # Custom context window
    python -m annotator.core.annotate --version v1 --context 30

    # Run on test split
    python -m annotator.core.annotate --version v1 --split test
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_valid_styles, get_annotation_types
from .storage import (
    load_all_transcripts, load_annotator_result, save_annotator_result,
    annotator_result_exists, get_annotator_result_path,
)
from .utils import format_excerpt, load_ground_truth, load_split_ids

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "annotator"

VALID_TARGETS = get_annotation_types()
VALID_ANNOTATION_TYPES = set(get_annotation_types())
VALID_ANNOTATOR_STYLES = get_valid_styles()


def load_conversations_map(split: str = "train") -> dict[str, dict]:
    """Load split transcripts as {conv_id: conversation} via storage layer."""
    split_ids = load_split_ids(split)
    return {
        conv_id: conv
        for conv_id, conv in load_all_transcripts().items()
        if conv.get("transcript_id", "") in split_ids
    }


def load_detections_from_version(version: str) -> dict[str, dict] | None:
    """Load detections from detect.py output via storage layer."""
    data = load_annotator_result(version, "detections.json")
    if data is None:
        return None
    return data["results"]


def load_gold_moments(targets: list[str],
                      annotator_style: str | None = None,
                      split: str = "train") -> dict[str, dict]:
    """Load split ground truth moments as detection-like dicts.

    Converts gold annotations into the same format as detect.py output
    so the rest of the pipeline works identically.
    """
    ground_truth = load_ground_truth(annotator_style=annotator_style)
    split_ids = load_split_ids(split)

    # Ground truth keys by UUID (transcript_id), but conversations_map keys by
    # the full conversation_id (e.g. {tutor_id}_{student_id}_{UUID}). Build a
    # mapping so detections use the same key as conversations_map.
    transcript_id_to_conv_id = {
        conv.get("transcript_id", ""): conv_id
        for conv_id, conv in load_all_transcripts().items()
        if conv.get("transcript_id")
    }

    detections_by_conv = {}
    for gt_id, conv_data in ground_truth.get("conversations", {}).items():
        if gt_id not in split_ids:
            continue
        conv_id = transcript_id_to_conv_id.get(gt_id, gt_id)

        # Deduplicate by (turn_start, turn_end, annotation_type), keeping
        # the longest situation text across all annotators who labeled that moment.
        best: dict[tuple, dict] = {}
        for moment in conv_data.get("key_moments", []):
            ann_type = moment.get("annotation_type", "")
            if ann_type not in targets:
                continue
            key = (moment["turn_start"], moment["turn_end"], ann_type)
            situation = moment.get("situation", "")
            if key not in best or len(situation) > len(best[key]["situation"]):
                best[key] = {
                    "turn_start": moment["turn_start"],
                    "turn_end": moment["turn_end"],
                    "annotation_type": ann_type,
                    "situation": situation or "Human-identified moment",
                }

        if best:
            detections_by_conv[conv_id] = {
                "detections": list(best.values()),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

    return detections_by_conv


def load_prompt(version: str, target: str) -> str:
    """Load a Pass 2 analysis prompt template."""
    for ext in ("md", "txt"):
        path = PROMPTS_DIR / version / "p2" / f"{target}.{ext}"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"Prompt not found: {PROMPTS_DIR / version / 'p2' / target}.{{md,txt}}"
    )



def build_analysis_entries(detections_by_conv: dict, conversations_map: dict,
                           context_window: int, version: str,
                           dialogue_only: bool = False,
                           annotator_style: str | None = None) -> list[dict]:
    """Build batch entries for analysis.

    annotator_style is accepted for API compatibility but NOT injected into
    prompts. Style calibration is achieved by iterating the prompt against
    archetype-filtered ground truth, not by injecting style text.
    """
    prompt_cache = {}
    entries = []

    for conv_id, conv_data in detections_by_conv.items():
        conversation = conversations_map.get(conv_id)
        if not conversation:
            logger.warning("No transcript found for %s, skipping", conv_id)
            continue

        for idx, det in enumerate(conv_data.get("detections", [])):
            ann_type = det.get("annotation_type", "scaffolding")
            if ann_type not in VALID_ANNOTATION_TYPES:
                ann_type = "scaffolding"

            turn_start = det.get("turn_start", 0)
            turn_end = det.get("turn_end", turn_start)
            situation = det.get("situation", "") or det.get("brief_description", "")

            if ann_type not in prompt_cache:
                prompt_cache[ann_type] = load_prompt(version, ann_type)

            excerpt = format_excerpt(
                conversation, turn_start, turn_end,
                context_before=context_window, context_after=context_window,
                dialogue_only=dialogue_only
            )

            prompt = prompt_cache[ann_type]
            prompt = prompt.replace("{annotator_style}", "")
            prompt = prompt.replace("{situation}", situation)
            prompt = prompt.replace("{excerpt}", excerpt)
            prompt = prompt.replace("{turn_start}", str(turn_start))
            prompt = prompt.replace("{turn_end}", str(turn_end))

            key = f"{conv_id}__{ann_type}__{idx}"
            entries.append(build_batch_entry(key, prompt))

    return entries


def parse_and_merge(raw_entries: dict, detections_by_conv: dict) -> dict[str, dict]:
    """Parse batch results and merge with detections into final annotations.

    Ported from archive_per_annotator/pipeline/pass2_analyze.py
    """
    # Parse raw results
    analyses = {}
    errors = []

    for key, data in raw_entries.items():
        if "error" in data:
            errors.append({"key": key, "error": data["error"]})
            continue

        text = data.get("text", "")
        if not text:
            errors.append({"key": key, "error": "Empty response"})
            continue

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            parsed["_usage"] = data.get("usage", {})
            analyses[key] = parsed
        except json.JSONDecodeError as e:
            errors.append({"key": key, "error": f"JSON parse error: {e}", "raw": text[:500]})

    if errors:
        logger.warning("Parse errors: %d", len(errors))
        for err in errors[:5]:
            logger.warning("  %s: %s", err["key"], err["error"])

    # Merge into final results
    results = {}

    for conv_id, conv_data in detections_by_conv.items():
        detections = conv_data.get("detections", [])
        p1_usage = conv_data.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        total_usage = dict(p1_usage)

        annotations = []
        for idx, det in enumerate(detections):
            ann_type = det.get("annotation_type", "scaffolding")
            key = f"{conv_id}__{ann_type}__{idx}"

            if key in analyses:
                a = analyses[key]
                annotations.append({
                    "annotation_type": a.get("annotation_type", ann_type),
                    "turn_start": a.get("turn_start", det.get("turn_start")),
                    "turn_end": a.get("turn_end", det.get("turn_end")),
                    "situation": a.get("situation", "") or det.get("situation", "") or det.get("brief_description", ""),
                    "action": a.get("action", ""),
                    "result": a.get("result", ""),
                })

                p2_usage = a.get("_usage", {})
                for field in ("input_tokens", "output_tokens", "total_tokens"):
                    total_usage[field] += p2_usage.get(field, 0)
            else:
                annotations.append({
                    "annotation_type": ann_type,
                    "turn_start": det.get("turn_start", 0),
                    "turn_end": det.get("turn_end", 0),
                    "situation": det.get("situation", ""),
                    "action": "[Analysis unavailable -- batch failed for this moment]",
                    "result": "",
                })

        results[conv_id] = {
            "conversation_id": conv_id,
            "annotations": annotations,
            "usage": total_usage,
            "pass1_detections": len(detections),
            "pass2_analyzed": sum(
                1 for i, d in enumerate(detections)
                if f"{conv_id}__{d.get('annotation_type', 'scaffolding')}__{i}" in analyses
            ),
        }

    return results


def run_annotate(version: str, model: str, mode: str, prompt_version: str,
                 targets: list[str], phase_cfg: dict,
                 dialogue_only: bool = False, context_window: int = 20,
                 gold: bool = False, annotator_style: str | None = None,
                 detections_by_conv: dict | None = None,
                 dry_run: bool = False,
                 profile: str | None = None,
                 split: str = "train") -> dict:
    """Run annotation pass. Returns the full output dict (with 'results' key).

    If detections_by_conv is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_detect().

    If dry_run is True, loads data and builds all entries but stops before
    any API call. Writes annotate_requests.jsonl and prints the first prompt.
    """
    output_dir = get_annotator_result_path(version)

    conversations_map = load_conversations_map(split=split)
    logger.info("Loaded %d transcripts", len(conversations_map))

    if detections_by_conv is None:
        if gold:
            logger.info("Using gold truth moments")
            detections_by_conv = load_gold_moments(targets, annotator_style=annotator_style, split=split)
        else:
            detections_by_conv = load_detections_from_version(version)
            if detections_by_conv is None:
                logger.error("detections.json not found for version %s. Run detect first, or use --gold.", version)
                return None
            logger.info("Loaded detections for version %s", version)

    total_moments = sum(len(d["detections"]) for d in detections_by_conv.values())
    logger.info("Moments to annotate: %d across %d conversations", total_moments, len(detections_by_conv))
    style_str = f" | Style: {annotator_style}" if annotator_style else ""
    logger.info("Model: %s | Mode: %s | Context: +/-%d turns%s", model, mode, context_window, style_str)

    client = ModelClient(model)

    enrichment_str = "dialogue only" if dialogue_only else "enriched (all turns)"
    logger.info("Transcript mode: %s", enrichment_str)
    entries = build_analysis_entries(
        detections_by_conv, conversations_map, context_window, prompt_version,
        dialogue_only=dialogue_only, annotator_style=annotator_style
    )
    profile_suffix = f"_{profile}" if profile else ""
    jsonl_path = str(output_dir / f"annotate_requests{profile_suffix}.jsonl")
    write_jsonl(entries, jsonl_path)
    logger.info("Wrote %d analysis entries", len(entries))

    if dry_run:
        logger.info("DRY RUN: stopping before API call. Requests written to %s", jsonl_path)
        return None

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, display_name="annotate", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0),
                       reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        logger.info("Running %d entries in sync mode...", len(entries))
        raw = run_sync_entries(client, entries)
    results = parse_and_merge(raw, detections_by_conv)

    total_annotations = sum(len(r["annotations"]) for r in results.values())
    total_input = sum(r["usage"]["input_tokens"] for r in results.values())
    total_output = sum(r["usage"]["output_tokens"] for r in results.values())
    error_count = sum(
        1 for r in results.values()
        if any(a.get("action", "").startswith("[Analysis unavailable")
               for a in r["annotations"])
    )

    output = {
        "version": version,
        "model": model,
        "source": "gold_truth" if gold else "detections",
        "annotator_style": annotator_style,
        "targets": targets,
        "thinking": phase_cfg.get("thinking", False),
        "thinking_budget": phase_cfg.get("thinking_budget", 0),
        "total_conversations": len(results),
        "total_annotations": total_annotations,
        "results": results,
        "token_summary": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": error_count,
        },
    }

    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    all_types = set(get_annotation_types())
    target_suffix = "" if set(targets) == all_types else "_" + "_".join(sorted(targets))
    if gold:
        filename = f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}{target_suffix}.json"
    else:
        filename = f"annotations{profile_suffix}{style_suffix}{split_suffix}{target_suffix}.json"
    save_annotator_result(version, filename, output)

    logger.info("Saved: %s (version: %s)", filename, version)
    logger.info("  %d annotations across %d conversations", total_annotations, len(results))
    logger.info("  Errors: %d", error_count)
    logger.info("  Tokens: %s", f"{total_input + total_output:,}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Pass 2: Annotate key moments")
    parser.add_argument("--version", default=None,
                        help="Results version (e.g. v1, v2). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--gold", action="store_true",
                        help="Use gold truth moments instead of detect.py output")
    parser.add_argument("--target", nargs="+", choices=VALID_TARGETS,
                        default=VALID_TARGETS,
                        help="Annotation targets to process")
    parser.add_argument("--context", type=int, default=None,
                        help="Context window (turns before/after) for excerpts")
    parser.add_argument("--dialogue-only", action="store_true",
                        help="Exclude non-dialogue turns (enrichments) from excerpts")
    parser.add_argument("--prompt-version", default=None,
                        help="Prompt version to use (defaults to --version)")
    parser.add_argument("--annotator-style", "--style", choices=VALID_ANNOTATOR_STYLES,
                        default=None, dest="annotator_style",
                        help="Annotator archetype to simulate (generous/balanced/demanding)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and write all entries but stop before any API call")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    args = parser.parse_args()

    from common.logging_setup import setup_logging
    setup_logging()

    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.annotator_style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]

    phase_cfg = get_phase_config("annotate", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")
    context_window = args.context if args.context is not None else phase_cfg.get("context_window", 20)

    # When style is set, override prompt version to per-style profiles
    if style and not args.prompt_version:
        prompt_version = f"profiles/{style}"

    output = run_annotate(version=version, model=model, mode=mode,
                          prompt_version=prompt_version, targets=args.target,
                          phase_cfg=phase_cfg, dialogue_only=args.dialogue_only,
                          context_window=context_window, gold=args.gold,
                          annotator_style=style, dry_run=args.dry_run,
                          profile=profile, split=args.split)
    if output:
        gold_flag = " --gold" if args.gold else ""
        style_flag = f" --annotator-style {style}" if style else ""
        logger.info("Next: python -m annotator.core.label --version %s%s%s", version, gold_flag, style_flag)


if __name__ == "__main__":
    main()
