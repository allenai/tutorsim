"""
Pass 2 -- Annotate key moments with situation/action/result.

Reads detected moments (from detect.py output OR gold truth) and sends
focused excerpts to the model provider's batch API for detailed analysis.

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
import hashlib
import json
import logging
from pathlib import Path

from common.logging_setup import setup_logging
from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_valid_styles, get_annotation_types
from .storage import (
    load_all_transcripts, load_annotator_result, save_annotator_result,
    annotator_result_exists, get_annotator_result_path,
    save_annotator_shard, list_annotator_shard_ids, load_annotator_shards,
    save_inflight_batch, load_inflight_batch, clear_inflight_batch,
)
from .utils import format_excerpt, load_ground_truth, load_split_ids

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "annotator"


def _entries_keys_hash(entries: list[dict]) -> str:
    """Stable short hash of an entries list, keyed on entry order + keys.
    Mirrors detect._entries_keys_hash for in-flight batch matching."""
    joined = "\n".join(e["key"] for e in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

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


def load_detections_from_version(version: str,
                                  targets: list[str] | None = None,
                                  profile: str | None = None,
                                  annotator_style: str | None = None,
                                  split: str = "train") -> dict[str, dict] | None:
    """Load detections from detect.py output, merging per-target files."""
    effective_targets = targets if targets else get_annotation_types()
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""

    merged: dict = {}
    for target in effective_targets:
        fname = f"detections{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
        data = load_annotator_result(version, fname)
        if data is not None:
            for conv_id, conv_data in data["results"].items():
                if conv_id not in merged:
                    merged[conv_id] = {"detections": [], "usage": dict(conv_data.get("usage", {}))}
                merged[conv_id]["detections"].extend(conv_data.get("detections", []))
    if merged:
        return merged

    fallbacks = [f"detections{profile_suffix}{style_suffix}{split_suffix}.json"]
    if split == "train":
        fallbacks += [f"detections{profile_suffix}{style_suffix}.json",
                      f"detections{profile_suffix}.json",
                      "detections.json"]
    for fname in fallbacks:
        data = load_annotator_result(version, fname)
        if data is not None:
            return data["results"]
    return None


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
            if moment.get("strategy_label") == "unclear":
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
            logger.info("Loaded prompt: %s (%d chars)", path, path.stat().st_size)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"Prompt not found: {PROMPTS_DIR / version / 'p2' / target}.{{md,txt}}"
    )



def build_analysis_entries(detections_by_conv: dict, conversations_map: dict,
                           context_window: int, version: str,
                           dialogue_only: bool = False,
                           annotator_style: str | None = None,
                           with_screenshots: bool = False,
                           screenshots_by_conv: dict[str, list[dict]] | None = None) -> list[dict]:
    """Build batch entries for analysis.

    annotator_style is accepted for API compatibility but NOT injected into
    prompts. Style calibration is achieved by iterating the prompt against
    archetype-filtered ground truth, not by injecting style text.

    When with_screenshots=True, attaches per-moment images whose anchor turn
    falls inside the excerpt window (excerpt_start <= anchor_turn <= excerpt_end,
    inclusive). If screenshots_by_conv is provided, it overrides per-conv lookup
    by conv_id -- the caller has already done the loading, possibly with a
    different conv_id than the iteration key (e.g. the benchmark bridge passes
    scenario_id-keyed entries with screenshots loaded from the original conv_id).
    """
    prompt_cache = {}
    entries = []

    for conv_id, conv_data in detections_by_conv.items():
        conversation = conversations_map.get(conv_id)
        if not conversation:
            logger.warning("No transcript found for %s, skipping", conv_id)
            continue

        if screenshots_by_conv is not None:
            all_screenshots = screenshots_by_conv.get(conv_id, [])
        elif with_screenshots:
            from .screenshots import load_anchored_screenshots
            all_screenshots = load_anchored_screenshots(conv_id, conversation["turns"])
        else:
            all_screenshots = []

        turns = conversation.get("turns", [])
        min_turn = turns[0]["turn_number"] if turns else 1
        max_turn = turns[-1]["turn_number"] if turns else 1

        for idx, det in enumerate(conv_data.get("detections", [])):
            ann_type = det.get("annotation_type", "scaffolding")
            if ann_type not in VALID_ANNOTATION_TYPES:
                ann_type = "scaffolding"

            turn_start = det.get("turn_start", 0)
            turn_end = det.get("turn_end", turn_start)
            situation = det.get("situation", "") or det.get("brief_description", "")

            excerpt_start = max(min_turn, turn_start - context_window)
            excerpt_end = min(max_turn, turn_end + context_window)
            in_scope = [
                s for s in all_screenshots
                if excerpt_start <= s["anchor_turn"] <= excerpt_end
            ]
            image_paths = [s["storage_path"] for s in in_scope]

            if ann_type not in prompt_cache:
                prompt_cache[ann_type] = load_prompt(version, ann_type)

            excerpt = format_excerpt(
                conversation, turn_start, turn_end,
                context_before=context_window, context_after=context_window,
                dialogue_only=dialogue_only,
                screenshots=in_scope if in_scope else None,
            )

            prompt = prompt_cache[ann_type]
            prompt = prompt.replace("{annotator_style}", "")
            prompt = prompt.replace("{situation}", situation)
            prompt = prompt.replace("{excerpt}", excerpt)
            prompt = prompt.replace("{turn_start}", str(turn_start))
            prompt = prompt.replace("{turn_end}", str(turn_end))

            key = f"{conv_id}__{ann_type}__{idx}"
            entries.append(build_batch_entry(
                key, prompt, images=image_paths or None,
            ))

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
                    "annotation_type": ann_type,
                    "turn_start": det.get("turn_start"),
                    "turn_end": det.get("turn_end"),
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
                 split: str = "train",
                 with_screenshots: bool = False,
                 rerun: bool = False) -> dict:
    """Run annotation pass. Returns the full output dict (with 'results' key).

    If detections_by_conv is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_detect().

    If dry_run is True, loads data and builds all entries but stops before
    any API call. Writes annotate_requests{profile_suffix}.jsonl and returns None.

    Resumable: per-conv results write to shards under

    results/annotator/{version}/shards/{basename}/{conv_id}.json as they parse
    (basename = output filename without .json, e.g. "annotations_generous").
    A re-run with the same flags skips conv_ids that already have a shard.
    """
    output_dir = get_annotator_result_path(version)

    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    gold_prefix = "annotations_gold" if gold else "annotations"
    output_basename = f"{gold_prefix}{profile_suffix}{style_suffix}{split_suffix}"

    conversations_map = load_conversations_map(split=split)

    logger.info("Loaded %d transcripts", len(conversations_map))

    if detections_by_conv is None:
        if gold:
            logger.info("Using gold truth moments")
            detections_by_conv = load_gold_moments(targets, annotator_style=annotator_style, split=split)
        else:
            detections_by_conv = load_detections_from_version(
                version, targets=targets, profile=profile,
                annotator_style=annotator_style, split=split)
            if detections_by_conv is None:
                logger.error("No detections found for version %s. Run detect first, or use --gold.", version)
                return None
            logger.info("Loaded detections for version %s", version)

    total_moments = sum(len(d["detections"]) for d in detections_by_conv.values())
    logger.info("Moments to annotate: %d across %d conversations", total_moments, len(detections_by_conv))
    style_str = f" | Style: {annotator_style}" if annotator_style else ""
    logger.info("Model: %s | Mode: %s | Context: +/-%d turns%s", model, mode, context_window, style_str)

    existing_ids = set() if rerun else set(list_annotator_shard_ids(version, output_basename))
    detections_to_process = {
        cid: d for cid, d in detections_by_conv.items() if cid not in existing_ids
    }
    if rerun:
        logger.info("Rerun mode: ignoring %d existing shards, processing all %d conversations",
                    len(list_annotator_shard_ids(version, output_basename)), len(detections_to_process))
    elif existing_ids:
        logger.info("Resuming version %s/%s: %d shards already on disk, %d to process",
                    version, output_basename, len(existing_ids), len(detections_to_process))

    if detections_to_process:
        client = ModelClient(model)

        if with_screenshots:
            from .client import validate_vision_support
            validate_vision_support(model)
            logger.info("Screenshots: enabled -- vision model validated, caching ON")

        enrichment_str = "dialogue only" if dialogue_only else "enriched (all turns)"
        logger.info("Transcript mode: %s", enrichment_str)
        entries = build_analysis_entries(
            detections_to_process, conversations_map, context_window, prompt_version,
            dialogue_only=dialogue_only, annotator_style=annotator_style,
            with_screenshots=with_screenshots,
        )
        jsonl_path = str(output_dir / f"annotate_requests{profile_suffix}.jsonl")
        write_jsonl(entries, jsonl_path)
        logger.info("Wrote %d analysis entries", len(entries))

        if dry_run:
            logger.info("DRY RUN: stopping before API call. Requests written to %s", jsonl_path)
            return None

        # Per-annotation `images_seen` = images attached to that single prompt.
        # Per-conv `images_attached` = sum across this conv's prompts -- matches
        # the same field on detection shards.
        images_per_key = {
            e["key"]: len(e["request"].get("images", []))
            for e in entries
        }

        def _stamp_and_shard(conv_results: dict) -> None:
            for cid, cresult in conv_results.items():
                for i, ann in enumerate(cresult["annotations"]):
                    ann_type = ann.get("annotation_type", "scaffolding")
                    k = f"{cid}__{ann_type}__{i}"
                    ann["images_seen"] = images_per_key.get(k, 0)
                cresult["images_attached"] = sum(
                    a.get("images_seen", 0) for a in cresult["annotations"]
                )
                save_annotator_shard(version, output_basename, cid, cresult)
                logger.debug("Shard saved: %s", cid)

        if mode == "batch":
            poll_interval = phase_cfg["poll_interval"]

            inflight = load_inflight_batch(version, output_basename)
            existing_batch_id = None
            if inflight:
                expected = inflight.get("entry_keys_hash")
                actual = _entries_keys_hash(entries)
                if expected == actual:
                    existing_batch_id = inflight["batch_id"]
                    logger.info("Found in-flight %s batch %s (submitted %s). Resuming poll.",
                                output_basename, existing_batch_id,
                                inflight.get("submitted_at", "?"))
                else:
                    logger.error(
                        "In-flight %s batch sidecar exists but entry-keys hash differs "
                        "(sidecar=%s, current=%s). Detections may have changed between runs. "
                        "Delete %s/in_flight/%s.json to start a fresh batch.",
                        output_basename, expected, actual, version, output_basename,
                    )
                    raise RuntimeError("entry-keys mismatch on in-flight batch resume")

            def _record(batch_id: str) -> None:
                save_inflight_batch(version, output_basename, {
                    "provider": client.provider,
                    "model": model,
                    "batch_id": batch_id,
                    "n_entries": len(entries),
                    "entry_keys_hash": _entries_keys_hash(entries),
                    "display_name": "annotate",
                    "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })

            raw = run_batch(client, entries, display_name="annotate",
                            poll_interval=poll_interval,
                            thinking=phase_cfg.get("thinking", False),
                            thinking_budget=phase_cfg.get("thinking_budget", 0),
                            reasoning_effort=phase_cfg.get("reasoning_effort", ""),
                            enable_cache=with_screenshots,
                            existing_batch_id=existing_batch_id,
                            on_batch_created=_record)
            _stamp_and_shard(parse_and_merge(raw, detections_to_process))
            clear_inflight_batch(version, output_basename)
        else:
            # Per-conv sync: shard after each conv's entries return so a
            # ctrl-C between convs leaves valid partial state on disk.
            entries_by_conv: dict[str, list[dict]] = {}
            for e in entries:
                cid = e["key"].split("__", 1)[0]
                entries_by_conv.setdefault(cid, []).append(e)
            logger.info("Running %d convs in sync mode...", len(entries_by_conv))
            for i, (conv_id, conv_entries) in enumerate(entries_by_conv.items(), start=1):
                logger.info("Conv %d/%d: %s", i, len(entries_by_conv), conv_id)
                raw_conv = run_sync_entries(client, conv_entries)
                conv_dets = {conv_id: detections_to_process[conv_id]}
                _stamp_and_shard(parse_and_merge(raw_conv, conv_dets))
    else:
        logger.info("All conversations already have shards -- nothing to send")

    # Aggregate the monolithic JSON from the union of all shards.
    results = load_annotator_shards(version, output_basename)

    total_annotations = sum(len(r.get("annotations", [])) for r in results.values())
    total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in results.values())
    total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in results.values())
    total_images_sent = sum(r.get("images_attached", 0) for r in results.values())
    convs_with_images = sum(1 for r in results.values() if r.get("images_attached", 0) > 0)
    annotations_with_images = sum(
        1 for r in results.values()
        for a in r.get("annotations", [])
        if a.get("images_seen", 0) > 0
    )
    error_count = sum(
        1 for r in results.values()
        if any(a.get("action", "").startswith("[Analysis unavailable")
               for a in r.get("annotations", []))
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
        "with_screenshots": with_screenshots,
        "convs_with_images": convs_with_images,
        "annotations_with_images": annotations_with_images,
        "total_images_sent": total_images_sent,
        "results": results,
        "token_summary": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": error_count,
        },
    }

    if len(targets) > 1:
        logger.info("")
    for target in sorted(targets):
        target_results = {
            conv_id: {
                **conv_data,
                "annotations": [a for a in conv_data["annotations"] if a.get("annotation_type") == target],
            }
            for conv_id, conv_data in results.items()
        }
        n = sum(len(r["annotations"]) for r in target_results.values())
        target_output = {**output, "targets": [target], "total_annotations": n, "results": target_results}
        if gold:
            filename = f"annotations_gold{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
        else:
            filename = f"annotations{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
        save_annotator_result(version, filename, target_output)
        logger.info("Saved: %s | %d annotations across %d conversations", filename, n, len(target_results))

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
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Include anchored screenshots from each moment's "
                             "context window. Requires a vision-capable model.")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore existing shards and re-annotate all inputs from scratch")
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

    setup_logging(version=version)

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
                          profile=profile, split=args.split, with_screenshots=args.with_screenshots,
                          rerun=args.rerun)
    if output:
        gold_flag = " --gold" if args.gold else ""
        style_flag = f" --annotator-style {style}" if style else ""
        profile_flag = f" --profile {profile}" if profile else ""
        split_flag = f" --split {args.split}" if args.split != "train" else ""
        logger.info("Next: python -m annotator.core.label --version %s%s%s%s%s", version, profile_flag, gold_flag, style_flag, split_flag)


if __name__ == "__main__":
    main()
