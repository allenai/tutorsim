"""
Pass 1 -- Key moment detection.

Reads transcripts, sends full transcripts to the model provider's batch API,
outputs detected moments (turn ranges + brief descriptions).

Usage:
    python -m annotator.core.detect --version v1
    python -m annotator.core.detect --version v1 --test 3
    python -m annotator.core.detect --version v1 --target scaffolding
    python -m annotator.core.detect --version v1 --split test
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
    load_all_transcripts, save_annotator_result, get_annotator_result_path,
    save_annotator_shard, list_annotator_shard_ids, load_annotator_shards,
    save_inflight_batch, load_inflight_batch, clear_inflight_batch,
)
from .utils import format_transcript, load_split_ids

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "annotator"


def _entries_keys_hash(entries: list[dict]) -> str:
    """Stable short hash of an entries list, keyed on entry order + keys.
    Used to detect when an in-flight batch's entry set diverges from the
    current run's entry set (e.g. user added/removed transcripts mid-resume)."""
    joined = "\n".join(e["key"] for e in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

VALID_TARGETS = get_annotation_types()
VALID_ANNOTATION_TYPES = set(get_annotation_types())


def load_conversations(limit: int = 0, split: str = "train") -> list[dict]:
    """Load split transcripts via storage layer."""
    transcripts = load_all_transcripts()
    if not transcripts:
        raise FileNotFoundError(
            "No transcripts found. Ensure data/transcripts/ contains JSON files, "
            "or configure transcript paths in config.yaml under storage.paths.transcripts."
        )
    split_ids = load_split_ids(split)
    conversations = sorted(
        (c for c in transcripts.values() if c.get("transcript_id", "") in split_ids),
        key=lambda c: c.get("conversation_id", ""),
    )
    if limit > 0:
        conversations = conversations[:limit]
    return conversations


def load_prompt(version: str, target: str) -> str:
    """Load a Pass 1 detection prompt template."""
    for ext in ("md", "txt"):
        path = PROMPTS_DIR / version / "p1" / f"{target}.{ext}"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"Prompt not found: {PROMPTS_DIR / version / 'p1' / target}.{{md,txt}}"
    )


def build_detection_entries(conversations: list[dict], targets: list[str],
                            version: str, dialogue_only: bool = False,
                            with_screenshots: bool = False,
                            screenshots_by_conv: dict[str, list[dict]] | None = None) -> list[dict]:
    """Build batch entries for detection.

    When with_screenshots=True, attaches every image for the conversation.
    If screenshots_by_conv is provided, the function uses it directly instead
    of looking up by conv_id -- symmetric with build_analysis_entries.
    """
    prompt_cache = {}
    entries = []

    for conv in conversations:
        conv_id = conv["conversation_id"]

        if screenshots_by_conv is not None:
            screenshots = screenshots_by_conv.get(conv_id, [])
        elif with_screenshots:
            from .screenshots import load_anchored_screenshots
            screenshots = load_anchored_screenshots(conv_id, conv["turns"])
        else:
            screenshots = []
        image_paths = [s["storage_path"] for s in screenshots]

        transcript_text = format_transcript(
            conv, dialogue_only=dialogue_only,
            screenshots=screenshots if screenshots else None,
        )

        for target in targets:
            if target not in prompt_cache:
                prompt_cache[target] = load_prompt(version, target)

            prompt = prompt_cache[target].replace("{transcript}", transcript_text)
            key = f"{conv_id}__{target}"
            entries.append(build_batch_entry(
                key, prompt, images=image_paths or None,
            ))

    return entries


def parse_detection_results(raw_entries: dict,
                            images_per_key: dict[str, int] | None = None) -> dict[str, dict]:
    """Parse batch results into detections per conversation.

    Ported from archive_per_annotator/pipeline/pass1_detect.py
    """
    images_per_key = images_per_key or {}
    detections_by_conv = {}
    errors = []

    for key, data in raw_entries.items():
        if "__" in key:
            conv_id, ann_type = key.rsplit("__", 1)
        else:
            conv_id = key
            ann_type = "unknown"

        if conv_id not in detections_by_conv:
            detections_by_conv[conv_id] = {
                "detections": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "images_seen": 0,
                "images_attached": 0,
            }
        # images_seen = max across targets (unique images per conv).
        # images_attached = sum across targets (total API attachments).
        n_imgs = images_per_key.get(key, 0)
        detections_by_conv[conv_id]["images_seen"] = max(
            detections_by_conv[conv_id]["images_seen"], n_imgs,
        )
        detections_by_conv[conv_id]["images_attached"] += n_imgs

        if "error" in data:
            errors.append({"key": key, "error": data["error"]})
            continue

        text = data.get("text", "")
        if not text:
            errors.append({"key": key, "error": "Empty response"})
            continue

        try:
            parsed = json.loads(text)
            # Some models return a bare array instead of {"detections": [...]}
            detections = parsed if isinstance(parsed, list) else parsed.get("detections", [])
            for det in detections:
                # Normalize new compact field names to canonical names
                if "start" in det and "turn_start" not in det:
                    det["turn_start"] = det.pop("start")
                if "end" in det and "turn_end" not in det:
                    det["turn_end"] = det.pop("end")
                if "description" in det and "brief_description" not in det:
                    det["brief_description"] = det.pop("description")
                det["annotation_type"] = ann_type
                # Validate and enforce suggested_cut_turn
                sct = det.get("suggested_cut_turn")
                ts = det.get("turn_start", 0)
                te = det.get("turn_end", ts)
                lower_bound = max(1, ts - 2)
                if sct is None or not (lower_bound <= sct <= te):
                    # Missing or out of bounds -- default to turn_start - 1
                    det["suggested_cut_turn"] = max(1, ts - 1)
                detections_by_conv[conv_id]["detections"].append(det)
        except json.JSONDecodeError as e:
            errors.append({"key": key, "error": f"JSON parse error: {e}", "raw": text[:500]})

        usage = data.get("usage", {})
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            detections_by_conv[conv_id]["usage"][field] += usage.get(field, 0)

    if errors:
        logger.warning("Parse errors: %d", len(errors))
        for err in errors[:5]:
            logger.warning("  %s: %s", err["key"], err["error"])

    return detections_by_conv


def run_detect(version: str, model: str, mode: str, prompt_version: str,
               targets: list[str], phase_cfg: dict,
               test: int = 0, dialogue_only: bool = False,
               profile: str | None = None,
               annotator_style: str | None = None,
               split: str = "train", with_screenshots: bool = False,
               rerun: bool = False) -> dict:
    """Run detection pass. Returns the full output dict (with 'results' key).

    Resumable: per-conv results are written to shards under
    results/annotator/{version}/shards/{shard_namespace}/{conv_id}.json as they parse,
    where shard_namespace matches the output filename prefix (e.g. detections_anthropic_test).
    A re-run with the same flags skips conv_ids that already have a shard
    and only sends the remainder to the model. Pass rerun=True to ignore
    existing shards and reprocess (overwriting) all conv_ids; alternatively,
    delete the version directory for a clean re-run.
    """
    output_dir = get_annotator_result_path(version)

    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    shard_namespace = f"detections{profile_suffix}{style_suffix}{split_suffix}"

    conversations = load_conversations(limit=test, split=split)
    if test > 0:
        logger.info("TEST MODE: %d conversations", test)
    logger.info("Loaded %d conversations", len(conversations))
    logger.info("Model: %s | Mode: %s | Targets: %s", model, mode, targets)

    on_disk_ids = set(list_annotator_shard_ids(version, shard_namespace))
    existing_ids = set() if rerun else on_disk_ids
    to_process = [c for c in conversations if c["conversation_id"] not in existing_ids]
    if rerun and on_disk_ids:
        logger.info("Rerun mode: ignoring %d existing shards, reprocessing all %d conversations",
                    len(on_disk_ids), len(to_process))
    elif existing_ids:
        logger.info("Resuming version %s: %d shards already on disk, %d to process",
                    version, len(existing_ids), len(to_process))

    if to_process:
        client = ModelClient(model)

        if with_screenshots:
            from .client import validate_vision_support
            validate_vision_support(model)
            logger.info("Screenshots: enabled -- vision model validated")

        enrichment_str = "dialogue only" if dialogue_only else "enriched (all turns)"
        logger.info("Transcript mode: %s", enrichment_str)
        entries = build_detection_entries(to_process, targets, prompt_version,
                                          dialogue_only=dialogue_only,
                                          with_screenshots=with_screenshots)
        jsonl_path = str(output_dir / "detect_requests.jsonl")
        write_jsonl(entries, jsonl_path)
        logger.info("Wrote %d detection entries", len(entries))

        images_per_key = {
            e["key"]: len(e["request"].get("images", []))
            for e in entries
        }

        if mode == "batch":
            poll_interval = phase_cfg["poll_interval"]

            # Resume an in-flight batch if the sidecar matches our current entries.
            inflight = load_inflight_batch(version, shard_namespace)
            existing_batch_id = None
            if inflight:
                expected = inflight.get("entry_keys_hash")
                actual = _entries_keys_hash(entries)
                if expected == actual:
                    existing_batch_id = inflight["batch_id"]
                    logger.info("Found in-flight detect batch %s (submitted %s). Resuming poll.",
                                existing_batch_id, inflight.get("submitted_at", "?"))
                else:
                    logger.error(
                        "In-flight detect batch sidecar exists but entry-keys hash differs "
                        "(sidecar=%s, current=%s). Convs may have changed between runs. "
                        "Delete %s/in_flight/%s.json to start a fresh batch.",
                        expected, actual, version, shard_namespace,
                    )
                    raise RuntimeError("entry-keys mismatch on in-flight batch resume")

            def _record(batch_id: str) -> None:
                save_inflight_batch(version, shard_namespace, {
                    "provider": client.provider,
                    "model": model,
                    "batch_id": batch_id,
                    "n_entries": len(entries),
                    "entry_keys_hash": _entries_keys_hash(entries),
                    "display_name": "detect",
                    "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })

            raw = run_batch(client, entries, display_name="detect",
                            poll_interval=poll_interval,
                            thinking=phase_cfg.get("thinking", False),
                            thinking_budget=phase_cfg.get("thinking_budget", 0),
                            reasoning_effort=phase_cfg.get("reasoning_effort", ""),
                            existing_batch_id=existing_batch_id,
                            on_batch_created=_record)
            new_by_conv = parse_detection_results(raw, images_per_key=images_per_key)
            for conv_id, conv_data in new_by_conv.items():
                save_annotator_shard(version, shard_namespace, conv_id, conv_data)
                logger.debug("Shard saved: %s", conv_id)
            clear_inflight_batch(version, shard_namespace)
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
                parsed_conv = parse_detection_results(raw_conv, images_per_key=images_per_key)
                if conv_id in parsed_conv:
                    save_annotator_shard(version, shard_namespace, conv_id, parsed_conv[conv_id])
                    logger.debug("Shard saved: %s", conv_id)
    else:
        logger.info("All %d conversations already have shards -- nothing to send", len(conversations))

    # Aggregate the monolithic JSON from the union of all shards.
    detections_by_conv = load_annotator_shards(version, shard_namespace)

    total_dets = sum(len(d.get("detections", [])) for d in detections_by_conv.values())
    avg = total_dets / len(detections_by_conv) if detections_by_conv else 0
    total_images_sent = sum(d.get("images_attached", 0) for d in detections_by_conv.values())
    convs_with_images = sum(
        1 for d in detections_by_conv.values() if d.get("images_seen", 0) > 0
    )

    output = {
        "pass": "detection",
        "version": version,
        "model": model,
        "targets": targets,
        "thinking": phase_cfg.get("thinking", False),
        "thinking_budget": phase_cfg.get("thinking_budget", 0),
        "total_conversations": len(detections_by_conv),
        "total_detections": total_dets,
        "avg_detections_per_conversation": round(avg, 1),
        "with_screenshots": with_screenshots,
        "convs_with_images": convs_with_images,
        "total_images_sent": total_images_sent,
        "results": detections_by_conv,
    }

    if len(targets) > 1:
        logger.info("")
        for target in sorted(targets):
            target_results = {
                conv_id: {
                    "detections": [d for d in data["detections"] if d.get("annotation_type") == target],
                    "usage": data["usage"],
                }
                for conv_id, data in detections_by_conv.items()
            }
            n = sum(len(d["detections"]) for d in target_results.values())
            avg_t = n / len(target_results) if target_results else 0
            target_output = {**output, "targets": [target], "total_detections": n,
                             "avg_detections_per_conversation": round(avg_t, 1),
                             "results": target_results}
            filename = f"detections{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
            save_annotator_result(version, filename, target_output)
            logger.info(f"Saved: {filename} | {n} detections across {len(target_results)} conversations (avg {avg_t:.1f}/conv)")
    else:
        filename = f"detections{profile_suffix}{style_suffix}{split_suffix}_{targets[0]}.json"
        save_annotator_result(version, filename, output)
        logger.info(f"\nSaved: {filename} (version: {version})")
        logger.info(f"  {total_dets} detections across {len(detections_by_conv)} conversations (avg {avg:.1f}/conv)")

    return output


def main():
    parser = argparse.ArgumentParser(description="Pass 1: Key moment detection")
    parser.add_argument("--version", default=None,
                        help="Results version (e.g. v1, v2). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--target", nargs="+", choices=get_annotation_types(),
                        default=get_annotation_types(),
                        help="Annotation targets to detect")
    parser.add_argument("--test", type=int, default=0,
                        help="Test on N conversations (0 = all)")
    parser.add_argument("--dialogue-only", action="store_true",
                        help="Exclude non-dialogue turns (enrichments) from transcripts")
    parser.add_argument("--prompt-version", default=None,
                        help="Prompt version to use (defaults to --version)")
    parser.add_argument("--style", choices=get_valid_styles(),
                        default=None,
                        help="Use per-style detection prompts from profiles/{style}/p1/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print an example prompt without calling the API")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Include anchored screenshots in detection prompts. "
                             "Requires a vision-capable model.")
    args = parser.parse_args()

    from common.logging_setup import setup_logging
    setup_logging()

    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]

    setup_logging(version=version)

    phase_cfg = get_phase_config("detect", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    # Override prompt version when style is set and p1 prompts exist
    if style and not args.prompt_version:
        style_p1_dir = PROMPTS_DIR / "profiles" / style / "p1"
        if style_p1_dir.exists():
            prompt_version = f"profiles/{style}"

    if args.dry_run:
        conversations = load_conversations(limit=args.test or 1, split=args.split)
        entries = build_detection_entries(conversations, args.target, prompt_version,
                                         dialogue_only=args.dialogue_only)
        print(f"\n--- DRY RUN: showing first of {len(entries)} prompt(s) ---\n")
        print(entries[0]["request"]["contents"][0]["parts"][0]["text"])
        return

    output = run_detect(version=version, model=model, mode=mode,
                        prompt_version=prompt_version, targets=args.target,
                        phase_cfg=phase_cfg, test=args.test,
                        dialogue_only=args.dialogue_only,
                        profile=profile, annotator_style=style,
                        split=args.split, with_screenshots=args.with_screenshots)
    style_flag = f" --annotator-style {style}" if style else ""
    profile_flag = f" --profile {profile}" if profile else ""
    split_flag = f" --split {args.split}" if args.split != "train" else ""
    with_screenshots_flag = f" --with-screenshots" if args.with_screenshots else ""
    print(f"\nNext: python -m annotator.core.annotate --version {version}{profile_flag}{style_flag}{split_flag}{with_screenshots_flag}")
    logger.info(f"Next: python -m annotator.core.annotate --version {version}{profile_flag}{style_flag}{split_flag}{with_screenshots_flag}")


if __name__ == "__main__":
    main()
