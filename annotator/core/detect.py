"""
Pass 1 -- Key moment detection.

Reads transcripts, sends full transcripts to Gemini batch API,
outputs detected moments (turn ranges + brief descriptions).

Usage:
    python -m annotator.core.detect --version v1
    python -m annotator.core.detect --version v1 --test 3
    python -m annotator.core.detect --version v1 --target scaffolding
    python -m annotator.core.detect --version v1 --split test
"""

import argparse
import datetime
import json
from pathlib import Path

from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_valid_styles, get_annotation_types
from .storage import load_all_transcripts, save_annotator_result, get_annotator_result_path
from .utils import format_transcript, load_split_ids

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "annotator"

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
                            version: str, dialogue_only: bool = False) -> list[dict]:
    """Build batch entries for detection.

    Ported from archive_per_annotator/pipeline/pass1_detect.py
    """
    prompt_cache = {}
    entries = []

    for conv in conversations:
        conv_id = conv["conversation_id"]
        transcript_text = format_transcript(conv, dialogue_only=dialogue_only)

        for target in targets:
            if target not in prompt_cache:
                prompt_cache[target] = load_prompt(version, target)

            prompt = prompt_cache[target].replace("{transcript}", transcript_text)
            key = f"{conv_id}__{target}"
            entries.append(build_batch_entry(key, prompt))

    return entries


def parse_detection_results(raw_entries: dict) -> dict[str, dict]:
    """Parse batch results into detections per conversation.

    Ported from archive_per_annotator/pipeline/pass1_detect.py
    """
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
            }

        if "error" in data:
            errors.append({"key": key, "error": data["error"]})
            continue

        text = data.get("text", "")
        if not text:
            errors.append({"key": key, "error": "Empty response"})
            continue

        try:
            parsed = json.loads(text)
            for det in parsed.get("detections", []):
                if "annotation_type" not in det:
                    det["annotation_type"] = ann_type
                if det["annotation_type"] not in VALID_ANNOTATION_TYPES:
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
        print(f"Parse errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  {err['key']}: {err['error']}")

    return detections_by_conv


def run_detect(version: str, model: str, mode: str, prompt_version: str,
               targets: list[str], phase_cfg: dict,
               test: int = 0, dialogue_only: bool = False,
               profile: str | None = None,
               annotator_style: str | None = None,
               split: str = "train") -> dict:
    """Run detection pass. Returns the full output dict (with 'results' key)."""
    output_dir = get_annotator_result_path(version)

    conversations = load_conversations(limit=test, split=split)
    if test > 0:
        print(f"TEST MODE: {test} conversations")
    print(f"Loaded {len(conversations)} conversations")
    print(f"Model: {model} | Mode: {mode} | Targets: {targets}")

    client = ModelClient(model)

    enrichment_str = "dialogue only" if dialogue_only else "enriched (all turns)"
    print(f"Transcript mode: {enrichment_str}")
    entries = build_detection_entries(conversations, targets, prompt_version,
                                     dialogue_only=dialogue_only)
    jsonl_path = str(output_dir / "detect_requests.jsonl")
    write_jsonl(entries, jsonl_path)
    print(f"Wrote {len(entries)} detection entries")

    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, display_name="detect", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0),
                       reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        print(f"Running {len(entries)} entries in sync mode...")
        raw = run_sync_entries(client, entries)
    detections_by_conv = parse_detection_results(raw)

    total_dets = sum(len(d["detections"]) for d in detections_by_conv.values())
    avg = total_dets / len(detections_by_conv) if detections_by_conv else 0

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
        "results": detections_by_conv,
    }

    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    all_types = set(get_annotation_types())
    target_suffix = "" if set(targets) == all_types else "_" + "_".join(sorted(targets))
    filename = f"detections{profile_suffix}{style_suffix}{split_suffix}{target_suffix}.json"
    save_annotator_result(version, filename, output)

    print(f"\nSaved: {filename} (version: {version})")
    print(f"  {total_dets} detections across {len(detections_by_conv)} conversations (avg {avg:.1f}/conv)")

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
                        split=args.split)
    print(f"\nNext: python -m annotator.core.annotate --version {version}")


if __name__ == "__main__":
    main()
