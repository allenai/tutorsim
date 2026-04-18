"""
Pass 2 -- Annotate key moments with situation/action/result.

Reads detected moments (from detect.py output OR gold truth) and sends
focused excerpts to Gemini batch API for detailed analysis.

Usage:
    # Annotate moments found by detect.py
    python -m pipeline.core.annotate --version v1

    # Annotate gold truth moments (explanations mode)
    python -m pipeline.core.annotate --version v1 --gold

    # Custom context window
    python -m pipeline.core.annotate --version v1 --context 30
"""

import argparse
import datetime
import json
from pathlib import Path

from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config
from .storage import (
    load_all_transcripts, load_annotator_result, save_annotator_result,
    annotator_result_exists, get_annotator_result_path,
)
from .utils import format_excerpt, load_ground_truth

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "annotator"

VALID_TARGETS = ["scaffolding", "rapport"]
VALID_ANNOTATION_TYPES = {"scaffolding", "rapport"}
VALID_ANNOTATOR_STYLES = ["generous", "balanced", "demanding"]


def load_conversations_map() -> dict[str, dict]:
    """Load transcripts as {conv_id: conversation} via storage layer."""
    return load_all_transcripts()


def load_detections_from_version(version: str) -> dict[str, dict] | None:
    """Load detections from detect.py output via storage layer."""
    data = load_annotator_result(version, "detections.json")
    if data is None:
        return None
    return data["results"]


def load_gold_moments(targets: list[str],
                      annotator_style: str | None = None) -> dict[str, dict]:
    """Load ground truth moments as detection-like dicts.

    Converts gold annotations into the same format as detect.py output
    so the rest of the pipeline works identically.
    """
    ground_truth = load_ground_truth(annotator_style=annotator_style)

    detections_by_conv = {}
    for conv_id, conv_data in ground_truth.get("conversations", {}).items():
        dets = []
        for moment in conv_data.get("key_moments", []):
            ann_type = moment.get("annotation_type", "")
            if ann_type not in targets:
                continue
            dets.append({
                "turn_start": moment["turn_start"],
                "turn_end": moment["turn_end"],
                "annotation_type": ann_type,
                "brief_description": moment.get("situation", "Human-identified moment"),
            })
        if dets:
            detections_by_conv[conv_id] = {
                "detections": dets,
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
            print(f"WARNING: No transcript found for {conv_id}, skipping")
            continue

        for idx, det in enumerate(conv_data.get("detections", [])):
            ann_type = det.get("annotation_type", "scaffolding")
            if ann_type not in VALID_ANNOTATION_TYPES:
                ann_type = "scaffolding"

            turn_start = det.get("turn_start", 0)
            turn_end = det.get("turn_end", turn_start)
            brief_desc = det.get("brief_description", "")

            if ann_type not in prompt_cache:
                prompt_cache[ann_type] = load_prompt(version, ann_type)

            excerpt = format_excerpt(
                conversation, turn_start, turn_end,
                context_before=context_window, context_after=context_window,
                dialogue_only=dialogue_only
            )

            prompt = prompt_cache[ann_type]
            prompt = prompt.replace("{annotator_style}", "")
            prompt = prompt.replace("{brief_description}", brief_desc)
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
        print(f"Parse errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  {err['key']}: {err['error']}")

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
                    "situation": a.get("situation", ""),
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
                    "situation": det.get("brief_description", ""),
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
                 detections_by_conv: dict | None = None) -> dict:
    """Run annotation pass. Returns the full output dict (with 'results' key).

    If detections_by_conv is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_detect().
    """
    output_dir = get_annotator_result_path(version)

    conversations_map = load_conversations_map()
    print(f"Loaded {len(conversations_map)} transcripts")

    if detections_by_conv is None:
        if gold:
            print("Using gold truth moments")
            detections_by_conv = load_gold_moments(targets, annotator_style=annotator_style)
        else:
            detections_by_conv = load_detections_from_version(version)
            if detections_by_conv is None:
                print(f"ERROR: detections.json not found for version {version}. Run detect first, or use --gold.")
                return None
            print(f"Loaded detections for version {version}")

    total_moments = sum(len(d["detections"]) for d in detections_by_conv.values())
    print(f"Moments to annotate: {total_moments} across {len(detections_by_conv)} conversations")
    style_str = f" | Style: {annotator_style}" if annotator_style else ""
    print(f"Model: {model} | Mode: {mode} | Context: +/-{context_window} turns{style_str}")

    client = ModelClient(model)

    enrichment_str = "dialogue only" if dialogue_only else "enriched (all turns)"
    print(f"Transcript mode: {enrichment_str}")
    entries = build_analysis_entries(
        detections_by_conv, conversations_map, context_window, prompt_version,
        dialogue_only=dialogue_only, annotator_style=annotator_style
    )
    jsonl_path = str(output_dir / "annotate_requests.jsonl")
    write_jsonl(entries, jsonl_path)
    print(f"Wrote {len(entries)} analysis entries")

    if mode == "batch":
        poll_interval = phase_cfg.get("poll_interval", 30)
        raw = run_batch(client, entries, display_name="annotate", poll_interval=poll_interval,
                       thinking=phase_cfg.get("thinking", False),
                       thinking_budget=phase_cfg.get("thinking_budget", 0),
                       reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        print(f"Running {len(entries)} entries in sync mode...")
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
    if gold:
        filename = f"annotations_gold{style_suffix}.json"
    else:
        filename = f"annotations{style_suffix}.json"
    save_annotator_result(version, filename, output)

    print(f"\nSaved: {filename} (version: {version})")
    print(f"  {total_annotations} annotations across {len(results)} conversations")
    print(f"  Errors: {error_count}")
    print(f"  Tokens: {total_input + total_output:,}")

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
    args = parser.parse_args()

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
                          annotator_style=style)
    if output:
        gold_flag = " --gold" if args.gold else ""
        style_flag = f" --annotator-style {style}" if style else ""
        print(f"\nNext: python -m annotator.core.label --version {version}{gold_flag}{style_flag}")


if __name__ == "__main__":
    main()
