#!/usr/bin/env python3
"""Temp script: decompose action and result for a sample of 100 ground truth moments.

Reads from data/ground_truth_hybrid/, picks the first N moments across conversations,
submits action + result decomposition via batch API, writes results to
data/temp_decompose_output.json.

Usage:
    python data/temp_decompose.py
    python data/temp_decompose.py --sample 50
    python data/temp_decompose.py --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from annotator.core.decompose import (
    JUNK_TEXTS as DECOMPOSE_JUNK_TEXTS,
    _load_prompt as _load_decompose_prompt,
    _parse_decomposed,
)
from annotator.core.utils import load_split_ids

GT_DIR = DATA_DIR / "ground_truth_hybrid"
OUTPUT_PATH = DATA_DIR / "temp_decompose_output.json"


def load_sample(n: int) -> list[dict]:
    """Return first n moments from train-split conversations in sorted GT files."""
    train_ids = load_split_ids("train")
    sample = []
    for f in sorted(GT_DIR.glob("*.json")):
        if len(sample) >= n:
            break
        if f.stem not in train_ids:
            continue
        d = json.load(open(f))
        conv_id = d["conversation_id"]
        for idx, m in enumerate(d["key_moments"]):
            if len(sample) >= n:
                break
            sample.append({
                "conv_id": conv_id,
                "idx": idx,
                "annotation_type": m.get("annotation_type", ""),
                "action": m.get("action", ""),
                "result": m.get("result", ""),
            })
    return sample


def run(sample_size: int, dry_run: bool):
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("label")
    model = cfg["model"]

    action_template = _load_decompose_prompt("decompose_action.md")
    result_template = _load_decompose_prompt("decompose_result.md")

    print(f"Loading {sample_size} moments from {GT_DIR}...")
    sample = load_sample(sample_size)
    print(f"Loaded {len(sample)} moments")

    entries = []
    skipped = {"action": 0, "result": 0}
    for item in sample:
        key_base = f"{item['conv_id']}__{item['idx']}"

        action = (item["action"] or "").strip()
        if action.lower() in DECOMPOSE_JUNK_TEXTS:
            item["action_decomposed"] = []
            skipped["action"] += 1
        else:
            entries.append(build_batch_entry(
                key=f"{key_base}__action",
                prompt_text=action_template.replace("{action}", action),
                json_mode=True,
            ))

        result_text = (item["result"] or "").strip()
        if result_text.lower() in DECOMPOSE_JUNK_TEXTS:
            item["result_decomposed"] = []
            skipped["result"] += 1
        else:
            entries.append(build_batch_entry(
                key=f"{key_base}__result",
                prompt_text=(result_template
                             .replace("{situation}", item.get("situation", ""))
                             .replace("{action}", item.get("action", ""))
                             .replace("{result}", result_text)),
                json_mode=True,
            ))

    print(f"Batch entries: {len(entries)}  (skipped action={skipped['action']}, result={skipped['result']})")
    print(f"Model: {model}")

    if dry_run:
        print("\nDry run — first 2 prompts:")
        for e in entries[:2]:
            body = e.get("body", e)
            req = e.get("request", {})
            contents = req.get("contents") or req.get("messages") or []
            first = contents[0] if contents else {}
            parts = first.get("parts") or []
            text = parts[0].get("text", "") if parts else first.get("content", "")
            print(f"\n  [{e['key']}]\n  {text[:300]}{'...' if len(text) > 300 else ''}")
        return

    client = ModelClient(model)
    raw = run_batch(
        client, entries,
        json_mode=True,
        display_name="temp_decompose",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=cfg.get("thinking", False),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    total_input = total_output = errors = 0
    for item in sample:
        key_base = f"{item['conv_id']}__{item['idx']}"
        for field in ("action", "result"):
            if f"{field}_decomposed" in item:
                continue  # already set (junk)
            key = f"{key_base}__{field}"
            entry = raw.get(key, {})
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            if "error" in entry or not entry.get("text"):
                print(f"  WARNING: error for {key}: {entry.get('error', 'no text')}")
                item[f"{field}_decomposed"] = []
                errors += 1
            else:
                facets, had_error = _parse_decomposed(entry["text"])
                if had_error:
                    print(f"  WARNING: parse error for {key}: {entry['text'][:80]!r}")
                    errors += 1
                item[f"{field}_decomposed"] = facets

    output = {
        "sample_size": len(sample),
        "model": model,
        "token_summary": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": errors,
        },
        "moments": sample,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDone! Written to {OUTPUT_PATH}")
    print(f"  Tokens in/out: {total_input:,} / {total_output:,}  (errors: {errors})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100, help="Number of moments to decompose")
    parser.add_argument("--dry-run", action="store_true", help="Print entries without calling API")
    args = parser.parse_args()
    run(args.sample, args.dry_run)


if __name__ == "__main__":
    main()
