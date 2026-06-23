"""Combine v2 (scaffolding) + v6 (rapport) test_v2 predictions into a per-type
hybrid, dump the errors as readable markdown with full SAR context.

Reads existing prediction files -- no LLM calls. Output is meant for human
inspection: do the errors look genuinely-ambiguous (humans would disagree too)
or fixable-by-prompt (a smarter classifier could catch them)?
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
OUT = EVAL / "test_v2_hybrid_errors.md"

V2_PREDS = EVAL / "labeller_predictions_test_v2_anthropic.jsonl"
V6_PREDS = EVAL / "labeller_predictions_test_v2_anthropic_v6.jsonl"
SAR_FILE = ROOT / "step_up_annotations.jsonl"

HUMAN_TO_LLM = {
    "effective": "effective",
    "partially_effective": "partial",
    "ineffective": "ineffective",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotation_key(transcript_id, source_annotator_id, annotation_type, ts, te):
    return f"{transcript_id}|{source_annotator_id}|{annotation_type}|{ts}|{te}"


def build_sar_lookup():
    lookup = {}
    for row in load_jsonl(SAR_FILE):
        if row.get("annotation_type") not in ("scaffolding", "rapport"):
            continue
        for ta in row.get("turn_annotations", []):
            key = annotation_key(
                row["transcript_id"], row["source_annotator_id"],
                row["annotation_type"], ta["turn_number_start"], ta["turn_number_end"],
            )
            lookup[key] = {
                "annotation_type": row["annotation_type"],
                "situation": ta.get("situation", ""),
                "action": ta.get("action", ""),
                "result": ta.get("result", ""),
            }
    return lookup


def main():
    v2 = {r["annotation_key"]: r for r in load_jsonl(V2_PREDS)}
    v6 = {r["annotation_key"]: r for r in load_jsonl(V6_PREDS)}
    sar = build_sar_lookup()

    v2_only = set(v2) - set(v6)
    v6_only = set(v6) - set(v2)
    if v2_only or v6_only:
        print(f"Warning: {len(v2_only)} v2-only keys, {len(v6_only)} v6-only keys -- excluded from analysis")

    rows = []
    for key, r2 in v2.items():
        r6 = v6.get(key)
        if not r6:
            continue
        ann_type = r2["annotation_type"]
        hybrid_pred = r2["predicted_label"] if ann_type == "scaffolding" else r6["predicted_label"]
        human = HUMAN_TO_LLM[r2["human_rating"]]
        rows.append({
            "key": key,
            "annotation_type": ann_type,
            "human_rating": r2["human_rating"],
            "human_3way": human,
            "hybrid_pred": hybrid_pred,
            "v2_pred": r2["predicted_label"],
            "v6_pred": r6["predicted_label"],
            "is_error": hybrid_pred != human,
        })

    errors = [r for r in rows if r["is_error"]]
    by_type = {"scaffolding": [], "rapport": []}
    for r in errors:
        by_type[r["annotation_type"]].append(r)

    lines = []
    lines.append(f"# Per-Type Hybrid Errors on test_v2 (n={len(rows)})\n")
    lines.append(f"Total errors: {len(errors)} | scaffolding: {len(by_type['scaffolding'])} | rapport: {len(by_type['rapport'])}\n")
    lines.append("Hybrid rule: scaffolding uses v2 (classify_v2), rapport uses v6 (unprimed Claude meta-prompt).\n")

    for ann_type, items in by_type.items():
        lines.append(f"\n## {ann_type} ({len(items)} errors)\n")
        for i, r in enumerate(items, 1):
            s = sar.get(r["key"], {})
            lines.append(f"### {ann_type} #{i}  --  human: `{r['human_3way']}`, hybrid: `{r['hybrid_pred']}`")
            if r["v2_pred"] != r["v6_pred"]:
                lines.append(f"  (v2 said `{r['v2_pred']}`, v6 said `{r['v6_pred']}` -- prompts disagreed)")
            lines.append(f"- **annotation_key:** `{r['key']}`")
            lines.append(f"- **situation:** {s.get('situation','').strip()}")
            lines.append(f"- **action:** {s.get('action','').strip()}")
            lines.append(f"- **result:** {s.get('result','').strip()}")
            lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(errors)} errors to {OUT}")


if __name__ == "__main__":
    main()
