#!/usr/bin/env python3
"""Build ground truth files from step_up_annotations.jsonl.

Reads teacher annotations from a JSONL file (default: data/teacher_annotations/step_up_annotations.jsonl).
For each conversation:
  - Reuse strategy_label for moments unchanged from the existing ground_truth file
    (matched by annotator_id + turn_start + turn_end + annotation_type + result text)
  - Classify new/changed moments via Anthropic batch API
  - Write the merged result to data/ground_truth_<labeller>/<conv_id>.json

Only scaffolding and rapport records are processed; caption records are skipped.

Output format — one JSON file per conversation:
  {
    "conversation_id": "<uuid>",
    "num_turns": <int>,           # max turn_end seen across all moments
    "key_moments": [
      {
        "turn_start": <int>,
        "turn_end": <int>,
        "annotation_type": "scaffolding" | "rapport",
        "annotator_id": "<str>",
        "situation": "<str>",
        "action": "<str>",
        "result": "<str>",
        "strategy_label": "effective" | "partial" | "ineffective",
        "situation_label": {      # scaffolding moments only
          "scaffolding": "yes" | "no" | "unclear" | "no_mention",
          "rigor": "yes" | "no" | "unclear" | "no_mention"
        },
        "situation_label_agg": "both" | "scaffolding" | "rigor" | "neither" | "mixed" | "unknown",  # scaffolding only; majority-voted across overlapping annotators (IoU >= 0.7), no_mention/unclear → no; "mixed" = tie, "unknown" = no annotator gave signal
        "action_decomposed": ["<str>", ...],  # atomic facets of the action field
        "result_decomposed": ["<str>", ...],  # atomic facets of the result field
        "cut_turn": <int>,          # optional — annotator-chosen benchmark cut point
        "moment_id": "<str>"        # optional — links cut point to its parent moment
      },
      ...
    ]
  }

Usage:
    python data/build_ground_truth.py
    python data/build_ground_truth.py --dry-run
    python data/build_ground_truth.py --labeller v2
    python data/build_ground_truth.py --labeller hybrid   # routes per annotation_type via config
    python data/build_ground_truth.py --input path/to/annotations.jsonl
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from annotator.core.label import JUNK_TEXTS, load_labeller_templates, pick_template
from annotator.core.situate import (
    JUNK_TEXTS as SIT_JUNK_TEXTS,
    _load_prompt as _load_situation_prompt,
    _parse_situation_label,
)
from annotator.core.decompose import (
    JUNK_TEXTS as DECOMPOSE_JUNK_TEXTS,
    _load_prompt as _load_decompose_prompt,
    _parse_decomposed,
)
from annotator.core.utils import compute_iou

ANNOTATIONS_JSONL = DATA_DIR / "teacher_annotations" / "step_up_annotations.jsonl"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "annotator" / "labeller"


def load_from_jsonl(path):
    """Load annotations from a step_up_annotations.jsonl file.

    Returns list of (conv_id, conv_data) sorted by conv_id, where conv_data is:
      {"annotations": [...], "num_turns": <int>}

    Caption records are skipped. turn_number_start/end are mapped to turn_start/end.
    annotator_id and annotation_type are promoted from the record level to each moment.
    num_turns is the max turn_end seen across all moments for that conversation.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("annotation_type") not in ("scaffolding", "rapport"):
                continue
            conv_id = record["transcript_id"]
            annotator_id = record.get("annotator_id", "")
            annotation_type = record["annotation_type"]
            for ta in record.get("turn_annotations", []):
                if ta.get("turn_number_start") is None or ta.get("turn_number_end") is None:
                    continue
                entry = {
                    "annotator_id": annotator_id,
                    "turn_start": ta["turn_number_start"],
                    "turn_end": ta["turn_number_end"],
                    "annotation_type": annotation_type,
                    "situation": ta.get("situation", ""),
                    "action": ta.get("action", ""),
                    "result": ta.get("result", ""),
                    "_timestamp": ta.get("annotation_timestamp", ""),
                }
                if "cut_turn" in ta:
                    entry["cut_turn"] = ta["cut_turn"]
                if "moment_id" in ta:
                    entry["moment_id"] = ta["moment_id"]
                groups[conv_id].append(entry)

    result = []
    for conv_id, annotations in sorted(groups.items()):
        annotations.sort(key=lambda a: a["_timestamp"])
        for a in annotations:
            del a["_timestamp"]
        num_turns = max((a["turn_end"] for a in annotations if a["turn_end"] is not None), default=0)
        result.append((conv_id, {"annotations": annotations, "num_turns": num_turns}))
    return result


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

VALID_LABELS = {"effective", "partial", "ineffective"}


def moment_key(m):
    """Stable key identifying an annotation moment across runs."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("result", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def situation_key(m):
    """Stable key for a situation label — same as moment_key but hashes situation text."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("situation", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def load_existing_labels():
    """Return {conv_id: {moment_key: strategy_label}} from current ground truth."""
    existing = {}
    if not GROUND_TRUTH_DIR.exists():
        return existing
    for f in GROUND_TRUTH_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        existing[f.stem] = {
            moment_key(m): m.get("strategy_label")
            for m in d.get("key_moments", [])
            if m.get("strategy_label")
        }
    return existing


def load_existing_situation_labels():
    """Return {conv_id: {situation_key: situation_label}} from current ground truth."""
    existing = {}
    if not GROUND_TRUTH_DIR.exists():
        return existing
    for f in GROUND_TRUTH_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        existing[f.stem] = {
            situation_key(m): m["situation_label"]
            for m in d.get("key_moments", [])
            if m.get("annotation_type") == "scaffolding" and m.get("situation_label")
        }
    return existing


def action_decompose_key(m):
    """Stable key for caching action decomposition by content hash."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("action", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def result_decompose_key(m):
    """Stable key for caching result decomposition by content hash."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("result", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def load_existing_decompositions():
    """Return {conv_id: {"action": {action_decompose_key: facets}, "result": {result_decompose_key: facets}}}."""
    existing = {}
    if not GROUND_TRUTH_DIR.exists():
        return existing
    for f in GROUND_TRUTH_DIR.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        action_cache = {}
        result_cache = {}
        for m in d.get("key_moments", []):
            if "action_decomposed" in m:
                action_cache[action_decompose_key(m)] = m["action_decomposed"]
            if "result_decomposed" in m:
                result_cache[result_decompose_key(m)] = m["result_decomposed"]
        if action_cache or result_cache:
            existing[f.stem] = {"action": action_cache, "result": result_cache}
    return existing


def decompose_batch(items):
    """Batch decompose action and result fields into atomic facets.

    items: list of {key, field ("action"|"result"), text}.
    Returns {key: [facets]}.
    """
    if not items:
        return {}
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("label")
    client = ModelClient(cfg["model"])

    action_template = _load_decompose_prompt("decompose_action.md")
    result_template = _load_decompose_prompt("decompose_result.md")

    entries = []
    results = {}
    for it in items:
        text = (it["text"] or "").strip()
        if text.lower() in DECOMPOSE_JUNK_TEXTS:
            results[it["key"]] = []
            continue
        if it["field"] == "action":
            prompt = action_template.replace("{action}", text)
        else:
            prompt = result_template.replace("{result}", text)
        entries.append(build_batch_entry(key=it["key"], prompt_text=prompt, json_mode=True))

    if not entries:
        return results

    print(f"  Submitting {len(entries)} decomposition requests to batch API "
          f"(model={cfg['model']})...")
    raw = run_batch(
        client, entries,
        json_mode=True,
        display_name="decomposition",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=cfg.get("thinking", False),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in raw.items():
        if "error" in result or not result.get("text"):
            print(f"  WARNING: error for {key}: {result.get('error', 'no text')}")
            results[key] = []
            continue
        facets, had_error = _parse_decomposed(result["text"])
        if had_error:
            print(f"  WARNING: could not parse decomposition for {key}: {result['text'][:100]!r}")
        results[key] = facets

    return results


def situation_classify_batch(items):
    """Batch classify situation appropriateness for scaffolding moments.

    items: list of {key, situation}. Returns {key: {"scaffolding": ..., "rigor": ...}}.
    """
    if not items:
        return {}
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("label")
    client = ModelClient(cfg["model"])
    prompt_template = _load_situation_prompt()

    entries = []
    labels = {}
    for it in items:
        situation = it["situation"]
        if (situation or "").strip().lower() in SIT_JUNK_TEXTS:
            labels[it["key"]] = {"scaffolding": "unclear", "rigor": "unclear"}
            continue
        prompt = prompt_template.replace("{situation}", situation)
        entries.append(build_batch_entry(key=it["key"], prompt_text=prompt, json_mode=True))

    if not entries:
        return labels

    print(f"  Submitting {len(entries)} situation classifications to batch API "
          f"(model={cfg['model']})...")
    results = run_batch(
        client, entries,
        json_mode=True,
        display_name="situation_classification",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=cfg.get("thinking", False),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in results.items():
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            labels[key] = {"scaffolding": "unclear", "rigor": "unclear"}
            continue
        sit_label, had_error = _parse_situation_label(result["text"])
        if had_error:
            print(f"  WARNING: could not parse situation label for {key}: {result['text'][:100]!r}")
        labels[key] = sit_label

    return labels


def classify_batch(items, labeller="hybrid"):
    """Run batch classification. `items` is list of dicts with keys:
    key, annotation_type, situation, action, result_text.
    Returns {key: label}.

    labeller="hybrid" routes per annotation_type using the `annotator.labeller`
    dict in config.yaml. Any other value loads classify_{labeller}.txt as a
    single shared template (legacy behavior)."""
    if not items:
        return {}
    from annotator.core.client import ModelClient, run_batch, build_batch_entry
    from annotator.core.config import get_phase_config, get_annotator_defaults

    cfg = get_phase_config("label")
    client = ModelClient(cfg["model"])

    if labeller == "hybrid":
        templates = load_labeller_templates(get_annotator_defaults()["labeller"])
    else:
        templates = {None: _load_prompt(f"classify_{labeller}")}

    entries = []
    labels = {}
    for it in items:
        text = it["result_text"]
        stripped = (text or "").strip().lower()
        if stripped in JUNK_TEXTS:
            labels[it["key"]] = "unclear"
            continue
        annotation_type = it.get("annotation_type", "unknown")
        template = pick_template(templates, annotation_type)
        prompt = (template
                  .replace("{annotation_type}", annotation_type)
                  .replace("{situation}", it.get("situation", ""))
                  .replace("{action}", it.get("action", ""))
                  .replace("{result_text}", text))
        entries.append(build_batch_entry(
            key=it["key"],
            prompt_text=prompt,
            json_mode=False,
            max_tokens=32,
        ))

    if not entries:
        return labels

    print(f"  Submitting {len(entries)} classifications to Anthropic batch API "
          f"(model={cfg['model']})...")
    results = run_batch(
        client, entries,
        json_mode=False,
        display_name="effectiveness_classification_refresh",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=cfg.get("thinking", False),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in results.items():
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            labels[key] = "unclear"
            continue
        label = result["text"].strip().lower().rstrip(".")
        labels[key] = label if label in VALID_LABELS else "unclear"

    return labels


def _normalize_sit(val):
    """Normalize unclear/None → no_mention (mirrors notebook _sit helper)."""
    if val in ("unclear", None, ""):
        return "no_mention"
    return val  # "yes", "no", or "no_mention"


def _cluster_by_iou(moments, threshold=0.7):
    """Group moment indices into IoU-based connected-component clusters.

    Same-annotator pairs are not directly linked (but may be transitively grouped).
    Returns a list of lists of indices into `moments`.
    """
    n = len(moments)
    if n == 0:
        return []
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if moments[i].get("annotator_id") == moments[j].get("annotator_id"):
                continue
            iou = compute_iou(
                (moments[i]["turn_start"], moments[i]["turn_end"]),
                (moments[j]["turn_start"], moments[j]["turn_end"]),
            )
            if iou >= threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters = defaultdict(list)
    for i in range(n):
        clusters[_find(i)].append(i)
    return list(clusters.values())


def _majority_vote_tuple(tuples):
    """Return the majority (scaf, rigor) tuple, or None on a tie."""
    if not tuples:
        return None
    counts = Counter(tuples).most_common(2)
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return None  # tie


_TUPLE_TO_AGG = {
    ("yes", "yes"): "both",
    ("yes", "no"):  "scaffolding",
    ("no",  "yes"): "rigor",
    ("no",  "no"):  "neither",
}


def compute_situation_label_agg(moments):
    """Return {idx: agg_label} for each scaffolding moment in `moments`.

    Groups overlapping scaffolding moments (IoU >= 0.7) into clusters,
    majority-votes the (scaffolding, rigor) tuple (remapping no_mention/unclear → no),
    and maps the winner to 'both'/'scaffolding'/'rigor'/'neither', 'mixed' for ties,
    or 'unknown' when every annotator in the cluster had both slots as no_mention.
    """
    scaf_idxs = [i for i, m in enumerate(moments) if m.get("annotation_type") == "scaffolding"]
    if not scaf_idxs:
        return {}
    scaf_moments = [moments[i] for i in scaf_idxs]
    clusters = _cluster_by_iou(scaf_moments)

    result = {}
    for cluster in clusters:
        seen_ann = set()
        vote_tuples = []
        for ci in cluster:
            m = scaf_moments[ci]
            ann_id = m.get("annotator_id", "")
            if ann_id in seen_ann:
                continue
            seen_ann.add(ann_id)
            sl = m.get("situation_label") or {}
            scaf = _normalize_sit(sl.get("scaffolding"))
            rigor = _normalize_sit(sl.get("rigor"))
            if scaf == "no_mention" and rigor == "no_mention":
                continue  # annotator gave no signal — exclude from vote
            scaf = "no" if scaf == "no_mention" else scaf
            rigor = "no" if rigor == "no_mention" else rigor
            vote_tuples.append((scaf, rigor))
        if not vote_tuples:
            label = "unknown"
        else:
            winner = _majority_vote_tuple(vote_tuples)
            label = "mixed" if winner is None else _TUPLE_TO_AGG.get(winner, "mixed")
        for ci in cluster:
            result[scaf_idxs[ci]] = label
    return result


def build_moment(ann, label):
    moment = {
        "turn_start": ann.get("turn_start"),
        "turn_end": ann.get("turn_end"),
        "annotation_type": ann.get("annotation_type", ""),
        "annotator_id": ann.get("annotator_id", ""),
        "situation": ann.get("situation", ""),
        "action": ann.get("action", ""),
        "result": ann.get("result", ""),
        "strategy_label": label,
    }
    if "cut_turn" in ann:
        moment["cut_turn"] = ann["cut_turn"]
    if "moment_id" in ann:
        moment["moment_id"] = ann["moment_id"]
    return moment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without submitting batch or writing files")
    parser.add_argument("--labeller", default="hybrid",
                        help="Labeller version. 'hybrid' routes per annotation_type using "
                             "config.yaml's annotator.labeller dict. Any other value loads "
                             "classify_{labeller}.txt as a single template. Determines output "
                             "dir (ground_truth_{labeller}/).")
    parser.add_argument("--input", default=str(ANNOTATIONS_JSONL),
                        help="Path to annotations JSONL file (default: teacher_annotations/step_up_annotations.jsonl)")
    args = parser.parse_args()

    global GROUND_TRUTH_DIR
    if args.labeller != "v1":
        GROUND_TRUTH_DIR = DATA_DIR / f"ground_truth_{args.labeller}"

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return

    print(f"Loading annotations from {input_path}...")
    conversations = load_from_jsonl(input_path)
    print(f"Loaded {len(conversations)} conversations")

    existing_labels = load_existing_labels()
    print(f"Loaded existing strategy labels for {len(existing_labels)} conversations")
    existing_situation_labels = load_existing_situation_labels()
    sit_cache_total = sum(len(v) for v in existing_situation_labels.values())
    print(f"Loaded existing situation labels for {len(existing_situation_labels)} conversations ({sit_cache_total} moments)")
    existing_decompositions = load_existing_decompositions()
    decomp_action_total = sum(len(v.get("action", {})) for v in existing_decompositions.values())
    decomp_result_total = sum(len(v.get("result", {})) for v in existing_decompositions.values())
    print(f"Loaded existing decompositions: {decomp_action_total} action, {decomp_result_total} result")

    # First pass: build per-conv plan (reuse vs classify) for both strategy and situation labels
    conv_plans = []
    to_classify = []          # [{key, annotation_type, situation, action, result_text}]
    to_situation_classify = []  # [{key, situation}] — scaffolding moments only
    to_decompose = []         # [{key, field, text}]
    situation_plans = {}      # {conv_id: [s_item, ...]} parallel to plan
    decompose_plans = {}      # {conv_id: [(action_item, result_item), ...]} parallel to plan

    for conv_id, conv_data in conversations:
        annotations = conv_data.get("annotations", [])
        known = existing_labels.get(conv_id, {})
        known_s = existing_situation_labels.get(conv_id, {})

        known_decomp = existing_decompositions.get(conv_id, {})
        known_action_decomp = known_decomp.get("action", {})
        known_result_decomp = known_decomp.get("result", {})

        plan = []
        s_plan = []
        d_plan = []
        for idx, ann in enumerate(annotations):
            k = moment_key(ann)
            if k in known:
                plan.append(("reuse", ann, known[k]))
            else:
                ckey = f"{conv_id}__{idx}"
                to_classify.append({
                    "key": ckey,
                    "annotation_type": ann.get("annotation_type", "unknown"),
                    "situation": ann.get("situation", ""),
                    "action": ann.get("action", ""),
                    "result_text": ann.get("result", ""),
                })
                plan.append(("classify", ann, ckey))

            if ann.get("annotation_type") == "scaffolding":
                sk = situation_key(ann)
                if sk in known_s:
                    s_plan.append(("reuse", known_s[sk]))
                else:
                    skey = f"{conv_id}__{idx}__sit"
                    to_situation_classify.append({"key": skey, "situation": ann.get("situation", "")})
                    s_plan.append(("classify", skey))
            else:
                s_plan.append(None)

            ak = action_decompose_key(ann)
            if ak in known_action_decomp:
                action_item = ("reuse", known_action_decomp[ak])
            else:
                dkey = f"{conv_id}__{idx}__action"
                to_decompose.append({"key": dkey, "field": "action", "text": ann.get("action", "")})
                action_item = ("classify", dkey)

            rk = result_decompose_key(ann)
            if rk in known_result_decomp:
                result_item = ("reuse", known_result_decomp[rk])
            else:
                dkey = f"{conv_id}__{idx}__result"
                to_decompose.append({"key": dkey, "field": "result", "text": ann.get("result", "")})
                result_item = ("classify", dkey)

            d_plan.append((action_item, result_item))

        conv_plans.append((conv_id, conv_data, plan))
        situation_plans[conv_id] = s_plan
        decompose_plans[conv_id] = d_plan

    total_moments = sum(len(p) for _, _, p in conv_plans)
    reused = sum(1 for _, _, p in conv_plans for kind, *_ in p if kind == "reuse")
    to_class = total_moments - reused
    new_convs = sum(1 for cid, _, _ in conv_plans if cid not in existing_labels)
    sit_reused = sum(1 for sp in situation_plans.values() for item in sp if item and item[0] == "reuse")
    decomp_action_reused = sum(1 for dp in decompose_plans.values() for a, _ in dp if a[0] == "reuse")
    decomp_result_reused = sum(1 for dp in decompose_plans.values() for _, r in dp if r[0] == "reuse")

    print(f"Plan: {len(conv_plans)} conversations, {total_moments} moments")
    print(f"  Reuse existing strategy labels:   {reused}")
    print(f"  Classify new strategy labels:     {to_class}")
    print(f"  Reuse existing situation labels:  {sit_reused}")
    print(f"  Classify new situation labels:    {len(to_situation_classify)}")
    print(f"  Reuse existing action decomps:    {decomp_action_reused}")
    print(f"  Reuse existing result decomps:    {decomp_result_reused}")
    print(f"  New decompositions (action+result): {len(to_decompose)}")
    print(f"  Brand new conversations:          {new_convs}")

    if args.dry_run:
        print("\nDry run — exiting without classifying or writing.")
        return

    # Second pass: batch classify strategy labels, situation labels, and decompositions
    new_labels = classify_batch(to_classify, labeller=args.labeller)
    new_situation_labels = situation_classify_batch(to_situation_classify)
    new_decompositions = decompose_batch(to_decompose)

    def _resolve_situation(s_item):
        if s_item is None:
            return None
        if s_item[0] == "reuse":
            return s_item[1]
        return new_situation_labels.get(s_item[1], {"scaffolding": "unclear", "rigor": "unclear"})

    # Third pass: write ground truth files
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    gt_written = 0
    for conv_id, conv_data, plan in conv_plans:
        s_plan = situation_plans[conv_id]
        d_plan = decompose_plans[conv_id]
        moments = []
        for (kind, ann, val), s_item, (action_item, result_item) in zip(plan, s_plan, d_plan):
            label = val if kind == "reuse" else new_labels.get(val, "unclear")
            moment = build_moment(ann, label)
            sit = _resolve_situation(s_item)
            if sit is not None:
                moment["situation_label"] = sit
            moment["action_decomposed"] = (
                action_item[1] if action_item[0] == "reuse"
                else new_decompositions.get(action_item[1], [])
            )
            moment["result_decomposed"] = (
                result_item[1] if result_item[0] == "reuse"
                else new_decompositions.get(result_item[1], [])
            )
            moments.append(moment)
        agg = compute_situation_label_agg(moments)
        for idx, agg_label in agg.items():
            moments[idx]["situation_label_agg"] = agg_label

        out = {
            "conversation_id": conv_id,
            "num_turns": conv_data.get("num_turns", 0),
            "key_moments": moments,
        }
        gt_path = GROUND_TRUTH_DIR / f"{conv_id}.json"
        with open(gt_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        gt_written += 1

    print(f"\nDone!")
    print(f"  Ground truth files written: {gt_written}")
    print(f"  Total ground truth: {len(list(GROUND_TRUTH_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
