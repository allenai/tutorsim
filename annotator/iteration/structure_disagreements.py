"""
View disagreements (or agreements, via --show) between gold structure-label
aggregates and the LM's structure labels (output of structure.py --gold).

Compares:
  - action_direction_agg (gold, aggregated from human action_decomposed
    judgments) vs action_label (LM, from structure.classify_action)
  - student_outcome_agg  (gold, aggregated from human result_decomposed
    judgments) vs result_label (LM, from structure.classify_student_result)

Both comparisons are scaffolding-only and matched 1:1 by exact
(turn_start, turn_end) span -- the same direct lookup eval.py's
compute_action_direction_f1 / compute_student_outcome_f1 use
(structure_labels_gold is produced 1:1 from deduplicated gold spans, see
annotate.load_gold_moments). Each is filtered and framed to match exactly what
that F1 scores, so every disagreement shown is one eval counts:

  - action direction: both 4-way labels are decomposed into independent
    (scaffolding, rigor) yes/no dimensions; sentinel labels ("unclear"/"unknown")
    are excluded, and a span is reported only when the sides differ on at least
    one dimension. Output is grouped by dimension, not by 4-way confusion.
  - student outcome: restricted to gold spans whose student_outcome_agg is a
    substantive "pos"/"neg" verdict (the only spans the F1 scores).

Usage:
    python -m annotator.iteration.structure_disagreements --version v13 --profile anthropic
    python -m annotator.iteration.structure_disagreements --version v13 --profile anthropic \\
        --field action_direction
    python -m annotator.iteration.structure_disagreements --version v13 --profile anthropic \\
        --field student_outcome
    python -m annotator.iteration.structure_disagreements --version v13 --profile anthropic \\
        --limit 5 --summary-only
    python -m annotator.iteration.structure_disagreements --version v13 --profile anthropic \\
        --show agreements --summary-only
"""

import argparse
from collections import defaultdict

from ..core.config import get_valid_styles
from ..core.utils import (
    load_ground_truth, load_split_ids, load_transcripts, get_excerpt,
    EXAMPLE_CONV_IDS,
)
from ..eval.eval import (
    load_structure_labels_gold,
    ground_truth_has_action_direction_agg,
    ground_truth_has_student_outcome_agg,
    _ACTION_LABEL_TO_DIMENSIONS,
)

ACTION_DIMENSIONS = ("scaffolding", "rigor")
from data.build_ground_truth import _scaffolding_clusters, _unify_facets

CONTEXT_TURNS = 10


def _load_gold(annotator_style=None, split="train"):
    """Load ground truth filtered to an archetype and split -- mirrors eval.py's loading."""
    gt = load_ground_truth(annotator_style=annotator_style)
    split_ids = load_split_ids(split)
    gt["conversations"] = {
        conv_id: conv_data
        for conv_id, conv_data in gt["conversations"].items()
        if conv_id in split_ids
    }
    return gt


def _load_transcripts_by_transcript_id():
    """Load transcripts keyed by bare transcript_id (UUID), not conversation_id.

    Ground truth and structure_labels_gold both key by transcript_id (see
    eval.load_structure_labels_gold's `conv_id.rsplit("_", 1)[-1]`), but
    load_transcripts() keys by the full conversation_id
    ({tutor_id}_{student_id}_{transcript_id}) -- see annotate.load_split_gold_moments
    for the same re-keying. Without this, get_excerpt looks up the wrong key and
    returns "[transcript not found]" for every gold span.
    """
    return {
        conv["transcript_id"]: conv
        for conv in load_transcripts().values()
        if conv.get("transcript_id")
    }


def _unified_gold_facets_by_span(ground_truth, eval_conv_ids):
    """{(conv_id, turn_start, turn_end): (unified_action_facets, unified_result_facets)}.

    Re-derives the same IoU>=1.0 clusters and per-cluster unified facet lists
    (one contribution per annotator) that build_ground_truth.plan_action_result_agg
    fed to the classifier when computing action_direction_agg / student_outcome_agg
    -- see _scaffolding_clusters / _unify_facets. A gold moment's own
    action_decomposed/result_decomposed is frequently empty (its annotator's text
    didn't decompose into facets) even though action_direction_agg/student_outcome_agg
    is set from a cluster-mate's facets; showing the unified list is what gold's
    label was actually based on.
    """
    by_span = {}
    for conv_id in eval_conv_ids:
        moments = ground_truth["conversations"].get(conv_id, {}).get("key_moments", [])
        for _, cluster_moments in _scaffolding_clusters(moments):
            unified_action, unified_result = _unify_facets(cluster_moments)
            for m in cluster_moments:
                span = (conv_id, m["turn_start"], m["turn_end"])
                by_span.setdefault(span, (unified_action, unified_result))
    return by_span


# ===================================================================
# Disagreement collection
# ===================================================================

def collect_action_direction_cases(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                   unified_facets_by_span, mode="disagree"):
    """Spans where gold and LM agree (mode="agree") or differ (mode="disagree")
    on at least one action-direction dimension.

    Mirrors eval.compute_action_direction_f1's scoring rather than a raw 4-way
    label comparison: both gold action_direction_agg and LM action_label are
    decomposed into independent (scaffolding, rigor) yes/no dimensions via
    _ACTION_LABEL_TO_DIMENSIONS, and each dimension is judged independently.

    Sentinel labels ("unclear" parse-failure fallback, "unknown" gold-only
    no-action-facets) don't decompose to a per-dimension verdict and are
    excluded from F1 -- so they're excluded here in both modes. A span like
    gold "both" vs LM "scaffolding" agrees on scaffolding and disagrees on
    rigor; it appears in the disagree view (disagree_dims == ["rigor"]) AND in
    the agree view (agree_dims == ["scaffolding"]).

    One entry per unique gold span. Each entry carries gold_dims/llm_dims
    ({"scaffolding": yes/no, "rigor": yes/no}), disagree_dims, and agree_dims
    (the subsets of ACTION_DIMENSIONS in scaffolding-then-rigor order) so callers
    can group by the dimension whose F1 the case actually feeds. In "disagree"
    mode the entry is kept only when disagree_dims is non-empty; in "agree" mode
    only when agree_dims is non-empty.
    """
    llm_by_span = {}
    for conv_id in eval_conv_ids:
        for a in structure_labels_by_conv.get(conv_id, []):
            if a.get("annotation_type") != "scaffolding":
                continue
            llm_by_span[(conv_id, a["turn_start"], a["turn_end"])] = a

    cases = []
    seen_spans = set()
    for conv_id in eval_conv_ids:
        for m in ground_truth["conversations"].get(conv_id, {}).get("key_moments", []):
            if m.get("annotation_type") != "scaffolding":
                continue
            gold_label = m.get("action_direction_agg")
            if gold_label is None:
                continue
            span = (conv_id, m["turn_start"], m["turn_end"])
            if span in seen_spans:
                continue
            seen_spans.add(span)

            llm_ann = llm_by_span.get(span)
            if llm_ann is None:
                continue
            llm_label = llm_ann.get("action_label")

            gold_dims_pair = _ACTION_LABEL_TO_DIMENSIONS.get(gold_label)
            llm_dims_pair = _ACTION_LABEL_TO_DIMENSIONS.get(llm_label)
            if gold_dims_pair is None or llm_dims_pair is None:
                continue  # sentinel on either side -- not scored by eval

            gold_dims = dict(zip(ACTION_DIMENSIONS, gold_dims_pair))
            llm_dims = dict(zip(ACTION_DIMENSIONS, llm_dims_pair))
            disagree_dims = [d for d in ACTION_DIMENSIONS if gold_dims[d] != llm_dims[d]]
            agree_dims = [d for d in ACTION_DIMENSIONS if gold_dims[d] == llm_dims[d]]
            relevant_dims = agree_dims if mode == "agree" else disagree_dims
            if not relevant_dims:
                continue

            cases.append({
                "conv_id": conv_id,
                "turn_start": m["turn_start"],
                "turn_end": m["turn_end"],
                "gold_label": gold_label,
                "llm_label": llm_label,
                "gold_dims": gold_dims,
                "llm_dims": llm_dims,
                "disagree_dims": disagree_dims,
                "agree_dims": agree_dims,
                "gold_moment": m,
                "gold_unified_action": unified_facets_by_span.get(span, ([], []))[0],
                "llm_annotation": llm_ann,
            })
    return cases


def collect_action_direction_disagreements(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                            unified_facets_by_span):
    """Spans where gold and LM disagree on at least one action-direction dimension."""
    return collect_action_direction_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode="disagree")


def collect_action_direction_agreements(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                         unified_facets_by_span):
    """Spans where gold and LM agree on at least one action-direction dimension."""
    return collect_action_direction_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode="agree")


def collect_student_outcome_cases(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                  unified_facets_by_span, mode="disagree"):
    """(gold, llm) pairs where student_outcome_agg == result_label (mode="agree")
    or != result_label (mode="disagree").

    One entry per unique gold span, mirroring eval.compute_student_outcome_f1's
    matching AND its gold filter: only spans where student_outcome_agg is "pos"
    or "neg" are considered (the substantive trending-toward-understanding /
    misconceptions-remain verdict that F1 is scored against). Gold "no_evidence"
    (no result facets to classify) and "unclear" (parse-failure fallback) spans
    carry no pos/neg signal -- surfacing them in either mode would show cases the
    F1 score never counted.
    """
    llm_by_span = {}
    for conv_id in eval_conv_ids:
        for a in structure_labels_by_conv.get(conv_id, []):
            if a.get("annotation_type") != "scaffolding":
                continue
            llm_by_span[(conv_id, a["turn_start"], a["turn_end"])] = a

    cases = []
    seen_spans = set()
    for conv_id in eval_conv_ids:
        for m in ground_truth["conversations"].get(conv_id, {}).get("key_moments", []):
            if m.get("annotation_type") != "scaffolding":
                continue
            gold_label = m.get("student_outcome_agg")
            if gold_label not in ("pos", "neg"):
                continue
            span = (conv_id, m["turn_start"], m["turn_end"])
            if span in seen_spans:
                continue
            seen_spans.add(span)

            llm_ann = llm_by_span.get(span)
            if llm_ann is None:
                continue
            llm_label = llm_ann.get("result_label")
            is_match = llm_label == gold_label
            if is_match != (mode == "agree"):
                continue

            cases.append({
                "conv_id": conv_id,
                "turn_start": m["turn_start"],
                "turn_end": m["turn_end"],
                "gold_label": gold_label,
                "llm_label": llm_label,
                "gold_moment": m,
                "gold_unified_result": unified_facets_by_span.get(span, ([], []))[1],
                "llm_annotation": llm_ann,
            })
    return cases


def collect_student_outcome_disagreements(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                          unified_facets_by_span):
    """(gold, llm) pairs where student_outcome_agg != result_label."""
    return collect_student_outcome_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode="disagree")


def collect_student_outcome_agreements(ground_truth, structure_labels_by_conv, eval_conv_ids,
                                       unified_facets_by_span):
    """(gold, llm) pairs where student_outcome_agg == result_label."""
    return collect_student_outcome_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode="agree")


# ===================================================================
# Printing
# ===================================================================

def _print_facets(label, facets):
    print(f"\n  {label}:")
    if not facets:
        print(f"    (none)")
        return
    for fact in facets:
        print(f"    - {fact}")


def print_action_direction_disagreement(entry, transcripts, idx, dim, noun="Disagreement"):
    conv_id = entry["conv_id"]
    llm_a = entry["llm_annotation"]

    print(f"\n  --- {noun} {idx}: {conv_id[:50]} "
          f"turns {entry['turn_start']}-{entry['turn_end']} "
          f"({dim}: gold={entry['gold_dims'][dim]}, llm={entry['llm_dims'][dim]}) ---")
    print(f"  4-way labels: gold={entry['gold_label']}, llm={entry['llm_label']}  "
          f"(agree on: {', '.join(entry['agree_dims']) or 'none'}; "
          f"disagree on: {', '.join(entry['disagree_dims']) or 'none'})")

    print(f"\n  TRANSCRIPT EXCERPT:")
    excerpt = get_excerpt(transcripts, conv_id, entry["turn_start"], entry["turn_end"],
                          context=CONTEXT_TURNS, bold_range=True)
    for line in excerpt.split("\n"):
        print(f"    {line}")

    # GOLD's action_direction_agg was classified from the cluster-unified facet
    # list (build_ground_truth.plan_action_result_agg), not this single moment's
    # own action_decomposed -- show that list so the label is explicable.
    _print_facets(f"GOLD action facets (cluster-unified, -> {entry['gold_label']})",
                  entry["gold_unified_action"])
    _print_facets(f"LLM action facets (-> {entry['llm_label']})",
                  llm_a.get("action_decomposed"))
    print()


def print_student_outcome_disagreement(entry, transcripts, idx, noun="Disagreement"):
    conv_id = entry["conv_id"]
    llm_a = entry["llm_annotation"]

    print(f"\n  --- {noun} {idx}: {conv_id[:50]} "
          f"turns {entry['turn_start']}-{entry['turn_end']} "
          f"(gold={entry['gold_label']}, llm={entry['llm_label']}) ---")

    print(f"\n  TRANSCRIPT EXCERPT:")
    excerpt = get_excerpt(transcripts, conv_id, entry["turn_start"], entry["turn_end"],
                          context=CONTEXT_TURNS, bold_range=True)
    for line in excerpt.split("\n"):
        print(f"    {line}")

    # GOLD's student_outcome_agg was classified from the cluster-unified facet
    # list (build_ground_truth.plan_action_result_agg), not this single moment's
    # own result_decomposed -- show that list so the label is explicable.
    _print_facets(f"GOLD result facets (cluster-unified, -> {entry['gold_label']})",
                  entry["gold_unified_result"])
    _print_facets(f"LLM result facets (-> {entry['llm_label']})",
                  llm_a.get("result_decomposed"))
    print()


# ===================================================================
# Reporting (summary + capped examples, symmetric across modes)
# ===================================================================

# noun[mode] = (PLURAL header word, singular example label)
_MODE_NOUNS = {"disagree": ("DISAGREEMENTS", "Disagreement"),
               "agree": ("AGREEMENTS", "Agreement")}


def report_action_direction(ground_truth, structure_labels_by_conv, eval_conv_ids,
                            unified_facets_by_span, mode, transcripts, limit, summary_only):
    """Print the action-direction summary + capped examples for one mode.

    Grouped by dimension (scaffolding/rigor), matching how
    eval.compute_action_direction_f1 scores each independently, then by the
    yes/no confusion within that dimension. A span that qualifies on both
    dimensions appears under both. dim_key selects which per-dimension subset
    (agree_dims / disagree_dims) drives the grouping.
    """
    plural, singular = _MODE_NOUNS[mode]
    dim_key = "agree_dims" if mode == "agree" else "disagree_dims"
    verb = "agreeing" if mode == "agree" else "disagreeing"

    cases = collect_action_direction_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode=mode)

    print(f"\n{'=' * 70}")
    print(f"  ACTION-DIRECTION {plural} ({len(cases)} spans, scored per dimension)")
    print(f"{'=' * 70}")

    def grouped(dim):
        dim_cases = [c for c in cases if dim in c[dim_key]]
        by_confusion = defaultdict(list)
        for c in dim_cases:
            by_confusion[f"gold={c['gold_dims'][dim]} -> llm={c['llm_dims'][dim]}"].append(c)
        return dim_cases, by_confusion

    for dim in ACTION_DIMENSIONS:
        dim_cases, by_confusion = grouped(dim)
        print(f"\n  {dim.upper()} dimension ({len(dim_cases)} {verb} spans)")
        for confusion_type, group in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
            print(f"    {confusion_type:>22s}: {len(group)}")

    if summary_only:
        return
    idx = 0
    for dim in ACTION_DIMENSIONS:
        _, by_confusion = grouped(dim)
        for confusion_type, group in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
            print(f"\n{'~' * 70}")
            print(f"  {dim.upper()} {confusion_type.upper()} "
                  f"({len(group)} total, showing {min(limit, len(group))})")
            print(f"{'~' * 70}")
            for entry in group[:limit]:
                idx += 1
                print_action_direction_disagreement(entry, transcripts, idx, dim, noun=singular)


def report_student_outcome(ground_truth, structure_labels_by_conv, eval_conv_ids,
                           unified_facets_by_span, mode, transcripts, limit, summary_only):
    """Print the student-outcome summary + capped examples for one mode."""
    plural, singular = _MODE_NOUNS[mode]

    cases = collect_student_outcome_cases(
        ground_truth, structure_labels_by_conv, eval_conv_ids,
        unified_facets_by_span, mode=mode)
    by_confusion = defaultdict(list)
    for c in cases:
        by_confusion[f"gold={c['gold_label']} -> llm={c['llm_label']}"].append(c)

    print(f"\n{'=' * 70}")
    print(f"  STUDENT-OUTCOME {plural} ({len(cases)} total)")
    print(f"{'=' * 70}")
    for confusion_type, group in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        print(f"  {confusion_type:>34s}: {len(group)}")

    if summary_only:
        return
    idx = 0
    for confusion_type, group in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        print(f"\n{'~' * 70}")
        print(f"  {confusion_type.upper()} "
              f"({len(group)} total, showing {min(limit, len(group))})")
        print(f"{'~' * 70}")
        for entry in group[:limit]:
            idx += 1
            print_student_outcome_disagreement(entry, transcripts, idx, noun=singular)


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="View disagreements or agreements (--show) between gold "
                    "structure-label aggregates (action_direction_agg / "
                    "student_outcome_agg) and the LM's structure labels "
                    "(action_label / result_label)")
    parser.add_argument("--version", required=True, help="Results version (e.g. v13)")
    parser.add_argument("--profile", default=None, help="Config profile (e.g. anthropic)")
    parser.add_argument("--annotator-style", choices=get_valid_styles(), default=None,
                        help="Filter ground truth to this annotator archetype")
    parser.add_argument("--split", default="train", help="Split to evaluate (default: train)")
    parser.add_argument("--field", choices=["action_direction", "student_outcome", "both"],
                        default="both", help="Which comparison to show (default: both)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max examples to print per confusion group (default: 10)")
    parser.add_argument("--show", choices=["disagreements", "agreements", "both"],
                        default="disagreements",
                        help="Show spans where gold and the LM disagree, agree, or both "
                             "(default: disagreements)")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary counts")
    args = parser.parse_args()

    modes = {"disagreements": ["disagree"], "agreements": ["agree"],
             "both": ["disagree", "agree"]}[args.show]

    ground_truth = _load_gold(annotator_style=args.annotator_style, split=args.split)
    print(f"Loaded ground truth: {len(ground_truth['conversations'])} conversations "
          f"({args.split} split"
          + (f", '{args.annotator_style}' annotators)" if args.annotator_style else ")"))

    structure_labels_by_conv, structure_filename = load_structure_labels_gold(
        args.version, profile=args.profile, annotator_style=args.annotator_style, split=args.split)
    if structure_labels_by_conv is None:
        print(f"ERROR: no structure_labels_gold output found for version={args.version} "
              f"profile={args.profile} style={args.annotator_style} split={args.split}")
        return
    print(f"Loaded structure labels: {structure_filename}")

    eval_conv_ids = (
        set(ground_truth["conversations"].keys())
        & set(structure_labels_by_conv.keys())
    ) - EXAMPLE_CONV_IDS
    print(f"Evaluating {len(eval_conv_ids)} conversations")

    unified_facets_by_span = _unified_gold_facets_by_span(ground_truth, eval_conv_ids)

    transcripts = None if args.summary_only else _load_transcripts_by_transcript_id()

    show_action_direction = args.field in ("action_direction", "both")
    show_student_outcome = args.field in ("student_outcome", "both")

    if show_action_direction:
        if not ground_truth_has_action_direction_agg(ground_truth):
            print("\nGround truth has no action_direction_agg -- skipping action-direction comparison")
        else:
            for mode in modes:
                report_action_direction(
                    ground_truth, structure_labels_by_conv, eval_conv_ids,
                    unified_facets_by_span, mode, transcripts, args.limit, args.summary_only)

    if show_student_outcome:
        if not ground_truth_has_student_outcome_agg(ground_truth):
            print("\nGround truth has no student_outcome_agg -- skipping student-outcome comparison")
        else:
            for mode in modes:
                report_student_outcome(
                    ground_truth, structure_labels_by_conv, eval_conv_ids,
                    unified_facets_by_span, mode, transcripts, args.limit, args.summary_only)


if __name__ == "__main__":
    main()
