#!/usr/bin/env python3
"""One-off migration: rewrite stale "neither" action_direction_agg defaults to
"unknown" in existing ground truth files.

Background: plan_action_result_agg (build_ground_truth.py) used to default
action_direction_agg to "neither" for scaffolding clusters with no action
facets to classify. It now defaults to "unknown" -- "neither" is a substantive
scaffolding-vs-rigor verdict the model never had a chance to make for these
clusters, whereas "unknown" mirrors situation_label_agg's "no annotator gave
signal" sentinel (see plan_action_result_agg's docstring).

Ground truth files written before that change still have "neither" baked in
for those clusters, and rerunning build_ground_truth.py won't fix them on its
own: load_existing_action_result_agg's cache-reuse check only compares cached
unified_action against freshly computed unified_action, not the cached label
against what the *current* code would produce as a default -- so an unchanged
empty-facets cluster reuses its stale cached "neither" forever.

This migration finds exactly those clusters (action_direction_agg == "neither"
AND unified action_decomposed is empty -- i.e. "neither" could only have come
from the old no-facets default, never a real model classification) and rewrites
every moment in them to "unknown", in place, with no LLM calls.

Usage:
  python -m data.migrate_action_direction_unknown [--dry-run]
"""

import argparse
import json

from data.build_ground_truth import DATA_DIR, _scaffolding_clusters, _unify_facets


def find_stale_neither_indices(key_moments):
    """Return indices into `key_moments` whose action_direction_agg should be
    rewritten from "neither" to "unknown".

    A cluster qualifies when every member's action_direction_agg is "neither"
    (the label plan_action_result_agg writes identically across a cluster) and
    the cluster's unified action_decomposed is empty -- meaning no annotator
    contributed an action facet, so "neither" can only be the old no-facets
    default rather than a real classification.
    """
    stale = []
    for cluster_indices, cluster_moments in _scaffolding_clusters(key_moments):
        agg_values = {m.get("action_direction_agg") for m in cluster_moments}
        if agg_values != {"neither"}:
            continue
        unified_action, _ = _unify_facets(cluster_moments)
        if unified_action:
            continue
        stale.extend(cluster_indices)
    return stale


def migrate_file(path, dry_run=False):
    """Rewrite stale "neither" entries in one ground truth file.

    Returns the number of moments flipped (0 if the file needed no changes).
    Writes the file back in place unless `dry_run` is set.
    """
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    key_moments = data.get("key_moments", [])
    stale_indices = find_stale_neither_indices(key_moments)
    if not stale_indices:
        return 0

    for idx in stale_indices:
        key_moments[idx]["action_direction_agg"] = "unknown"

    if not dry_run:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)

    return len(stale_indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files")
    parser.add_argument("--labeller", default="hybrid",
                        help="Labeller version -- selects data/ground_truth_<labeller>/, "
                             "mirroring build_ground_truth.py's --labeller (default: hybrid)")
    args = parser.parse_args()

    ground_truth_dir = DATA_DIR / f"ground_truth_{args.labeller}"
    if not ground_truth_dir.exists():
        print(f"Ground truth dir not found: {ground_truth_dir}")
        return

    files_changed = 0
    moments_flipped = 0
    for path in sorted(ground_truth_dir.glob("*.json")):
        n = migrate_file(path, dry_run=args.dry_run)
        if n:
            files_changed += 1
            moments_flipped += n
            print(f"  {'would flip' if args.dry_run else 'flipped'} {n} moment(s) in {path.name}")

    verb = "Would flip" if args.dry_run else "Flipped"
    print(f"\n{verb} {moments_flipped} moment(s) across {files_changed} file(s) "
          f"in {ground_truth_dir} (neither -> unknown).")


if __name__ == "__main__":
    main()
