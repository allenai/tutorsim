"""Tests for the one-off migration that rewrites stale "neither" defaults to
"unknown" in existing ground truth files.

Background: plan_action_result_agg used to default action_direction_agg to
"neither" for scaffolding clusters with no action facets to classify; it now
defaults to "unknown" (see plan_action_result_agg's docstring). Ground truth
files written before that change still have "neither" baked in for those
clusters, and rerunning build_ground_truth.py won't fix them -- the cache
(load_existing_action_result_agg) reuses the cached label whenever the cached
unified_action still matches. find_stale_neither_indices identifies exactly
the moments that need rewriting: every moment in a cluster whose
action_direction_agg is "neither" but whose unified action_decomposed is empty
(i.e. "neither" could only have come from the old no-facets default, never a
real classification).
"""

from data.migrate_action_direction_unknown import find_stale_neither_indices


def _moment(annotator_id, turn_start, turn_end, annotation_type="scaffolding",
            action_decomposed=None, action_direction_agg=None):
    m = {
        "annotator_id": annotator_id,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "annotation_type": annotation_type,
        "action_decomposed": action_decomposed if action_decomposed is not None else [],
    }
    if action_direction_agg is not None:
        m["action_direction_agg"] = action_direction_agg
    return m


def test_flags_cluster_with_no_facets_and_stale_neither():
    moments = [
        _moment("t1", 1, 10, action_direction_agg="neither"),
        _moment("t2", 1, 10, action_direction_agg="neither"),
    ]
    assert find_stale_neither_indices(moments) == [0, 1]


def test_does_not_flag_substantive_neither_with_facets():
    moments = [
        _moment("t1", 1, 10, action_decomposed=["a1"], action_direction_agg="neither"),
        _moment("t2", 1, 10, action_decomposed=["b1"], action_direction_agg="neither"),
    ]
    assert find_stale_neither_indices(moments) == []


def test_does_not_flag_other_labels_with_no_facets():
    # Shouldn't occur in practice (the no-facets default only ever produced
    # "neither"), but the migration must stay narrowly scoped to "neither".
    moments = [
        _moment("t1", 1, 10, action_direction_agg="scaffolding"),
        _moment("t2", 1, 10, action_direction_agg="scaffolding"),
    ]
    assert find_stale_neither_indices(moments) == []


def test_ignores_rapport_moments():
    moments = [
        _moment("t1", 1, 10, annotation_type="rapport", action_direction_agg="neither"),
    ]
    assert find_stale_neither_indices(moments) == []


def test_handles_multiple_clusters_independently():
    moments = [
        _moment("t1", 1, 10, action_direction_agg="neither"),               # cluster 1: no facets -> flip
        _moment("t2", 1, 10, action_direction_agg="neither"),
        _moment("t3", 20, 30, action_decomposed=["a1"],                      # cluster 2: has facets -> keep
                action_direction_agg="neither"),
        _moment("t4", 20, 30, action_decomposed=["b1"], action_direction_agg="neither"),
    ]
    assert find_stale_neither_indices(moments) == [0, 1]
