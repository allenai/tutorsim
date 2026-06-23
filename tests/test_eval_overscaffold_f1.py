"""Tests for the over-scaffolding F1 evaluation in annotator/eval/eval.py.

compute_overscaffold_f1 treats over-scaffolding as a binary signal on the gold
scaffolding spans (positive = "over-scaffolding present"):
  - The LM is positive when its overscaffold_decomposed (from the decomposed_gold
    side file, keyed by span) is non-empty.
  - The gold span is positive when >= min_teachers DISTINCT annotators flagged
    over-scaffolding (non-empty overscaffold_decomposed in the ground truth).
Denominator is every gold scaffolding span the LM also decomposed;
precision/recall/F1 are for the positive class.

Mirrors compute_student_outcome_f1: teacher signal from ground_truth, LM signal
from a span-keyed decomposed file.
"""
import pytest

pytest.importorskip("krippendorff")  # annotator.eval.eval imports krippendorff

from annotator.eval.eval import compute_overscaffold_f1, decomposed_has_overscaffold


def _gt(spans):
    """spans: list of (turn_start, turn_end, [(annotator_id, over_list), ...])."""
    moments = []
    for ts, te, teachers in spans:
        for ann_id, over in teachers:
            moments.append({
                "annotation_type": "scaffolding",
                "turn_start": ts, "turn_end": te,
                "annotator_id": ann_id,
                "overscaffold_decomposed": over,
            })
    return {"conversations": {"c1": {"key_moments": moments}}}


def _decomp(spans):
    """spans: list of (turn_start, turn_end, over_list) for the LM."""
    return {"c1": [
        {"annotation_type": "scaffolding", "turn_start": ts, "turn_end": te,
         "overscaffold_decomposed": over}
        for ts, te, over in spans
    ]}


def test_min1_confusion_and_metrics():
    gt = _gt([
        (1, 2, [("t1", ["a"])]),   # gold yes
        (3, 4, [("t1", ["a"])]),   # gold yes
        (5, 6, [("t1", [])]),      # gold no
        (7, 8, [("t1", [])]),      # gold no
    ])
    dec = _decomp([
        (1, 2, ["x"]),  # llm yes -> TP
        (3, 4, []),     # llm no  -> FN
        (5, 6, ["x"]),  # llm yes -> FP
        (7, 8, []),     # llm no  -> TN
    ])
    r = compute_overscaffold_f1(gt, dec, ["c1"], min_teachers=1)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["n_units"] == 4
    assert r["n_gold_positive"] == 2
    assert r["n_llm_positive"] == 2
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5
    assert r["f1_by_label"]["yes"]["support"] == 2
    assert r["confusion"]["yes"]["yes"] == 1
    assert r["confusion"]["no"]["yes"] == 1


def test_min2_requires_two_distinct_teachers():
    gt = _gt([(1, 2, [("t1", ["a"])])])     # one flagger
    dec = _decomp([(1, 2, ["x"])])
    assert compute_overscaffold_f1(gt, dec, ["c1"], 1)["n_gold_positive"] == 1
    r2 = compute_overscaffold_f1(gt, dec, ["c1"], 2)
    assert r2["n_gold_positive"] == 0
    assert r2["fp"] == 1
    assert r2["precision"] == 0.0


def test_distinct_annotator_dedup():
    # Same annotator flagging twice on a span is still ONE teacher.
    gt = _gt([(1, 2, [("t1", ["a"]), ("t1", ["b"])])])
    dec = _decomp([(1, 2, [])])
    assert compute_overscaffold_f1(gt, dec, ["c1"], 2)["n_gold_positive"] == 0
    assert compute_overscaffold_f1(gt, dec, ["c1"], 1)["n_gold_positive"] == 1


def test_min2_two_distinct_teachers_positive():
    gt = _gt([(1, 2, [("t1", ["a"]), ("t2", ["b"])])])
    dec = _decomp([(1, 2, [])])
    assert compute_overscaffold_f1(gt, dec, ["c1"], 2)["n_gold_positive"] == 1


def test_span_not_decomposed_is_skipped():
    # Gold span the LM never decomposed -> can't compare -> excluded.
    gt = _gt([(1, 2, [("t1", ["a"])]), (3, 4, [("t1", ["a"])])])
    dec = _decomp([(1, 2, ["x"])])  # missing span (3,4)
    r = compute_overscaffold_f1(gt, dec, ["c1"], 1)
    assert r["n_units"] == 1


def test_no_positives_does_not_crash():
    gt = _gt([(1, 2, [("t1", [])]), (3, 4, [("t1", [])])])
    dec = _decomp([(1, 2, []), (3, 4, [])])
    r = compute_overscaffold_f1(gt, dec, ["c1"], 1)
    assert r["n_units"] == 2
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    assert r["tn"] == 2


def test_decomposed_has_overscaffold_detects_backfill():
    # Field present on a scaffolding annotation -> backfilled.
    assert decomposed_has_overscaffold(_decomp([(1, 2, [])])) is True
    # Field absent entirely (predates the pass) -> not backfilled.
    not_backfilled = {"c1": [{"annotation_type": "scaffolding",
                             "turn_start": 1, "turn_end": 2}]}
    assert decomposed_has_overscaffold(not_backfilled) is False


def test_non_scaffolding_gold_ignored():
    gt = _gt([(1, 2, [("t1", ["a"])])])
    gt["conversations"]["c1"]["key_moments"].append({
        "annotation_type": "rapport", "turn_start": 9, "turn_end": 9,
        "annotator_id": "t1", "overscaffold_decomposed": ["x"],
    })
    dec = _decomp([(1, 2, ["x"])])
    assert compute_overscaffold_f1(gt, dec, ["c1"], 1)["n_units"] == 1
