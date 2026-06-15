"""Tests for _binary_metrics in validation/overscaffold_validation.py.

The over-scaffolding prompt eval is a binary classification: gold = human
overscaffolding? label (positive=True), pred = the prompt returned >=1 span.
_binary_metrics turns (gold, pred) pairs into a confusion matrix plus
precision/recall/F1/accuracy, with safe division when a class is empty.
"""

from validation.overscaffold_validation import _binary_metrics


def test_perfect_classification():
    m = _binary_metrics([(True, True), (False, False), (True, True)])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 0, 0, 1)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_mixed_confusion_and_rates():
    # 1 TP, 1 FN, 1 FP, 2 TN
    pairs = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    m = _binary_metrics(pairs)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5          # 1 / (1+1)
    assert m["recall"] == 0.5             # 1 / (1+1)
    assert m["f1"] == 0.5
    assert m["accuracy"] == 3 / 5         # (1 TP + 2 TN) / 5


def test_no_positive_predictions_no_zero_division():
    # All negatives, predicted all negative: precision/recall/f1 undefined -> 0.0.
    m = _binary_metrics([(False, False), (False, False)])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 1.0


def test_recall_zero_when_all_positives_missed():
    m = _binary_metrics([(True, False), (True, False)])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (0, 2, 0, 0)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # no positive predictions
    assert m["accuracy"] == 0.0
