"""Per-scenario benchmark scoring under the new annotator pipeline.

Mirrors the annotator-validation pattern in `annotator/eval/eval.py`:
both the gold tag (`situation_label_agg`) and the AI's annotation
(`action_label`) are 4-way labels in `{both, scaffolding, rigor, neither}`,
collapsed from two independent yes/no judgements (`scaffolding=yes/no` and
`rigor=yes/no`). We decompose both sides back to (scaffolding_yn, rigor_yn)
via `_ACTION_LABEL_TO_DIMENSIONS` and compute binary F1 per dimension.

This avoids the asymmetric trap where, e.g., a gold "both" vs LM "scaffolding"
(partial agreement: both say scaffolding=yes, only the rigor side disagrees)
gets counted as a total miss under a 4-way one-vs-rest.

Scenarios whose collapsed label is a non-substantive sentinel (`mixed`,
`unknown`, `unclear`) on either side are excluded from per-dimension F1 --
they don't decompose into yes/no on either dimension. They still contribute
to the outcome rate.
"""
from __future__ import annotations

# Canonical decomposition from collapsed 4-way label -> (scaffolding_yn, rigor_yn).
# Mirrors `annotator.eval.eval._ACTION_LABEL_TO_DIMENSIONS` -- we inline here
# to avoid pulling that module's heavy dependencies (krippendorff, etc.).
# "unclear" (LM parse-failure) and "unknown" (gold-side missing-facets) are
# non-substantive sentinels and intentionally absent from this map -- any
# label not in it gets None and is excluded from per-dimension F1.
_ACTION_LABEL_TO_DIMENSIONS = {
    "both":        ("yes", "yes"),
    "scaffolding": ("yes", "no"),
    "rigor":       ("no",  "yes"),
    "neither":     ("no",  "no"),
}


def _to_dims(label):
    """Return (scaffolding_yn, rigor_yn) for a collapsed action label,
    or None if the label is a non-substantive sentinel."""
    return _ACTION_LABEL_TO_DIMENSIONS.get(label)


def _action_label_for_scenario(annotation_data: dict):
    """Return the AI's action_label for the scenario.

    Lucy's structure.py emits ONE collapsed `action_label` per annotation
    (string, one of {both, scaffolding, rigor, neither, unclear}).
    Benchmark Phase 2 produces one annotation per scenario, so we read the
    first annotation's label. If multiple annotations are ever present we
    fall back to the first substantive one.
    """
    for ann in annotation_data.get("annotations", []) or []:
        label = ann.get("action_label")
        if isinstance(label, str) and label:
            return label
    return None


def _has_pos_result(annotation_data: dict) -> bool:
    for ann in annotation_data.get("annotations", []) or []:
        rl = ann.get("result_label")
        if isinstance(rl, list):
            if "pos" in rl:
                return True
        elif isinstance(rl, str):
            if rl == "pos":
                return True
    return False


def _precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def score_scenarios(scenarios: list[dict], annotations: list[dict]) -> dict:
    """Compute per-dimension binary F1 (scaffolding / rigor) + outcome rate.

    Args:
        scenarios: list of scenario dicts (must include detection.situation_label_agg).
        annotations: aligned list of annotation dicts (each has 'annotations' list
                     with 'action_label' (str) + 'result_label' populated).

    Returns:
        {
          "scaffolding": {tp, fp, fn, precision, recall, f1},
          "rigor": {tp, fp, fn, precision, recall, f1},
          "outcome_pos_rate": float,
          "n_scenarios": int,
          "n_scored_for_f1": int,
        }
    """
    counts = {
        "scaffolding": {"tp": 0, "fp": 0, "fn": 0},
        "rigor": {"tp": 0, "fp": 0, "fn": 0},
    }
    outcome_pos = 0
    n_total = 0
    n_scored = 0

    for scenario, ann in zip(scenarios, annotations):
        n_total += 1

        gt_label = (scenario.get("detection") or {}).get("situation_label_agg")
        pred_label = _action_label_for_scenario(ann)

        gt_dims = _to_dims(gt_label)
        pred_dims = _to_dims(pred_label)

        # Score only when BOTH sides decompose to yes/no on each dimension.
        if gt_dims is not None and pred_dims is not None:
            n_scored += 1
            gt_scaf, gt_rig = gt_dims
            pred_scaf, pred_rig = pred_dims
            for cls, gtv, predv in (
                ("scaffolding", gt_scaf, pred_scaf),
                ("rigor", gt_rig, pred_rig),
            ):
                if gtv == "yes" and predv == "yes":
                    counts[cls]["tp"] += 1
                elif gtv == "yes" and predv == "no":
                    counts[cls]["fn"] += 1
                elif gtv == "no" and predv == "yes":
                    counts[cls]["fp"] += 1
                # else: tn -- not counted in binary F1

        if _has_pos_result(ann):
            outcome_pos += 1

    result: dict = {"n_scenarios": n_total, "n_scored_for_f1": n_scored}
    for cls in ("scaffolding", "rigor"):
        tp, fp, fn = counts[cls]["tp"], counts[cls]["fp"], counts[cls]["fn"]
        result[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _precision(tp, fp),
            "recall": _recall(tp, fn),
            "f1": _f1(tp, fp, fn),
        }
    result["outcome_pos_rate"] = (outcome_pos / n_total) if n_total else 0.0
    return result
