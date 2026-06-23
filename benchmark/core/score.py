"""Per-scenario benchmark scoring.

Lucy's three-axis scoring (proposed 2026-06-15):

  scaffolding_did_rate    yes_count / (count where gold says scaffolding-appropriate). Higher better.
  rigor_did_rate          yes_count / (count where gold says rigor-appropriate).      Higher better.
  overscaffold_rate       count where any overscaffold_decomposed facet was emitted / total scenarios. Lower better.

These replace the prior per-dimension F1 numbers. The shift drops the
precision penalty (a tutor that scaffolds-on-rigor-moments no longer gets
penalized on the scaffolding dimension) and trusts the over-scaffold signal
to catch the always-both exploit instead. See `prompts/annotator/decomposer/
decompose_overscaffold.md` and Lucy's `compute_overscaffold_f1` for the
detection mechanism.

The collapsed action label decomposes via `_ACTION_LABEL_TO_DIMENSIONS`:
  both        -> scaffolding=yes, rigor=yes
  scaffolding -> scaffolding=yes, rigor=no
  rigor       -> scaffolding=no,  rigor=yes
  neither     -> scaffolding=no,  rigor=no
"unclear" (LM parse-failure) and "unknown" (gold missing-facets) are
non-substantive sentinels: scenarios with those on either side are excluded
from did-rate denominators (the gold doesn't carry a definite direction).
"""
from __future__ import annotations

_ACTION_LABEL_TO_DIMENSIONS = {
    "both":        ("yes", "yes"),
    "scaffolding": ("yes", "no"),
    "rigor":       ("no",  "yes"),
    "neither":     ("no",  "no"),
}


def _to_dims(label):
    return _ACTION_LABEL_TO_DIMENSIONS.get(label)


def _action_label_for_scenario(annotation_data: dict):
    """First substantive action_label across annotations in this scenario."""
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


def _has_overscaffold(annotation_data: dict) -> bool:
    """True if ANY annotation in the scenario emitted a non-empty
    overscaffold_decomposed list. The list is produced by Lucy's
    decompose_overscaffold step (PR #18); empty means no over-scaffolding
    detected; missing key means the over-scaffolding pass didn't run yet."""
    for ann in annotation_data.get("annotations", []) or []:
        if ann.get("overscaffold_decomposed"):
            return True
    return False


def score_scenarios(scenarios: list[dict], annotations: list[dict]) -> dict:
    """Compute Lucy's three-axis scoring + outcome+ rate.

    Returns:
        {
          "scaffolding_did": {n_yes, n_total, rate},
          "rigor_did":       {n_yes, n_total, rate},
          "overscaffold":    {n_yes, n_total, rate, available},
          "outcome_pos_rate": float,
          "n_scenarios": int,
        }

      scaffolding_did.rate  = scenarios where gold==scaffolding-yes and LM scaffolded
                              (collapsed action_label in {scaffolding, both})
                              divided by all scaffolding-appropriate scenarios.
      rigor_did.rate        = analogous for rigor-appropriate scenarios.
      overscaffold.rate     = scenarios with non-empty overscaffold_decomposed
                              divided by total scenarios. `available=False` if
                              none of the annotation files carried the field
                              (PR #18 wasn't run on this batch).
      outcome_pos_rate      = scenarios with at least one result facet labeled
                              "pos" / total.

    Scenarios where gold is mixed/unknown/neither/unclear are excluded from
    BOTH did-rate denominators -- they don't carry a clean direction.
    """
    scaf_yes = scaf_total = 0
    rig_yes = rig_total = 0
    over_yes = 0
    outcome_pos = 0
    any_overscaffold_field = False
    n_total = 0

    # Calibrated scoring (Lucy + Ryan final spec, 2026-06-17):
    #   scaffold_calibrated = n_scaffolded_cleanly / n_scaffold_moments
    #   rigor_calibrated    = n_rigor_pushed_cleanly / n_rigor_moments
    # "cleanly" = right action direction AND no over-scaffold facets.
    # Both axes are symmetric: count clean moments / total moments. Range [0, 1].
    #
    # NOTE: an earlier version subtracted n_over_scaffolded from the scaffold
    # numerator. That double-penalized -- a moment that over-scaffolds is
    # already excluded from n_clean_yes (clean requires no over-scaffold), so
    # subtracting it again counted it twice. Lucy/Ryan removed the subtraction;
    # scaf_over_yes is kept as a reported component only (not in the score).
    scaf_clean_yes = 0    # scaffold-gold + action right + no over-scaffold
    scaf_over_yes = 0     # scaffold-gold + over-scaffold emitted (reported, not scored)
    rig_clean_yes = 0     # rigor-gold + action right + no over-scaffold

    for scenario, ann in zip(scenarios, annotations):
        n_total += 1

        gt_label = (scenario.get("detection") or {}).get("situation_label_agg")
        pred_label = _action_label_for_scenario(ann)
        pred_dims = _to_dims(pred_label)
        has_over = _has_overscaffold(ann)

        if gt_label == "scaffolding":
            scaf_total += 1
            action_right = pred_dims is not None and pred_dims[0] == "yes"
            if action_right:
                scaf_yes += 1
                if not has_over:
                    scaf_clean_yes += 1
            if has_over:
                scaf_over_yes += 1
        elif gt_label == "rigor":
            rig_total += 1
            action_right = pred_dims is not None and pred_dims[1] == "yes"
            if action_right:
                rig_yes += 1
                if not has_over:
                    rig_clean_yes += 1
        # mixed / both / neither / unknown / unclear -> excluded from both
        # did-rate denominators. Still counts toward outcome and overscaffold.

        if _has_pos_result(ann):
            outcome_pos += 1

        for a in ann.get("annotations", []) or []:
            if "overscaffold_decomposed" in a:
                any_overscaffold_field = True
                break
        if has_over:
            over_yes += 1

    def _rate(yes, total):
        return (yes / total) if total else None

    return {
        "n_scenarios": n_total,
        "scaffolding_did": {
            "n_yes": scaf_yes,
            "n_total": scaf_total,
            "rate": _rate(scaf_yes, scaf_total),
        },
        "rigor_did": {
            "n_yes": rig_yes,
            "n_total": rig_total,
            "rate": _rate(rig_yes, rig_total),
        },
        "overscaffold": {
            "n_yes": over_yes,
            "n_total": n_total,
            "rate": _rate(over_yes, n_total),
            "available": any_overscaffold_field,
        },
        "outcome_pos_rate": (outcome_pos / n_total) if n_total else 0.0,
        # Calibrated scores -- subsume did-rate + over-scaffold into one
        # number per axis. Components exposed so other formulas can be
        # recomputed from the same data without re-running annotation.
        "scaffold_calibrated": {
            "n_clean_yes": scaf_clean_yes,
            "n_overscaffold": scaf_over_yes,
            "n_total": scaf_total,
            "score": _rate(scaf_clean_yes, scaf_total),
        },
        "rigor_calibrated": {
            "n_clean_yes": rig_clean_yes,
            "n_total": rig_total,
            "score": _rate(rig_clean_yes, rig_total),
        },
    }
