"""Aggregate annotation scores across styles into composite ratings."""

from dataclasses import dataclass, field, asdict


@dataclass
class ScenarioScore:
    scenario_id: str
    tutor_model: str
    style_labels: dict = field(default_factory=dict)   # {style: {scaffolding: label, rapport: label}}
    composite_score: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelSummary:
    tutor_model: str
    n_scenarios: int = 0
    mean_score: float = 0.0
    by_mode: dict = field(default_factory=dict)           # {key_moment: mean, random: mean}
    style_breakdown: dict = field(default_factory=dict)   # {generous: mean, balanced: mean, ...}
    type_breakdown: dict = field(default_factory=dict)    # {scaffolding: mean, rapport: mean}

    def to_dict(self):
        return asdict(self)


DEFAULT_LABEL_WEIGHTS = {
    "effective": 1.0,
    "partial": 0.5,
    "ineffective": 0.0,
    "unclear": 0.0,
}

DEFAULT_STYLE_WEIGHTS = {
    "generous": 0.25,
    "balanced": 0.50,
    "demanding": 0.25,
}


def label_to_score(label: str, weights: dict | None = None) -> float:
    """Convert an effectiveness label to a numeric score."""
    w = weights or DEFAULT_LABEL_WEIGHTS
    return w.get(label, 0.0)


def extract_effectiveness_by_type(annotations_data: dict) -> dict[str, str]:
    """Extract effectiveness labels per annotation_type from labeled annotations.

    Returns: {annotation_type: effectiveness_label}
    e.g. {"scaffolding": "effective", "rapport": "partial"}
    """
    if not annotations_data:
        return {}

    labels = {}
    results = annotations_data.get("results", {})
    for conv_id, conv_data in results.items():
        for ann in conv_data.get("annotations", []):
            ann_type = ann.get("annotation_type", "unknown")
            label = ann.get("effectiveness", "unclear")
            if label in ("effective", "partial", "ineffective"):
                labels[ann_type] = label
    return labels


def compute_composite_score(
    style_type_labels: dict[str, dict[str, str]],
    label_weights: dict | None = None,
    style_weights: dict | None = None,
) -> float:
    """Compute weighted composite score from per-style, per-type labels.

    Args:
        style_type_labels: {style: {annotation_type: label}}
        label_weights: {label: numeric_score}
        style_weights: {style: weight}

    Returns:
        Weighted average score in [0, 1]
    """
    lw = label_weights or DEFAULT_LABEL_WEIGHTS
    sw = style_weights or DEFAULT_STYLE_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for style, type_labels in style_type_labels.items():
        style_weight = sw.get(style, 1.0 / len(style_type_labels))

        if not type_labels:
            continue

        # Average across annotation types within this style
        type_scores = [label_to_score(label, lw) for label in type_labels.values()]
        style_score = sum(type_scores) / len(type_scores)

        weighted_sum += style_weight * style_score
        total_weight += style_weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def aggregate_scenario(
    scenario_id: str,
    tutor_model: str,
    style_annotations: dict[str, dict],
    label_weights: dict | None = None,
    style_weights: dict | None = None,
) -> ScenarioScore:
    """Aggregate 3-style annotations into a single scenario score."""
    style_type_labels = {}
    for style, ann_data in style_annotations.items():
        style_type_labels[style] = extract_effectiveness_by_type(ann_data)

    composite = compute_composite_score(style_type_labels, label_weights, style_weights)

    return ScenarioScore(
        scenario_id=scenario_id,
        tutor_model=tutor_model,
        style_labels=style_type_labels,
        composite_score=composite,
    )


def aggregate_model(
    tutor_model: str,
    scenario_scores: list[ScenarioScore],
    scenario_modes: dict[str, str] | None = None,
) -> ModelSummary:
    """Aggregate all scenario scores for a single tutor model."""
    if not scenario_scores:
        return ModelSummary(tutor_model=tutor_model)

    scores = [s.composite_score for s in scenario_scores]
    mean = sum(scores) / len(scores)

    # Per-mode breakdown
    by_mode = {}
    if scenario_modes:
        mode_scores = {}
        for s in scenario_scores:
            m = scenario_modes.get(s.scenario_id, "unknown")
            mode_scores.setdefault(m, []).append(s.composite_score)
        for m, ms in mode_scores.items():
            by_mode[m] = sum(ms) / len(ms) if ms else 0.0

    # Per-style breakdown
    style_scores = {}
    for s in scenario_scores:
        for style, type_labels in s.style_labels.items():
            if not type_labels:
                continue
            type_score_vals = [label_to_score(l) for l in type_labels.values()]
            avg = sum(type_score_vals) / len(type_score_vals)
            style_scores.setdefault(style, []).append(avg)
    style_breakdown = {
        style: sum(ss) / len(ss) if ss else 0.0
        for style, ss in style_scores.items()
    }

    # Per-annotation-type breakdown
    type_scores = {}
    for s in scenario_scores:
        for style, type_labels in s.style_labels.items():
            for ann_type, label in type_labels.items():
                type_scores.setdefault(ann_type, []).append(label_to_score(label))
    type_breakdown = {
        ann_type: sum(ss) / len(ss) if ss else 0.0
        for ann_type, ss in type_scores.items()
    }

    return ModelSummary(
        tutor_model=tutor_model,
        n_scenarios=len(scenario_scores),
        mean_score=mean,
        by_mode=by_mode,
        style_breakdown=style_breakdown,
        type_breakdown=type_breakdown,
    )
