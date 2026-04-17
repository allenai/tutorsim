"""Per-style scoring utilities for benchmark annotations."""


DEFAULT_LABEL_WEIGHTS = {
    "effective": 1.0,
    "partial": 0.5,
    "ineffective": 0.0,
    "unclear": 0.0,
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
