"""
Dimension descriptions for characterizing student behavior in tutoring conversations.

Each dimension's text lives in `prompts/benchmark/v6/students/dimensions/{name}.txt`
and is loaded on demand. This module owns the dispatch and aggregation logic
(joined, all, all_engagement, all_cognitive) -- the prose is on disk.
"""

from pathlib import Path
from typing import Dict, List


_DIMENSIONS_DIR = (
    Path(__file__).parent.parent.parent
    / "prompts" / "benchmark" / "v6" / "students" / "dimensions"
)


def _load_dimension_text(name: str) -> str:
    path = _DIMENSIONS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


# Registry of dimension names by group. Text is loaded from disk on demand
# via `get_dimension_description(name)` below.
ENGAGEMENT_STATIC_DIMENSIONS: Dict[str, None] = {
    "distractedness": None,
    "active_vs_passive_learning": None,
    "affect": None,
}

ENGAGEMENT_CHANGE_DIMENSIONS: Dict[str, None] = {}

COGNITIVE_STATIC_DIMENSIONS: Dict[str, None] = {
    "misconceptions": None,
}

COGNITIVE_CHANGE_DIMENSIONS: Dict[str, None] = {
    "learning_efficiency": None,
}

ALL_ENGAGEMENT_DIMENSIONS: Dict[str, None] = {
    **ENGAGEMENT_STATIC_DIMENSIONS,
    **ENGAGEMENT_CHANGE_DIMENSIONS,
}

ALL_COGNITIVE_DIMENSIONS: Dict[str, None] = {
    **COGNITIVE_STATIC_DIMENSIONS,
    **COGNITIVE_CHANGE_DIMENSIONS,
}

ALL_DIMENSIONS = {**ALL_ENGAGEMENT_DIMENSIONS, **ALL_COGNITIVE_DIMENSIONS}

ALL_DIMENSION_NAMES = list(ALL_DIMENSIONS.keys()) + [
    "all",
    "all_engagement",
    "all_cognitive",
    "joined",
    "joined_misconceptions_distractedness",
    "joined_misconceptions_affect",
]


def get_dimension_description(persona_dimension: str) -> str:
    """
    Get a description of what a persona dimension represents.

    Single dimensions load from `prompts/.../dimensions/{name}.txt`. The
    aggregate names ("all", "all_engagement", "all_cognitive") compose
    numbered descriptions across the relevant group. "joined" forms are
    handled in `TraitGenerator`, not here.
    """
    if persona_dimension not in ALL_DIMENSION_NAMES:
        raise ValueError(
            f"Invalid persona dimension: {persona_dimension}. Must be one of: {ALL_DIMENSION_NAMES}"
        )

    if persona_dimension in [
        "joined",
        "joined_misconceptions_distractedness",
        "joined_misconceptions_affect",
    ]:
        raise ValueError(
            f"'{persona_dimension}' persona dimension should not call get_dimension_description()"
        )

    if persona_dimension == "all":
        texts = sorted(_load_dimension_text(n) for n in ALL_DIMENSIONS.keys())
        return "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_engagement":
        texts = [get_dimension_description(n) for n in sorted(ALL_ENGAGEMENT_DIMENSIONS.keys())]
        return "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_cognitive":
        texts = [get_dimension_description(n) for n in sorted(ALL_COGNITIVE_DIMENSIONS.keys())]
        return "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    return _load_dimension_text(persona_dimension)
