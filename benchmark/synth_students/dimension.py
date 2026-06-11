"""
Dimension descriptions for characterizing student behavior in tutoring conversations.

Each dimension below describes what to attend to in the transcript and how to
summarize the student's behavior along a continuous spectrum (not as a discrete label).
"""

from typing import Dict, List


# ============================================================================
# ENGAGEMENT DIMENSION DESCRIPTIONS
# ============================================================================

DISTRACTEDNESS_DESC = """Distractedness: How consistently the student stays focused on the learning task over the course of the conversation. 
Consider:
- Topical focus: Are their responses generally tied to the current problem or concept, or do they drift to unrelated topics?
- Evidence of listening: Do they reference, paraphrase, or build on what the tutor has said, or do they ignore key information and repeat earlier confusion?
- Topic shifting: When topics change, is this motivated by the learning task (e.g., trying a new strategy), or abrupt and unrelated to the goal?
- Signs of divided attention: Do delays, very short replies, or obviously rushed answers suggest that their attention is split?"""

ACTIVE_VS_PASSIVE_LEARNING_DESC = """Active Learning: How much initiative the student takes in their own learning during this conversation. 
Consider:
- Question asking: Do they ask clarifying or exploratory questions to resolve confusion or deepen understanding, or do they mainly answer when prompted?
- Initiation of ideas: Do they propose their own approaches, checks, or hypotheses, or mostly follow the tutor's suggested steps?
- Response depth: Do they explain their reasoning and reflect on their choices, or give brief, surface-level answers?"""

AFFECT_DESC = """Affect: The emotions exhibited by the student throughout the conversation. 
Consider:
- Confusion: Does the student express uncertainty, admit they don't understand, or show signs of being lost or overwhelmed by the material?
- Happiness: Does the student express positive emotions, enthusiasm, satisfaction with progress, or excitement about learning?
- Anxiety: Does the student show nervousness, worry about performance, express self-doubt, or display signs of stress or pressure?
- Frustration: Does the student express annoyance, impatience, exasperation, or dissatisfaction with the difficulty of the task or their progress?"""

# ============================================================================
# COGNITIVE DIMENSION DESCRIPTIONS
# ============================================================================

MISCONCEPTIONS_DESC = """Misconceptions: The student's misconceptions about the subject matter as revealed in this conversation. 
Consider:
- Specific incorrect beliefs or mental models such as systematic misuse of a rule, confusion between two concepts, or misinterpretation of notation.
- Systematic vs. isolated errors: Do their mistakes form a consistent pattern suggesting a deeper misconception, or do they look like rare slips or calculation errors?"""

LEARNING_EFFICIENCY_DESC = """Learning Efficiency: How readily the student revises incorrect beliefs and integrates new understanding when provided with feedback or guidance.
Consider:
- Speed of learning: How many explanations or examples does the student need before showing signs of understanding? Do they quickly grasp corrections or require repeated instruction?
- Generalization: When they correct one instance of an error, do they apply that correction to similar problems, or does the misconception resurface in new contexts?
- Evidence of conceptual change: Do their later responses show genuine understanding of why their initial approach was wrong, or do they mechanically follow corrections without deeper insight?"""

# ============================================================================
# DIMENSION DICTIONARIES
# ============================================================================

ENGAGEMENT_STATIC_DIMENSIONS: Dict[str, str] = {
    "distractedness": DISTRACTEDNESS_DESC,
    "active_vs_passive_learning": ACTIVE_VS_PASSIVE_LEARNING_DESC,
    "affect": AFFECT_DESC,
}

ENGAGEMENT_CHANGE_DIMENSIONS: Dict[str, str] = {}

COGNITIVE_STATIC_DIMENSIONS: Dict[str, str] = {
    "misconceptions": MISCONCEPTIONS_DESC,
}

COGNITIVE_CHANGE_DIMENSIONS: Dict[str, str] = {
    "learning_efficiency": LEARNING_EFFICIENCY_DESC,
}

ALL_ENGAGEMENT_DIMENSIONS: Dict[str, str] = {
    **ENGAGEMENT_STATIC_DIMENSIONS,
    **ENGAGEMENT_CHANGE_DIMENSIONS,
}

ALL_COGNITIVE_DIMENSIONS: Dict[str, str] = {
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

    Args:
        persona_dimension: The persona dimension to describe

    Returns:
        Description string for the dimension
    """

    if persona_dimension not in ALL_DIMENSION_NAMES:
        raise ValueError(
            f"Invalid persona dimension: {persona_dimension}. Must be one of: {ALL_DIMENSION_NAMES}"
        )

    # 'joined' should never reach here - it's handled specially in TraitGenerator
    if persona_dimension in [
        "joined",
        "joined_misconceptions_distractedness",
        "joined_misconceptions_affect",
    ]:
        raise ValueError(
            f"'{persona_dimension}' persona dimension should not call get_dimension_description()"
        )

    # if all, return all dimension descriptions
    if persona_dimension == "all":

        # number them
        all_descriptions = []
        sorted_descriptions = sorted(list(ALL_DIMENSIONS.values()))
        # random shuffle
        for i, description in enumerate(sorted_descriptions):
            all_descriptions.append(f"{i+1}. {description}")
        return "\n\n".join(all_descriptions)

    elif persona_dimension == "all_engagement":
        all_descriptions = []
        sorted_descriptions = [
            get_dimension_description(dimension)
            for dimension in sorted(ALL_ENGAGEMENT_DIMENSIONS.keys())
        ]
        # random shuffle
        for i, description in enumerate(sorted_descriptions):
            all_descriptions.append(f"{i+1}. {description}")
        return "\n\n".join(all_descriptions)

    elif persona_dimension == "all_cognitive":
        all_descriptions = []
        sorted_descriptions = [
            get_dimension_description(dimension)
            for dimension in sorted(ALL_COGNITIVE_DIMENSIONS.keys())
        ]
        for i, description in enumerate(sorted_descriptions):
            all_descriptions.append(f"{i+1}. {description}")
        return "\n\n".join(all_descriptions)

    return ALL_DIMENSIONS[persona_dimension]
