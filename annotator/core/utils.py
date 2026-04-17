"""Shared pipeline utilities: IoU, clustering, transcript loading/formatting, excerpts."""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

RESULTS_DIR = REPO_ROOT / "results" / "annotator"

from .config import get_iou_threshold

# Re-export for backwards compatibility with scripts that import from utils
IOU_THRESHOLD = get_iou_threshold()

# Conversations used as few-shot examples in prompts.
# These MUST be excluded from evaluation to prevent data leakage.
# Each ID corresponds to a verbatim ground-truth annotation used in p2 prompts.
EXAMPLE_CONV_IDS = frozenset([
    # Scaffolding prompt examples (p2/scaffolding.txt)
    "2024-t24830_2024-s10698_29025873-9511-491c-b386-7f63baaa42ee",  # effective: pizza slices 6/6=1
    "2025-t27246_2025-s12454_8d1395e1-e407-4f5f-b077-fd0f0a2fa949",  # partial: two-step equations
    "2024-t22698_2025-s11513_6b8d3f6f-77c1-48e1-b093-bec862e28cb9",  # ineffective: step-by-step without student
    # Rapport prompt examples (p2/rapport.txt)
    "2025-t27030_2024-s7147_1e5ce9d7-b75a-4329-934a-11141633c27e",   # effective: adapts to sick student
    "2025-t27253_2025-s12492_c790ab4b-97be-4670-926b-631157179be6",   # partial: weekend question no follow-up
    "2024-t23317_2024-s8511_21050adf-8b91-45c9-a536-46816d9daa08",    # ineffective: missed "been better" cue
])


def compute_iou(range_a, range_b):
    """Compute IoU between two (start, end) ranges."""
    set_a = set(range(range_a[0], range_a[1] + 1))
    set_b = set(range(range_b[0], range_b[1] + 1))
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def merge_overlapping_ranges(moments):
    """Merge overlapping/adjacent human moments (same type) into clusters."""
    if not moments:
        return []

    by_type = defaultdict(list)
    for m in moments:
        by_type[m.get("annotation_type", "unknown")].append(m)

    clusters = []
    for ann_type, type_moments in by_type.items():
        sorted_moments = sorted(type_moments, key=lambda m: (m["turn_start"], m["turn_end"]))
        current = {
            "turn_start": sorted_moments[0]["turn_start"],
            "turn_end": sorted_moments[0]["turn_end"],
            "annotation_type": ann_type,
            "moments": [sorted_moments[0]],
        }
        for m in sorted_moments[1:]:
            if m["turn_start"] <= current["turn_end"] + 1:
                current["turn_end"] = max(current["turn_end"], m["turn_end"])
                current["moments"].append(m)
            else:
                clusters.append(current)
                current = {
                    "turn_start": m["turn_start"],
                    "turn_end": m["turn_end"],
                    "annotation_type": ann_type,
                    "moments": [m],
                }
        clusters.append(current)

    return clusters


def load_transcripts():
    """Load all transcripts into a dict keyed by conversation ID."""
    from .storage import load_all_transcripts
    return load_all_transcripts()


def load_ground_truth(annotator_style: str | None = None) -> dict:
    """Load ground truth from per-conversation files.

    Args:
        annotator_style: If provided, filter to moments from annotators
            matching this archetype (generous/balanced/demanding).

    Returns:
        Dict in legacy format: {"conversations": {conv_id: {"num_turns": N, "key_moments": [...]}}}
    """
    from .storage import load_all_ground_truth_files

    from .config import get_archetype_annotators
    filter_ids = get_archetype_annotators(annotator_style) if annotator_style else None

    conversations = {}
    for data in load_all_ground_truth_files():
        conv_id = data["conversation_id"]
        moments = data["key_moments"]

        if filter_ids:
            moments = [m for m in moments if m.get("annotator_id") in filter_ids]

        if moments:
            conversations[conv_id] = {
                "num_turns": data["num_turns"],
                "key_moments": moments,
            }

    return {"conversations": conversations}


def get_excerpt(transcripts, conv_id, turn_start, turn_end, context=5,
                bold_range=False):
    """Get a transcript excerpt around a moment.

    Args:
        bold_range: If True, use ** prefix for turns in the moment range
                    and add START/END markers. If False, use <<< suffix.
    """
    conv = transcripts.get(conv_id)
    if not conv:
        return "[transcript not found]"
    turns = conv.get("turns", [])
    if not turns:
        return "[no turns]"
    min_turn = turns[0]["turn_number"]
    max_turn = turns[-1]["turn_number"]
    start = max(min_turn, turn_start - context)
    end = min(max_turn, turn_end + context)
    lines = []
    for turn in turns:
        n = turn["turn_number"]
        if n < start or n > end:
            continue
        text = turn["text"][:200]
        if bold_range:
            marker = ""
            if n == turn_start:
                marker = " >>> MOMENT START <<<"
            if n == turn_end:
                marker = " >>> MOMENT END <<<"
            prefix = "**" if turn_start <= n <= turn_end else "  "
            lines.append(f"{prefix}Turn {n}. {turn['role']}: {text}{marker}")
        else:
            marker = " <<<" if turn_start <= n <= turn_end else ""
            lines.append(f"  Turn {n}. {turn['role']}: {text}{marker}")
    return "\n".join(lines)


# ===================================================================
# Transcript formatting for pipeline passes
# ===================================================================

def _filter_turns(turns: list[dict], dialogue_only: bool) -> list[dict]:
    """Filter turns based on dialogue_only flag.

    When dialogue_only=True, only include turns with type == "DIALOGUE".
    When False (enriched mode), include all turns.
    """
    if not dialogue_only:
        return turns
    return [t for t in turns if t.get("type", "DIALOGUE") == "DIALOGUE"]


def format_transcript(conversation: dict, dialogue_only: bool = False) -> str:
    """Format conversation turns as: Turn {n}. {ROLE}: {text}

    Args:
        conversation: Consolidated conversation dict with "turns" key.
        dialogue_only: If True, exclude non-dialogue turns (enrichments).
    """
    lines = []
    for turn in _filter_turns(conversation["turns"], dialogue_only):
        n = turn["turn_number"]
        role = turn["role"]
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)


def format_excerpt(conversation: dict, turn_start: int, turn_end: int,
                   context_before: int = 20, context_after: int = 20,
                   dialogue_only: bool = False) -> str:
    """Extract a transcript excerpt around a detected moment, with context.

    Outputs the detected range with >>> markers, surrounded by context turns.
    This lets Pass 2 see what happened before and after for look-ahead analysis.

    Args:
        conversation: Consolidated conversation dict.
        turn_start: First turn of the detected moment.
        turn_end: Last turn of the detected moment.
        context_before: Number of turns before the detection to include.
        context_after: Number of turns after the detection to include.
        dialogue_only: If True, exclude non-dialogue turns (enrichments).
    """
    turns = _filter_turns(conversation["turns"], dialogue_only)
    if not turns:
        return ""

    # Find the actual min/max turn numbers in the transcript
    all_turn_nums = [t["turn_number"] for t in turns]
    min_turn = min(all_turn_nums)
    max_turn = max(all_turn_nums)

    # Calculate excerpt boundaries
    excerpt_start = max(min_turn, turn_start - context_before)
    excerpt_end = min(max_turn, turn_end + context_after)

    lines = []

    # Header if we're not starting from the beginning
    if excerpt_start > min_turn:
        lines.append(f"[... turns 1-{excerpt_start - 1} omitted ...]")
        lines.append("")

    for turn in turns:
        n = turn["turn_number"]
        if n < excerpt_start or n > excerpt_end:
            continue

        # Add markers around the detected moment
        if n == turn_start:
            lines.append(f">>> DETECTED MOMENT START (Turn {turn_start}) <<<")

        role = turn["role"]
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")

        if n == turn_end:
            lines.append(f">>> DETECTED MOMENT END (Turn {turn_end}) <<<")

    # Footer if we're not ending at the last turn
    if excerpt_end < max_turn:
        lines.append("")
        lines.append(f"[... turns {excerpt_end + 1}-{max_turn} omitted ...]")

    return "\n".join(lines)
