"""Human-baseline replay for the leaderboard's Human row.

Scores the real human tutor's post-cut continuation (not an AI mimic) on the same
horizon the AI tutors get: the first 5 speaking turns after the cut, alternating
TUTOR/STUDENT/TUTOR/STUDENT/TUTOR. The real transcript is messy, so extraction
drops non-dialogue markers (e.g. ``[PAUSE: 217 seconds]``) and merges consecutive
same-speaker lines into one speaking turn before taking the first 5.
"""

import re

from tutor_bench.benchmark.conversation import Transcript

_LINE_RE = re.compile(r"^Turn\s+(\d+)\.\s+(\w+):\s*(.*)$")


def extract_human_turns(reference: str, max_turns: int = 5) -> list[dict]:
    """Parse a reference transcript into up to `max_turns` speaking turns.

    `reference` is the ``Turn N. ROLE: text`` block from `scenario.student["reference"]`.
    Returns ``[{turn_number, role, text}]`` with role UPPERCASE (matching what the
    scorer expects for generated turns), pause/enrichment-only lines dropped, and
    consecutive same-speaker lines merged.
    """
    merged: list[dict] = []
    for line in (reference or "").splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        n, role, text = int(m.group(1)), m.group(2).upper(), m.group(3).strip()
        if not text:
            continue
        # A line that is entirely a bracketed marker ("[PAUSE: ...]", "[IMAGE]")
        # is non-dialogue; the AI replay has none, so drop it for comparability.
        if text.startswith("[") and text.endswith("]"):
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["text"] = (merged[-1]["text"] + " " + text).strip()
        else:
            merged.append({"turn_number": n, "role": role, "text": text})
    # Tutor-first: the AI replay ALWAYS generates a TUTOR turn first, so for a 1:1
    # comparison drop any leading STUDENT turn(s) before taking the 5-turn TSTST
    # window. (Affects the references that open on a student turn.)
    while merged and merged[0]["role"] != "TUTOR":
        merged.pop(0)
    return merged[:max_turns]


def build_human_transcript(scenario, max_turns: int = 5) -> Transcript:
    """Build a Transcript whose generated turns are the human's first 5 speaking turns.

    Turn numbers are reassigned sequentially from ``cut_turn + 1`` -- identical to how
    the AI run numbers its generated turns -- so ``score()`` sees the same turn
    numbering it would for an AI tutor. (After dropping pauses and merging
    same-speaker lines the real source numbers go non-contiguous, which would
    otherwise make the scorer input differ from an AI run.)
    """
    reference = (scenario.student or {}).get("reference", "")
    turns = extract_human_turns(reference, max_turns)
    cut_turn = scenario.provenance["cut_turn"]
    for i, turn in enumerate(turns):
        turn["turn_number"] = cut_turn + 1 + i
    return Transcript(
        scenario_id=scenario.id,
        tutor_model="human",
        generated_turns=turns,
        ended_via="HUMAN_REPLAY",
    )
