"""Human-baseline replay: first 5 speaking turns (TSTST) after the cut point.

The AI tutors are scored on a 5-speaking-turn continuation; the human row must use
the SAME horizon -- the real human's post-cut turns, bounded to 5 speaking turns,
alternating TUTOR/STUDENT/TUTOR/STUDENT/TUTOR -- so it's directly comparable. The
real transcript is messy (PAUSE pseudo-turns, split/consecutive same-speaker lines),
so extraction drops non-dialogue markers and merges consecutive same-speaker lines.
"""

from tutor_bench.benchmark.human import extract_human_turns

REF = (
    "Turn 28. TUTOR: Skip nine for now.\n"
    "Turn 29. STUDENT: Oh okay.\n"
    "Turn 30. TUTOR: [PAUSE: 217 seconds]\n"
    "Turn 30. STUDENT: I'm confused on fourteen.\n"
    "Turn 31. TUTOR: [PAUSE: 3 seconds]\n"
    "Turn 31. TUTOR: Have we done algebra?\n"
    "Turn 32. STUDENT: I remember subtraction.\n"
    "Turn 33. TUTOR: What is the first step?\n"
    "Turn 34. STUDENT: Um.\n"
    "Turn 35. TUTOR: Try isolating x.\n"
)


class TestExtractHumanTurns:
    def test_caps_at_five_speaking_turns(self):
        assert len(extract_human_turns(REF)) == 5

    def test_drops_pause_markers(self):
        assert all("PAUSE" not in t["text"] for t in extract_human_turns(REF))

    def test_alternates_tstst_uppercase_roles(self):
        roles = [t["role"] for t in extract_human_turns(REF)]
        assert roles == ["TUTOR", "STUDENT", "TUTOR", "STUDENT", "TUTOR"]

    def test_merges_consecutive_same_speaker(self):
        # Student turns 29 + 30 (consecutive after the dropped pause) become one.
        second = extract_human_turns(REF)[1]
        assert "Oh okay." in second["text"] and "fourteen" in second["text"]

    def test_empty_reference_returns_empty(self):
        assert extract_human_turns("") == []

    def test_student_first_reference_is_made_tutor_first(self):
        # The AI replay is always tutor-first; a student-first reference must
        # drop the leading student so the human is 1:1 (T,S,T,S,T).
        ref = (
            "Turn 50. STUDENT: wait I'm lost.\n"
            "Turn 51. TUTOR: okay, what's the first step?\n"
            "Turn 52. STUDENT: add them?\n"
            "Turn 53. TUTOR: try it.\n"
            "Turn 54. STUDENT: ok.\n"
            "Turn 55. TUTOR: nice.\n"
        )
        turns = extract_human_turns(ref)
        assert turns[0]["role"] == "TUTOR"
        assert "lost" not in turns[0]["text"]
        assert [t["role"] for t in turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT", "TUTOR"]


class TestBuildHumanTranscript:
    def test_renumbers_sequentially_from_cut_turn_like_the_ai(self):
        """The AI numbers generated turns cut_turn+1, +2, ... in conversation.py.
        The human transcript must use the SAME scheme so score() sees identical
        turn numbers -- not the real, post-merge non-contiguous source numbers."""
        from tutor_bench.benchmark.human import build_human_transcript

        class _Scenario:
            id = "balanced_520:00001"
            student = {"reference": REF}
            provenance = {"cut_turn": 27}

        t = build_human_transcript(_Scenario(), max_turns=5)
        nums = [x["turn_number"] for x in t.generated_turns]
        assert nums == [28, 29, 30, 31, 32]
        assert t.tutor_model == "human"
