from __future__ import annotations

from pathlib import Path

from tutor_bench.toolkit.transcript_utils import context_prefix, context_window, parse_gold_transcript, transcript_lines


def test_parse_gold_transcript_and_lines(tmp_path: Path) -> None:
    p = tmp_path / "sample_transcript.txt"
    p.write_text(
        "\n".join(
            [
                "[00:00:01.000 - 00:00:03.000] TUTOR: Hello there",
                "[00:00:03.500 - 00:00:06.000] STUDENT: Hi",
                "malformed line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    segs = parse_gold_transcript(p)
    assert len(segs) == 2
    assert segs[0].role == "TUTOR"
    assert segs[1].text == "Hi"

    lines = transcript_lines(segs)
    assert lines[0].startswith("[00:00:01.000 - 00:00:03.000] TUTOR:")
    assert "STUDENT: Hi" in lines[1]


def test_context_window_and_prefix(tmp_path: Path) -> None:
    p = tmp_path / "ctx_transcript.txt"
    p.write_text(
        "\n".join(
            [
                "[00:00:00.000 - 00:00:02.000] TUTOR: A",
                "[00:00:02.000 - 00:00:04.000] STUDENT: B",
                "[00:00:04.000 - 00:00:06.000] TUTOR: C",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    segs = parse_gold_transcript(p)
    prefix = context_prefix(segs, center_t=4.0, pre_s=1.5)
    window = context_window(segs, center_t=4.0, pre_s=3.0, post_s=1.5)

    assert "STUDENT: B" in prefix
    assert "TUTOR: C" in prefix
    assert "TUTOR: A" not in prefix
    assert "TUTOR: C" in window
