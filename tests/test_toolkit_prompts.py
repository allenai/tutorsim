from __future__ import annotations

from tutor_bench.toolkit.prompts import (
    PROMPT_IDS,
    build_dense_caption_prompt,
    build_moment_selection_pass1_prompt,
    build_moment_selection_pass2_prompt,
    build_qa_author_prompt,
    build_qa_solver_prompt,
)


def test_prompt_ids_are_present() -> None:
    expected = {"moment_selection_pass1", "moment_selection_pass2", "dense_caption", "qa_author", "qa_solver"}
    assert expected.issubset(set(PROMPT_IDS))


def test_build_moment_selection_prompts_include_expected_context() -> None:
    p1 = build_moment_selection_pass1_prompt(
        stem="stem_x",
        chunk={"chunk_id": 3, "start_ts": "00:01:00.000", "end_ts": "00:05:00.000"},
        transcript_text="[00:01:02.000 - 00:01:05.000] TUTOR: hello",
    )
    assert "Session stem: stem_x" in p1
    assert "Chunk 3 (00:01:00.000 to 00:05:00.000)" in p1

    p2 = build_moment_selection_pass2_prompt(
        stem="stem_x",
        transcript_lines=["[00:01:02.000 - 00:01:05.000] TUTOR: hello"],
        candidates=[
            {
                "timestamp": "00:01:03.000",
                "reason": "visual ref",
                "evidence_quote": "hello",
                "confidence": 0.8,
                "tags": ["visual_ref"],
                "source": "pass1",
                "chunk_id": 3,
            }
        ],
    )
    assert "Session stem: stem_x" in p2
    assert "Pass1 candidates JSON:" in p2


def test_build_qa_and_caption_prompts_include_fields() -> None:
    caption = build_dense_caption_prompt()
    assert "dense, image-only caption" in caption

    q_prompt = build_qa_author_prompt(
        {
            "moment_type": "selected",
            "timestamp": "00:10:00.000",
            "transcript_window_text": "ctx",
            "prediction_context_text": "pref",
            "future_turns_text": "future",
        },
        caption_text="dense cap",
    )
    assert "MOMENT_TYPE: selected" in q_prompt
    assert "IMAGE_CAPTION" in q_prompt

    s_prompt = build_qa_solver_prompt(
        {"task_type": "predict", "question": "Q?", "choices": ["A", "B", "C", "D"]},
        {"moment_type": "selected", "transcript_window_text": "ctx", "prediction_context_text": "pref"},
        caption_text=None,
    )
    assert "QUESTION (predict): Q?" in s_prompt
    assert "0) A" in s_prompt
