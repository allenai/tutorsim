"""Versioned, experiment-agnostic prompt templates for reusable data workflows."""

from __future__ import annotations

import json
from typing import Any

PROMPT_IDS = {
    "moment_selection_pass1": "moment_selection_pass1_v1",
    "moment_selection_pass2": "moment_selection_pass2_v1",
    "dense_caption": "dense_caption_v1",
    "qa_author": "qa_author_v1",
    "qa_solver": "qa_solver_v1",
    "mmtutor_keystep_selector": "mmtutor_keystep_selector_v1",
}


def build_moment_selection_pass1_prompt(stem: str, chunk: dict[str, Any], transcript_text: str) -> str:
    return (
        "You analyze tutoring transcripts to find moments where seeing the screen is necessary to understand the dialogue.\n"
        "Return JSON only, as an array of objects with fields:\n"
        "timestamp, reason, evidence_quote, confidence, tags.\n"
        "Rules:\n"
        "- Pick moments where visual context matters: drawing/writing/showing work/deictic references/problem transitions.\n"
        "- Exclude generic chat that is understandable without visuals.\n"
        "- Return at most 12 moments for this chunk.\n"
        "- Keep only strong, non-redundant moments.\n"
        "- timestamp must be from transcript timeline and in HH:MM:SS.mmm format.\n"
        "- confidence is a number 0 to 1.\n"
        "- tags is array of short strings.\n"
        f"Session stem: {stem}\n"
        f"Chunk {chunk['chunk_id']} ({chunk['start_ts']} to {chunk['end_ts']}) transcript:\n"
        f"{transcript_text}\n"
    )


def build_moment_selection_pass2_prompt(
    stem: str, transcript_lines: list[str], candidates: list[dict[str, Any]]
) -> str:
    tx = "\n".join(transcript_lines)
    cands_json = json.dumps(candidates, ensure_ascii=True)
    return (
        "You refine candidate moments for a tutoring transcript.\n"
        "Goal: keep moments where visual context is necessary to understand meaning.\n"
        "Return JSON only, as an array of objects with fields:\n"
        "timestamp, reason, evidence_quote, confidence, tags.\n"
        "Rules:\n"
        "- Keep semantically necessary visual-dependent moments, including subtle cues.\n"
        "- Remove redundant or weak moments.\n"
        "- Return at most 60 final moments for the full transcript.\n"
        "- Do not force a target density; include all necessary moments and no filler.\n"
        "- timestamp must be HH:MM:SS.mmm and within transcript timeline.\n"
        f"Session stem: {stem}\n"
        f"Pass1 candidates JSON: {cands_json}\n"
        "Full transcript:\n"
        f"{tx}\n"
    )


def build_dense_caption_prompt() -> str:
    return (
        "Generate a dense, image-only caption for this tutoring screenshot.\n"
        "Return JSON object with fields: dense_caption, key_entities, visible_math, ui_state.\n"
        "Rules:\n"
        "- Use only visual evidence in the image.\n"
        "- Do NOT use transcript or audio assumptions.\n"
        "- Include board/work content, symbols/equations, tools/cursor behavior, and salient UI state.\n"
        "- dense_caption should be concise but detailed (roughly 80-180 words).\n"
    )


def build_qa_author_prompt(moment: dict[str, Any], caption_text: str | None) -> str:
    cap_block = f"\nIMAGE_CAPTION:\n{caption_text}\n" if caption_text else ""
    return (
        "Create exactly 4 multiple-choice questions for transcript understanding.\n"
        "Return JSON array of 4 objects with fields:\n"
        "task_type, question, choices, answer_index, requires_visual, difficulty_tag.\n"
        "Rules:\n"
        "- Exactly 2 questions with task_type='comprehend'.\n"
        "- Exactly 2 questions with task_type='predict'.\n"
        "- Each question has exactly 4 choices.\n"
        "- answer_index is integer 0..3.\n"
        "- 'predict' questions should target what happens in the next 3-6 turns (see FUTURE_TURNS).\n"
        "- Make distractors plausible and non-trivial.\n"
        "- No free-form answers; all multiple choice only.\n"
        f"MOMENT_TYPE: {moment['moment_type']}\n"
        f"TIMESTAMP: {moment['timestamp']}\n"
        f"TRANSCRIPT_CONTEXT:\n{moment['transcript_window_text']}\n"
        f"PREDICTION_CONTEXT_PREFIX:\n{moment['prediction_context_text']}\n"
        f"FUTURE_TURNS (ground truth horizon):\n{moment['future_turns_text']}\n"
        f"{cap_block}"
    )


def build_qa_solver_prompt(question: dict[str, Any], moment: dict[str, Any], caption_text: str | None) -> str:
    cap_block = f"\nIMAGE_CAPTION:\n{caption_text}\n" if caption_text else ""
    choices = question["choices"]
    return (
        "Answer the multiple-choice question using the provided context.\n"
        'Return JSON object: {"answer_index": <0-3>, "reason_short": "..."}\n'
        f"MOMENT TYPE: {moment['moment_type']}\n"
        f"TRANSCRIPT_CONTEXT:\n{moment['transcript_window_text']}\n"
        f"PREDICTION_CONTEXT_PREFIX:\n{moment['prediction_context_text']}\n"
        f"{cap_block}"
        f"QUESTION ({question['task_type']}): {question['question']}\n"
        f"0) {choices[0]}\n"
        f"1) {choices[1]}\n"
        f"2) {choices[2]}\n"
        f"3) {choices[3]}\n"
    )


def build_mmtutor_keystep_selector_prompt(
    stem: str,
    timestamp: str,
    transcript_context: str,
) -> str:
    return (
        "You are analyzing a tutoring session frame to decide whether it is a key instructional step.\n"
        "Return JSON object only with fields:\n"
        "is_key_step, score, reason, prev_step_timestamp, quality_flags.\n"
        "Field requirements:\n"
        "- is_key_step: boolean\n"
        "- score: number in [0,1]\n"
        "- reason: concise string (<=220 chars)\n"
        "- prev_step_timestamp: HH:MM:SS.mmm or empty string if unknown\n"
        "- quality_flags: object with booleans: clear_handwriting, stable_frame, aligns_with_explanation.\n"
        "Selection rules:\n"
        "- Prefer pivotal mathematical transitions that change the solving state.\n"
        "- Exclude title writing, erasing transitions, blurry frames, and final answer states without explanation.\n"
        "- Use both visual evidence and local transcript context.\n"
        f"Session stem: {stem}\n"
        f"Candidate timestamp: {timestamp}\n"
        "TRANSCRIPT_CONTEXT:\n"
        f"{transcript_context}\n"
    )
