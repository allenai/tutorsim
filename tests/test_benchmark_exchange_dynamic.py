"""Tests for dynamic-length benchmark exchanges (tutor [END] + max_turns cap)."""

import pytest

from benchmark.core.exchange import (
    _check_end_token, END_TOKEN,
    _parse_tutor_tokens, NEXT_PROBLEM_TOKEN,
)


def test_check_end_token_trailing():
    text, ended = _check_end_token("Great job on that. [END]")
    assert ended is True
    assert text == "Great job on that."


def test_check_end_token_trailing_on_own_line():
    text, ended = _check_end_token("Great job on that.\n[END]")
    assert ended is True
    assert text == "Great job on that."


def test_check_end_token_absent():
    text, ended = _check_end_token("Keep going!")
    assert ended is False
    assert text == "Keep going!"


def test_check_end_token_mid_text():
    text, ended = _check_end_token("ok [END] bye")
    assert ended is True
    # Token stripped; surrounding text preserved (interior whitespace acceptable):
    assert "[END]" not in text
    assert "ok" in text and "bye" in text


def test_check_end_token_alone():
    text, ended = _check_end_token("[END]")
    assert ended is True
    assert text == ""


def test_check_end_token_constant_value():
    assert END_TOKEN == "[END]"


def test_parse_tutor_tokens_end_only():
    text, ended, next_p = _parse_tutor_tokens("Wrap up. [END]")
    assert ended is True and next_p is False
    assert text == "Wrap up."


def test_parse_tutor_tokens_next_problem_only():
    text, ended, next_p = _parse_tutor_tokens("Great work. [NEXT_PROBLEM]")
    assert ended is False and next_p is True
    assert text == "Great work."


def test_parse_tutor_tokens_both_end_wins():
    text, ended, next_p = _parse_tutor_tokens("done [NEXT_PROBLEM] [END]")
    assert ended is True
    assert next_p is False   # END takes precedence
    assert "[END]" not in text
    assert "[NEXT_PROBLEM]" not in text


def test_parse_tutor_tokens_neither():
    text, ended, next_p = _parse_tutor_tokens("Try this next.")
    assert ended is False and next_p is False
    assert text == "Try this next."


def test_parse_tutor_tokens_constant_values():
    assert END_TOKEN == "[END]"
    assert NEXT_PROBLEM_TOKEN == "[NEXT_PROBLEM]"


# ---------------------------------------------------------------------------
# Sync run_exchange dynamic tests (Task 3)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from unittest.mock import MagicMock

from benchmark.core.exchange import run_exchange
from benchmark.core.scenarios import Scenario


def _make_scenario():
    return Scenario(
        scenario_id="s1",
        conv_id="conv1",
        cut_turn=3,
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok",
        student_context="Grade 5, fractions",
        last_student_message="hello",
        mode="human",
        detection={"turn_start": 2, "turn_end": 3,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def _stub_client(replies):
    """ModelClient stub whose .generate(...) returns reply objects in order."""
    client = MagicMock()
    client.model = "stub-model"
    iterator = iter(replies)

    def _generate(*args, **kwargs):
        text = next(iterator)
        resp = MagicMock()
        resp.text = text
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp

    client.generate = _generate
    return client


def test_run_exchange_ends_on_first_tutor_end_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # avoid touching real prompt files
    # Patch prompt loader to return a fixed string, so we don't need v3 on disk in test
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    tutor = _stub_client(["Great work! [END]"])
    student = _stub_client([])  # never called

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    assert len(ex.generated_turns) == 1
    assert ex.generated_turns[0]["role"] == "TUTOR"
    assert ex.generated_turns[0]["text"] == "Great work!"


def test_run_exchange_runs_full_turn_then_ends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client([
        "What did you get for x?",          # round 1 tutor
        "Nice! That matches. [END]",        # round 2 tutor, ends
    ])
    student = _stub_client(["I got 5"])     # round 1 student only

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    roles = [t["role"] for t in ex.generated_turns]
    assert roles == ["TUTOR", "STUDENT", "TUTOR"]
    assert ex.generated_turns[-1]["text"] == "Nice! That matches."


def test_run_exchange_respects_max_turns_when_no_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )
    # Both clients always reply with single-message text and never emit [END].
    tutor = _stub_client([f"tutor msg {i}" for i in range(50)])
    student = _stub_client([f"student msg {i}" for i in range(50)])

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    assert len(ex.generated_turns) == 4
    # Alternates tutor/student starting with tutor:
    assert [t["role"] for t in ex.generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]


def test_run_exchange_end_token_alone_skips_empty_turn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client(["What did you get?", "[END]"])
    student = _stub_client(["I got 5"])

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v3",
    )

    assert ex.completed is True
    # Tutor-only [END] adds no turn for the empty wrap-up:
    roles = [t["role"] for t in ex.generated_turns]
    assert roles == ["TUTOR", "STUDENT"]
    assert ex.generated_turns[0]["text"] == "What did you get?"
    assert ex.generated_turns[1]["text"] == "I got 5"


# ---------------------------------------------------------------------------
# Batch run_exchanges_batch dynamic tests (Task 4)
# ---------------------------------------------------------------------------

from benchmark.core.exchange import run_exchanges_batch


def _scenario(sid):
    return Scenario(
        scenario_id=sid, conv_id=sid, cut_turn=3,
        transcript_prefix=f"Turn 1. TUTOR: a\nTurn 2. STUDENT: b\nTurn 3. TUTOR: c",
        student_context="ctx", last_student_message="b", mode="human",
        detection={"turn_start": 2, "turn_end": 3,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def test_run_exchanges_batch_ends_per_scenario(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    # Two scenarios: s1 ends on first tutor reply, s2 runs through all rounds (no [END]).
    # Batch driver should drop s1 from active_ids after round 1 and keep going for s2.

    # Capture the order of run_batch calls and their entries.
    calls = []

    def fake_run_batch(client, entries, json_mode, display_name, poll_interval):
        calls.append({"display_name": display_name,
                      "sids": [e["custom_id"] for e in entries]})
        results = {}
        for e in entries:
            sid = e["custom_id"]
            # tutor batches: alternate end-state per scenario
            if display_name.startswith("tutor"):
                if sid == "s1":
                    text = "wrap [END]"
                else:
                    text = "more please"
            else:  # student
                text = "ok"
            results[sid] = {"text": text,
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        return results

    def fake_build_batch_entry(custom_id, prompt, json_mode, max_tokens, images=None, cacheable_prefix=None):
        return {"custom_id": custom_id}

    monkeypatch.setattr("benchmark.core.exchange.run_batch", fake_run_batch)
    monkeypatch.setattr("benchmark.core.exchange.build_batch_entry", fake_build_batch_entry)

    tutor = MagicMock(); tutor.model = "t"
    student = MagicMock(); student.model = "s"

    exchanges = run_exchanges_batch(
        scenarios=[_scenario("s1"), _scenario("s2")],
        tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=64, student_max_tokens=64,
        poll_interval=0, prompt_version="v3",
    )

    # s1 ended after round 1 tutor -> exactly 1 turn (TUTOR only), no student.
    assert [t["role"] for t in exchanges["s1"].generated_turns] == ["TUTOR"]
    assert exchanges["s1"].generated_turns[0]["text"] == "wrap"
    assert exchanges["s1"].completed is True

    # s2 ran until generated_turns hit max_turns=4: T, S, T, S
    assert [t["role"] for t in exchanges["s2"].generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]
    assert exchanges["s2"].completed is True

    # After s1 ends, subsequent batches must NOT include s1:
    later_calls = calls[1:]  # everything after the first tutor batch
    for c in later_calls:
        assert "s1" not in c["sids"]


# ---------------------------------------------------------------------------
# [NEXT_PROBLEM] = second end-token tests
# ---------------------------------------------------------------------------

def test_next_problem_ends_exchange_with_distinct_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client(["Ready for next? [NEXT_PROBLEM]"])
    student = _stub_client([])  # never called

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
    )
    assert ex.completed is True
    assert ex.ended_via == "PROBLEM_CHANGE"
    # No student turn after [NEXT_PROBLEM]; tutor wrap-up text is kept.
    assert [t["role"] for t in ex.generated_turns] == ["TUTOR"]
    assert ex.generated_turns[0]["text"] == "Ready for next?"


def test_end_token_takes_precedence_over_next_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )
    tutor = _stub_client(["Wrap up [NEXT_PROBLEM] [END]"])
    student = _stub_client([])

    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=256, student_max_tokens=256,
        prompt_version="v4",
    )
    assert ex.completed is True
    assert ex.ended_via == "END"


def test_batch_records_ended_via_per_scenario(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    # s1 emits [NEXT_PROBLEM] round 1; s2 hits max_turns.
    def fake_run_batch(client, entries, json_mode, display_name, poll_interval):
        results = {}
        for e in entries:
            sid = e["custom_id"]
            if display_name.startswith("tutor"):
                text = "p1 done [NEXT_PROBLEM]" if sid == "s1" else "more please"
            else:
                text = "ok"
            results[sid] = {"text": text,
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        return results

    def fake_build_batch_entry(custom_id, prompt, json_mode, max_tokens, images=None, cacheable_prefix=None):
        return {"custom_id": custom_id}

    monkeypatch.setattr("benchmark.core.exchange.run_batch", fake_run_batch)
    monkeypatch.setattr("benchmark.core.exchange.build_batch_entry", fake_build_batch_entry)

    tutor = MagicMock(); tutor.model = "t"
    student = MagicMock(); student.model = "st"

    exchanges = run_exchanges_batch(
        scenarios=[_scenario("s1"), _scenario("s2")],
        tutor_client=tutor, student_client=student,
        max_turns=4, tutor_max_tokens=64, student_max_tokens=64,
        poll_interval=0, prompt_version="v4",
    )

    # s1 emitted [PROBLEM_CHANGE] -> single tutor turn, ended_via=PROBLEM_CHANGE
    assert exchanges["s1"].ended_via == "PROBLEM_CHANGE"
    assert [t["role"] for t in exchanges["s1"].generated_turns] == ["TUTOR"]
    assert exchanges["s1"].generated_turns[0]["text"] == "p1 done"

    # s2 hit cap: T, S, T, S
    assert exchanges["s2"].ended_via == "MAX_TURNS"
    assert [t["role"] for t in exchanges["s2"].generated_turns] == ["TUTOR", "STUDENT", "TUTOR", "STUDENT"]


# ---------------------------------------------------------------------------
# trait student_mode tests (Task 3)
# ---------------------------------------------------------------------------

def test_build_role_prompt_trait_mode_substitutes_persona(tmp_path, monkeypatch):
    """When student_mode='trait', _build_role_prompt resolves a persona via
    get_or_generate_trait and the persona text appears in the assembled head."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    from unittest.mock import MagicMock
    fake_client = MagicMock()
    fake_client.model = "m1"
    fake_response = MagicMock()
    fake_response.text = "A determined 5th grader who skips multiplication facts."
    fake_response.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    fake_client.generate.return_value = fake_response

    from benchmark.core.exchange import _build_role_prompt
    scenario = _make_scenario()
    head, tail = _build_role_prompt(
        "STUDENT",
        transcript_prefix=scenario.transcript_prefix,
        extra="",
        student_context=scenario.student_context,
        prompt_version="v5",
        student_mode="trait",
        scenario=scenario,
        trait_client=fake_client,
        trait_model="m1",
    )
    out = head + tail

    assert "A determined 5th grader" in out
    # synth_students prompts use [[PERSONA_DESCRIPTION_HERE]] / [[NEXT_CONVERSATION_INFORMATION_HERE]];
    # all placeholders must be filled.
    assert "[[PERSONA_DESCRIPTION_HERE]]" not in out
    assert "[[NEXT_CONVERSATION_INFORMATION_HERE]]" not in out
    assert scenario.student_context in out


def test_run_exchange_with_trait_mode(tmp_path, monkeypatch):
    """run_exchange with student_mode='trait' should resolve a persona via
    trait_client and produce student turns that include the persona text in
    the student prompt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    fake_response = MagicMock()
    fake_response.text = "calm but distracted 5th grader"
    fake_response.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    trait_client = MagicMock()
    trait_client.model = "stub-trait"
    trait_client.generate.return_value = fake_response

    # The student client records the cacheable_prefix (head) it sees so we can
    # assert the persona was substituted. The persona lives in `head` (system
    # prompt portion), not `tail` (the per-round suffix).
    student_heads = []
    def _student_generate(prompt, **kw):
        student_heads.append(kw.get("cacheable_prefix", ""))
        resp = MagicMock()
        resp.text = "I'll try the next step"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    student_client = MagicMock()
    student_client.model = "stub-student"
    student_client.generate = _student_generate

    tutor = _stub_client(["Try this problem.", "Great, [NEXT_PROBLEM]"])

    scenario = _make_scenario()
    ex = run_exchange(
        scenario=scenario,
        tutor_client=tutor,
        student_client=student_client,
        max_turns=10,
        tutor_max_tokens=128,
        student_max_tokens=128,
        prompt_version="v5",
        student_mode="trait",
        trait_client=trait_client,
        trait_model="stub-trait",
    )

    assert ex.completed is True
    assert trait_client.generate.called
    # The student's head (cacheable_prefix) for at least one turn should include
    # the generated persona text.
    assert any("calm but distracted" in h for h in student_heads), (
        f"persona not found in any student head; saw {len(student_heads)} head(s)"
    )


# ---------------------------------------------------------------------------
# _build_role_prompt cache-tuple tests (PC Task 2)
# ---------------------------------------------------------------------------

def test_build_role_prompt_returns_head_tail_tuple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    from benchmark.core.exchange import _build_role_prompt
    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello",
        extra="\nTurn 3. TUTOR: ok",
        student_context="Grade 5",
        prompt_version="v5",
    )
    assert isinstance(head, str) and isinstance(tail, str)
    assert "Grade 5" in head             # system + context is in the head
    assert "Turn 1." in head             # prefix is in the head
    assert "Turn 3." in tail             # extra is in the tail
    assert "Turn 3." not in head


def test_build_role_prompt_head_invariant_across_extras(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    from benchmark.core.exchange import _build_role_prompt
    head1, _ = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx", prompt_version="v5",
    )
    head2, _ = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok",
        student_context="ctx", prompt_version="v5",
    )
    # Head must be byte-identical -- this is what enables cache hits.
    assert head1 == head2


def test_run_exchange_sends_same_cacheable_prefix_each_round(tmp_path, monkeypatch):
    """Every tutor call within one scenario should pass the same cacheable_prefix.
    This is what makes the prompt cache hit on round 2+."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "SYS {student_context}",
    )

    seen_prefixes = []
    def _tutor_generate(prompt, **kwargs):
        seen_prefixes.append(kwargs.get("cacheable_prefix"))
        resp = MagicMock()
        resp.text = "next" if len(seen_prefixes) < 3 else "wrap [NEXT_PROBLEM]"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    tutor = MagicMock(); tutor.model = "stub"; tutor.generate = _tutor_generate

    student = _stub_client(["one", "two"])

    from benchmark.core.exchange import run_exchange
    ex = run_exchange(
        scenario=_make_scenario(), tutor_client=tutor, student_client=student,
        max_turns=100, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
    )
    assert ex.completed is True
    assert len(seen_prefixes) >= 3
    # All non-None cacheable_prefix values must be identical.
    non_none = [p for p in seen_prefixes if p is not None]
    assert len(non_none) == len(seen_prefixes), "every call should send a cacheable_prefix"
    assert len(set(non_none)) == 1, f"prefix changed across rounds: {set(non_none)}"
