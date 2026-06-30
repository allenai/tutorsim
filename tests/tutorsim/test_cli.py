"""Tests for tutorsim.cli -- run_cell end-to-end (TDD: Red phase).

All tests mock run_conversation and score so there are no network calls.
Uses tmp_path as results_root so nothing writes to the real results/ directory.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tutorsim import results as results_mod
from tutorsim.scenarios import Scenario
from tutorsim.scoring import Annotation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scenario(sid: str, dimension: str = "scaffolding") -> Scenario:
    """Build a minimal Scenario for testing."""
    return Scenario(
        id=sid,
        context=[{"turn_number": 1, "role": "tutor", "text": "Hello"}],
        dimension=dimension,
        student={"context": "Student context", "reference": ""},
        rubric={"gold": dimension, "hint": "test hint"},
        provenance={"conv_id": f"conv_{sid}", "cut_turn": 1},
    )


def _make_annotation(sid: str) -> Annotation:
    """Build a minimal Annotation for testing."""
    return Annotation(
        scenario_id=sid,
        annotation_type="scaffolding",
        turn_start=2,
        turn_end=4,
        situation="situation text",
        action="action text",
        result="result text",
        action_decomposed=["facet a"],
        result_decomposed=["result facet"],
        overscaffold_decomposed=[],
        action_label="scaffolding",
        result_label="pos",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _make_transcript(sid: str, tutor_latencies=None, student_latencies=None,
                     tutor_usage=None, student_usage=None) -> MagicMock:
    """Build a mock Transcript-like object with optional latency/usage data."""
    t = MagicMock()
    t.scenario_id = sid
    t.tutor_model = "claude-opus-4-8"
    t.generated_turns = [{"turn_number": 2, "role": "TUTOR", "text": "Hi"}]
    t.to_dict.return_value = {"scenario_id": sid, "completed": True, "generated_turns": []}
    t.tutor_latencies = tutor_latencies if tutor_latencies is not None else []
    t.student_latencies = student_latencies if student_latencies is not None else []
    t.tutor_usage = tutor_usage if tutor_usage is not None else {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    }
    t.student_usage = student_usage if student_usage is not None else {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    }
    return t


FIXTURE_SCENARIOS = [
    _make_scenario("scenario_001", "scaffolding"),
    _make_scenario("scenario_002", "rigor"),
]


# ---------------------------------------------------------------------------
# Helper: patch targets
# ---------------------------------------------------------------------------

_CONV_PATCH = "tutorsim.cli.conversation.run_conversation"
_SCORE_PATCH = "tutorsim.cli.scoring.score"
_LOAD_PATCH = "tutorsim.cli.load_scenarios"
_CFG_PATCH = "tutorsim.cli.build_run_config"


def _make_run_config(sample=2, dataset="test_ds", max_turns=4):
    """Build a fake RunConfig-like object."""
    cfg = MagicMock()
    cfg.dataset = dataset
    cfg.sample = sample
    cfg.max_turns = max_turns
    cfg.tutors = ["claude-opus-4-8"]
    cfg.modes = ["plain"]
    cfg.trials = 1
    cfg.seed = 42
    cfg.student = {"model": "claude-haiku", "mode": "oracle", "thinking": "adaptive"}
    cfg.scorer = {"model": "claude-opus-4-6", "thinking": "adaptive"}
    cfg.resolved_tutors = {"claude-opus-4-8": {}}
    cfg.config_source = "test"
    return cfg


# ---------------------------------------------------------------------------
# Test 1: run_cell writes all expected files and correct summary
# ---------------------------------------------------------------------------

def test_run_cell_writes_all_files(tmp_path):
    """run_cell over a 2-scenario fixture writes 2 transcripts + 2 scores +
    config.json + summary.json; summary metrics == aggregate of the 2 scores."""
    from tutorsim.cli import run_cell
    import tutorsim.report as report_mod

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]
    expected_summary = report_mod.aggregate(scenarios, annotations)

    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock) as mock_build_cfg,
        patch(_LOAD_PATCH, return_value=list(scenarios)) as mock_load,
        patch(_CONV_PATCH, side_effect=transcripts) as mock_conv,
        patch(_SCORE_PATCH, side_effect=annotations) as mock_score,
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    # run_id must be non-empty
    assert run_id

    run_dir = tmp_path / run_id

    # config.json
    assert (run_dir / "config.json").exists()
    cfg_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert isinstance(cfg_data, dict)

    # 2 transcripts
    for s in scenarios:
        assert (run_dir / "transcripts" / f"{s.id}.json").exists(), \
            f"Missing transcript for {s.id}"

    # 2 scores
    for s in scenarios:
        assert (run_dir / "scores" / f"{s.id}.json").exists(), \
            f"Missing score for {s.id}"

    # summary.json
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # summary metrics == aggregate of the 2 annotations
    assert summary["n_scenarios"] == expected_summary["n_scenarios"]
    assert summary["outcome_pos_rate"] == expected_summary["outcome_pos_rate"]
    scaf = summary["scaffolding_did"]
    exp_scaf = expected_summary["scaffolding_did"]
    assert scaf["n_yes"] == exp_scaf["n_yes"]
    assert scaf["n_total"] == exp_scaf["n_total"]


# ---------------------------------------------------------------------------
# Test 1b: run_cell writes latency + tokens blocks into summary.json (spec S7)
# ---------------------------------------------------------------------------

def test_run_cell_writes_latency_and_tokens(tmp_path):
    """run_cell aggregates tutor_latencies/usage from transcripts into
    summary.json latency.tutor and tokens.total blocks (spec S7).

    Mocked transcripts carry non-zero latencies and usage so the aggregation
    path is exercised end-to-end.
    """
    from tutorsim.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)

    # Two transcripts with known latencies and token counts
    transcripts = [
        _make_transcript(
            "scenario_001",
            tutor_latencies=[1.0, 3.0],
            student_latencies=[0.5],
            tutor_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            student_usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        ),
        _make_transcript(
            "scenario_002",
            tutor_latencies=[2.0],
            student_latencies=[0.8],
            tutor_usage={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
            student_usage={"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
        ),
    ]
    annotations = [_make_annotation(s.id) for s in scenarios]

    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=annotations),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # latency.tutor block must be present and non-None
    assert "latency" in summary, "summary must have 'latency' key"
    assert "tutor" in summary["latency"], "summary['latency'] must have 'tutor' key"
    lat = summary["latency"]["tutor"]
    assert lat is not None, "latency.tutor must not be None (transcripts had latencies)"
    assert "p50_seconds" in lat, "latency.tutor must have p50_seconds"
    assert "p95_seconds" in lat, "latency.tutor must have p95_seconds"
    # tutor_latencies = [1.0, 3.0, 2.0] -> sorted = [1.0, 2.0, 3.0]
    # n=3, p50 = s[1] = 2.0, p95_idx = max(0, min(2, round(2.85)-1)) = max(0,min(2,2)) = 2 -> 3.0
    assert lat["p50_seconds"] == pytest.approx(2.0), f"p50 expected 2.0, got {lat['p50_seconds']}"
    assert lat["p95_seconds"] == pytest.approx(3.0), f"p95 expected 3.0, got {lat['p95_seconds']}"

    # tokens.total block must be present
    assert "tokens" in summary, "summary must have 'tokens' key"
    assert "total" in summary["tokens"], "summary['tokens'] must have 'total' key"
    tok = summary["tokens"]["total"]
    assert "total_tokens" in tok, "tokens.total must have total_tokens"
    # tutor: 150+300=450, student: 100+200=300, total: 750
    assert tok["total_tokens"] == 750, f"total_tokens expected 750, got {tok['total_tokens']}"


# ---------------------------------------------------------------------------
# Test 2: run_cell resumes (second call skips already-done scenarios)
# ---------------------------------------------------------------------------

def test_run_cell_resumes(tmp_path):
    """Second run_cell call finds both scenarios already done; 0 new conversation
    calls are made (is_done returns True for both)."""
    from tutorsim.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]

    cfg_mock = _make_run_config(sample=2)

    # First run: write all files
    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=list(transcripts)),
        patch(_SCORE_PATCH, side_effect=list(annotations)),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    # Second run: both scenarios are done, no new conv/score calls
    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH) as mock_conv2,
        patch(_SCORE_PATCH) as mock_score2,
    ):
        run_id2 = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    assert run_id2 == run_id
    mock_conv2.assert_not_called()
    mock_score2.assert_not_called()

    # summary.json still written on resume
    run_dir = tmp_path / run_id
    assert (run_dir / "summary.json").exists()


# ---------------------------------------------------------------------------
# Test 3: score error -> logged + skipped; run completes with partial summary
# ---------------------------------------------------------------------------

def test_run_cell_skips_on_score_error(tmp_path):
    """A scenario whose score() raises is logged + skipped; run still completes
    and writes summary.json over the successful scenarios only."""
    from tutorsim.cli import run_cell
    import tutorsim.report as report_mod

    scenarios = list(FIXTURE_SCENARIOS)
    good_scenario = scenarios[0]
    bad_scenario = scenarios[1]

    good_transcript = _make_transcript(good_scenario.id)
    bad_transcript = _make_transcript(bad_scenario.id)
    good_annotation = _make_annotation(good_scenario.id)

    cfg_mock = _make_run_config(sample=2)

    def _score_side_effect(scenario, transcript):
        if scenario.id == bad_scenario.id:
            raise RuntimeError("Scorer exploded!")
        return good_annotation

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=[good_transcript, bad_transcript]),
        patch(_SCORE_PATCH, side_effect=_score_side_effect),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    run_dir = tmp_path / run_id

    # Good scenario: transcript + score written
    assert (run_dir / "transcripts" / f"{good_scenario.id}.json").exists()
    assert (run_dir / "scores" / f"{good_scenario.id}.json").exists()

    # Bad scenario: transcript may or may not exist, score must NOT exist
    assert not (run_dir / "scores" / f"{bad_scenario.id}.json").exists()

    # summary.json written (over the 1 successful scenario)
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # Only 1 scenario made it to summary
    assert summary["n_scenarios"] == 1
    assert summary["run_counts"] == {
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
        "resumed": 0,
    }
    assert summary["failed_scenarios"][0]["id"] == bad_scenario.id


def test_run_cell_raises_when_all_scenarios_fail(tmp_path):
    """A run with zero completed scenarios must not write an empty valid summary."""
    from tutorsim.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=RuntimeError("Scorer exploded!")),
    ):
        with pytest.raises(RuntimeError, match="No scenarios completed"):
            run_cell(
                tutor="claude-opus-4-8",
                mode="plain",
                run_cfg=None,
                date="20260626",
                results_root=str(tmp_path),
                trait_cache_dir=str(tmp_path / "_trait_cache"),
            )


# ---------------------------------------------------------------------------
# Test 4: cell expansion -- tutors x modes = cells with correct lane assignment
# ---------------------------------------------------------------------------

def test_cell_expansion_and_lane_assignment():
    """3 tutors x 2 modes = 6 cells; each cell gets the correct provider lane."""
    from tutorsim.cli import expand_cells

    tutors = ["claude-opus-4-8", "gemini-3.1-pro-preview", "gpt-5.4"]
    modes = ["plain", "scaffolding_rigor"]

    cells = expand_cells(tutors, modes)

    assert len(cells) == 6

    # Check all (tutor, mode) pairs present
    pairs = {(c["tutor"], c["mode"]) for c in cells}
    for t in tutors:
        for m in modes:
            assert (t, m) in pairs, f"Missing cell ({t}, {m})"

    # Check lane assignment by provider
    lane_map = {c["tutor"]: c["lane"] for c in cells}
    assert lane_map["claude-opus-4-8"] == "anthropic"
    assert lane_map["gemini-3.1-pro-preview"] == "gemini"
    assert lane_map["gpt-5.4"] == "openai"


# ---------------------------------------------------------------------------
# Test 5: scheduler -- within-lane sequential, lanes can run independently
# ---------------------------------------------------------------------------

def test_scheduler_within_lane_sequential(tmp_path):
    """Cells within a lane are called in order; all 6 run_cell calls complete."""
    from tutorsim.cli import expand_cells, run_sweep

    # 2 tutors in the same lane (anthropic), 1 mode each -> 2 cells in 1 lane
    tutors = ["claude-opus-4-8", "claude-haiku-3-5"]
    modes = ["plain"]

    cells = expand_cells(tutors, modes)
    # Both claude models -> anthropic lane
    assert all(c["lane"] == "anthropic" for c in cells)

    call_order = []

    def fake_run_cell(tutor, mode, run_cfg, *, date, results_root, trait_cache_dir):
        call_order.append((tutor, mode))
        return f"{tutor}_{mode}_run_id"

    run_ids = run_sweep(
        cells=cells,
        run_cfg=MagicMock(),
        date="20260626",
        results_root=str(tmp_path),
        trait_cache_dir=str(tmp_path / "_trait_cache"),
        _run_cell_fn=fake_run_cell,
    )

    # All 2 cells produced a run_id
    assert len(run_ids) == 2

    # Within-lane sequential: claude-opus-4-8 before claude-haiku-3-5 (sweep order)
    tutors_in_order = [t for t, m in call_order]
    assert tutors_in_order == ["claude-opus-4-8", "claude-haiku-3-5"]


def test_scheduler_multiple_lanes_all_cells_run(tmp_path):
    """3 tutors x 2 modes = 6 cells; all 6 get a run_id regardless of parallelism."""
    from tutorsim.cli import expand_cells, run_sweep

    tutors = ["claude-opus-4-8", "gemini-3.1-pro-preview", "gpt-5.4"]
    modes = ["plain", "scaffolding_rigor"]

    cells = expand_cells(tutors, modes)

    def fake_run_cell(tutor, mode, run_cfg, *, date, results_root, trait_cache_dir):
        return f"{tutor}_{mode}_run_id"

    run_ids = run_sweep(
        cells=cells,
        run_cfg=MagicMock(),
        date="20260626",
        results_root=str(tmp_path),
        trait_cache_dir=str(tmp_path / "_trait_cache"),
        _run_cell_fn=fake_run_cell,
    )

    assert len(run_ids) == 6


# ---------------------------------------------------------------------------
# Test 6: --trials N -- conversation+score called N times; summary has mean+spread
# ---------------------------------------------------------------------------

def _make_run_config_trials(n_trials: int, sample=2, dataset="test_ds", max_turns=4):
    cfg = MagicMock()
    cfg.dataset = dataset
    cfg.sample = sample
    cfg.max_turns = max_turns
    cfg.tutors = ["claude-opus-4-8"]
    cfg.modes = ["plain"]
    cfg.trials = n_trials
    cfg.seed = 42
    cfg.student = {"model": "claude-haiku", "mode": "oracle", "thinking": "adaptive"}
    cfg.scorer = {"model": "claude-opus-4-6", "thinking": "adaptive"}
    cfg.resolved_tutors = {"claude-opus-4-8": {}}
    cfg.config_source = "test"
    return cfg


def test_trials_3_calls_conversation_n_times(tmp_path):
    """trials=3: run_conversation is called 3x per scenario (not 1x)."""
    from tutorsim.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    n_scenarios = len(scenarios)
    n_trials = 3

    # We need n_trials * n_scenarios transcripts and annotations
    transcripts = [_make_transcript(s.id) for s in scenarios for _ in range(n_trials)]
    annotations = [_make_annotation(s.id) for s in scenarios for _ in range(n_trials)]

    cfg_mock = _make_run_config_trials(n_trials=n_trials, sample=n_scenarios)

    conv_mock = MagicMock(side_effect=transcripts)
    score_mock = MagicMock(side_effect=annotations)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, new=conv_mock),
        patch(_SCORE_PATCH, new=score_mock),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    # conversation called n_trials times per scenario
    assert conv_mock.call_count == n_trials * n_scenarios
    assert score_mock.call_count == n_trials * n_scenarios


def test_trials_summary_has_mean_and_spread(tmp_path):
    """trials=3: summary.json has mean and spread keys for numeric metrics."""
    from tutorsim.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    n_scenarios = len(scenarios)
    n_trials = 3

    transcripts = [_make_transcript(s.id) for s in scenarios for _ in range(n_trials)]
    annotations = [_make_annotation(s.id) for s in scenarios for _ in range(n_trials)]

    cfg_mock = _make_run_config_trials(n_trials=n_trials, sample=n_scenarios)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=annotations),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # Multi-trial summaries must carry mean and spread at the top level
    assert "trials" in summary, "summary must record number of trials"
    assert summary["trials"] == n_trials
    assert "mean" in summary, "summary must have 'mean' sub-dict for trials>1"
    assert "spread" in summary, "summary must have 'spread' sub-dict for trials>1"
    # The mean block should have the same shape as a single-run summary
    assert "n_scenarios" in summary["mean"]
    assert "outcome_pos_rate" in summary["mean"]


def test_trials_1_summary_matches_single_run(tmp_path):
    """trials=1: summary.json must be byte-compatible with the Task-4 single-run format
    (no 'mean'/'spread' keys -- it is the plain aggregate dict)."""
    from tutorsim.cli import run_cell
    import tutorsim.report as report_mod

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]
    expected_summary = report_mod.aggregate(scenarios, annotations)

    cfg_mock = _make_run_config_trials(n_trials=1, sample=len(scenarios))

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=list(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=annotations),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
            trait_cache_dir=str(tmp_path / "_trait_cache"),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # trials=1: no mean/spread wrapper -- plain aggregate
    assert "mean" not in summary, "trials=1 must NOT have 'mean' key"
    assert "spread" not in summary, "trials=1 must NOT have 'spread' key"
    # Metrics match expected
    assert summary["n_scenarios"] == expected_summary["n_scenarios"]
    assert summary["outcome_pos_rate"] == expected_summary["outcome_pos_rate"]


# ---------------------------------------------------------------------------
# Helper: build a fake run directory with a summary.json
# ---------------------------------------------------------------------------

def _make_fake_run(
    root: Path,
    run_id: str,
    tutor_model: str,
    mode: str,
    n: int = 10,
    scaffold_cal: float = 0.6,
) -> Path:
    """Write a minimal summary.json inside root/run_id/."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tutor_model": tutor_model,
        "mode": mode,
        "n_scenarios": n,
        "scaffold_calibrated": {"score": scaffold_cal, "n_clean_yes": 6, "n_total": n, "n_overscaffold": 1},
        "rigor_calibrated": {"score": 0.5, "n_clean_yes": 5, "n_total": n},
        "overscaffold": {"rate": 0.1, "n_yes": 1, "n_total": n, "available": True},
        "outcome_pos_rate": 0.4,
        "scaffolding_did": {"rate": 0.7, "n_yes": 7, "n_total": n},
        "rigor_did": {"rate": 0.5, "n_yes": 5, "n_total": n},
        "latency": {"tutor": {"p50_seconds": 1.0, "p95_seconds": 2.5, "mean_seconds": 1.2, "n": n}},
        "tokens": {"total": {"total_tokens": 100000, "input_tokens": 80000, "output_tokens": 20000}},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# Test 7: report subcommand writes leaderboard.md + .csv with both rows
# ---------------------------------------------------------------------------

def test_report_writes_leaderboard_md_and_csv(tmp_path):
    """main(['report', '--results-root', ..., '--out', ...]) writes leaderboard.md + .csv
    containing both run rows."""
    from tutorsim.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(results_root, "model_alpha_plain_ds_20260626", "model-alpha", "plain", scaffold_cal=0.75)
    _make_fake_run(results_root, "model_beta_plain_ds_20260626", "model-beta", "plain", scaffold_cal=0.50)

    out_stem = str(tmp_path / "leaderboard")

    main(["report", "--results-root", str(results_root), "--out", out_stem])

    md_path = tmp_path / "leaderboard.md"
    csv_path = tmp_path / "leaderboard.csv"

    assert md_path.exists(), "leaderboard.md must be written"
    assert csv_path.exists(), "leaderboard.csv must be written"

    md_text = md_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    # Both model names must appear in both files
    assert "model-alpha" in md_text
    assert "model-beta" in md_text
    assert "model-alpha" in csv_text
    assert "model-beta" in csv_text

    # CSV must have header + 2 data rows (+ possible trailing newline)
    csv_lines = [l for l in csv_text.strip().splitlines() if l.strip()]
    assert len(csv_lines) == 3, f"Expected header + 2 rows, got: {csv_lines}"


# ---------------------------------------------------------------------------
# Test 8: view subcommand writes non-empty HTML
# ---------------------------------------------------------------------------

def test_view_writes_html(tmp_path):
    """main(['view', '--results-root', ..., '--out', ...]) writes a non-empty HTML file."""
    from tutorsim.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(results_root, "model_alpha_plain_ds_20260626", "model-alpha", "plain")

    out_html = str(tmp_path / "viewer.html")

    main(["view", "--results-root", str(results_root), "--out", out_html])

    html_path = tmp_path / "viewer.html"
    assert html_path.exists(), "viewer.html must be written"

    html_text = html_path.read_text(encoding="utf-8")
    assert len(html_text) > 100, "HTML must be non-trivially non-empty"
    assert "<!DOCTYPE html>" in html_text or "<html" in html_text


# ---------------------------------------------------------------------------
# Test 9: build-scenarios subcommand dispatches to scenarios._cli_build
# ---------------------------------------------------------------------------

def test_build_scenarios_dispatches(tmp_path):
    """main(['build-scenarios', ...]) calls scenarios._cli_build with correct args."""
    from tutorsim.cli import main

    ids_file = tmp_path / "ids.json"
    ids_file.write_text("[]", encoding="utf-8")

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()

    with patch("tutorsim.cli._cmd_build_scenarios") as mock_bs:
        main([
            "build-scenarios",
            "--set", "balanced_520",
            "--ids", str(ids_file),
            "--ground-truth", str(gt_dir),
            "--transcripts", str(tx_dir),
            "--created", "2026-06-26",
        ])

    mock_bs.assert_called_once()
    # Verify the args namespace passed in has the expected fields
    call_args = mock_bs.call_args[0][0]
    assert call_args.set == "balanced_520"
    assert call_args.ids == str(ids_file)
    assert call_args.ground_truth == str(gt_dir)
    assert call_args.transcripts == str(tx_dir)
    assert call_args.created == "2026-06-26"


def test_dataset_validate_cli(capsys):
    """main(['dataset', 'validate', ...]) validates the mini fixture dataset."""
    from tutorsim.cli import main

    main([
        "dataset",
        "validate",
        "--set", "mini_set",
        "--root", "tests/tutorsim/fixtures",
    ])

    captured = capsys.readouterr()
    assert "Dataset valid: mini_set" in captured.out


# ---------------------------------------------------------------------------
# Test 10: --help and run --help exit 0 and list subcommands
# ---------------------------------------------------------------------------

def test_main_help_exits_0():
    """main(['--help']) raises SystemExit(0)."""
    from tutorsim.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_run_help_exits_0():
    """main(['run', '--help']) raises SystemExit(0)."""
    from tutorsim.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Test 11: end-to-end smoke -- main(['run', ...]) with mocks
# ---------------------------------------------------------------------------

def test_main_run_smoke(tmp_path):
    """main(['run', '--tutors', 'claude-opus-4-8', '--sample', '1', ...]) with mocked
    run_conversation and score produces a run dir with summary.json."""
    from tutorsim.cli import main

    scenario = _make_scenario("smoke_001", "scaffolding")
    transcript = _make_transcript("smoke_001")
    annotation = _make_annotation("smoke_001")

    cfg_mock = _make_run_config(sample=1, dataset="test_ds")

    with (
        patch(_CFG_PATCH, return_value=cfg_mock) as _mock_cfg,
        patch(_LOAD_PATCH, return_value=[scenario]),
        patch(_CONV_PATCH, return_value=transcript),
        patch(_SCORE_PATCH, return_value=annotation),
        patch("tutorsim.cli.run_sweep") as mock_sweep,
    ):
        # run_sweep returns a list of run_ids; stub it so we test main() wiring
        mock_sweep.return_value = ["claude-opus-4-8_plain_test_ds_20260626"]
        main([
            "run",
            "--tutors", "claude-opus-4-8",
            "--sample", "1",
            "--dataset", "test_ds",
            "--trait-cache-dir", str(tmp_path / "_trait_cache"),
        ])

    # Verify run_sweep was called (proving main() wired through correctly)
    mock_sweep.assert_called_once()


def test_main_run_missing_dataset_exits_cleanly(capsys):
    """DatasetNotFoundError should be reported without a Python traceback."""
    from tutorsim.cli import main
    from tutorsim.scenarios import DatasetNotFoundError

    cfg_mock = _make_run_config(sample=1, dataset="missing_ds")

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch("tutorsim.cli.run_sweep", side_effect=DatasetNotFoundError("missing dataset")),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run",
                "--tutors", "claude-opus-4-8",
                "--sample", "1",
                "--dataset", "missing_ds",
            ])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Error: missing dataset" in captured.err
    assert "Traceback" not in captured.err


def test_run_config_argument_becomes_active_config_for_late_resolution(tmp_path, monkeypatch):
    """`tutorsim run --config FILE` also affects later no-arg config lookups."""
    from tutorsim.cli import main
    from tutorsim.config import _reset_config_cache

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  openai: { env: OPENAI_API_KEY }

models:
  gpt-4o-mini: {}

student: { model: mock-student, mode: oracle, thinking: false }
scorer: { model: gpt-4o-mini, thinking: false }

defaults: { seed: 10, trials: 1, max_turns: 1 }
retry: { max_retries: 1, base_delay: 1 }
batch: { timeout: 60 }
""".strip(),
        encoding="utf-8",
    )

    observed = {}

    def fake_run_sweep(cells, run_cfg, *, date, results_root, trait_cache_dir):
        from tutorsim.config import resolve_model, student_spec, scorer_spec

        observed["provider"] = resolve_model("gpt-4o-mini")["provider"]
        observed["student"] = student_spec()["model"]
        observed["scorer"] = scorer_spec()["model"]
        observed["run_cfg_source"] = run_cfg.config_source
        return ["fake_run"]

    monkeypatch.setattr("tutorsim.cli.run_sweep", fake_run_sweep)
    original_config = os.environ.pop("TUTORSIM_CONFIG", None)
    _reset_config_cache()

    try:
        main([
            "run",
            "--config", str(config_path),
            "--tutors", "gpt-4o-mini",
            "--dataset", "readme_mock",
        ])

        assert observed == {
            "provider": "openai",
            "student": "mock-student",
            "scorer": "gpt-4o-mini",
            "run_cfg_source": str(config_path),
        }
        assert os.environ["TUTORSIM_CONFIG"] == str(config_path)
    finally:
        if original_config is None:
            os.environ.pop("TUTORSIM_CONFIG", None)
        else:
            os.environ["TUTORSIM_CONFIG"] = original_config
        _reset_config_cache()
