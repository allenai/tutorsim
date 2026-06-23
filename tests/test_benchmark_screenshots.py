"""Bridge loads screenshots per scenario when with_screenshots=True."""
from unittest.mock import patch, MagicMock


def test_prepare_bulk_entries_loads_screenshots_per_scenario():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0",
        conv_id="conv_xyz",
        cut_turn=10,
        transcript_prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello",
        student_context="ctx",
        last_student_message="hello",
        mode="detected",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0",
        tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )
    fake_screenshots = [
        {"filename": "s1.jpg", "anchor_turn": 6, "storage_path": "deidentified/screenshots/REAL/s1.jpg", "timestamp_seconds": 6.0},
    ]

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots",
               return_value=fake_screenshots) as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
            with_screenshots=True,
        )

    # load_anchored_screenshots called with original conv_id, not scenario_id
    mock_load.assert_called_once()
    assert mock_load.call_args.args[0] == "conv_xyz"

    # build_analysis_entries got screenshots_by_conv keyed by scenario_id
    kwargs = mock_build.call_args.kwargs
    sbc = kwargs.get("screenshots_by_conv")
    assert sbc == {"conv_xyz__det_0": fake_screenshots}
    assert kwargs.get("with_screenshots") is True


def test_prepare_bulk_entries_default_no_screenshots():
    from benchmark.core.annotator_bridge import prepare_bulk_entries
    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenario = Scenario(
        scenario_id="conv_xyz__det_0", conv_id="conv_xyz", cut_turn=10,
        transcript_prefix="Turn 1. TUTOR: hi", student_context="ctx",
        last_student_message="hi", mode="detected",
        detection={"turn_start": 5, "turn_end": 8, "annotation_type": "scaffolding"},
    )
    exchange = Exchange(
        scenario_id="conv_xyz__det_0", tutor_model="claude-opus-4-6",
        generated_turns=[{"turn_number": 11, "role": "TUTOR", "text": "ok"}],
        completed=True,
    )

    with patch("benchmark.core.annotator_bridge.load_anchored_screenshots") as mock_load, \
         patch("benchmark.core.annotator_bridge.build_analysis_entries",
               return_value=[]) as mock_build:
        prepare_bulk_entries(
            scenarios=[scenario],
            exchanges={"conv_xyz__det_0": exchange},
            annotator_style="balanced",
            prompt_version="profiles/balanced",
            context_window=20,
        )

    mock_load.assert_not_called()
    assert mock_build.call_args.kwargs.get("screenshots_by_conv") is None
    assert mock_build.call_args.kwargs.get("with_screenshots") is False


def test_run_exchange_attaches_images_to_tutor_call():
    """Sync mode: when images= is passed, tutor calls receive them."""
    from benchmark.core.exchange import run_exchange
    from benchmark.core.scenarios import Scenario
    from unittest.mock import MagicMock

    scenario = Scenario(
        scenario_id="conv_xyz__det_0", conv_id="conv_xyz", mode="detected",
        cut_turn=10, transcript_prefix="Turn 1. TUTOR: hi",
        student_context="ctx", last_student_message="hi",
        detection=None,
    )
    images = [
        "deidentified/screenshots/REAL/s1.jpg",
        "deidentified/screenshots/REAL/s2.jpg",
    ]

    tutor_resp = MagicMock(text="answer", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    tutor_client = MagicMock()
    tutor_client.model = "claude-opus-4-6"
    tutor_client.generate.return_value = tutor_resp
    student_client = MagicMock()
    student_client.model = "claude-opus-4-6"
    student_client.generate.return_value = MagicMock(text="ok", usage={})

    run_exchange(
        scenario=scenario,
        tutor_client=tutor_client, student_client=student_client,
        max_turns=1, tutor_max_tokens=100, student_max_tokens=100,
        prompt_version="v1", images=images,
    )

    # With max_turns=1, exactly one tutor call (no student turn before the cap is hit)
    assert tutor_client.generate.called
    for call in tutor_client.generate.call_args_list:
        assert call.kwargs.get("images") == images


def test_run_exchange_default_no_images():
    """Sync mode without images= still works (back-compat)."""
    from benchmark.core.exchange import run_exchange
    from benchmark.core.scenarios import Scenario
    from unittest.mock import MagicMock

    scenario = Scenario(
        scenario_id="x__0", conv_id="x", mode="detected",
        cut_turn=5, transcript_prefix="Turn 1. TUTOR: hi",
        student_context="ctx", last_student_message="hi",
        detection=None,
    )
    tutor_resp = MagicMock(text="answer", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    tutor_client = MagicMock()
    tutor_client.model = "x"
    tutor_client.generate.return_value = tutor_resp
    student_client = MagicMock()
    student_client.model = "x"
    student_client.generate.return_value = MagicMock(text="", usage={})

    run_exchange(
        scenario=scenario,
        tutor_client=tutor_client, student_client=student_client,
        max_turns=1, tutor_max_tokens=100, student_max_tokens=100,
        prompt_version="v1",
    )
    for call in tutor_client.generate.call_args_list:
        assert call.kwargs.get("images") in (None, [])
