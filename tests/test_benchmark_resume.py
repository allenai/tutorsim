"""Resume + sidecar wire-through tests for the benchmark annotator bridge."""
from unittest.mock import patch, MagicMock


def test_execute_and_parse_bulk_forwards_existing_batch_id():
    from benchmark.core.annotator_bridge import execute_and_parse_bulk
    entries = [{"key": "scen1__scaffolding__0", "request": {"prompt": "p"}}]
    all_detections = {"scen1": {"scen1": {"detections": [
        {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding"}
    ]}}}

    with patch("benchmark.core.annotator_bridge.run_batch") as mock_rb, \
         patch("benchmark.core.annotator_bridge.ModelClient"), \
         patch("benchmark.core.annotator_bridge.get_phase_config",
               return_value={"model": "claude-opus-4-6", "poll_interval": 60}):
        mock_rb.return_value = {}
        execute_and_parse_bulk(
            entries=entries,
            all_detections=all_detections,
            annotator_profile="anthropic",
            mode="batch",
            existing_batch_id="msgbatch_resumed",
            on_batch_created=lambda bid: None,
        )

    kwargs = mock_rb.call_args.kwargs
    assert kwargs["existing_batch_id"] == "msgbatch_resumed"
    assert callable(kwargs["on_batch_created"])


def test_execute_and_parse_bulk_default_kwargs_are_none():
    from benchmark.core.annotator_bridge import execute_and_parse_bulk
    entries = [{"key": "scen1__scaffolding__0", "request": {"prompt": "p"}}]
    all_detections = {"scen1": {"scen1": {"detections": [
        {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding"}
    ]}}}
    with patch("benchmark.core.annotator_bridge.run_batch") as mock_rb, \
         patch("benchmark.core.annotator_bridge.ModelClient"), \
         patch("benchmark.core.annotator_bridge.get_phase_config",
               return_value={"model": "claude-opus-4-6", "poll_interval": 60}):
        mock_rb.return_value = {}
        execute_and_parse_bulk(
            entries=entries,
            all_detections=all_detections,
            annotator_profile="anthropic",
            mode="batch",
        )
    kwargs = mock_rb.call_args.kwargs
    assert kwargs.get("existing_batch_id") is None
    assert kwargs.get("on_batch_created") is None
