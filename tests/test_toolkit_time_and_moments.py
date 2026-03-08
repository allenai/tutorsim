from __future__ import annotations

from tutor_bench.toolkit.moment_ops import pairwise_overlap, unmatched_timestamps
from tutor_bench.toolkit.time_utils import fmt_ts, nearest_within, ts_to_seconds


def test_ts_roundtrip_precision() -> None:
    ts = "01:02:03.456"
    seconds = ts_to_seconds(ts)
    assert abs(seconds - 3723.456) < 1e-6
    assert fmt_ts(seconds) == ts


def test_fmt_ts_carries_rounding_boundary() -> None:
    assert fmt_ts(59.9996) == "00:01:00.000"


def test_nearest_within_returns_none_when_outside_tolerance() -> None:
    assert nearest_within(10.0, [0.0, 1.0, 2.0], tolerance_seconds=0.5) is None
    assert nearest_within(10.0, [8.9, 10.4, 11.1], tolerance_seconds=0.5) == 10.4


def test_unmatched_and_pairwise_overlap() -> None:
    a = [10.0, 20.0, 30.0]
    b = [9.8, 25.0, 30.2]
    unmatched = unmatched_timestamps(a, b, tolerance_seconds=0.5)
    assert unmatched == [20.0]

    overlap = pairwise_overlap(a, b, tolerance_seconds=0.5)
    assert overlap["n_a"] == 3
    assert overlap["n_b"] == 3
    assert overlap["covered_a"] == 2
    assert overlap["covered_b"] == 2
    assert round(overlap["recall_a_to_b"], 4) == round(2 / 3, 4)
    assert round(overlap["recall_b_to_a"], 4) == round(2 / 3, 4)
