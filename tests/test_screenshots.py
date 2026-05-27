"""Tests for annotator.core.screenshots."""
import pytest


class TestTimestampFromFilename:
    def test_parses_decimal(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("603.834.jpg") == pytest.approx(603.834)

    def test_parses_integer(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("120.jpg") == pytest.approx(120.0)

    def test_raises_on_junk(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        with pytest.raises(ValueError):
            timestamp_seconds_from_filename("notanumber.jpg")

    def test_accepts_png(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("50.5.png") == pytest.approx(50.5)


class TestAnchorScreenshots:
    def _turns(self, *ss_pairs):
        return [
            {"turn_number": n, "start_seconds": s}
            for n, s in ss_pairs
        ]

    def test_anchors_to_latest_turn_at_or_before(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0), (3, 10.0))
        # Screenshot at 7s should anchor to turn 2 (5.0 <= 7 < 10.0)
        result = anchor_screenshots(["7.000.jpg"], turns)
        assert len(result) == 1
        assert result[0]["anchor_turn"] == 2
        assert result[0]["filename"] == "7.000.jpg"
        assert result[0]["timestamp_seconds"] == pytest.approx(7.0)

    def test_boundary_exact_match(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0), (3, 10.0))
        result = anchor_screenshots(["5.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 2  # <= is inclusive

    def test_before_all_turns_anchors_to_first(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 10.0), (2, 20.0))
        result = anchor_screenshots(["3.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 1

    def test_after_all_turns_anchors_to_last(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0))
        result = anchor_screenshots(["100.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 2

    def test_sorted_by_anchor_turn(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 10.0), (3, 20.0))
        result = anchor_screenshots(["25.000.jpg", "5.000.jpg", "15.000.jpg"], turns)
        assert [r["anchor_turn"] for r in result] == [1, 2, 3]

    def test_empty_screenshots_returns_empty(self):
        from annotator.core.screenshots import anchor_screenshots
        assert anchor_screenshots([], []) == []


class TestLoadAnchoredScreenshots:
    def test_filters_flagged_and_eedi_ip(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        from annotator.core.storage import load_transcript

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)
        result = load_anchored_screenshots(conv_id, conv["turns"])
        # Only 4.000.jpg is usable -- 11.500.jpg has eedi_ip=True
        assert len(result) == 1
        assert result[0]["filename"] == "4.000.jpg"
        # Fixture turn starts: 1->0s, 2->3s, 3->10s. 4s anchors to turn 2
        # (latest start_seconds <= 4.0 is 3.0 -> turn 2).
        assert result[0]["anchor_turn"] == 2

    def test_empty_when_no_screenshots(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        conv = {"turns": [{"turn_number": 1, "start_seconds": 0.0}]}
        result = load_anchored_screenshots("nonexistent_conv_id", conv["turns"])
        assert result == []

    def test_storage_path_composed(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        from annotator.core.storage import load_transcript

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)
        result = load_anchored_screenshots(conv_id, conv["turns"])
        assert result[0]["storage_path"] == "deidentified/screenshots/099bf759-abcd/4.000.jpg"
