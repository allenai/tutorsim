"""Test that grade school math fixtures load correctly."""

import json
from pathlib import Path

import pytest


class TestGradeSchoolFixtures:
    """Test grade school math transcript fixtures."""

    @pytest.fixture
    def fixture_path(self):
        """Path to the grade school math transcripts fixture."""
        return Path(__file__).parent / "fixtures" / "grade_school_math_transcripts.jsonl"

    def test_fixture_exists(self, fixture_path):
        """Test that the fixture file exists."""
        assert fixture_path.exists(), f"Fixture file not found at {fixture_path}"

    def test_fixture_format(self, fixture_path):
        """Test that all transcripts have the correct format."""
        with open(fixture_path) as f:
            transcripts = [json.loads(line) for line in f]

        assert len(transcripts) == 5, "Should have exactly 5 transcripts"

        for idx, transcript in enumerate(transcripts):
            # Check required fields
            assert "messages" in transcript, f"Transcript {idx} missing 'messages'"
            assert "metadata" in transcript, f"Transcript {idx} missing 'metadata'"

            # Check messages structure
            messages = transcript["messages"]
            assert len(messages) > 0, f"Transcript {idx} has no messages"

            # First message should be system
            assert messages[0]["role"] == "system", f"Transcript {idx} should start with system message"

            # Should have alternating user/assistant messages after system
            for i in range(1, len(messages)):
                if i % 2 == 1:  # Odd indices should be user
                    assert messages[i]["role"] == "user", f"Transcript {idx}, message {i} should be 'user'"
                else:  # Even indices should be assistant
                    assert messages[i]["role"] == "assistant", f"Transcript {idx}, message {i} should be 'assistant'"

            # Check metadata
            metadata = transcript["metadata"]
            required_metadata = ["session_id", "timestamp", "grade_level", "topic", "student_id"]
            for field in required_metadata:
                assert field in metadata, f"Transcript {idx} missing metadata field '{field}'"

    def test_math_topics_covered(self, fixture_path):
        """Test that different math topics are covered."""
        with open(fixture_path) as f:
            transcripts = [json.loads(line) for line in f]

        topics = [t["metadata"]["topic"] for t in transcripts]
        expected_topics = [
            "division",
            "addition_with_regrouping",
            "fractions",
            "multiplication",
            "subtraction_with_regrouping",
        ]

        assert set(topics) == set(expected_topics), f"Topics don't match. Got: {topics}"

    def test_grade_levels(self, fixture_path):
        """Test that appropriate grade levels are represented."""
        with open(fixture_path) as f:
            transcripts = [json.loads(line) for line in f]

        grade_levels = [t["metadata"]["grade_level"] for t in transcripts]

        # Should have grade levels between 2 and 4 (elementary)
        assert all(2 <= level <= 4 for level in grade_levels), "All grade levels should be between 2 and 4"
        assert len(set(grade_levels)) >= 2, "Should have at least 2 different grade levels"

    def test_conversation_flow(self, fixture_path):
        """Test that conversations have realistic back-and-forth."""
        with open(fixture_path) as f:
            transcripts = [json.loads(line) for line in f]

        for idx, transcript in enumerate(transcripts):
            messages = transcript["messages"]

            # Should have at least 4 exchanges (system + 3+ back-and-forth)
            assert len(messages) >= 4, f"Transcript {idx} should have at least 4 messages"

            # Check that assistant responses are substantial
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            for msg in assistant_messages:
                assert len(msg["content"]) >= 50, "Assistant messages should be substantial (teaching)"

            # Check that there's actual math content
            all_content = " ".join([m["content"] for m in messages])
            # Should contain numbers and math operations
            assert any(char.isdigit() for char in all_content), f"Transcript {idx} should contain numbers"
            assert any(
                op in all_content for op in ["+", "-", "x", "×", "÷", "/", "times", "plus", "minus", "divided"]
            ), f"Transcript {idx} should contain math operations"

    def test_load_with_api(self, fixture_path):
        """Test that fixtures work with the tutor_bench API."""
        from tutor_bench import Annotator, Evaluator

        # Test that annotator can process the fixture
        annotator = Annotator(verbose=False)
        annotations = annotator.process_transcripts(str(fixture_path))
        assert len(annotations) == 5, "Should generate annotations for all 5 transcripts"

        # Save annotations to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            annotations.save(f.name)
            temp_annotations = f.name

        # Test that evaluator can process both files
        evaluator = Evaluator(verbose=False)
        metrics = evaluator.evaluate(transcripts=str(fixture_path), annotations=temp_annotations)

        assert metrics.num_transcripts == 5
        assert metrics.num_annotations == 5
        assert metrics.avg_quality_score > 0

        # Clean up
        Path(temp_annotations).unlink()
