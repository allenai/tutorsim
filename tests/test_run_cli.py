"""Tests for the top-level annotator pipeline CLI (annotator.run)."""
import inspect


class TestRerunFlag:
    def test_rerun_defaults_false(self):
        from annotator.run import build_parser
        args = build_parser().parse_args([])
        assert args.rerun is False

    def test_rerun_sets_true(self):
        from annotator.run import build_parser
        args = build_parser().parse_args(["--rerun"])
        assert args.rerun is True

    def test_detect_and_annotate_accept_rerun(self):
        # The pipeline threads args.rerun into these two phases; both must accept
        # the kwarg or the wiring raises TypeError at call time.
        from annotator.core.detect import run_detect
        from annotator.core.annotate import run_annotate
        assert "rerun" in inspect.signature(run_detect).parameters
        assert "rerun" in inspect.signature(run_annotate).parameters
