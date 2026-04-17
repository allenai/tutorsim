"""Tests for annotator.core.config."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear the config cache before each test."""
    import annotator.core.config as cfg
    cfg._loaded_config = None
    yield
    cfg._loaded_config = None


class TestGetPhaseConfig:
    def test_returns_model(self):
        from annotator.core.config import get_phase_config
        cfg = get_phase_config("detect", "anthropic")
        assert "model" in cfg
        assert cfg["model"] == "claude-opus-4-6"

    def test_phase_overrides_merge(self):
        from annotator.core.config import get_phase_config
        cfg = get_phase_config("annotate", "anthropic")
        assert "max_tokens" in cfg
        assert "context_window" in cfg
        assert cfg["context_window"] == 20

    def test_unknown_profile_raises(self):
        from annotator.core.config import get_phase_config
        with pytest.raises(ValueError, match="Unknown profile"):
            get_phase_config("detect", "nonexistent_profile")

    def test_no_profile_raises_when_missing(self):
        from annotator.core.config import get_phase_config, load_config
        config = load_config()
        original = config.get("profile")
        try:
            config.pop("profile", None)
            with pytest.raises(ValueError, match="No profile specified"):
                get_phase_config("detect", None)
        finally:
            if original is not None:
                config["profile"] = original


class TestGetArchetypeAnnotators:
    def test_returns_full_mapping(self):
        from annotator.core.config import get_archetype_annotators
        mapping = get_archetype_annotators()
        assert "generous" in mapping
        assert "balanced" in mapping
        assert "demanding" in mapping
        assert isinstance(mapping["generous"], set)

    def test_returns_set_for_archetype(self):
        from annotator.core.config import get_archetype_annotators
        generous = get_archetype_annotators("generous")
        assert isinstance(generous, set)
        assert "Gerber" in generous

    def test_returns_none_for_unknown(self):
        from annotator.core.config import get_archetype_annotators
        assert get_archetype_annotators("nonexistent") is None


class TestGetIouThreshold:
    def test_returns_float(self):
        from annotator.core.config import get_iou_threshold
        val = get_iou_threshold()
        assert isinstance(val, float)
        assert val == 0.3


class TestResolveRunParams:
    def test_cli_overrides_all(self):
        from annotator.core.config import resolve_run_params
        params = resolve_run_params(
            cli_version="test_v1",
            cli_profile="gemini",
            cli_style="generous",
            cli_prompt_version="v4",
        )
        assert params["version"] == "test_v1"
        assert params["profile"] == "gemini"
        assert params["style"] == "generous"
        assert params["prompt_version"] == "v4"

    def test_auto_generates_version(self):
        from annotator.core.config import resolve_run_params
        params = resolve_run_params(
            cli_version=None,
            cli_profile="anthropic",
            cli_style=None,
            cli_prompt_version=None,
        )
        assert "anthropic_" in params["version"]
        assert params["profile"] == "anthropic"
