from tutorsim import config as cfgmod


def test_packaged_default_config_parses_and_has_expected_roster():
    cfgmod._reset_config_cache()
    cfg = cfgmod.load_config()
    assert set(cfg["providers"]) == {"anthropic", "openai", "gemini", "together"}
    assert set(cfg["models"]) == {
        "claude-opus-4-8", "claude-sonnet-4-6", "gemini-2.5-pro", "gemini-3.5-flash",
        "gpt-5.4-mini-2026-03-17", "gpt-5.5-2026-04-23", "deepseek-ai/DeepSeek-V4-Pro",
    }
    assert cfg["models"]["claude-opus-4-8"] == {"thinking": True, "effort": "xhigh"}
    assert cfg["models"]["deepseek-ai/DeepSeek-V4-Pro"] == {}
    assert cfg["student"] == {"model": "claude-opus-4-6", "mode": "oracle", "thinking": False}
    assert cfg["scorer"] == {"model": "claude-opus-4-6", "thinking": "adaptive"}
    assert cfg["defaults"] == {"seed": 10, "trials": 1, "max_turns": 5}
    assert cfg["retry"] == {"max_retries": 5, "base_delay": 5}
    assert cfg["batch"] == {"timeout": 86400}


def test_get_retry_config_reads_yaml():
    cfgmod._reset_config_cache()
    assert cfgmod.get_retry_config() == {"max_retries": 5, "base_delay": 5}


def test_get_batch_timeout_reads_yaml():
    cfgmod._reset_config_cache()
    assert cfgmod.get_batch_timeout() == 86400


def test_load_config_returns_parsed_dict():
    cfgmod._reset_config_cache()
    c = cfgmod.load_config()
    assert c["scorer"]["model"] == "claude-opus-4-6"
    assert "models" in c and "providers" in c
    assert cfgmod.describe_config_source() == "tutorsim:default_config.yaml"


def test_load_config_accepts_explicit_override(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  openai:    { env: OPENAI_API_KEY }
  gemini:    { env: GEMINI_API_KEY }
  together:  { env: TOGETHER_API_KEY }
models:
  claude-opus-4-8: { thinking: false }
student: { model: claude-opus-4-6, mode: oracle, thinking: false }
scorer:  { model: claude-opus-4-6, thinking: adaptive }
defaults: { seed: 99, trials: 1, max_turns: 2 }
retry:    { max_retries: 1, base_delay: 1 }
batch:    { timeout: 123 }
""",
        encoding="utf-8",
    )

    cfgmod._reset_config_cache()
    cfg = cfgmod.load_config(config_path)
    assert cfg["defaults"]["seed"] == 99
    rc = cfgmod.build_run_config(
        tutors=["claude-opus-4-8"],
        config_path=config_path,
    )
    assert rc.seed == 99
    assert rc.config_source == str(config_path)


def test_resolve_model_known():
    r = cfgmod.resolve_model("claude-opus-4-8")
    assert r["provider"] == "anthropic"
    assert r["env"] == "ANTHROPIC_API_KEY"
    assert r["kwargs"] == {"thinking": True, "effort": "xhigh"}


def test_resolve_model_together_empty_kwargs():
    r = cfgmod.resolve_model("deepseek-ai/DeepSeek-V4-Pro")
    assert r["provider"] == "together"
    assert r["kwargs"] == {}


def test_resolve_model_gemini():
    r = cfgmod.resolve_model("gemini-2.5-pro")
    assert r["provider"] == "gemini"
    assert r["env"] == "GEMINI_API_KEY"
    assert r["kwargs"] == {"thinking": True, "thinking_budget": -1}


def test_resolve_model_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        cfgmod.resolve_model("gpt-9-imaginary")


def test_scorer_and_student_specs():
    assert cfgmod.scorer_spec()["model"] == "claude-opus-4-6"
    assert cfgmod.student_spec()["thinking"] is False


def test_build_run_config_defaults():
    rc = cfgmod.build_run_config(tutors=["claude-opus-4-8"])
    assert rc.modes == ["plain", "scaffolding_rigor"]
    assert rc.dataset == "balanced_520"
    assert rc.max_turns == 5 and rc.trials == 1 and rc.seed == 10
    assert rc.sample is None
    assert rc.resolved_tutors["claude-opus-4-8"] == {"thinking": True, "effort": "xhigh"}


def test_build_run_config_overrides():
    rc = cfgmod.build_run_config(tutors=["gpt-5.5-2026-04-23"], modes=["plain"], sample=10, trials=3)
    assert rc.modes == ["plain"] and rc.sample == 10 and rc.trials == 3
    assert rc.resolved_tutors["gpt-5.5-2026-04-23"] == {"thinking": True, "reasoning_effort": "high"}


def test_register_and_lookup_tutor():
    from tutorsim import register_tutor

    @register_tutor("my-model")
    def my_tutor(conversation):
        return "next turn"

    assert cfgmod.get_registered_tutor("my-model") is my_tutor
    rc = cfgmod.build_run_config(tutors=["my-model"])
    assert "my-model" in rc.tutors
    assert rc.resolved_tutors["my-model"] == {}


def test_register_and_lookup_student():
    from tutorsim import register_student

    @register_student("my-student")
    def my_student(conversation):
        return "student turn"

    assert cfgmod.get_registered_student("my-student") is my_student


def test_registered_tutor_lookup_returns_none_for_unknown():
    assert cfgmod.get_registered_tutor("nonexistent") is None


def test_registered_student_lookup_returns_none_for_unknown():
    assert cfgmod.get_registered_student("nonexistent") is None
