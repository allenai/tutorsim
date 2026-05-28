"""
Pipeline configuration loader.

Reads pipeline/config.yaml once, resolves the active profile,
and provides per-phase config dicts to callers.

Usage:
    from annotator.core.config import get_phase_config

    cfg = get_phase_config("detect")           # uses default profile
    cfg = get_phase_config("detect", "openai")  # uses openai profile
    model = cfg["model"]
    mode = cfg.get("mode", "batch")
"""

import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
_loaded_config = None


def load_config(path: Optional[Path] = None) -> dict:
    """Load and cache the pipeline config. Returns full config dict."""
    global _loaded_config
    if _loaded_config is not None and path is None:
        return _loaded_config
    config_path = path or _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if path is None:
        _loaded_config = config
    return config


def get_phase_config(phase: str, profile: Optional[str] = None) -> dict:
    """Get config for a specific phase (detect/annotate/label/advisor/tutor).

    Profile-level defaults are merged with phase-specific overrides.
    For example, if the profile defines model and max_tokens at the top level,
    and the 'annotate' section adds context_window, the returned dict contains
    all three keys.

    Args:
        phase: One of 'detect', 'annotate', 'label', 'advisor', 'tutor'.
        profile: Profile name. If None, uses config's 'profile' field.

    Returns:
        Dict with keys like 'model', 'max_tokens', 'mode', etc.
    """
    config = load_config()
    profile_name = profile or config.get("profile")
    if not profile_name:
        raise ValueError("No profile specified and no 'profile' set in config.yaml")
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available: {', '.join(profiles.keys())}"
        )
    profile_data = profiles[profile_name]

    # Profile-level defaults: all non-dict (scalar) values
    base = {k: v for k, v in profile_data.items() if not isinstance(v, dict)}

    # Phase-specific overrides (if any)
    phase_overrides = profile_data.get(phase, {})
    if isinstance(phase_overrides, dict):
        base.update(phase_overrides)

    if "model" not in base:
        raise ValueError(
            f"Profile '{profile_name}' has no 'model' configured"
        )
    return base


def get_retry_config() -> dict:
    """Get retry settings."""
    config = load_config()
    return config.get("retry", {"max_retries": 5, "base_delay": 5})


def get_archetype_annotators(archetype: str | None = None) -> dict[str, set[str]] | set[str] | None:
    """Get annotator IDs by archetype from config.

    Args:
        archetype: If given, return the set of annotator IDs for that archetype.
                   If None, return the full mapping {archetype: set(annotator_ids)}.

    Returns:
        Full mapping dict, or a set of annotator IDs, or None if archetype not found.
    """
    config = load_config()
    raw = config.get("archetype_annotators", {})
    mapping = {k: set(v) for k, v in raw.items()}
    if archetype is None:
        return mapping
    return mapping.get(archetype)


def get_annotator_defaults() -> dict:
    """Get annotator pipeline defaults from config."""
    config = load_config()
    return config.get("annotator", {})


def get_iou_threshold() -> float:
    """Get IoU threshold for detection/effectiveness matching."""
    config = load_config()
    return config.get("eval", {}).get("iou_threshold", 0.3)


def get_batch_timeout() -> int:
    """Get max seconds to poll a batch job before raising."""
    config = load_config()
    return config.get("batch", {}).get("timeout", 86400)


def get_valid_styles() -> list[str]:
    """Get valid annotator styles from archetype_annotators config keys."""
    config = load_config()
    return list(config.get("archetype_annotators", {}).keys())


def get_annotation_types() -> list[str]:
    """Get valid annotation types from config."""
    config = load_config()
    return config.get("annotator", {}).get("annotation_types", ["scaffolding", "rapport"])


def get_benchmark_config(overrides: dict | None = None) -> dict:
    """Get benchmark config section with optional CLI overrides.

    This is the single entry point for benchmark configuration.
    Replaces benchmark/run.py's local load_config().
    """
    import copy
    config = load_config()
    bm = copy.deepcopy(config.get("benchmark", {}))

    if overrides:
        # Direct key overrides (e.g. tutor_profiles, exchange, etc.)
        for key, value in overrides.items():
            if key in bm:
                bm[key] = value

        # CLI-style shorthand overrides mapped to nested keys
        if overrides.get("scenario_mode"):
            bm["scenarios"]["mode"] = overrides["scenario_mode"]
        if overrides.get("max_scenarios"):
            bm["scenarios"]["max_scenarios"] = overrides["max_scenarios"]
        if overrides.get("max_per_conv"):
            bm["scenarios"]["max_per_conv"] = overrides["max_per_conv"]
        if overrides.get("tutor_profile"):
            bm["tutor_profiles"] = [overrides["tutor_profile"]]
        if overrides.get("mode"):
            bm["annotator"]["mode"] = overrides["mode"]
        if overrides.get("test_transcripts"):
            bm["scenarios"]["test_transcripts"] = overrides["test_transcripts"]
        if overrides.get("with_screenshots"):
            bm["with_screenshots"] = True

    return bm


def resolve_run_params(
    cli_version: str | None,
    cli_profile: str | None,
    cli_style: str | None,
    cli_prompt_version: str | None,
) -> dict:
    """Resolve version, profile, style, and prompt_version from CLI + config.

    Resolution order for each: CLI > config > auto-generate/None.
    Returns dict with keys: version, profile, style, prompt_version.
    """
    import datetime

    config = load_config()
    defaults = get_annotator_defaults()

    profile = cli_profile or config.get("profile", "anthropic")

    if cli_version:
        version = cli_version
    elif defaults.get("version"):
        version = defaults["version"]
    else:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        version = f"{profile}_{date_str}"
        logger.info("Auto-generated version: %s", version)

    style = cli_style
    if style is None:
        cfg_style = defaults.get("style")
        if cfg_style is not None:
            style = cfg_style

    # Explicit --prompt-version wins; then explicit --version wins over config default;
    # finally fall back to config prompt_version or the resolved version.
    prompt_version = cli_prompt_version or (cli_version if cli_version else defaults.get("prompt_version")) or version

    return {
        "version": version,
        "profile": profile,
        "style": style,
        "prompt_version": prompt_version,
    }
