"""
Pipeline configuration loader.

Reads pipeline/config.yaml once, resolves the active profile,
and provides per-phase config dicts to callers.

Usage:
    from pipeline.core.config import get_phase_config

    cfg = get_phase_config("detect")           # uses default profile
    cfg = get_phase_config("detect", "openai")  # uses openai profile
    model = cfg["model"]
    mode = cfg.get("mode", "batch")
"""

import yaml
from pathlib import Path
from typing import Optional

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
    """Get config for a specific phase (detect/annotate/label/advisor).

    Args:
        phase: One of 'detect', 'annotate', 'label', 'advisor'.
        profile: Profile name. If None, uses config's 'profile' field.

    Returns:
        Dict with keys like 'model', 'max_tokens', 'mode', etc.
    """
    config = load_config()
    profile_name = profile or config.get("profile", "gemini")
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available: {', '.join(profiles.keys())}"
        )
    phase_config = profiles[profile_name].get(phase, {})
    if not phase_config:
        raise ValueError(
            f"Profile '{profile_name}' has no config for phase '{phase}'"
        )
    return dict(phase_config)


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
