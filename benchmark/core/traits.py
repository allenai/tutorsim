"""Per-scenario trait generator + cache for trait-mode synthetic students.

The generator reads ONLY the scenario's transcript_prefix and student_context.
It never touches the full conversation object -- this is what keeps trait mode
from being an oracle on the real student's post-cut turns.
"""
import datetime
import logging

from annotator.core.storage import _get_backend
from benchmark.core.scenarios import Scenario
from benchmark.core.exchange import _load_prompt

logger = logging.getLogger(__name__)


_TRAIT_CACHE_DIR_NAME = "_trait_cache"


def _trait_cache_filename(scenario: Scenario) -> str:
    """Cache file relpath: <conv_id>__<cut_turn>.json"""
    safe_conv = scenario.conv_id.replace("/", "_")
    return f"{safe_conv}__{scenario.cut_turn}.json"


def _trait_cache_relpath(scenario: Scenario) -> str:
    """Storage-backend relative path for the cache file."""
    return f"results/benchmark/{_TRAIT_CACHE_DIR_NAME}/{_trait_cache_filename(scenario)}"


def _load_cached_persona(scenario: Scenario) -> "str | None":
    be = _get_backend()
    data = be.read_json(_trait_cache_relpath(scenario))
    if data and isinstance(data, dict) and "persona" in data:
        return data["persona"]
    return None


def _save_cached_persona(scenario: Scenario, persona: str, *,
                         generator_model: str, prompt_version: str,
                         usage: dict) -> None:
    payload = {
        "conv_id": scenario.conv_id,
        "cut_turn": scenario.cut_turn,
        "persona": persona,
        "generator_model": generator_model,
        "prompt_version": prompt_version,
        "prefix_length_chars": len(scenario.transcript_prefix),
        "usage": usage,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    be = _get_backend()
    be.write_json(_trait_cache_relpath(scenario), payload)


def get_or_generate_trait(
    scenario: Scenario,
    prompt_version: str,
    model_client,
    model_name: str,
) -> str:
    """Return cached persona for (conv_id, cut_turn), else generate via the LLM.

    The generator sees only scenario.transcript_prefix and scenario.student_context.
    It NEVER reads the full conversation object -- oracle-safe by construction.
    """
    cached = _load_cached_persona(scenario)
    if cached is not None:
        logger.info("trait cache hit: %s cut=%d", scenario.conv_id[:24], scenario.cut_turn)
        return cached

    template = _load_prompt(prompt_version, "trait_generator.txt")
    prompt = (
        template
        .replace("{student_context}", scenario.student_context or "")
        .replace("{transcript_prefix}", scenario.transcript_prefix or "")
    )

    response = model_client.generate(prompt, json_mode=False, max_tokens=1024)
    persona = (response.text or "").strip()
    usage = response.usage or {}

    _save_cached_persona(
        scenario, persona,
        generator_model=model_name,
        prompt_version=prompt_version,
        usage=usage,
    )
    logger.info("trait generated: %s cut=%d (%d chars)",
                scenario.conv_id[:24], scenario.cut_turn, len(persona))
    return persona
