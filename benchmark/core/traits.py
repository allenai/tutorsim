"""Per-scenario trait generator + cache for trait-mode synthetic students.

Wraps `benchmark.synth_students.TraitGenerator` (verbatim port of Alexis et
al.'s student-trait pipeline) with a persistent file cache keyed by
(conv_id, cut_turn). The generator reads ONLY scenario.transcript_prefix and
scenario.student_context -- it never touches the post-cut conversation, so
trait mode does not leak oracle information about the real student.

Trait modes accepted (passed straight to TraitGenerator.generate_trait):
- "<dimension>"            -> default sentence count
- "<dimension>-<n>"        -> n sentences
- "joined" / "joined-<n>"  -> all five dimensions, concatenated
where <dimension> is one of synth_students.dimension.ALL_DIMENSION_NAMES.

The bare alias "trait" resolves to "joined-3" (the default behavior baked
into the original v5 trait_generator prompt).
"""
from __future__ import annotations

import datetime
import logging

from annotator.core.storage import _get_backend
from benchmark.core.scenarios import Scenario
from benchmark.synth_students._adapter import ModelWrapperAdapter
from benchmark.synth_students.traits import TraitGenerator

logger = logging.getLogger(__name__)


_TRAIT_CACHE_DIR_NAME = "_trait_cache"
_DEFAULT_TRAIT_MODE = "joined-3"


def _resolve_trait_mode(student_mode: str) -> str:
    """Map the exchange-layer `student_mode` to a TraitGenerator trait_mode.

    'trait' is our shorthand for the default consolidated persona ('joined-3').
    Everything else is passed through verbatim ('joined-2', 'affect-3', etc.).
    """
    return _DEFAULT_TRAIT_MODE if student_mode == "trait" else student_mode


def _trait_cache_filename(scenario: Scenario, trait_mode: str) -> str:
    """Cache file relpath: <conv_id>__<cut_turn>__<trait_mode>.json"""
    safe_conv = scenario.conv_id.replace("/", "_")
    safe_mode = trait_mode.replace("/", "_")
    return f"{safe_conv}__{scenario.cut_turn}__{safe_mode}.json"


def _trait_cache_relpath(scenario: Scenario, trait_mode: str) -> str:
    return f"results/benchmark/{_TRAIT_CACHE_DIR_NAME}/{_trait_cache_filename(scenario, trait_mode)}"


def _load_cached_persona(scenario: Scenario, trait_mode: str) -> "str | None":
    be = _get_backend()
    data = be.read_json(_trait_cache_relpath(scenario, trait_mode))
    if data and isinstance(data, dict) and "persona" in data:
        return data["persona"]
    return None


def _save_cached_persona(scenario: Scenario, trait_mode: str, persona: str,
                         *, generator_model: str) -> None:
    payload = {
        "conv_id": scenario.conv_id,
        "cut_turn": scenario.cut_turn,
        "trait_mode": trait_mode,
        "persona": persona,
        "generator_model": generator_model,
        "prefix_length_chars": len(scenario.transcript_prefix),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    be = _get_backend()
    be.write_json(_trait_cache_relpath(scenario, trait_mode), payload)


def get_or_generate_trait(
    scenario: Scenario,
    student_mode: str,
    model_client,
    model_name: str,
) -> str:
    """Return the cached persona for (conv_id, cut_turn, trait_mode), else
    generate via synth_students.TraitGenerator.

    The generator sees only scenario.transcript_prefix and scenario.student_context;
    it NEVER reads the full conversation object -- oracle-safe by construction.
    """
    trait_mode = _resolve_trait_mode(student_mode)

    cached = _load_cached_persona(scenario, trait_mode)
    if cached is not None:
        logger.info("trait cache hit: %s cut=%d mode=%s",
                    scenario.conv_id[:24], scenario.cut_turn, trait_mode)
        return cached

    adapter = ModelWrapperAdapter(model_client)
    generator = TraitGenerator(adapter)
    persona = generator.generate_trait(
        conversation_text=scenario.transcript_prefix or "",
        trait_mode=trait_mode,
    )
    if not isinstance(persona, str):
        # generate_trait can return a tuple when do_return_message_history /
        # do_return_raw_output are set; we never set those, so this is defensive.
        persona = persona[0] if persona else ""
    persona = persona.strip()

    _save_cached_persona(scenario, trait_mode, persona, generator_model=model_name)
    logger.info("trait generated: %s cut=%d mode=%s (%d chars)",
                scenario.conv_id[:24], scenario.cut_turn, trait_mode, len(persona))
    return persona
