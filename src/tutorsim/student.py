"""Student trait generation and oracle simulator prompt for tutorsim.

Ports the live trait-generation path from:
  _archive/benchmark/synth_students/traits.py  (TraitGenerator)
  _archive/benchmark/synth_students/dimension.py (dimension loading)
  _archive/benchmark/synth_students/_adapter.py  (ModelWrapperAdapter)
  _archive/benchmark/core/traits.py              (get_or_generate_trait + disk cache)

Dead code dropped: BaseTraitEditor, DistractorTraitEditor, UnconditionalTraitEditor,
generate_batch_traits, _generate_batch_joined_traits, generate_trait_with_metadata,
get_history_str, get_default_trait_modes.

Public API:
    generate_trait(transcript_prefix, mode, *, model_client, ...) -> str
    get_or_generate_trait(scenario, mode, model_client, model_name, cache_dir) -> str
"""
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from tutorsim.resources import resource_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt resources (flattened -- no v6 subdir)
# ---------------------------------------------------------------------------

_PROMPTS_DIR = "prompts/student"
_DIMENSIONS_DIR = f"{_PROMPTS_DIR}/dimensions"
_TRAIT_GEN_DIR = f"{_PROMPTS_DIR}/trait_generator"

# ---------------------------------------------------------------------------
# Dimension registry (ported verbatim from dimension.py)
# ---------------------------------------------------------------------------

ENGAGEMENT_STATIC_DIMENSIONS: dict = {
    "distractedness": None,
    "active_vs_passive_learning": None,
    "affect": None,
}

ENGAGEMENT_CHANGE_DIMENSIONS: dict = {}

COGNITIVE_STATIC_DIMENSIONS: dict = {
    "misconceptions": None,
}

COGNITIVE_CHANGE_DIMENSIONS: dict = {
    "learning_efficiency": None,
}

ALL_ENGAGEMENT_DIMENSIONS: dict = {
    **ENGAGEMENT_STATIC_DIMENSIONS,
    **ENGAGEMENT_CHANGE_DIMENSIONS,
}

ALL_COGNITIVE_DIMENSIONS: dict = {
    **COGNITIVE_STATIC_DIMENSIONS,
    **COGNITIVE_CHANGE_DIMENSIONS,
}

ALL_DIMENSIONS: dict = {**ALL_ENGAGEMENT_DIMENSIONS, **ALL_COGNITIVE_DIMENSIONS}

ALL_DIMENSION_NAMES: list = list(ALL_DIMENSIONS.keys()) + [
    "all",
    "all_engagement",
    "all_cognitive",
    "joined",
    "joined_misconceptions_distractedness",
    "joined_misconceptions_affect",
]


def _load_dimension_text(name: str) -> str:
    """Load and return the dimension description text from disk.

    Ported verbatim from dimension.py:_load_dimension_text.
    Points at prompts/student/dimensions/ (flattened, no v6 subdir).
    """
    return resource_text(f"{_DIMENSIONS_DIR}/{name}.txt").rstrip("\n")


def get_dimension_description(persona_dimension: str) -> str:
    """Return the text description for a single dimension or aggregate group.

    Ported verbatim from dimension.py:get_dimension_description.
    Aggregate names ("all", "all_engagement", "all_cognitive") compose
    numbered descriptions. "joined" forms are handled in TraitGenerator,
    not here.
    """
    if persona_dimension not in ALL_DIMENSION_NAMES:
        raise ValueError(
            f"Invalid persona dimension: {persona_dimension}. "
            f"Must be one of: {ALL_DIMENSION_NAMES}"
        )

    if persona_dimension in [
        "joined",
        "joined_misconceptions_distractedness",
        "joined_misconceptions_affect",
    ]:
        raise ValueError(
            f"'{persona_dimension}' should not call get_dimension_description()"
        )

    if persona_dimension == "all":
        texts = sorted(_load_dimension_text(n) for n in ALL_DIMENSIONS.keys())
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_engagement":
        texts = [
            get_dimension_description(n) for n in sorted(ALL_ENGAGEMENT_DIMENSIONS.keys())
        ]
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    if persona_dimension == "all_cognitive":
        texts = [
            get_dimension_description(n) for n in sorted(ALL_COGNITIVE_DIMENSIONS.keys())
        ]
        return "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    return _load_dimension_text(persona_dimension)


# ---------------------------------------------------------------------------
# Adapter (ported verbatim from _adapter.py, but wraps tutorsim.client.ModelClient)
# ---------------------------------------------------------------------------

class _ModelWrapperAdapter:
    """Thin shim around tutorsim ModelClient mimicking TraitGenerator's model interface.

    TraitGenerator calls self.model.call(non_system_messages, system_prompt, **kw) -> str.
    This adapter translates that into client.generate(user_text, ..., cacheable_prefix=...).

    Ported verbatim from _adapter.py:ModelWrapperAdapter.
    """

    def __init__(self, client) -> None:
        self._client = client

    def call(
        self,
        non_system_messages: list,
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **_kwargs,
    ) -> str:
        """Send a single user-message + system-prompt to the wrapped ModelClient.

        Ported verbatim from _adapter.py:ModelWrapperAdapter.call.
        system_prompt forwarded as cacheable_prefix (prompt cache friendly).
        """
        user_text = "\n\n".join(
            m.get("content", "")
            for m in (non_system_messages or [])
            if m.get("role") in (None, "user")
        )
        resp = self._client.generate(
            user_text,
            json_mode=False,
            max_tokens=max_tokens or 1024,
            cacheable_prefix=system_prompt or None,
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# TraitGenerator (ported verbatim from traits.py -- live path only)
# ---------------------------------------------------------------------------

class TraitGenerator:
    """Generate student traits from example conversations.

    Ported from _archive/benchmark/synth_students/traits.py.
    Dead code dropped: batch methods, trait editors, get_history_str usage,
    do_return_message_history, do_return_raw_output return paths.
    Live path only: generate_trait() for single + joined modes.
    """

    def __init__(self, model: _ModelWrapperAdapter) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # System / user prompt builders (verbatim from traits.py)
    # ------------------------------------------------------------------

    def get_trait_system_prompt(
        self, trait_type: str, num_sentences: Optional[int] = None
    ) -> str:
        """Build the system prompt for a single dimension.

        Loads prompts/student/trait_generator/system.txt and substitutes
        {dimension_description} and {sentence_count_suffix}.
        Ported verbatim from traits.py:TraitGenerator.get_trait_system_prompt.
        """
        dimension_description = get_dimension_description(trait_type)

        template = resource_text(f"{_TRAIT_GEN_DIR}/system.txt").rstrip("\n")

        if num_sentences is None:
            suffix = ""
        elif num_sentences > 1:
            suffix = f" The description should be {num_sentences} sentences long."
        else:
            suffix = " The description should be 1 sentence long."

        return (
            template
            .replace("{dimension_description}", dimension_description)
            .replace("{sentence_count_suffix}", suffix)
        )

    def _get_user_prompt(
        self, conversation_text: str, num_sentences: Optional[int] = None
    ) -> str:
        """Build the user prompt for trait generation.

        Loads prompts/student/trait_generator/user.txt and substitutes
        {conversation_text} and {num_sentences}.
        Ported verbatim from traits.py:TraitGenerator._get_user_prompt.
        """
        template = resource_text(f"{_TRAIT_GEN_DIR}/user.txt").rstrip("\n")
        return (
            template
            .replace("{conversation_text}", conversation_text)
            .replace("{num_sentences}", str(num_sentences))
        )

    # ------------------------------------------------------------------
    # Parsing (verbatim from traits.py)
    # ------------------------------------------------------------------

    def parse_trait_output(self, raw_output: str) -> tuple:
        """Parse raw model output to extract description after 'DESCRIPTION:' marker.

        Ported verbatim from traits.py:TraitGenerator.parse_trait_output.
        Returns (description, raw_output).
        """
        if "DESCRIPTION:" in raw_output:
            parts = raw_output.split("DESCRIPTION:", 1)
            description = parts[1].strip()
        else:
            description = raw_output.strip()
        return description, raw_output

    def parse_trait_type(self, trait_type: str) -> tuple:
        """Parse trait mode to (base_trait_type, num_sentences).

        Ported verbatim from traits.py:TraitGenerator.parse_trait_type.
        """
        if "-" not in trait_type:
            base_trait_type = trait_type
            num_sentences = None
        else:
            base_trait_type, num_sentences_str = trait_type.split("-", 1)
            num_sentences = int(num_sentences_str)

        if base_trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid base trait type: {base_trait_type}. "
                f"Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        if num_sentences is not None and num_sentences <= 0:
            raise ValueError(
                f"Invalid number of sentences: {num_sentences}. Must be greater than 0."
            )

        return base_trait_type, num_sentences

    # ------------------------------------------------------------------
    # Joined trait generation (verbatim from traits.py:_generate_joined_trait)
    # ------------------------------------------------------------------

    def _generate_joined_trait(
        self,
        conversation_text: str,
        num_sentences: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        individual_dimensions: Optional[list] = None,
    ) -> str:
        """Generate one trait per dimension, then sort + join.

        Ported verbatim from traits.py:TraitGenerator._generate_joined_trait.
        Dropped: do_return_message_history / do_return_raw_output paths (dead
        in the live benchmark path).
        """
        if individual_dimensions is None:
            individual_dimensions = list(ALL_DIMENSIONS.keys())

        dimension_traits: dict = {}
        for dimension in individual_dimensions:
            dimension_mode = dimension
            if num_sentences is not None:
                dimension_mode = f"{dimension}-{num_sentences}"

            trait = self.generate_trait(
                conversation_text=conversation_text,
                trait_mode=dimension_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            dimension_traits[dimension] = trait

        return "\n\n".join(sorted(dimension_traits.values()))

    # ------------------------------------------------------------------
    # Main entry point (verbatim from traits.py:generate_trait, live path)
    # ------------------------------------------------------------------

    def generate_trait(
        self,
        conversation_text: str,
        trait_mode: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a student trait description from a conversation.

        Ported verbatim from traits.py:TraitGenerator.generate_trait.
        Dropped: do_return_message_history, do_return_raw_output return paths.
        """
        trait_type, num_sentences = self.parse_trait_type(trait_mode)

        if trait_type not in ALL_DIMENSION_NAMES:
            raise ValueError(
                f"Invalid trait mode: {trait_mode}. "
                f"Must start with one of: {ALL_DIMENSION_NAMES}"
            )

        # Joined modes: generate per-dimension, then consolidate.
        if trait_type in [
            "joined",
            "joined_misconceptions_distractedness",
            "joined_misconceptions_affect",
        ]:
            if trait_type == "joined_misconceptions_distractedness":
                individual_dimensions = ["misconceptions", "distractedness"]
            elif trait_type == "joined_misconceptions_affect":
                individual_dimensions = ["misconceptions", "affect"]
            else:  # "joined"
                individual_dimensions = list(ALL_DIMENSIONS.keys())

            return self._generate_joined_trait(
                conversation_text=conversation_text,
                num_sentences=num_sentences,
                temperature=temperature,
                max_tokens=max_tokens,
                individual_dimensions=individual_dimensions,
            )

        # Single-dimension path.
        system_prompt = self.get_trait_system_prompt(trait_type, num_sentences)
        user_prompt = self._get_user_prompt(conversation_text, num_sentences)
        non_system_messages = [{"role": "user", "content": user_prompt}]

        trait_response = self.model.call(
            non_system_messages=non_system_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        description, _raw = self.parse_trait_output(trait_response)
        return description


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

def generate_trait(
    transcript_prefix: str,
    mode: str = "joined-3",
    *,
    model_client,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a student trait description from a transcript prefix.

    Thin wrapper around TraitGenerator for module-level ergonomics.

    Args:
        transcript_prefix: Pre-cut conversation text to analyse.
        mode: Trait mode string (default "joined-3").
        model_client: A tutorsim ModelClient instance.
        temperature: Model temperature (default 0.7, adapter default).
        max_tokens: Max tokens (default None -> adapter uses 1024).

    Returns:
        Parsed trait description string.
    """
    adapter = _ModelWrapperAdapter(model_client)
    generator = TraitGenerator(adapter)
    return generator.generate_trait(
        conversation_text=transcript_prefix,
        trait_mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Disk cache + get_or_generate_trait
# ---------------------------------------------------------------------------

_DEFAULT_TRAIT_MODE = "joined-3"
_DEFAULT_CACHE_DIR = "results/tutorsim/_trait_cache"


def _resolve_trait_mode(student_mode: str) -> str:
    """Map 'trait' alias to 'joined-3'; pass everything else through verbatim.

    Ported verbatim from _archive/benchmark/core/traits.py:_resolve_trait_mode.
    """
    return _DEFAULT_TRAIT_MODE if student_mode in ("trait", "oracle") else student_mode


def _trait_cache_filename(scenario, trait_mode: str) -> str:
    """Return the cache filename for a (scenario, trait_mode) key.

    Key: (conv_id, cut_turn, trait_mode) -- ported verbatim from
    _archive/benchmark/core/traits.py:_trait_cache_filename.
    """
    safe_conv = str(scenario.conv_id).replace("/", "_")
    safe_mode = trait_mode.replace("/", "_")
    return f"{safe_conv}__{scenario.cut_turn}__{safe_mode}.json"


def get_or_generate_trait(
    scenario,
    student_mode: str,
    model_client,
    model_name: str,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> str:
    """Return cached persona or generate via TraitGenerator.

    Cache key: (conv_id, cut_turn, trait_mode) stored as JSON on disk.
    Ported verbatim from _archive/benchmark/core/traits.py:get_or_generate_trait.

    Args:
        scenario: Object with .conv_id, .cut_turn, .transcript_prefix attributes.
        student_mode: Mode string ('trait', 'joined-3', 'affect-3', etc.).
        model_client: A tutorsim ModelClient instance.
        model_name: Model name string (stored in cache for provenance).
        cache_dir: Directory for cache files (default: results/tutorsim/_trait_cache/).

    Returns:
        Trait persona string (cached or freshly generated).
    """
    trait_mode = _resolve_trait_mode(student_mode)
    cache_path = Path(cache_dir) / _trait_cache_filename(scenario, trait_mode)

    # Cache hit.
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "persona" in payload:
                logger.info(
                    "trait cache hit: %s cut=%d mode=%s",
                    str(scenario.conv_id)[:24],
                    scenario.cut_turn,
                    trait_mode,
                )
                return payload["persona"]
        except Exception as e:
            logger.warning("Corrupt cache file %s: %s -- regenerating", cache_path, e)

    # Cache miss: generate.
    persona = generate_trait(
        transcript_prefix=scenario.transcript_prefix or "",
        mode=trait_mode,
        model_client=model_client,
    )
    if not isinstance(persona, str):
        persona = persona[0] if persona else ""
    persona = persona.strip()

    # Write cache.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conv_id": scenario.conv_id,
        "cut_turn": scenario.cut_turn,
        "trait_mode": trait_mode,
        "persona": persona,
        "generator_model": model_name,
        "prefix_length_chars": len(scenario.transcript_prefix or ""),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        "trait generated: %s cut=%d mode=%s (%d chars)",
        str(scenario.conv_id)[:24],
        scenario.cut_turn,
        trait_mode,
        len(persona),
    )
    return persona


# ---------------------------------------------------------------------------
# Oracle student system prompt builder (Task 4)
# ---------------------------------------------------------------------------

_STUDENT_PROMPTS_DIR = _PROMPTS_DIR

_EMPTY_REFERENCE_FALLBACK = (
    "(The real conversation ends at the cut point — the student made no "
    "further turns. Acknowledge any final tutor message briefly, then end "
    "the session.)"
)


def _get_shared_generation_instructions() -> str:
    """Return the shared generation instructions text.

    Ported verbatim from _archive/benchmark/synth_students/prompts.py:
    get_shared_generation_instructions (oracle mode; convo_length unused).
    """
    return (
        "You may generate multiple student turns in a row as needed. "
        "Do *not* generate any turns as the tutor. "
        "You should only generate turns that involve student utterances or actions. "
        "Remember to wait for the tutor to respond before generating your next turns."
    )


def _render_oracle_template() -> str:
    """Load prompts/student/oracle.txt and substitute {shared_generation_instructions}.

    Ported verbatim from _archive/benchmark/synth_students/prompts.py:_render
    (oracle mode, include_example=True).
    """
    template = resource_text(f"{_STUDENT_PROMPTS_DIR}/oracle.txt").rstrip("\n")
    shared = _get_shared_generation_instructions()
    return template.replace("{shared_generation_instructions}", shared)


def build_student_system_prompt(
    *,
    student_context: str,
    reference_transcript: str | None,
    persona: str,
) -> str:
    """Assemble the STUDENT system prompt for oracle mode.

    Loads prompts/student/oracle.txt, substitutes {shared_generation_instructions},
    then fills the three bracket placeholders:
      [[REFERENCE_TRANSCRIPT_HERE]]           <- reference_transcript (fallback if empty)
      [[PERSONA_DESCRIPTION_HERE]]            <- persona
      [[NEXT_CONVERSATION_INFORMATION_HERE]]  <- student_context

    Ported verbatim-in-behavior from:
      _archive/benchmark/synth_students/prompts.py  (OracleMomentStudentPrompt / _render)
      _archive/benchmark/core/students.py           (build_student_system_prompt oracle branch)

    Args:
        student_context: Text for the student's current context / conversation info.
        reference_transcript: Full post-cut real transcript the oracle student imitates.
            If empty or None, the empty-reference fallback message is used instead.
        persona: Trait persona description (generated by generate_trait / get_or_generate_trait).

    Returns:
        Fully-substituted system prompt string (ASCII-safe, no unreplaced placeholders).
    """
    out = _render_oracle_template()

    # Substitute shared + bracket placeholders (order mirrors students.py).
    out = out.replace("[[NEXT_CONVERSATION_INFORMATION_HERE]]", student_context or "")
    out = out.replace("[[PERSONA_DESCRIPTION_HERE]]", persona or "")

    # Empty reference fallback: same logic as students.py lines 155-160.
    ref = reference_transcript or _EMPTY_REFERENCE_FALLBACK
    out = out.replace("[[REFERENCE_TRANSCRIPT_HERE]]", ref)

    return out


# ---------------------------------------------------------------------------
# resolve_student (Task 4)
# ---------------------------------------------------------------------------

def resolve_student(student_id: str | None = None) -> dict:
    """Decide what the student is: registered callable or hosted model.

    Mirrors resolve_tutor() in tutor.py. The default hosted student is the
    model in the active config's student.model block.

    Args:
        student_id: Student name registered via @register_student, or None
            to use the default hosted student from config.

    Returns:
        Dictionary with:
        - kind == "registered": {"kind": "registered", "fn": <callable>}
        - kind == "hosted": {"kind": "hosted", "client": ModelClient, "kwargs": dict}
    """
    from tutorsim.config import get_registered_student, student_spec
    from tutorsim.client import ModelClient

    spec = student_spec()
    model = student_id or spec["model"]

    # FIRST: check registry for either an explicit student_id or config student.model.
    if model is not None:
        registered_fn = get_registered_student(model)
        if registered_fn is not None:
            return {"kind": "registered", "fn": registered_fn}

    # ELSE: use the hosted student model from config or the explicit id.
    client = ModelClient(model)
    kwargs = {k: v for k, v in spec.items() if k not in {"model", "mode"}}
    kwargs.setdefault("thinking", False)
    return {
        "kind": "hosted",
        "client": client,
        "kwargs": kwargs,
    }
