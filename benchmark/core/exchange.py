"""Multi-turn exchange orchestration: tutor and student alternate.

Supports two execution modes:
- sync: run_exchange() for one scenario at a time
- batch: run_exchanges_batch() processes all scenarios in parallel,
  batching each round of tutor/student calls across scenarios

Supports multi-message turns (split by [NEXT] delimiter) to match
real transcript patterns where the same speaker sends consecutive messages.
"""

import math
from dataclasses import dataclass, field, asdict

from annotator.core.client import (
    ModelClient, build_batch_entry, run_batch, run_sync_entries,
)
import logging

from .scenarios import Scenario

logger = logging.getLogger(__name__)

NEXT_DELIMITER = "[NEXT]"
NEW_MESSAGE_DELIMITER = "[NEW_MESSAGE]"

END_TOKEN = "[END]"


def _check_end_token(text: str) -> tuple[str, bool]:
    """Strip [END] token from text and report whether it was present.

    The token usually appears at the end of the message but may appear
    anywhere; either way the scenario should end. Trailing whitespace
    after the strip is removed.
    """
    if END_TOKEN not in text:
        return text, False
    cleaned = text.replace(END_TOKEN, "").rstrip()
    return cleaned, True


PROBLEM_CHANGE_TOKEN = "[PROBLEM_CHANGE]"
NEXT_PROBLEM_TOKEN = "[NEXT_PROBLEM]"  # legacy alias for v5 and earlier prompts


def _parse_tutor_tokens(text: str) -> tuple[str, bool, bool]:
    """Strip tutor control tokens and report which were present.

    Returns (cleaned_text, ended, problem_change).
    END takes precedence: if both kinds of tokens appear, ended=True,
    problem_change=False.

    Both [PROBLEM_CHANGE] (v6+) and [NEXT_PROBLEM] (v5 legacy) trigger
    problem_change=True. v6 prompts the tutor to emit [PROBLEM_CHANGE]
    because the "scene change" framing reads more clearly as "this problem
    is over" than "give me a new one to work on" -- the latter wording
    seems to encourage the AI tutor to keep going onto follow-up problems.
    """
    has_end = END_TOKEN in text
    has_change = (PROBLEM_CHANGE_TOKEN in text) or (NEXT_PROBLEM_TOKEN in text)
    cleaned = (
        text.replace(END_TOKEN, "")
            .replace(PROBLEM_CHANGE_TOKEN, "")
            .replace(NEXT_PROBLEM_TOKEN, "")
            .rstrip()
    )
    if has_end:
        return cleaned, True, False
    return cleaned, False, has_change


@dataclass
class Exchange:
    scenario_id: str
    tutor_model: str
    generated_turns: list[dict] = field(default_factory=list)
    tutor_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    student_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    # Per-call latency samples. Only populated in sync mode (batch latency
    # = queue + processing, not useful for model comparison). Each list is
    # one float per LLM call.
    tutor_latencies: list[float] = field(default_factory=list)
    student_latencies: list[float] = field(default_factory=list)
    completed: bool = False
    # "END" | "PROBLEM_CHANGE" | "MAX_TURNS" | "" (older runs may have "NEXT_PROBLEM")
    ended_via: str = ""

    def to_dict(self):
        return asdict(self)


def _build_role_prompt(
    role: str,
    transcript_prefix: str,
    extra: str,
    student_context: str,
    prompt_version: str = "v1",
    student_mode: str | None = None,
    scenario=None,
    trait_client=None,
    trait_model: str | None = None,
    tutor_mode: str | None = None,
    reference_transcript: str | None = None,
) -> tuple[str, str]:
    """Build (cacheable_head, tail) for either tutor or student.

    Thin coordinator: delegates system-prompt assembly to
    `benchmark.core.tutors.build_tutor_system_prompt` /
    `benchmark.core.students.build_student_system_prompt`, then wraps the
    result with the running conversation history.

    head = system_prompt + "Here is the conversation so far:\\n" + transcript_prefix
    tail = extra + "\\n\\n" + role_instruction

    The head is byte-stable across rounds (system + static cut), so the
    prompt cache hits on round 2+. Generated turns flow through `extra` (tail).
    """
    from benchmark.core.tutors import build_tutor_system_prompt
    from benchmark.core.students import build_student_system_prompt, needs_persona

    if role == "TUTOR":
        system_prompt = build_tutor_system_prompt(
            tutor_mode,
            prompt_version=prompt_version,
            student_context=student_context,
            reference_transcript=reference_transcript,
        )
        role_instruction = "Respond as the TUTOR. Give only your response, no labels or prefixes."
    else:
        mode = student_mode or "simple"
        persona = None
        if needs_persona(mode):
            if scenario is None or trait_client is None or trait_model is None:
                raise ValueError(
                    f"_build_role_prompt: student_mode={mode!r} requires scenario, "
                    "trait_client, and trait_model"
                )
            from benchmark.core.traits import get_or_generate_trait
            # Oracle student uses the default trait persona ('trait' resolves
            # to joined-3 in traits.py); trait/<dim>-<n>/joined-<n> modes pass
            # through their own spec verbatim.
            persona_mode = "trait" if mode == "oracle" else mode
            persona = get_or_generate_trait(scenario, persona_mode, trait_client, trait_model)
        system_prompt = build_student_system_prompt(
            mode,
            student_context=student_context,
            transcript_prefix=transcript_prefix,
            persona=persona,
            reference_transcript=reference_transcript,
        )
        role_instruction = "Respond as the STUDENT. Give only your response, no labels or prefixes."

    head = f"{system_prompt}\n\nHere is the conversation so far:\n\n{transcript_prefix}"
    tail = f"{extra}\n\n{role_instruction}"
    return head, tail


def _split_messages(text: str) -> list[str]:
    """Split LLM output into multiple messages on either delimiter.

    Recognizes [NEXT] (v1-v4) and [NEW_MESSAGE] (v5+). Either token splits
    the text into separate chat messages.
    """
    # Normalize both delimiters to one before splitting.
    normalized = text.replace(NEW_MESSAGE_DELIMITER, NEXT_DELIMITER)
    parts = normalized.split(NEXT_DELIMITER)
    messages = [p.strip() for p in parts]
    return [m for m in messages if m]


def _add_usage(total: dict, new: dict):
    """Accumulate token usage."""
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + new.get(key, 0)


def _append_turns(
    exchange: Exchange,
    messages: list[str],
    role: str,
    running_transcript: str,
    next_turn_num: int,
) -> tuple[str, int]:
    """Append one or more messages as consecutive turns for the same role."""
    for msg in messages:
        turn = {"turn_number": next_turn_num, "role": role, "text": msg}
        exchange.generated_turns.append(turn)
        running_transcript += f"\nTurn {next_turn_num}. {role}: {msg}"
        next_turn_num += 1
    return running_transcript, next_turn_num


def _append_turns_to_extra(
    exchange: Exchange,
    messages: list[str],
    role: str,
    extra: str,
    next_turn_num: int,
) -> tuple[str, int]:
    """Append messages as turns and grow the `extra` suffix.

    Used by the cache-aware exchange loop. transcript_prefix stays fixed;
    this only mutates the per-round growing portion.
    """
    for msg in messages:
        turn = {"turn_number": next_turn_num, "role": role, "text": msg}
        exchange.generated_turns.append(turn)
        extra += f"\nTurn {next_turn_num}. {role}: {msg}"
        next_turn_num += 1
    return extra, next_turn_num


def _build_reference_transcript(conversation: dict, cut_turn: int) -> str:
    """Format the post-cut real human turns from a full conversation.

    Returns a newline-joined string of `Turn N. ROLE: text` lines for every
    turn whose turn_number > cut_turn. Empty string if no post-cut turns.

    Used by oracle tutor mode -- the reference shown to the AI so it can
    mimic the real tutor's continuation.
    """
    lines = []
    for turn in conversation.get("turns", []):
        n = turn.get("turn_number")
        if n is None or n <= cut_turn:
            continue
        role = turn.get("role", "")
        text = turn.get("text", "")
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sync mode: one scenario at a time
# ---------------------------------------------------------------------------

def run_exchange(
    scenario: Scenario,
    tutor_client: ModelClient,
    student_client: ModelClient,
    max_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    prompt_version: str,
    images: list[str] | None = None,
    student_mode: str | None = None,
    trait_client: ModelClient | None = None,
    trait_model: str | None = None,
    tutor_mode: str | None = None,
    transcripts: dict[str, dict] | None = None,
    tutor_kwargs: dict | None = None,
    student_kwargs: dict | None = None,
) -> Exchange:
    """Sync mode multi-turn exchange.

    Both [END] and [PROBLEM_CHANGE] (or legacy [NEXT_PROBLEM]) terminate;
    recorded on Exchange.ended_via.
    Each tutor/student call passes scenario.transcript_prefix's head as
    cacheable_prefix so the static head hits the prompt cache on round 2+.

    When tutor_mode is set, transcripts must include scenario.conv_id; the
    post-cut reference is computed once and substituted into the tutor prompt.
    """
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    extra = ""
    next_turn_num = scenario.cut_turn + 1
    ended_via = ""

    # Compute reference once per scenario when oracle (or any tutor_mode) is on.
    reference_transcript = None
    # Both oracle tutor and oracle student need the post-cut real transcript.
    if tutor_mode == "oracle" or student_mode == "oracle":
        if not transcripts:
            raise ValueError(
                "run_exchange: oracle tutor or oracle student requires transcripts"
            )
        conv = transcripts.get(scenario.conv_id)
        if conv is None:
            raise ValueError(
                "run_exchange: oracle mode set but no transcript loaded for "
                f"conv_id={scenario.conv_id!r}"
            )
        reference_transcript = _build_reference_transcript(conv, scenario.cut_turn)

    # max_turns counts SPEAKING TURNS (one per LLM call, alternating T-S),
    # not generated message entries. [NEW_MESSAGE] splits don't count as new
    # speaking turns -- they're multiple messages within ONE speaking turn.
    speaking_turns = 0

    while speaking_turns < max_turns:
        # --- Tutor turn ---
        head, tail = _build_role_prompt(
            "TUTOR", scenario.transcript_prefix, extra, scenario.student_context,
            prompt_version,
            tutor_mode=tutor_mode,
            reference_transcript=reference_transcript,
        )
        response = tutor_client.generate(
            tail, json_mode=False, max_tokens=tutor_max_tokens,
            images=images, cacheable_prefix=head,
            **(tutor_kwargs or {}),
        )
        _add_usage(exchange.tutor_usage, response.usage)
        if response.latency_seconds is not None:
            exchange.tutor_latencies.append(response.latency_seconds)
        speaking_turns += 1

        text, ended, problem_change = _parse_tutor_tokens(response.text)
        messages = _split_messages(text)
        if not messages and not (ended or problem_change):
            messages = ["..."]
        if messages:
            extra, next_turn_num = _append_turns_to_extra(
                exchange, messages, "TUTOR", extra, next_turn_num,
            )

        if ended:
            ended_via = "END"
            break
        if problem_change:
            ended_via = "PROBLEM_CHANGE"
            break
        if speaking_turns >= max_turns:
            ended_via = "MAX_TURNS"
            break

        # --- Student turn ---
        head, tail = _build_role_prompt(
            "STUDENT", scenario.transcript_prefix, extra, scenario.student_context,
            prompt_version, student_mode=student_mode,
            scenario=scenario, trait_client=trait_client, trait_model=trait_model,
            reference_transcript=reference_transcript,
        )
        response = student_client.generate(
            tail, json_mode=False, max_tokens=student_max_tokens,
            images=images, cacheable_prefix=head,
            **(student_kwargs or {}),
        )
        _add_usage(exchange.student_usage, response.usage)
        if response.latency_seconds is not None:
            exchange.student_latencies.append(response.latency_seconds)
        speaking_turns += 1

        messages = _split_messages(response.text) or ["..."]
        extra, next_turn_num = _append_turns_to_extra(
            exchange, messages, "STUDENT", extra, next_turn_num,
        )

    if not ended_via:
        ended_via = "MAX_TURNS"

    exchange.completed = True
    exchange.ended_via = ended_via
    return exchange


# ---------------------------------------------------------------------------
# Batch mode: all scenarios in parallel, one batch per round
# ---------------------------------------------------------------------------

def run_exchanges_batch(
    scenarios: list[Scenario],
    tutor_client: ModelClient,
    student_client: ModelClient,
    max_turns: int,
    tutor_max_tokens: int,
    student_max_tokens: int,
    poll_interval: int,
    save_callback: callable = None,
    prompt_version: str = "v1",
    images_by_scenario: dict[str, list[str]] | None = None,
    student_mode: str | None = None,
    trait_client: ModelClient | None = None,
    trait_model: str | None = None,
    tutor_mode: str | None = None,
    transcripts: dict[str, dict] | None = None,
) -> dict[str, Exchange]:
    """Batch mode multi-turn exchanges.

    Per-scenario state tracks `extra` (growing suffix) separate from the
    static scenario.transcript_prefix; the head is passed as cacheable_prefix
    on every per-scenario batch entry.

    When tutor_mode is set, transcripts must include every scenario's conv_id;
    the post-cut reference is computed once per scenario and reused across
    rounds.
    """
    exchanges = {}
    extras: dict[str, str] = {}
    next_turns = {}
    ended_via: dict[str, str] = {}
    refs: dict[str, str] = {}

    # Both oracle tutor and oracle student need the post-cut reference; build
    # it once per scenario and share the dict.
    needs_reference = tutor_mode == "oracle" or student_mode == "oracle"
    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        extras[scenario.scenario_id] = ""
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1
        if needs_reference:
            if not transcripts:
                raise ValueError(
                    "run_exchanges_batch: oracle tutor or oracle student "
                    "requires transcripts"
                )
            conv = transcripts.get(scenario.conv_id)
            if conv is None:
                raise ValueError(
                    "run_exchanges_batch: oracle mode set but no transcript "
                    f"loaded for conv_id={scenario.conv_id!r}"
                )
            refs[scenario.scenario_id] = _build_reference_transcript(conv, scenario.cut_turn)

    scenario_map = {s.scenario_id: s for s in scenarios}
    active_ids = list(scenario_map.keys())

    # max_turns counts SPEAKING TURNS (LLM calls), alternating T-S-T-S-...
    # Each round = 1 tutor LLM call + 1 student LLM call = 2 speaking turns.
    # We may skip the student batch in the last round if max_turns is odd.
    for round_num in range(math.ceil(max_turns / 2)):
        if not active_ids:
            break

        # --- Tutor batch ---
        logger.info("Round %d - tutor batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "TUTOR", scenario.transcript_prefix, extras[sid], scenario.student_context,
                prompt_version,
                tutor_mode=tutor_mode,
                reference_transcript=refs.get(sid),
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            tutor_entries.append(
                build_batch_entry(sid, tail, json_mode=False, max_tokens=tutor_max_tokens,
                                  images=scenario_images, cacheable_prefix=head)
            )

        tutor_raw = run_batch(
            tutor_client, tutor_entries, json_mode=False,
            display_name=f"tutor_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = tutor_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("tutor failed for %s", sid[:50])
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.tutor_usage, result["usage"])

            text, ended, problem_change = _parse_tutor_tokens(result["text"])
            messages = _split_messages(text)
            if not messages and not (ended or problem_change):
                messages = ["..."]
            if messages:
                extras[sid], next_turns[sid] = _append_turns_to_extra(
                    exchange, messages, "TUTOR", extras[sid], next_turns[sid],
                )

            if ended:
                ended_via[sid] = "END"
                ended_this_round.append(sid)
                continue
            if problem_change:
                ended_via[sid] = "PROBLEM_CHANGE"
                ended_this_round.append(sid)
                continue
            # Speaking-turns budget check: after the tutor batch in round_num,
            # each active scenario has done (2*round_num + 1) speaking turns.
            if (2 * round_num + 1) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        # --- Student batch ---
        if not active_ids:
            if save_callback:
                for sid in scenario_map:
                    save_callback(sid, exchanges[sid])
            continue

        logger.info("Round %d - student batch (%d scenarios)",
                    round_num + 1, len(active_ids))
        student_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "STUDENT", scenario.transcript_prefix, extras[sid], scenario.student_context,
                prompt_version, student_mode=student_mode,
                scenario=scenario, trait_client=trait_client, trait_model=trait_model,
                reference_transcript=refs.get(sid),
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            student_entries.append(
                build_batch_entry(sid, tail, json_mode=False, max_tokens=student_max_tokens,
                                  images=scenario_images, cacheable_prefix=head)
            )

        student_raw = run_batch(
            student_client, student_entries, json_mode=False,
            display_name=f"student_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = student_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("student failed for %s", sid[:50])
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.student_usage, result["usage"])

            messages = _split_messages(result["text"]) or ["..."]
            extras[sid], next_turns[sid] = _append_turns_to_extra(
                exchange, messages, "STUDENT", extras[sid], next_turns[sid],
            )

            # After the student batch in round_num, each active scenario has
            # done (2*round_num + 2) speaking turns.
            if (2 * round_num + 2) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        if save_callback:
            for sid in scenario_map:
                save_callback(sid, exchanges[sid])

    for sid in scenario_map:
        exchanges[sid].completed = True
        exchanges[sid].ended_via = ended_via.get(sid, "MAX_TURNS")

    logger.info("Exchanges complete: %d scenarios", len(scenario_map))
    return exchanges
