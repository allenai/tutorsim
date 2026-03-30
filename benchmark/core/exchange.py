"""Multi-turn exchange orchestration: tutor and student alternate.

Supports two execution modes:
- sync: run_exchange() for one scenario at a time
- batch: run_exchanges_batch() processes all scenarios in parallel,
  batching each round of tutor/student calls across scenarios

Supports multi-message turns (split by [NEXT] delimiter) to match
real transcript patterns where the same speaker sends consecutive messages.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

from annotator.core.client import (
    ModelClient, build_batch_entry, run_batch, run_sync_entries,
)
from .scenarios import Scenario

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "benchmark"

NEXT_DELIMITER = "[NEXT]"


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

    def to_dict(self):
        return asdict(self)


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _build_role_prompt(
    role: str, transcript_so_far: str, student_context: str,
) -> str:
    """Build a prompt for either tutor or student."""
    if role == "TUTOR":
        system_prompt = _load_prompt("tutor_system.txt")
        role_instruction = "Respond as the TUTOR. Give only your response, no labels or prefixes."
    else:
        system_prompt = _load_prompt("student_system.txt")
        role_instruction = "Respond as the STUDENT. Give only your response, no labels or prefixes."

    system_prompt = system_prompt.replace("{student_context}", student_context)

    return f"""{system_prompt}

Here is the conversation so far:

{transcript_so_far}

{role_instruction}"""


def _split_messages(text: str) -> list[str]:
    """Split LLM output into multiple messages on [NEXT] delimiter."""
    parts = text.split(NEXT_DELIMITER)
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


# ---------------------------------------------------------------------------
# Sync mode: one scenario at a time
# ---------------------------------------------------------------------------

def run_exchange(
    scenario: Scenario,
    tutor_client: ModelClient,
    student_client: ModelClient,
    num_turns: int = 4,
    tutor_max_tokens: int = 65536,
    student_max_tokens: int = 65536,
) -> Exchange:
    """Run a multi-turn exchange for a single scenario (sync mode)."""
    exchange = Exchange(
        scenario_id=scenario.scenario_id,
        tutor_model=tutor_client.model,
    )

    running_transcript = scenario.transcript_prefix
    next_turn_num = scenario.cut_turn + 1

    for i in range(num_turns):
        # Tutor turn(s)
        prompt = _build_role_prompt("TUTOR", running_transcript, scenario.student_context)
        response = tutor_client.generate(prompt, json_mode=False, max_tokens=tutor_max_tokens)
        _add_usage(exchange.tutor_usage, response.usage)

        messages = _split_messages(response.text) or ["..."]
        running_transcript, next_turn_num = _append_turns(
            exchange, messages, "TUTOR", running_transcript, next_turn_num,
        )

        # Student turn(s) — skip on last round
        if i < num_turns - 1:
            prompt = _build_role_prompt("STUDENT", running_transcript, scenario.student_context)
            response = student_client.generate(prompt, json_mode=False, max_tokens=student_max_tokens)
            _add_usage(exchange.student_usage, response.usage)

            messages = _split_messages(response.text) or ["..."]
            running_transcript, next_turn_num = _append_turns(
                exchange, messages, "STUDENT", running_transcript, next_turn_num,
            )

    return exchange


# ---------------------------------------------------------------------------
# Batch mode: all scenarios in parallel, one batch per round
# ---------------------------------------------------------------------------

def run_exchanges_batch(
    scenarios: list[Scenario],
    tutor_client: ModelClient,
    student_client: ModelClient,
    num_turns: int = 4,
    tutor_max_tokens: int = 65536,
    student_max_tokens: int = 65536,
    poll_interval: int = 30,
    save_callback: callable = None,
) -> dict[str, Exchange]:
    """Run multi-turn exchanges for all scenarios using batch API.

    For each round, submits one batch of tutor prompts (all scenarios),
    waits for results, then one batch of student prompts. Repeats for
    num_turns rounds.

    Args:
        save_callback: Optional function(scenario_id, exchange) called after
            each round to save progress incrementally. If provided, exchanges
            are saved after every tutor+student pair completes.

    Returns: {scenario_id: Exchange}
    """
    # Initialize state per scenario
    exchanges = {}
    transcripts = {}
    next_turns = {}

    for scenario in scenarios:
        exchanges[scenario.scenario_id] = Exchange(
            scenario_id=scenario.scenario_id,
            tutor_model=tutor_client.model,
        )
        transcripts[scenario.scenario_id] = scenario.transcript_prefix
        next_turns[scenario.scenario_id] = scenario.cut_turn + 1

    scenario_map = {s.scenario_id: s for s in scenarios}
    active_ids = list(scenario_map.keys())

    for round_num in range(num_turns):
        # --- Tutor batch ---
        print(f"\n    Round {round_num + 1}/{num_turns} - Tutor batch ({len(active_ids)} scenarios)...")
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            prompt = _build_role_prompt("TUTOR", transcripts[sid], scenario.student_context)
            tutor_entries.append(
                build_batch_entry(sid, prompt, json_mode=False, max_tokens=tutor_max_tokens)
            )

        tutor_raw = run_batch(
            tutor_client, tutor_entries, json_mode=False,
            display_name=f"tutor_round_{round_num + 1}",
            poll_interval=poll_interval,
        )

        # Process tutor results
        failed = []
        for sid in active_ids:
            result = tutor_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                print(f"      WARN: tutor failed for {sid[:50]}")
                failed.append(sid)
                continue

            exchange = exchanges[sid]
            if result.get("usage"):
                _add_usage(exchange.tutor_usage, result["usage"])

            messages = _split_messages(result["text"]) or ["..."]
            transcripts[sid], next_turns[sid] = _append_turns(
                exchange, messages, "TUTOR", transcripts[sid], next_turns[sid],
            )

        # Remove failed scenarios
        for sid in failed:
            active_ids.remove(sid)

        # --- Student batch (skip on last round) ---
        if round_num < num_turns - 1 and active_ids:
            print(f"    Round {round_num + 1}/{num_turns} - Student batch ({len(active_ids)} scenarios)...")
            student_entries = []
            for sid in active_ids:
                scenario = scenario_map[sid]
                prompt = _build_role_prompt("STUDENT", transcripts[sid], scenario.student_context)
                student_entries.append(
                    build_batch_entry(sid, prompt, json_mode=False, max_tokens=student_max_tokens)
                )

            student_raw = run_batch(
                student_client, student_entries, json_mode=False,
                display_name=f"student_round_{round_num + 1}",
                poll_interval=poll_interval,
            )

            # Process student results
            failed = []
            for sid in active_ids:
                result = student_raw.get(sid, {})
                if "error" in result or not result.get("text"):
                    print(f"      WARN: student failed for {sid[:50]}")
                    failed.append(sid)
                    continue

                exchange = exchanges[sid]
                if result.get("usage"):
                    _add_usage(exchange.student_usage, result["usage"])

                messages = _split_messages(result["text"]) or ["..."]
                transcripts[sid], next_turns[sid] = _append_turns(
                    exchange, messages, "STUDENT", transcripts[sid], next_turns[sid],
                )

            for sid in failed:
                active_ids.remove(sid)

        # Save progress after each round
        if save_callback:
            for sid in active_ids:
                save_callback(sid, exchanges[sid])

    print(f"\n    Exchanges complete: {len(active_ids)}/{len(scenarios)} succeeded")
    return exchanges
