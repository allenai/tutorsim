"""Scenario extraction: cut real transcripts at detected key moments."""

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

from annotator.core.utils import (
    load_transcripts, format_transcript, EXAMPLE_CONV_IDS,
)


@dataclass
class Scenario:
    scenario_id: str              # "{conv_id}__{ann_type}_{idx}"
    conv_id: str
    cut_turn: int                 # last turn included in the prefix (suggested_cut_turn)
    transcript_prefix: str        # formatted transcript up to cut_turn
    student_context: str          # grade, subject from conversation metadata
    last_student_message: str     # student's last utterance before cut
    mode: str                     # "detected" | "random"
    detection: dict | None        # the detection dict (turn_start, turn_end, annotation_type, etc.)

    def to_dict(self):
        d = asdict(self)
        # Don't serialize the full transcript prefix to save space in JSON
        d["transcript_prefix_length"] = len(self.transcript_prefix)
        del d["transcript_prefix"]
        return d


def _format_prefix(conversation: dict, cut_turn: int) -> str:
    """Format transcript turns up to and including cut_turn."""
    lines = []
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        n = turn["turn_number"]
        role = turn["role"]
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)


def _get_student_context(conversation: dict) -> str:
    """Extract student context from conversation metadata."""
    context = conversation.get("context", "")
    if context:
        return context
    # Fallback: construct from available fields
    parts = []
    if conversation.get("platform"):
        parts.append(f"Platform: {conversation['platform']}")
    return "; ".join(parts) if parts else "K-12 tutoring session"


def _last_student_msg(conversation: dict, cut_turn: int) -> str:
    """Find the last STUDENT message at or before cut_turn."""
    last = ""
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        if turn["role"] == "STUDENT":
            last = turn["text"]
    return last


def extract_detected_scenarios(
    transcripts: dict[str, dict],
    detections_by_conv: dict[str, dict],
) -> list[Scenario]:
    """Extract scenarios from synthetic detection results.

    Each detection becomes a scenario. The cut point is the detection's
    suggested_cut_turn (falling back to turn_start - 1). The synthetic
    tutor continues from there.
    """
    scenarios = []

    for conv_id, conv_data in detections_by_conv.items():
        if conv_id in EXAMPLE_CONV_IDS:
            continue
        if conv_id not in transcripts:
            continue

        conversation = transcripts[conv_id]
        detections = conv_data.get("detections", [])

        for idx, det in enumerate(detections):
            ann_type = det.get("annotation_type", "scaffolding")
            cut_turn = det.get("suggested_cut_turn", max(1, det.get("turn_start", 1) - 1))

            prefix = _format_prefix(conversation, cut_turn)
            if not prefix:
                continue

            scenarios.append(Scenario(
                scenario_id=f"{conv_id}__{ann_type}_{idx}",
                conv_id=conv_id,
                cut_turn=cut_turn,
                transcript_prefix=prefix,
                student_context=_get_student_context(conversation),
                last_student_message=_last_student_msg(conversation, cut_turn),
                mode="detected",
                detection=det,
            ))

    return scenarios


def extract_random_scenarios(
    transcripts: dict[str, dict],
    count: int = 50,
    seed: int = 42,
    min_turn: int = 10,
) -> list[Scenario]:
    """Pick random cut points where the next speaker should be TUTOR.

    Selects points where the last turn before the cut is a STUDENT turn,
    so the tutor model must respond to what the student just said.
    """
    rng = random.Random(seed)
    candidates = []

    for conv_id, conversation in transcripts.items():
        if conv_id in EXAMPLE_CONV_IDS:
            continue
        turns = conversation.get("turns", [])
        for i, turn in enumerate(turns):
            if turn["turn_number"] < min_turn:
                continue
            if turn["role"] != "STUDENT":
                continue
            # Check next turn exists and is TUTOR
            if i + 1 < len(turns) and turns[i + 1]["role"] == "TUTOR":
                candidates.append((conv_id, turn["turn_number"]))

    if len(candidates) > count:
        candidates = rng.sample(candidates, count)

    scenarios = []
    for conv_id, cut_turn in candidates:
        conversation = transcripts[conv_id]
        prefix = _format_prefix(conversation, cut_turn)
        if not prefix:
            continue

        scenarios.append(Scenario(
            scenario_id=f"{conv_id}__rnd_{cut_turn}",
            conv_id=conv_id,
            cut_turn=cut_turn,
            transcript_prefix=prefix,
            student_context=_get_student_context(conversation),
            last_student_message=_last_student_msg(conversation, cut_turn),
            mode="random",
            detection=None,
        ))

    return scenarios


def _sample_per_conversation(
    scenarios: list[Scenario],
    max_per_conv: int,
    seed: int = 42,
) -> list[Scenario]:
    """Randomly sample up to max_per_conv scenarios per conversation."""
    rng = random.Random(seed)
    by_conv: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_conv.setdefault(s.conv_id, []).append(s)

    sampled = []
    for conv_id, conv_scenarios in by_conv.items():
        if len(conv_scenarios) <= max_per_conv:
            sampled.extend(conv_scenarios)
        else:
            sampled.extend(rng.sample(conv_scenarios, max_per_conv))

    return sampled


def load_scenarios(config: dict, detections_by_conv: dict | None = None) -> list[Scenario]:
    """Load scenarios based on config settings.

    Args:
        config: Scenario config from config.yaml.
        detections_by_conv: Detection results from run_detect(). If provided,
            scenarios are extracted from detections. If None and mode is
            'detected', raises an error.
    """
    transcripts = load_transcripts()
    logger.info("Loaded %d transcripts", len(transcripts))
    if not transcripts:
        raise FileNotFoundError(
            "No transcripts found. Ensure data/transcripts/ contains JSON files, "
            "or configure transcript paths in config.yaml under storage.paths.transcripts."
        )

    mode = config.get("mode", "detected")
    max_scenarios = config.get("max_scenarios", 0)
    max_per_conv = config.get("max_per_conv", 0)

    scenarios = []

    if mode in ("detected", "both"):
        if detections_by_conv is None:
            raise ValueError(
                "Detection results required for mode='detected'. "
                "Run detection first or set mode='random'."
            )
        det_scenarios = extract_detected_scenarios(transcripts, detections_by_conv)
        logger.info("Detected scenarios: %d", len(det_scenarios))
        scenarios.extend(det_scenarios)

    if mode in ("random", "both"):
        count = config.get("random_count", 50)
        seed = config.get("random_seed", 42)
        min_turn = config.get("min_turn", 10)
        rnd_scenarios = extract_random_scenarios(transcripts, count, seed, min_turn)
        logger.info("Random scenarios: %d", len(rnd_scenarios))
        scenarios.extend(rnd_scenarios)

    if max_per_conv > 0:
        before = len(scenarios)
        scenarios = _sample_per_conversation(
            scenarios, max_per_conv, seed=config.get("random_seed", 42),
        )
        logger.info("Sampled %d/conv: %d -> %d scenarios", max_per_conv, before, len(scenarios))

    if max_scenarios > 0:
        scenarios = scenarios[:max_scenarios]

    logger.info("Total scenarios: %d", len(scenarios))
    return scenarios
