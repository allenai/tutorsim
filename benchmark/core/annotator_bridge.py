"""Bridge to the synthetic annotator pipeline.

Constructs in-memory transcripts and detections from benchmark exchanges,
then calls the existing annotation and labeling functions in bulk mode.
"""

from annotator.core.annotate import (
    build_analysis_entries, parse_and_merge,
)
from annotator.core.client import (
    ModelClient, run_sync_entries, run_batch,
)
from annotator.core.decompose import run_decompose
from annotator.core.label import run_label
from annotator.core.structure import run_structure_label
from annotator.core.config import get_phase_config
from annotator.core.screenshots import load_anchored_screenshots

from .scenarios import Scenario
from .exchange import Exchange


def build_synthetic_conversation(scenario: Scenario, exchange: Exchange) -> dict:
    """Build a full conversation dict with generated turns appended."""
    turns = []
    for line in scenario.transcript_prefix.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ". TUTOR: " in line:
            parts = line.split(". TUTOR: ", 1)
            turn_num = int(parts[0].replace("Turn ", ""))
            turns.append({
                "turn_number": turn_num, "role": "TUTOR",
                "text": parts[1], "type": "DIALOGUE", "timestamp": "",
            })
        elif ". STUDENT: " in line:
            parts = line.split(". STUDENT: ", 1)
            turn_num = int(parts[0].replace("Turn ", ""))
            turns.append({
                "turn_number": turn_num, "role": "STUDENT",
                "text": parts[1], "type": "DIALOGUE", "timestamp": "",
            })

    for gen_turn in exchange.generated_turns:
        turns.append({
            "turn_number": gen_turn["turn_number"], "role": gen_turn["role"],
            "text": gen_turn["text"], "type": "DIALOGUE", "timestamp": "",
        })

    return {
        "conversation_id": scenario.conv_id,
        "turns": turns,
        "context": scenario.student_context,
        "num_turns": len(turns),
    }


def build_synthetic_detections(scenario: Scenario, exchange: Exchange) -> dict:
    """Build detections spanning the AI tutor's generated turns only.

    The moment window (turn_start..turn_end) is what the Pass 2 prompt tells
    the annotator to summarize ("ONLY tutor actions between START and END").
    We scope it to the AI replay so the annotator scores the AI tutor, not
    the human tutor's pre-cut actions. Pre-cut context still reaches the
    annotator via the surrounding excerpt window (context_window turns).
    """
    if not exchange.generated_turns:
        return {}

    first_gen = exchange.generated_turns[0]["turn_number"]
    last_gen = exchange.generated_turns[-1]["turn_number"]

    if scenario.detection:
        ann_type = scenario.detection.get("annotation_type", "scaffolding")
        description = (
            f"AI tutor continuation from cut at turn {scenario.cut_turn}: "
            f"{scenario.detection.get('situation', scenario.mode)}"
        )
        # Use the detection's annotation type (scaffolding or rapport).
        # Propagate situation_label_agg so the new (v11/v12) annotator prompts
        # can render the correct teacher-consensus suggestion.
        detections = [
            {
                "turn_start": first_gen, "turn_end": last_gen,
                "annotation_type": ann_type,
                "situation": description,
                "situation_label_agg": scenario.detection.get("situation_label_agg"),
            },
        ]
    else:
        description = f"AI tutor continuation in a {scenario.mode} scenario"
        # Random scenarios: annotate for both types
        detections = [
            {
                "turn_start": first_gen, "turn_end": last_gen,
                "annotation_type": "scaffolding",
                "situation": description,
            },
            {
                "turn_start": first_gen, "turn_end": last_gen,
                "annotation_type": "rapport",
                "situation": description,
            },
        ]

    return {
        scenario.conv_id: {
            "detections": detections,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }


# ---------------------------------------------------------------------------
# Bulk mode (batch API across many scenarios)
# ---------------------------------------------------------------------------

def prepare_bulk_entries(
    scenarios: list[Scenario],
    exchanges: dict[str, Exchange],
    annotator_style: str,
    prompt_version: str,
    context_window: int = 20,
    with_screenshots: bool = False,
) -> tuple[list[dict], dict, dict]:
    """Prepare annotation entries for many scenarios at once.

    Since different scenarios may share the same conv_id, we use
    scenario_id as a namespace prefix in batch keys to keep them unique.

    When with_screenshots=True, loads anchored screenshots for each scenario
    using the *original* scenario.conv_id (not the remapped scenario_id) and
    passes them to build_analysis_entries via screenshots_by_conv keyed on
    scenario_id so the function's iteration matches.

    Returns:
        entries: List of batch entries ready for run_batch/run_sync_entries
        all_detections: {scenario_id: detections_dict} for parse_and_merge
        all_conversations: {scenario_id: conversations_map} for reference
    """
    all_entries = []
    all_detections = {}
    all_conversations = {}
    screenshots_by_conv: dict[str, list[dict]] | None = (
        {} if with_screenshots else None
    )

    for scenario in scenarios:
        exchange = exchanges.get(scenario.scenario_id)
        if not exchange:
            continue

        synth_conv = build_synthetic_conversation(scenario, exchange)
        detections = build_synthetic_detections(scenario, exchange)
        if not detections:
            continue

        # Use scenario_id as conv_id namespace to avoid collisions
        # Remap conv_id -> scenario_id in both conversations_map and detections
        remapped_conv = dict(synth_conv)
        remapped_conv["conversation_id"] = scenario.scenario_id
        remapped_conversations = {scenario.scenario_id: remapped_conv}

        remapped_detections = {
            scenario.scenario_id: detections[scenario.conv_id]
        }

        if with_screenshots:
            scenario_screenshots = load_anchored_screenshots(
                scenario.conv_id, synth_conv["turns"],
            )
            screenshots_by_conv[scenario.scenario_id] = scenario_screenshots

        entries = build_analysis_entries(
            remapped_detections, remapped_conversations,
            context_window, prompt_version,
            annotator_style=annotator_style,
            with_screenshots=with_screenshots,
            screenshots_by_conv=screenshots_by_conv,
        )

        all_entries.extend(entries)
        all_detections[scenario.scenario_id] = remapped_detections
        all_conversations[scenario.scenario_id] = remapped_conversations

    return all_entries, all_detections, all_conversations


def execute_and_parse_bulk(
    entries: list[dict],
    all_detections: dict,
    annotator_profile: str,
    mode: str = "batch",
    existing_batch_id: str | None = None,
    on_batch_created: callable = None,
) -> dict[str, dict]:
    """Execute bulk entries and parse results back to per-scenario annotations.

    When mode == "batch", existing_batch_id resumes polling on a previously
    submitted provider batch (skip submission). on_batch_created fires once
    immediately after submission with the new batch id, so the orchestrator
    can persist a sidecar before the poll loop starts.

    Returns: {scenario_id: parsed_results_dict}
    """
    if not entries:
        return {}

    annotate_cfg = get_phase_config("annotate", annotator_profile)
    client = ModelClient(annotate_cfg["model"])

    if mode == "batch":
        raw = run_batch(
            client, entries, display_name="benchmark_annotate",
            poll_interval=annotate_cfg["poll_interval"],
            existing_batch_id=existing_batch_id,
            on_batch_created=on_batch_created,
        )
    else:
        raw = run_sync_entries(client, entries)

    # Merge all detections into one dict and parse once (avoids repeated error logs)
    merged_detections = {}
    for scenario_id, remapped_detections in all_detections.items():
        merged_detections.update(remapped_detections)

    all_results = parse_and_merge(raw, merged_detections)

    # Split back to per-scenario
    per_scenario = {}
    for scenario_id in all_detections:
        if scenario_id in all_results:
            per_scenario[scenario_id] = {scenario_id: all_results[scenario_id]}
        else:
            per_scenario[scenario_id] = {}

    return per_scenario


def label_bulk(
    per_scenario_results: dict[str, dict],
    annotator_style: str,
    annotator_profile: str,
    annotator_model: str,
    mode: str = "batch",
) -> dict[str, dict]:
    """Label all scenario annotations in one batch.

    Merges all results into a single annotations_data, labels in one pass,
    then splits back to per-scenario.

    Returns: {scenario_id: labeled_annotations_data}
    """
    if not per_scenario_results:
        return {}

    # Merge all results into one dict for a single run_label call
    merged_results = {}
    for scenario_id, results in per_scenario_results.items():
        # results is {remapped_conv_id: {annotations: [...]}}
        merged_results.update(results)

    annotations_data = {
        "version": "benchmark",
        "model": annotator_model,
        "source": "benchmark_exchange",
        "annotator_style": annotator_style,
        "results": merged_results,
    }

    label_cfg = get_phase_config("label", annotator_profile)
    labeled = run_label(
        version="benchmark", model=label_cfg["model"],
        mode=mode, phase_cfg=label_cfg, annotations_data=annotations_data,
    )

    if not labeled:
        return {}

    # Split back to per-scenario
    labeled_results = labeled.get("results", {})
    per_scenario_labeled = {}
    for scenario_id in per_scenario_results:
        if scenario_id in labeled_results:
            scenario_labeled = dict(labeled)
            scenario_labeled["results"] = {scenario_id: labeled_results[scenario_id]}
            per_scenario_labeled[scenario_id] = scenario_labeled

    return per_scenario_labeled


def decompose_bulk(
    per_scenario_results: dict[str, dict],
    annotator_profile: str,
    mode: str = "batch",
) -> dict[str, dict]:
    """Run decompose on all scenarios' annotations in one in-memory pass.

    Input shape: {scenario_id: {scenario_id: {annotations: [...]}}}
    Returns same shape with action_decomposed / result_decomposed populated.

    The split filter inside run_decompose is bypassed for in-memory calls
    (decompose.py skips filtering when annotations_data is provided directly).
    """
    if not per_scenario_results:
        return {}

    merged_results = {}
    for sid, results in per_scenario_results.items():
        merged_results.update(results)

    annotations_data = {
        "version": "benchmark",
        "source": "benchmark_exchange",
        "results": merged_results,
    }

    phase_cfg = get_phase_config("annotate", annotator_profile)
    enriched = run_decompose(
        version="benchmark",
        model=phase_cfg["model"],
        mode=mode,
        phase_cfg=phase_cfg,
        annotations_data=annotations_data,
        profile=annotator_profile,
    )
    if not enriched:
        return per_scenario_results

    out: dict[str, dict] = {}
    enriched_results = enriched.get("results", {})
    for sid in per_scenario_results:
        if sid in enriched_results:
            out[sid] = {sid: enriched_results[sid]}
        else:
            out[sid] = per_scenario_results[sid]
    return out


def structure_bulk(
    per_scenario_results: dict[str, dict],
    annotator_profile: str,
    mode: str = "batch",
) -> dict[str, dict]:
    """Run structure labelling on all scenarios in one in-memory pass.

    Input requires action_decomposed / result_decomposed populated (from
    decompose_bulk). Returns same shape with action_label / result_label added.

    The split filter inside run_structure_label is bypassed for in-memory
    calls (structure.py skips filtering when annotations_data is provided).
    """
    if not per_scenario_results:
        return {}

    merged_results = {}
    for sid, results in per_scenario_results.items():
        merged_results.update(results)

    annotations_data = {
        "version": "benchmark",
        "source": "benchmark_exchange",
        "results": merged_results,
    }

    phase_cfg = get_phase_config("annotate", annotator_profile)
    enriched = run_structure_label(
        version="benchmark",
        model=phase_cfg["model"],
        mode=mode,
        phase_cfg=phase_cfg,
        annotations_data=annotations_data,
        profile=annotator_profile,
        target="scaffolding",
    )
    if not enriched:
        return per_scenario_results

    out: dict[str, dict] = {}
    enriched_results = enriched.get("results", {})
    for sid in per_scenario_results:
        if sid in enriched_results:
            out[sid] = {sid: enriched_results[sid]}
        else:
            out[sid] = per_scenario_results[sid]
    return out
