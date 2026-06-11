"""
Structure Labeller -- Classify decomposed action and result facets.

Reads decomposed annotations (from decompose.py output, which adds
action_decomposed / result_decomposed facet lists) and classifies:
  - action_decomposed facets as scaffolding / rigor / neither / both, using
    prompts/annotator/action_labeller/classify_action.md
  - result_decomposed facets as a single mutually-exclusive student-outcome
    verdict -- pos (statements trend toward demonstrated
    understanding/realization) or neg (misconceptions/
    misunderstandings predominantly remain), using
    prompts/annotator/student_result_classifier/classify_student_result.md

Facets are joined into bullet-point lists (one facet per line) and
substituted as {action_list} and {student_list} respectively. Adds
action_label (str) and result_label (str) to each annotation, then
saves to structure_labels_{target}.json.

Usage:
    python -m annotator.core.structure --version v1
    python -m annotator.core.structure --version v1 --gold
    python -m annotator.core.structure --version v1 --split test
    python -m annotator.core.structure --version v1 --style balanced
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from common.logging_setup import setup_logging
from .client import (
    ModelClient, build_batch_entry, write_jsonl, run_batch, run_sync_entries,
)
from .config import get_phase_config, get_valid_styles, get_annotation_types
from .storage import (
    load_annotator_result, save_annotator_result,
    get_annotator_result_path,
)
from .utils import load_split_ids

logger = logging.getLogger(__name__)

ACTION_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "annotator" / "action_labeller" / "classify_action.md"
)
RESULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "annotator" / "student_result_classifier" / "classify_student_result.md"
)

# classify_action.md asks the model to judge scaffolding and rigor as two
# independent yes/no dimensions -- JSON {"scaffolding": "yes"|"no", "rigor":
# "yes"|"no"} -- rather than picking one mutually-exclusive verdict. The four
# (scaffolding, rigor) combinations map onto action_label's four substantive
# values, mirroring build_ground_truth._TUPLE_TO_AGG, which combines the same
# kind of independent yes/no judgments (there, human-annotated
# situation_label) into the same "scaffolding"/"rigor"/"neither"/"both" space.
_YES_NO_TO_ACTION_LABEL = {
    ("yes", "yes"): "both",
    ("yes", "no"): "scaffolding",
    ("no", "yes"): "rigor",
    ("no", "no"): "neither",
}

VALID_ACTION_LABELS = set(_YES_NO_TO_ACTION_LABEL.values())

# classify_student_result.md asks for a single mutually-exclusive bare letter:
# "A" = the statements trend toward demonstrated understanding/realization,
# "B" = misconceptions/misunderstandings predominantly remain. We map these to
# semantic names (mirroring action_label's "scaffolding"/"rigor"/... rather than
# raw letters) to avoid confusion with other lettered schemas in this codebase.
RESULT_LABEL_MAP = {"a": "pos", "b": "neg"}
VALID_RESULT_LABELS = set(RESULT_LABEL_MAP.values())

# Sentinel for annotations with no result facets to classify -- distinct from
# "unclear" (a parse-failure fallback meaning "the model answered but we
# couldn't read it"). "no_evidence" means "nothing was sent to the model in
# the first place", the same no-signal/parse-failure distinction situation
# labels draw between "no_mention" and "unclear".
DEFAULT_RESULT_LABEL = "no_evidence"


def _load_prompt(path: Path) -> str:
    logger.info("Loading structure labeller prompt: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_facet_list(facets: list[str]) -> str:
    """Format facets as a bullet list, one facet per line, for {action_list}/{student_list}."""
    return "\n".join(f"- {facet}" for facet in facets)


def _parse_action_label(text: str) -> tuple[str, bool]:
    """Parse the action label from model output text.

    classify_action.md asks for JSON {"scaffolding": "yes"|"no", "rigor":
    "yes"|"no"} -- two independent per-dimension judgments. Tries json.loads
    first (a list-wrapped response like [{...}] is unwrapped), then falls
    back to regex field extraction for responses with extra surrounding text
    -- mirroring situate._parse_situation_label's tolerance strategy for the
    same {"scaffolding": ..., "rigor": ...} JSON shape. The resulting
    (scaffolding, rigor) tuple is mapped to a single action_label via
    _YES_NO_TO_ACTION_LABEL.

    Returns (label, had_error). Falls back to "unclear" if either dimension
    is missing or isn't "yes"/"no" -- a half-parsed answer can't be mapped to
    a substantive verdict.
    """
    def _coerce(val) -> str | None:
        v = str(val).strip().lower()
        return v if v in ("yes", "no") else None

    scaffolding = rigor = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict):
            scaffolding = _coerce(parsed.get("scaffolding"))
            rigor = _coerce(parsed.get("rigor"))
    except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
        pass

    if scaffolding is None or rigor is None:
        m_scaf = re.search(r'["\']?scaffolding["\']?\s*:\s*["\']?(yes|no)["\']?', text, re.IGNORECASE)
        m_rigor = re.search(r'["\']?rigor["\']?\s*:\s*["\']?(yes|no)["\']?', text, re.IGNORECASE)
        if scaffolding is None and m_scaf:
            scaffolding = m_scaf.group(1).lower()
        if rigor is None and m_rigor:
            rigor = m_rigor.group(1).lower()

    if scaffolding is not None and rigor is not None:
        return _YES_NO_TO_ACTION_LABEL[(scaffolding, rigor)], False

    return "unclear", True


def _parse_result_label(text: str) -> tuple[str, bool]:
    """Parse the student-outcome label (a single bare letter) from model output text.

    Tries an exact match first (the documented bare-letter format), then
    falls back to the first line with markdown emphasis stripped -- this
    recovers verbose responses that state the verdict up front before
    explaining, e.g.
    "**A**\n\nThe statements indicate...". Only the first line/token is
    checked, so a letter mentioned later while reasoning through the answer
    ("...traces of understanding (A), but the answer is B") isn't mistaken
    for the verdict.

    Returns (label, had_error), where label is "pos" | "neg"
    | "unclear" (RESULT_LABEL_MAP). Falls back to "unclear" if no valid letter
    can be recovered.
    """
    cleaned = text.strip().lower().rstrip(".")
    if not cleaned:
        # Whitespace-only response: nothing to recover. (The caller's `not text`
        # guard treats whitespace as truthy, so it reaches us here.)
        return "unclear", True
    if cleaned in RESULT_LABEL_MAP:
        return RESULT_LABEL_MAP[cleaned], False

    first_line = re.sub(r"[*_`]", "", cleaned.splitlines()[0]).strip().rstrip(".")
    if first_line in RESULT_LABEL_MAP:
        return RESULT_LABEL_MAP[first_line], False
    m = re.match(r"(a|b)\b", first_line)
    if m:
        return RESULT_LABEL_MAP[m.group(1)], False

    return "unclear", True


def run_structure_label(version: str, model: str, mode: str, phase_cfg: dict,
                        gold: bool = False,
                        annotator_style: str | None = None,
                        annotations_data: dict | None = None,
                        profile: str | None = None,
                        target: str = "scaffolding",
                        split: str = "train") -> dict | None:
    """Run structure labelling pass. Returns the labeled annotations data dict.

    If annotations_data is provided, uses it directly instead of reading
    from disk. This allows in-memory chaining from run_decompose().

    Reads decomposed_{target}.json (output of decompose.py). For each
    annotation of the target type: classifies action_decomposed facets with
    classify_action.md (writing action_label), and result_decomposed facets
    with classify_student_result.md (writing result_label). Annotations with
    no facets in a field are not sent to the model -- they get the documented
    default ("neither" for actions; "no_evidence" for results, since no
    statements means there's nothing to classify).
    """
    in_memory = annotations_data is not None
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    gold_prefix = "decomposed_gold" if gold else "decomposed"
    input_filename = f"{gold_prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"

    if in_memory:
        data = annotations_data
    else:
        data = load_annotator_result(version, input_filename)
        if data is None:
            logger.error("%s not found for version %s. Run decompose first.", input_filename, version)
            return None
        logger.info("Loaded: %s", input_filename)

    if in_memory:
        # In-memory path (benchmark/chaining): data is already scoped by the
        # caller; skip split filtering so synthetic scenario IDs aren't dropped.
        results = dict(data["results"])
    else:
        split_ids = load_split_ids(split)
        results = {
            conv_id: conv_data
            for conv_id, conv_data in data["results"].items()
            if conv_id.rsplit("_", 1)[-1] in split_ids
        }

    action_template = _load_prompt(ACTION_PROMPT_PATH)
    result_template = _load_prompt(RESULT_PROMPT_PATH)

    action_entries = []
    result_entries = []
    locations_action = []
    locations_result = []
    skipped_action = []
    skipped_result = []

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data["annotations"]):
            if ann.get("annotation_type", target) != target:
                continue

            action_facets = ann.get("action_decomposed") or []
            result_facets = ann.get("result_decomposed") or []

            if not action_facets:
                skipped_action.append((conv_id, idx))
            else:
                key = f"action__{conv_id}__{idx}"
                prompt = action_template.replace("{action_list}", _format_facet_list(action_facets))
                action_entries.append(build_batch_entry(key, prompt, json_mode=True))
                locations_action.append((conv_id, idx))

            if not result_facets:
                skipped_result.append((conv_id, idx))
            else:
                key = f"result__{conv_id}__{idx}"
                prompt = result_template.replace("{student_list}", _format_facet_list(result_facets))
                result_entries.append(build_batch_entry(key, prompt, json_mode=False))
                locations_result.append((conv_id, idx))

    entries = action_entries + result_entries
    logger.info(
        "Action entries: %d (%d skipped, no facets) | Result entries: %d (%d skipped, no facets)",
        len(action_entries), len(skipped_action), len(result_entries), len(skipped_result),
    )
    logger.info("Model: %s | Mode: %s", model, mode)

    for conv_id, idx in skipped_action:
        results[conv_id]["annotations"][idx]["action_label"] = "neither"
    for conv_id, idx in skipped_result:
        results[conv_id]["annotations"][idx]["result_label"] = DEFAULT_RESULT_LABEL

    client = ModelClient(model)
    if not in_memory:
        output_dir = get_annotator_result_path(version)
        jsonl_path = str(output_dir / f"structure_label_requests{profile_suffix}.jsonl")
        write_jsonl(entries, jsonl_path)

    # Action entries are JSON ({"scaffolding": ..., "rigor": ...}); result
    # entries are bare-token ("A"/"B"). json_mode is set per-entry in
    # build_batch_entry above -- _extract_entry reads it back from each
    # entry's generation_config, so the json_mode passed here is unused for
    # mixed-mode batches like this one.
    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, json_mode=False, display_name="structure_label",
                        poll_interval=poll_interval,
                        thinking=phase_cfg.get("thinking", False),
                        thinking_budget=phase_cfg.get("thinking_budget", 0),
                        reasoning_effort=phase_cfg.get("reasoning_effort", ""))
    else:
        logger.info("Running %d entries in sync mode...", len(entries))
        raw = run_sync_entries(client, entries, json_mode=False)

    _ZERO_ACTION_COUNTS = lambda: {"scaffolding": 0, "rigor": 0, "neither": 0, "both": 0, "unclear": 0}
    action_counts = _ZERO_ACTION_COUNTS()
    action_counts["neither"] += len(skipped_action)

    result_counts = {"pos": 0, "neg": 0, "no_evidence": 0, "unclear": 0}
    result_counts["no_evidence"] += len(skipped_result)

    errors = 0
    total_input = 0
    total_output = 0

    for conv_id, idx in locations_action:
        key = f"action__{conv_id}__{idx}"
        entry = raw.get(key, {})

        if "error" in entry or not entry.get("text"):
            label = "unclear"
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            label, had_error = _parse_action_label(entry["text"])
            if had_error:
                logger.warning("Could not parse action label for %s: %r", key, entry["text"][:200])
                errors += 1

        results[conv_id]["annotations"][idx]["action_label"] = label
        action_counts[label] += 1

    for conv_id, idx in locations_result:
        key = f"result__{conv_id}__{idx}"
        entry = raw.get(key, {})

        if "error" in entry or not entry.get("text"):
            result_label = "unclear"
            errors += 1
        else:
            usage = entry.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            result_label, had_error = _parse_result_label(entry["text"])
            if had_error:
                logger.warning("Could not parse result label for %s: %r", key, entry["text"][:200])
                errors += 1

        results[conv_id]["annotations"][idx]["result_label"] = result_label
        result_counts[result_label] += 1

    output = {
        **data,
        "results": results,
        "structure_labeled": True,
        "structure_label_stats": {
            "action": action_counts,
            "result": result_counts,
        },
        "token_summary": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "errors": errors,
        },
    }

    output_prefix = "structure_labels_gold" if gold else "structure_labels"
    output_filename = f"{output_prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
    save_annotator_result(version, output_filename, output)
    n_classified = len(locations_action) + len(skipped_action)
    logger.info("Saved: %s | %d annotations structure-labeled", output_filename, n_classified)

    logger.info("  Action labels:")
    for label, count in action_counts.items():
        logger.info("    %-12s %d", label + ":", count)
    logger.info("  Result labels:")
    for label, count in result_counts.items():
        logger.info("    %-14s %d", label + ":", count)
    logger.info("  Tokens: %s", f"{total_input + total_output:,}")
    if errors:
        logger.warning("  Errors: %d", errors)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Structure Labeller: classify decomposed action and result facets"
    )
    parser.add_argument("--version", default=None,
                        help="Version to label (reads decomposed_{target}.json). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--profile", default=None,
                        help="Config profile to use (overrides config.yaml default)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode (overrides config)")
    parser.add_argument("--gold", action="store_true",
                        help="Label gold truth decomposition (decomposed_gold_{target}.json)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Match the decomposed_{style}_{target}.json file from decompose --style")
    parser.add_argument("--target", choices=get_annotation_types(), default="scaffolding",
                        help="Annotation type to label (default: scaffolding)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    args = parser.parse_args()

    setup_logging()

    from .config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.annotator_style,
        cli_prompt_version=None,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]

    setup_logging(version=version)

    phase_cfg = get_phase_config("label", profile)
    model = args.model or phase_cfg["model"]
    mode = args.mode or phase_cfg.get("mode", "batch")

    result = run_structure_label(version=version, model=model, mode=mode,
                                 phase_cfg=phase_cfg, gold=args.gold,
                                 annotator_style=style, profile=profile,
                                 target=args.target, split=args.split)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
