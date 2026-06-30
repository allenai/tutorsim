"""Scenario dataclass and build_scenarios for tutorsim."""

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCAFF_AGGS = {"scaffolding", "rigor"}
DATASET_SCHEMA_VERSION = 1

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_CONV_ID_PREFIX_RE = re.compile(r"^\d{4}-t\d+_\d{4}-s\d+_")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested scenario set has not been installed locally."""


def scenario_set_dir(set_name: str, root: str = "scenarios") -> Path:
    """Return the directory for a named scenario set."""
    return Path(root) / set_name


def scenarios_jsonl_path(set_name: str, root: str = "scenarios") -> Path:
    """Return the scenarios.jsonl path for a named scenario set."""
    return scenario_set_dir(set_name, root) / "scenarios.jsonl"


def manifest_path(set_name: str, root: str = "scenarios") -> Path:
    """Return the manifest.json path for a named scenario set."""
    return scenario_set_dir(set_name, root) / "manifest.json"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_dataset_message(set_name: str, path: Path) -> str:
    return (
        f"Dataset '{set_name}' is not installed. Expected scenarios at {path}. "
        "Create it with `tutorsim dataset build ...` or install/download the "
        "published dataset release when it is available. The official "
        "balanced_520 release is intentionally external to git."
    )


@dataclass
class Scenario:
    """A declarative, self-contained scenario for tutor simulation.

    context: list of prefix turns, each {"turn_number": int, "role": str, "text": str}.
    turn_number is the REAL (non-sequential) turn number from the source transcript.
    role is lowercase ("tutor" or "student") as a structured value.
    """

    id: str
    context: list[dict]
    dimension: str
    student: dict
    rubric: dict
    provenance: dict

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict, preserving all fields (no dropping)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        """Reconstruct a Scenario from a dict."""
        return cls(
            id=d["id"],
            context=d["context"],
            dimension=d["dimension"],
            student=d["student"],
            rubric=d["rubric"],
            provenance=d["provenance"],
        )


def load_scenarios(set_name: str, root: str = "scenarios") -> list[Scenario]:
    """Load scenarios from a frozen scenarios.jsonl file.

    Args:
        set_name: Name of the scenario set (subdirectory).
        root: Root directory path (default "scenarios").

    Returns:
        List of Scenario objects in file order.

    Raises:
        FileNotFoundError: If the scenarios.jsonl file does not exist.
    """
    path = scenarios_jsonl_path(set_name, root)
    if not path.exists():
        raise DatasetNotFoundError(_missing_dataset_message(set_name, path))

    scenarios = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)
                scenarios.append(Scenario.from_dict(data))

    return scenarios


def load_manifest(set_name: str, root: str = "scenarios") -> dict | None:
    """Load a dataset manifest if present."""
    path = manifest_path(set_name, root)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_dataset(set_name: str, root: str = "scenarios") -> dict:
    """Validate the installed dataset files and manifest.

    Returns a small validation report. Raises ValueError/FileNotFoundError for
    missing files, count mismatches, hash mismatches, or malformed manifests.
    """
    jsonl_path = scenarios_jsonl_path(set_name, root)
    if not jsonl_path.exists():
        raise DatasetNotFoundError(_missing_dataset_message(set_name, jsonl_path))

    scenarios = load_scenarios(set_name, root=root)
    manifest = load_manifest(set_name, root=root)
    if manifest is None:
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path(set_name, root)}"
        )

    expected_name = manifest.get("name")
    if expected_name and expected_name != set_name:
        raise ValueError(
            f"Dataset manifest name mismatch: expected {set_name!r}, got {expected_name!r}"
        )

    schema_version = manifest.get("schema_version")
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema_version {schema_version!r}; "
            f"expected {DATASET_SCHEMA_VERSION}"
        )

    expected_count = manifest.get("record_count", manifest.get("n"))
    if expected_count is not None and int(expected_count) != len(scenarios):
        raise ValueError(
            f"Dataset count mismatch: manifest={expected_count}, loaded={len(scenarios)}"
        )

    content_hash = file_sha256(jsonl_path)
    expected_hash = manifest.get("content_hash") or manifest.get("scenarios_sha256")
    if expected_hash and expected_hash != content_hash:
        raise ValueError(
            f"Dataset content hash mismatch: manifest={expected_hash}, actual={content_hash}"
        )

    return {
        "name": set_name,
        "root": str(root),
        "path": str(jsonl_path),
        "record_count": len(scenarios),
        "content_hash": content_hash,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Local-only data loaders (ported from _archive/annotator/core/storage.py)
# Read from explicit directories; no S3.
# ---------------------------------------------------------------------------

def _conv_id_to_uuid(conv_id: str) -> str:
    """Extract the transcript-UUID component from a full conv_id.

    Ported verbatim from _archive/annotator/core/storage.py lines 80-96.
    Accepts bare UUIDs, legacy composites, and bench composites.
    """
    matches = _UUID_RE.findall(conv_id)
    if matches:
        return matches[-1]
    stripped = _CONV_ID_PREFIX_RE.sub("", conv_id)
    if stripped != conv_id:
        return stripped
    return conv_id


def _load_ground_truth_files(ground_truth_dir: str) -> list[dict]:
    """Load all ground-truth JSON files from a local directory, sorted by filename.

    Ported from _archive/annotator/core/storage.py load_all_ground_truth_files,
    local-only variant.
    """
    gt_dir = Path(ground_truth_dir)
    files = []
    for fpath in sorted(gt_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if data is not None:
            files.append(data)
    return files


def _load_transcripts(transcripts_dir: str) -> dict[str, dict]:
    """Load all transcript JSON files from a local directory, sorted by filename.

    Ported from _archive/annotator/core/storage.py load_all_transcripts,
    local-only variant. Returns {conv_id: conversation_dict}.
    """
    tx_dir = Path(transcripts_dir)
    transcripts = {}
    for fpath in sorted(tx_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if data and "conversation_id" in data:
            transcripts[data["conversation_id"]] = data
        elif data:
            transcripts[fpath.stem] = data
    return transcripts


# ---------------------------------------------------------------------------
# JSONL transcript loader (ported from _archive/annotator/core/storage.py)
# ---------------------------------------------------------------------------

def _transform_normalized_record(rec: dict) -> dict:
    """Transform an S3 normalized JSONL record to internal transcript format.

    Ported faithfully from _archive/annotator/core/storage.py lines 383-475.
    S3 format: turns[] (dialogue only) + enrichments[] (non-dialogue, with before_turn).
    Enrichments are inserted before their associated dialogue turn with
    is_enrichment=True so they appear in excerpts as context but don't shift
    turn numbering.

    Omitted: _annotate_turns_with_start_seconds (applied separately if needed).
    """
    from collections import defaultdict as _defaultdict

    sess = rec.get("session", {})
    source_id = rec.get("transcript_id", "") or rec.get("source_id", "")
    tutor_id = sess.get("tutor_id", "") or sess.get("source_tutor_id", "")
    student_id = sess.get("student_id", "") or sess.get("source_student_id", "")

    # source_id format varies by batch:
    #   older batches: UUID only (e.g. "69b80b21-...")
    #   newer batches: full conv_id (e.g. "2025-t27247_2025-s12069_69b80b21-...")
    if tutor_id and tutor_id in source_id:
        conv_id = source_id  # already includes tutor_student prefix
    else:
        conv_id = f"{tutor_id}_{student_id}_{source_id}"

    # Index dialogue turns by their original turn_number
    raw_turns = rec.get("turns", [])
    max_dialogue_turn = max((t["turn_number"] for t in raw_turns), default=0)

    # Group enrichments by the dialogue turn they precede
    enrichments_by_turn = _defaultdict(list)
    trailing_enrichments = []
    for e in rec.get("enrichments", []):
        etype = e.get("type", "").upper().replace(" ", "_")
        label = e.get("label", "")
        content = e.get("content", "")
        text = f"[{etype}]"
        if label:
            text = f"[{etype}: {label}]"
        if content:
            text = f"{text} {content}"
        ss = float(e.get("start_seconds", 0) or 0)
        entry = {
            "role": "TUTOR",
            "text": text,
            "type": etype,
            "timestamp": f"{ss}s",
            "start_seconds": ss,
            "is_enrichment": True,
        }
        before = e.get("before_turn") or 1  # None/0 -> before turn 1
        if before <= max_dialogue_turn:
            enrichments_by_turn[before].append(entry)
        else:
            trailing_enrichments.append(entry)

    # Build final list: enrichments before each dialogue turn, preserving original numbering
    final_turns = []
    for t in raw_turns:
        turn_num = t["turn_number"]
        for e_entry in enrichments_by_turn.get(turn_num, []):
            final_turns.append({**e_entry, "turn_number": turn_num})
        ss = float(t.get("start_seconds", 0) or 0)
        final_turns.append({
            "turn_number": turn_num,
            "role": t["role"].upper(),
            "text": t["text"],
            "type": "DIALOGUE",
            "timestamp": f"{ss}s",
            "start_seconds": ss,
            "is_enrichment": False,
        })
    for e_entry in trailing_enrichments:
        final_turns.append({**e_entry, "turn_number": max_dialogue_turn})

    # Build context from demographics if available
    demo = rec.get("demographics", {})
    student = demo.get("student", {})
    context_parts = []
    if student.get("grade"):
        context_parts.append(f"Grade {student['grade']}")
    if student.get("subject"):
        context_parts.append(student["subject"])
    context = ", ".join(context_parts)

    return {
        "conversation_id": conv_id,
        "transcript_id": source_id,
        "tutor_id": tutor_id,
        "student_id": student_id,
        "context": context,
        "platform": rec.get("source", "step_up"),
        "num_turns": len(raw_turns),  # dialogue turns only
        "turns": final_turns,
    }


def _load_jsonl_index(path: str) -> dict[str, dict]:
    """Load a local JSONL file, transform each record, and index by conv_id.

    Ported from _archive/annotator/core/storage.py lines 478-537.
    Local-only variant: no S3/backend, no caching across calls.
    Returns {conv_id: conversation_dict}.
    """
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        logger.warning("JSONL file not found: %s", path)
        return {}

    index: dict[str, dict] = {}
    errors = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                transformed = _transform_normalized_record(rec)
                conv_id = transformed["conversation_id"]
                index[conv_id] = transformed
            except Exception:
                errors += 1

    logger.info("Indexed %d transcripts from %s (%d parse errors)", len(index), path, errors)
    return index


# ---------------------------------------------------------------------------
# Cut/extraction helpers (ported from _archive/benchmark/core/scenarios.py)
# ---------------------------------------------------------------------------

def _pick_modal_cut(votes: list[int]) -> "int | None":
    """Return the most-voted cut. On tie, return the smallest.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 15-25.
    Returns None when votes is empty.
    """
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [c for c, n in counts.items() if n == max_count]
    return min(winners)


def _role_adjust_cut(cut_turn: int, conversation: dict) -> "int | None":
    """Adjust cut_turn based on the role of the turn at cut_turn.

    - STUDENT: cut stays as-is (prefix includes the student turn).
    - TUTOR: cut_turn -= 1 (prefix excludes the human tutor turn; AI replaces it).
    - turn not found OR adjustment falls below 1: returns None (caller drops the cluster).

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 28-44.
    """
    turns_by_n = {t["turn_number"]: t for t in conversation.get("turns", [])}
    turn = turns_by_n.get(cut_turn)
    if turn is None:
        return None
    if turn.get("role") == "TUTOR":
        adjusted = cut_turn - 1
        if adjusted < 1:
            return None
        return adjusted
    return cut_turn


def _pick_representative_member(members: list[dict], chosen_cut: int) -> dict:
    """Return the member to use for the scenario's detection payload.

    Preference: members whose own cut_turn equals the chosen (modal) cut,
    smallest annotator_id (lexicographic) on tie. Falls back to smallest
    annotator_id overall if none voted the chosen cut.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 47-56.
    """
    matching = [m for m in members if m.get("cut_turn") == chosen_cut]
    pool = matching if matching else members
    return min(pool, key=lambda m: (m.get("annotator_id") or ""))


def _resolve_cluster(
    members: list[dict], conversation: dict, turn_start: int, turn_end: int
) -> "dict | None":
    """Return resolved cluster data, or None if the cluster should be dropped.

    Applies vote filtering (cut in [ts, te]), modal selection, role adjustment,
    and representative-member pick.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 78-107.
    """
    votes: list[int] = []
    for m in members:
        cut = m.get("cut_turn")
        if not isinstance(cut, int):
            continue
        if cut < turn_start or cut > turn_end:
            continue
        votes.append(cut)
    chosen = _pick_modal_cut(votes)
    if chosen is None:
        return None
    adjusted = _role_adjust_cut(chosen, conversation)
    if adjusted is None:
        return None
    rep = _pick_representative_member(members, chosen)
    return {
        "cut_turn": adjusted,
        "chosen_cut_turn": chosen,
        "cut_votes": dict(Counter(votes)),
        "cluster_size": len(members),
        "representative": rep,
    }


def _format_prefix(conversation: dict, cut_turn: int) -> str:
    """Format transcript turns up to and including cut_turn.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 110-120.
    """
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
    """Extract student context from conversation metadata.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 123-132.
    """
    context = conversation.get("context", "")
    if context:
        return context
    parts = []
    if conversation.get("platform"):
        parts.append(f"Platform: {conversation['platform']}")
    return "; ".join(parts) if parts else "K-12 tutoring session"


def _last_student_msg(conversation: dict, cut_turn: int) -> str:
    """Find the last STUDENT message at or before cut_turn.

    Ported verbatim from _archive/benchmark/core/scenarios.py lines 135-143.
    """
    last = ""
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        if turn["role"] == "STUDENT":
            last = turn["text"]
    return last


def _build_reference_transcript(conversation: dict, cut_turn: int) -> str:
    """Format the post-cut real human turns from a full conversation.

    Returns a newline-joined string of 'Turn N. ROLE: text' lines for every
    turn whose turn_number > cut_turn. Empty string if no post-cut turns.

    Ported verbatim from _archive/benchmark/core/exchange.py lines 218-235.
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


def _prefix_turns_to_context(conversation: dict, cut_turn: int) -> list[dict]:
    """Convert prefix turns into the new Scenario context format: [{turn_number, role, text}].

    turn_number is preserved verbatim from the source turn (real, non-sequential).
    Role is lowercased (TUTOR -> tutor, STUDENT -> student).
    Only includes turns up to and including cut_turn.
    """
    result = []
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        result.append({
            "turn_number": turn["turn_number"],
            "role": turn["role"].lower(),
            "text": turn["text"],
        })
    return result


# ---------------------------------------------------------------------------
# Public API: build_scenarios
# ---------------------------------------------------------------------------

def build_scenarios(
    *,
    set_name: str,
    ids: list[str],
    ground_truth_dir: str,
    transcripts_dir: str,
    step_up_jsonl: "str | None" = None,
) -> "list[Scenario]":
    """Hydrate a list of scenario ids into Scenario objects.

    Each id must be a string of the form '{conv_id}__hum_{ts}_{te}'.
    Scenarios are returned in the order given by `ids`.

    Args:
        set_name: Prefix used for Scenario.id, e.g. "balanced_520".
        ids: Ordered list of scenario id strings to hydrate.
        ground_truth_dir: Path to directory of per-conversation GT JSON files.
        transcripts_dir: Path to directory of per-conversation transcript JSON files.
        step_up_jsonl: Optional path to a normalized JSONL transcript file (e.g.
            step_up_annotations.jsonl). When provided, JSONL-loaded convs are merged
            into the transcript pool alongside per-file *.json files. JSONL entries
            take lower precedence: per-file JSON wins on conv_id collision.

    Returns:
        List of Scenario objects in the same order as `ids`. IDs with no
        matching ground-truth cluster are silently skipped.
    """
    # Load data
    gt_files = _load_ground_truth_files(ground_truth_dir)
    transcripts = _load_transcripts(transcripts_dir)

    # Merge JSONL transcripts (lower precedence: per-file JSON wins on collision)
    if step_up_jsonl is not None:
        jsonl_index = _load_jsonl_index(step_up_jsonl)
        for conv_id, conv in jsonl_index.items():
            if conv_id not in transcripts:
                transcripts[conv_id] = conv

    # Build UUID->conv_id lookup (same technique as archived scenarios.py line 151)
    uuid_to_conv = {_conv_id_to_uuid(cid): cid for cid in transcripts}

    # First pass: build cluster map {(conv_id, ts, te): [members]}
    # Ported from _archive/benchmark/core/scenarios.py lines 154-167
    clusters: dict[tuple, list[dict]] = {}
    for gt in gt_files:
        gt_uuid = gt.get("conversation_id")
        full_conv_id = uuid_to_conv.get(_conv_id_to_uuid(gt_uuid or ""))
        if not full_conv_id:
            continue
        for m in gt.get("key_moments", []):
            if m.get("situation_label_agg") not in _SCAFF_AGGS:
                continue
            ts = m.get("turn_start")
            te = m.get("turn_end")
            if ts is None or te is None:
                continue
            key = (full_conv_id, ts, te)
            clusters.setdefault(key, []).append(m)

    # Resolve every cluster once (resolve_cluster may return None -> dropped)
    resolved: dict[tuple, dict] = {}
    for (conv_id, ts, te), members in clusters.items():
        if conv_id not in transcripts:
            continue
        conversation = transcripts[conv_id]
        r = _resolve_cluster(members, conversation, ts, te)
        if r is not None:
            resolved[(conv_id, ts, te)] = r

    # Parse requested ids and build an index for O(1) lookup
    # id format: "{conv_id}__hum_{ts}_{te}"
    _HUM_RE = re.compile(r"^(.+)__hum_(\d+)_(\d+)$")

    scenarios_by_key: dict[tuple, Scenario] = {}
    for scenario_id_str in ids:
        m = _HUM_RE.match(scenario_id_str)
        if not m:
            logger.warning("Unrecognized scenario id format: %s", scenario_id_str)
            continue
        conv_id, ts_s, te_s = m.group(1), int(m.group(2)), int(m.group(3))
        key = (conv_id, ts_s, te_s)

        if key in scenarios_by_key:
            continue  # already built (duplicate in ids list)

        r = resolved.get(key)
        if r is None:
            logger.warning("No resolved cluster for scenario id: %s", scenario_id_str)
            continue

        if conv_id not in transcripts:
            logger.warning("Transcript not found for conv_id: %s", conv_id)
            continue

        conversation = transcripts[conv_id]
        cut_turn = r["cut_turn"]
        rep = r["representative"]

        context_turns = _prefix_turns_to_context(conversation, cut_turn)
        if not context_turns:
            logger.warning("Empty prefix for %s, skipping", scenario_id_str)
            continue

        reference = _build_reference_transcript(conversation, cut_turn)
        student_ctx = _get_student_context(conversation)
        dimension = rep.get("situation_label_agg")

        scenario = Scenario(
            id=f"{set_name}:{scenario_id_str}",
            context=context_turns,
            dimension=dimension,
            student={
                "mode": "oracle",
                "reference": reference,
                "context": student_ctx,
            },
            rubric={
                "gold": dimension,
                "hint": rep.get("situation", ""),
            },
            provenance={
                "conv_id": conv_id,
                "cut_turn": cut_turn,
                "turn_start": ts_s,
                "turn_end": te_s,
                "moment_id": rep.get("moment_id"),
                "annotator_id": rep.get("annotator_id"),
                "chosen_cut_turn": r["chosen_cut_turn"],
                "cut_votes": r["cut_votes"],
                "cluster_size": r["cluster_size"],
            },
        )
        scenarios_by_key[key] = scenario

    # Return in the requested order (skipping unknown ids)
    result = []
    seen_keys = set()
    for scenario_id_str in ids:
        m = _HUM_RE.match(scenario_id_str)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        if key in scenarios_by_key and key not in seen_keys:
            result.append(scenarios_by_key[key])
            seen_keys.add(key)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_build(args: argparse.Namespace) -> None:
    """Write scenarios/<set>/{scenarios.jsonl, manifest.json, CHANGELOG.md}."""
    ids_path = Path(args.ids)
    with open(ids_path, encoding="utf-8") as f:
        ids = json.load(f)

    scenarios = build_scenarios(
        set_name=args.set,
        ids=ids,
        ground_truth_dir=args.ground_truth,
        transcripts_dir=args.transcripts,
        step_up_jsonl=getattr(args, "step_up_jsonl", None),
    )

    out_dir = Path(args.out_root) / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    # scenarios.jsonl
    jsonl_path = out_dir / "scenarios.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {len(scenarios)} scenarios to {jsonl_path}")

    content_hash = file_sha256(jsonl_path)

    # manifest.json
    manifest = {
        "name": args.set,
        "version": getattr(args, "version", None) or "0",
        "schema_version": DATASET_SCHEMA_VERSION,
        "record_count": len(scenarios),
        "content_hash": content_hash,
        "provenance": {
            "ids_file": str(ids_path),
            "ground_truth_dir": args.ground_truth,
            "transcripts_dir": args.transcripts,
            "cut_logic": "modal vote; tie->smallest; TUTOR-role decrement",
            "source_tags": ["human_annotated", "hybrid_ground_truth"],
        },
        "created": args.created,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")

    # CHANGELOG.md (stub)
    changelog_path = out_dir / "CHANGELOG.md"
    if not changelog_path.exists():
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(f"# CHANGELOG: {args.set}\n\n")
            f.write(f"## {args.created}\n\n")
            f.write(f"- Initial build: {len(scenarios)} scenarios\n")
        print(f"Wrote CHANGELOG to {changelog_path}")


def _cli_validate(args: argparse.Namespace) -> None:
    """Validate scenarios/<set>/{scenarios.jsonl, manifest.json}."""
    report = validate_dataset(args.set, root=args.root)
    print(f"Dataset valid: {report['name']}")
    print(f"  records: {report['record_count']}")
    print(f"  sha256 : {report['content_hash']}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="tutorsim scenario tools")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Hydrate and freeze scenarios")
    build_p.add_argument("--set", required=True, help="Set name, e.g. balanced_520")
    build_p.add_argument("--ids", required=True, help="Path to JSON list of scenario ids")
    build_p.add_argument("--ground-truth", required=True, help="Ground truth directory")
    build_p.add_argument("--transcripts", required=True, help="Transcripts directory")
    build_p.add_argument("--out-root", default="scenarios", help="Output root (default: scenarios/)")
    build_p.add_argument("--created", default="", help="ISO date string for manifest (default: empty)")
    build_p.add_argument("--version", default="0", help="Dataset version string for manifest")
    build_p.add_argument("--step-up-jsonl", dest="step_up_jsonl", default=None,
                         help="Path to normalized step_up JSONL transcript file")

    args = parser.parse_args()
    if args.command == "build":
        _cli_build(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
