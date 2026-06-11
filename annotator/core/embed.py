"""
Embed -- Encode action and result facets into dense vectors.

Reads decomposed facets (action_decomposed / result_decomposed) from either:
  - Ground truth files in data/ground_truth_{labeller}/
  - A decomposed_*.json result produced by decompose.py

Encodes all facets in a single batch with sentence-transformers/all-MiniLM-L6-v2
(384-dim vectors) and writes action_embeddings / result_embeddings (list[list[float]])
parallel to the facet lists. A progress bar is shown when there are >100 facets.

Output structure
----------------
Ground truth mode  →  data/embeddings_{labeller}.json
    {
      "<conv_id>": {
        "conversation_id": "...",
        "num_turns": N,
        "key_moments": [
          {
            "action_decomposed": ["facet1", "facet2", ...],
            "result_decomposed": ["facet1", ...],
            "action_embeddings": [[...384 floats...], ...],  # parallel to action_decomposed
            "result_embeddings": [[...384 floats...], ...],  # parallel to result_decomposed
            ...original moment fields...
          }
        ]
      }
    }

Decompose mode  →  results/annotator/{version}/embedded_{...}_{target}.json
    Same structure as the input decomposed_*.json but each annotation gains:
      "action_embeddings": [[...384 floats...], ...]  # parallel to action_decomposed
      "result_embeddings": [[...384 floats...], ...]  # parallel to result_decomposed

Usage:
    # Embed the ground truth
    python -m annotator.core.embed --ground-truth
    python -m annotator.core.embed --ground-truth --labeller v2

    # Embed decompose.py output
    python -m annotator.core.embed --version v1
    python -m annotator.core.embed --version v1 --gold
    python -m annotator.core.embed --version v1 --split test
    python -m annotator.core.embed --version v1 --target rapport
    python -m annotator.core.embed --version v1 --style balanced
"""

import argparse
import json
import logging
from pathlib import Path

from common.logging_setup import setup_logging
from .config import get_valid_styles, get_annotation_types
from .storage import (
    load_annotator_result, save_annotator_result,
    get_annotator_result_path,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )
    logger.info("Loading embedding model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def _encode_all(encoder, facets: list[str]) -> list[list[float]]:
    """Encode a list of strings; return list of float lists."""
    if not facets:
        return []
    vectors = encoder.encode(facets, show_progress_bar=len(facets) > 100)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Ground truth mode
# ---------------------------------------------------------------------------

def run_embed_ground_truth(labeller: str = "hybrid") -> None:
    gt_dir = DATA_DIR / f"ground_truth_{labeller}"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    conv_files = sorted(gt_dir.glob("*.json"))
    if not conv_files:
        raise FileNotFoundError(f"No JSON files found in {gt_dir}")

    logger.info("Loading ground truth from %s (%d files)", gt_dir, len(conv_files))

    # Load all conversations, validating required structural keys up front.
    # A missing key here means the ground-truth file is corrupt -- fail loudly
    # rather than silently embedding zero facets and writing a file that looks
    # successful.
    convs = {}
    for f in conv_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if "key_moments" not in data:
            raise ValueError(f"Ground truth file {f.name} is missing required key 'key_moments'")
        for m_idx, moment in enumerate(data["key_moments"]):
            for field in ("action_decomposed", "result_decomposed"):
                if field not in moment:
                    raise ValueError(
                        f"Ground truth file {f.name} moment {m_idx} is missing "
                        f"required key '{field}' (was it decomposed?)"
                    )
        convs[f.stem] = data

    # Collect all facets in order so we can do one encode() call
    # Record: (conv_id, moment_idx, field, facet_idx) → position in flat list
    flat_facets: list[str] = []
    locations: list[tuple[str, int, str, int]] = []

    for conv_id, data in convs.items():
        for m_idx, moment in enumerate(data["key_moments"]):
            for field in ("action_decomposed", "result_decomposed"):
                for f_idx, facet in enumerate(moment[field] or []):
                    flat_facets.append(facet)
                    locations.append((conv_id, m_idx, field, f_idx))

    logger.info("Encoding %d facets from %d conversations", len(flat_facets), len(convs))
    encoder = _load_encoder()
    all_vectors = _encode_all(encoder, flat_facets)

    # Distribute embeddings back — build embedding lists in parallel to facet lists
    embed_map: dict[tuple[str, int, str], list[list[float]]] = {}
    for (conv_id, m_idx, field, f_idx), vec in zip(locations, all_vectors):
        key = (conv_id, m_idx, field)
        if key not in embed_map:
            embed_map[key] = []
        embed_map[key].append(vec)

    # Write output — one merged structure per conv mirroring the source
    output: dict[str, dict] = {}
    for conv_id, data in convs.items():
        moments = []
        for m_idx, moment in enumerate(data["key_moments"]):
            m = dict(moment)
            m["action_embeddings"] = embed_map.get((conv_id, m_idx, "action_decomposed"), [])
            m["result_embeddings"] = embed_map.get((conv_id, m_idx, "result_decomposed"), [])
            moments.append(m)
        output[conv_id] = {**data, "key_moments": moments}

    out_path = DATA_DIR / f"embeddings_{labeller}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Saved embeddings for %d conversations (%d facets) → %s",
        len(output), len(flat_facets), out_path,
    )


# ---------------------------------------------------------------------------
# Decompose output mode
# ---------------------------------------------------------------------------

def run_embed_decomposed(
    version: str,
    profile: str | None = None,
    annotator_style: str | None = None,
    target: str = "scaffolding",
    split: str = "train",
    gold: bool = False,
) -> None:
    profile_suffix = f"_{profile}" if profile else ""
    style_suffix = f"_{annotator_style}" if annotator_style else ""
    split_suffix = f"_{split}" if split != "train" else ""
    gold_prefix = "decomposed_gold" if gold else "decomposed"
    input_filename = f"{gold_prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"

    data = load_annotator_result(version, input_filename)
    if data is None:
        raise FileNotFoundError(
            f"{input_filename} not found for version {version}. Run decompose first."
        )
    logger.info("Loaded: %s (version=%s)", input_filename, version)

    results = data.get("results", {})

    # Collect all facets in order
    flat_facets: list[str] = []
    locations: list[tuple[str, int, str, int]] = []

    for conv_id, conv_data in results.items():
        for ann_idx, ann in enumerate(conv_data.get("annotations", [])):
            for field in ("action_decomposed", "result_decomposed"):
                for f_idx, facet in enumerate(ann.get(field) or []):
                    flat_facets.append(facet)
                    locations.append((conv_id, ann_idx, field, f_idx))

    logger.info("Encoding %d facets from %d conversations", len(flat_facets), len(results))
    encoder = _load_encoder()
    all_vectors = _encode_all(encoder, flat_facets)

    # Distribute back
    embed_map: dict[tuple[str, int, str], list[list[float]]] = {}
    for (conv_id, ann_idx, field, f_idx), vec in zip(locations, all_vectors):
        key = (conv_id, ann_idx, field)
        if key not in embed_map:
            embed_map[key] = []
        embed_map[key].append(vec)

    # Enrich results in-place
    for conv_id, conv_data in results.items():
        for ann_idx, ann in enumerate(conv_data.get("annotations", [])):
            ann["action_embeddings"] = embed_map.get((conv_id, ann_idx, "action_decomposed"), [])
            ann["result_embeddings"] = embed_map.get((conv_id, ann_idx, "result_decomposed"), [])

    output = {**data, "results": results, "embedded": True}
    embedded_prefix = "embedded_gold" if gold else "embedded"
    output_filename = f"{embedded_prefix}{profile_suffix}{style_suffix}{split_suffix}_{target}.json"
    save_annotator_result(version, output_filename, output)
    logger.info(
        "Saved embeddings for %d conversations (%d facets) → %s",
        len(results), len(flat_facets), output_filename,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed action and result facets using sentence-transformers/all-MiniLM-L6-v2"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ground-truth", action="store_true",
                            help="Embed ground truth facets from data/ground_truth_{labeller}/")
    mode_group.add_argument("--version", default=None,
                            help="Embed decomposed_{...}.json for this version")

    # Ground truth options
    parser.add_argument("--labeller", default="hybrid",
                        help="Ground truth labeller variant (default: hybrid)")

    # Decompose output options (mirror decompose.py)
    parser.add_argument("--profile", default=None,
                        help="Config profile suffix (overrides config.yaml default)")
    parser.add_argument("--annotator-style", "--style", choices=get_valid_styles(),
                        default=None, dest="annotator_style",
                        help="Style suffix matching the decomposed file (e.g. balanced)")
    parser.add_argument("--target", choices=get_annotation_types(), default="scaffolding",
                        help="Annotation type (default: scaffolding)")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split (default: train)")
    parser.add_argument("--gold", action="store_true",
                        help="Embed gold truth decomposition (decomposed_gold_*.json)")

    args = parser.parse_args()
    setup_logging()

    if args.ground_truth:
        run_embed_ground_truth(labeller=args.labeller)
    else:
        run_embed_decomposed(
            version=args.version,
            profile=args.profile,
            annotator_style=args.annotator_style,
            target=args.target,
            split=args.split,
            gold=args.gold,
        )


if __name__ == "__main__":
    main()
