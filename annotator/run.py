"""
Unified pipeline runner: detect -> annotate -> label -> decompose -> structure.

Runs all passes in sequence with a single command.
Supports per-phase model selection via profiles (e.g., openai_claude
uses gpt-5.4 for detection and claude-opus-4-6 for annotation).

Usage:
    python -m annotator --version v1
    python -m annotator --version v1 --profile openai --mode sync
    python -m annotator --version v1 --skip-detect       # annotate+label only
    python -m annotator --version v1 --gold               # annotate gold moments
    python -m annotator --version v1 --test 3 --mode sync # quick test
"""

import argparse
import datetime
import logging
from pathlib import Path

from common.logging_setup import setup_logging
from .core.config import get_phase_config, get_valid_styles, get_annotation_types
from .core.detect import run_detect
from .core.annotate import run_annotate
from .core.label import run_label
from .core.decompose import run_decompose
from .core.structure import run_structure_label

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "annotator"


def build_parser():
    parser = argparse.ArgumentParser(description="Run full annotation pipeline")

    parser.add_argument("--version", default=None,
                        help="Results version directory (e.g. v1, v3_gemini). Auto-generates if not set.")
    parser.add_argument("--model", default=None,
                        help="Override model for all phases")
    parser.add_argument("--profile", default=None,
                        help="Config profile (gemini, openai, anthropic, openai_claude)")
    parser.add_argument("--mode", choices=["batch", "sync"], default=None,
                        help="Execution mode for all phases")
    parser.add_argument("--prompt-version", default=None,
                        help="Prompt version (defaults to --version)")
    parser.add_argument("--target", nargs="+", choices=get_annotation_types(),
                        default=get_annotation_types(),
                        help="Annotation targets")
    parser.add_argument("--test", type=int, default=0,
                        help="Test on N conversations (0 = all)")
    parser.add_argument("--dialogue-only", action="store_true",
                        help="Exclude non-dialogue turns from transcripts")

    parser.add_argument("--skip-detect", action="store_true",
                        help="Skip detection; use existing detections.json")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="Skip annotation; use existing annotations.json")
    parser.add_argument("--skip-decompose", action="store_true",
                        help="Stop after labeling; skip decompose + structure passes")
    parser.add_argument("--gold", action="store_true",
                        help="Use gold truth moments (skips detect automatically)")

    parser.add_argument("--style", choices=get_valid_styles(),
                        default=None,
                        help="Annotator style: use per-style prompts for annotation and labeling")
    parser.add_argument("--context", type=int, default=None,
                        help="Context window for annotation excerpts")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Which split to run on (default: train)")
    parser.add_argument("--rerun", action="store_true",
                        help="Reprocess from scratch: ignore existing detect/annotate "
                             "shards and overwrite them instead of resuming. "
                             "(Label/decompose/structure already overwrite every run.)")
    return parser


def main():
    args = build_parser().parse_args()

    from .core.config import resolve_run_params
    params = resolve_run_params(
        cli_version=args.version,
        cli_profile=args.profile,
        cli_style=args.style,
        cli_prompt_version=args.prompt_version,
    )
    profile = params["profile"]
    version = params["version"]
    style = params["style"]
    prompt_version = params["prompt_version"]

    setup_logging(version=version)

    if args.gold:
        args.skip_detect = True

    # When style is set, override prompts to per-style profiles (if they exist)
    annotation_prompt_version = prompt_version
    detection_prompt_version = prompt_version
    if style:
        annotation_prompt_version = f"profiles/{style}"
        # Use per-style detection prompts if p1/ dir exists for this style
        style_p1_dir = PROMPTS_DIR / "profiles" / style / "p1"
        if style_p1_dir.exists():
            detection_prompt_version = f"profiles/{style}"

    # --- Pass 1: Detect ---
    detections_data = None
    if not args.skip_detect:
        logger.info("=== PASS 1: Detection ===")
        detect_cfg = get_phase_config("detect", profile)
        detect_output = run_detect(
            version=version,
            model=args.model or detect_cfg["model"],
            mode=args.mode or detect_cfg.get("mode", "batch"),
            prompt_version=detection_prompt_version,
            targets=args.target,
            phase_cfg=detect_cfg,
            test=args.test,
            dialogue_only=args.dialogue_only,
            split=args.split,
            rerun=args.rerun,
        )
        detections_data = detect_output["results"]

    # --- Pass 2: Annotate ---
    annotations_data = None
    if not args.skip_annotate:
        logger.info("=== PASS 2: Annotation ===")
        annotate_cfg = get_phase_config("annotate", profile)
        context_window = (args.context if args.context is not None
                          else annotate_cfg.get("context_window", 20))
        annotations_data = run_annotate(
            version=version,
            model=args.model or annotate_cfg["model"],
            mode=args.mode or annotate_cfg.get("mode", "batch"),
            prompt_version=annotation_prompt_version,
            targets=args.target,
            phase_cfg=annotate_cfg,
            dialogue_only=args.dialogue_only,
            context_window=context_window,
            gold=args.gold,
            annotator_style=style,
            detections_by_conv=detections_data,
            profile=profile,
            split=args.split,
            rerun=args.rerun,
        )
        if annotations_data is None:
            logger.error("Annotation failed. Aborting.")
            return

    # --- Pass 3: Label ---
    logger.info("=== PASS 3: Labeling ===")
    label_cfg = get_phase_config("label", profile)
    labels_data = run_label(
        version=version,
        model=args.model or label_cfg["model"],
        mode=args.mode or label_cfg.get("mode", "batch"),
        phase_cfg=label_cfg,
        gold=args.gold,
        annotator_style=style,
        annotations_data=annotations_data,
        profile=profile,
        targets=args.target,
        split=args.split,
    )
    if labels_data is None:
        logger.error("Labeling failed (nothing labeled/saved). Aborting.")
        return

    # --- Pass 4 + 5: Decompose -> Structure (per target) ---
    # Both passes reuse the "label" phase config and operate on a single
    # target, reading their inputs from disk (written by the prior pass).
    # A None return means the prior pass wrote no input for this target -- a
    # silent no-op we surface instead of marching on to the next phase.
    if not args.skip_decompose:
        decompose_model = args.model or label_cfg["model"]
        decompose_mode = args.mode or label_cfg.get("mode", "batch")
        for target in args.target:
            logger.info("=== PASS 4: Decompose (%s) ===", target)
            decomposed = run_decompose(
                version=version,
                model=decompose_model,
                mode=decompose_mode,
                phase_cfg=label_cfg,
                gold=args.gold,
                annotator_style=style,
                profile=profile,
                target=target,
                split=args.split,
            )
            if decomposed is None:
                logger.error("Decompose failed for target '%s'. Aborting.", target)
                return

            logger.info("=== PASS 5: Structure (%s) ===", target)
            structured = run_structure_label(
                version=version,
                model=decompose_model,
                mode=decompose_mode,
                phase_cfg=label_cfg,
                gold=args.gold,
                annotator_style=style,
                profile=profile,
                target=target,
                split=args.split,
            )
            if structured is None:
                logger.error("Structure labeling failed for target '%s'. Aborting.", target)
                return

    logger.info("=== Pipeline complete ===")
    style_flag = f" --annotator-style {style}" if style else ""
    profile_flag = f" --profile {profile}" if profile else ""
    split_flag = f" --split {args.split}" if args.split != "train" else ""
    logger.info(f"  Next: python -m annotator.eval.eval --version {version}{profile_flag}{style_flag}{split_flag}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
