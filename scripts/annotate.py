#!/usr/bin/env python
"""Script to annotate tutoring transcripts."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tutor_bench import Annotator


def main():
    """Main annotation script."""
    parser = argparse.ArgumentParser(description="Annotate tutoring transcripts for evaluation")
    parser.add_argument("input", type=str, help="Path to input JSONLines file containing transcripts")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output JSONLines file for annotations (default: input_annotations.jsonl)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-3.5-turbo", "claude-2", "llama-2"],
        help="Model to use for annotation (mock parameter)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_annotations.jsonl"
    else:
        output_path = Path(args.output)

    # Create annotator
    print("🚀 tutor-bench Annotation Pipeline")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print("=" * 50)
    print()

    annotator = Annotator(model=args.model, verbose=not args.quiet)

    # Process transcripts
    try:
        annotations = annotator.process_transcripts(args.input)
        annotations.save(str(output_path))

        print()
        print("✅ Annotation complete!")
        print(f"   Generated {len(annotations)} annotations")
        print(f"   Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during annotation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
