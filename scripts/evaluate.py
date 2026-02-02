#!/usr/bin/env python
"""Script to evaluate annotated tutoring transcripts."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tutor_bench import Evaluator


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate tutoring transcripts with annotations")
    parser.add_argument("transcripts", type=str, help="Path to JSONLines file containing original transcripts")
    parser.add_argument("annotations", type=str, help="Path to JSONLines file containing annotations")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to output JSON file for metrics (optional)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")
    parser.add_argument("--no-summary", action="store_true", help="Don't print the metrics summary")

    args = parser.parse_args()

    # Create evaluator
    print("📊 tutor-bench Evaluation Pipeline")
    print("=" * 50)
    print(f"Transcripts: {args.transcripts}")
    print(f"Annotations: {args.annotations}")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 50)
    print()

    evaluator = Evaluator(verbose=not args.quiet)

    # Run evaluation
    try:
        metrics = evaluator.evaluate(
            transcripts=args.transcripts, annotations=args.annotations, output_path=args.output
        )

        if not args.no_summary:
            print()
            print(metrics.summary())

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
