#!/usr/bin/env python
"""Demo script to run the complete tutor-bench pipeline."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and print its output."""
    print(f"\n🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    """Run the complete demo pipeline."""
    print("=" * 60)
    print("tutor-bench Demo Pipeline".center(60))
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("tutor_bench").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Step 1: Generate sample transcripts
    run_command(["python", "scripts/generate_sample_transcripts.py"], "Step 1: Generating sample transcripts")

    # Step 2: Annotate the transcripts
    transcript_file = "data/sample_transcripts/mixed_transcripts.jsonl"
    annotation_file = "data/sample_transcripts/mixed_transcripts_annotations.jsonl"

    run_command(
        ["python", "scripts/annotate.py", transcript_file, "-o", annotation_file], "Step 2: Annotating transcripts"
    )

    # Step 3: Evaluate with annotations
    metrics_file = "data/sample_transcripts/evaluation_metrics.json"

    run_command(
        ["python", "scripts/evaluate.py", transcript_file, annotation_file, "-o", metrics_file],
        "Step 3: Evaluating transcripts with annotations",
    )

    print("\n" + "=" * 60)
    print("✅ Demo pipeline completed successfully!".center(60))
    print("=" * 60)

    print("\nGenerated files:")
    print(f"  📄 Transcripts: {transcript_file}")
    print(f"  📝 Annotations: {annotation_file}")
    print(f"  📊 Metrics: {metrics_file}")

    print("\nTo explore the results:")
    print(f"  cat {metrics_file} | python -m json.tool")


if __name__ == "__main__":
    main()
