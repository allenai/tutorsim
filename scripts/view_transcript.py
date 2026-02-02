#!/usr/bin/env python
"""View a transcript in a human-readable format."""

import argparse
import json
from typing import Any


def format_transcript(transcript: dict[str, Any], colored: bool = True) -> str:
    """Format a transcript for display."""
    lines = []

    # Colors for terminal output
    if colored:
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"  # noqa: F841
        GRAY = "\033[90m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
    else:
        BLUE = GREEN = GRAY = RESET = BOLD = ""  # noqa: F841

    # Header
    metadata = transcript.get("metadata", {})
    lines.append(f"{BOLD}{'='*70}{RESET}")
    lines.append(f"{BOLD}Session ID: {metadata.get('session_id', 'unknown')}{RESET}")
    lines.append(f"{GRAY}Timestamp: {metadata.get('timestamp', 'N/A')}{RESET}")

    if "grade_level" in metadata:
        lines.append(f"{GRAY}Grade Level: {metadata['grade_level']}{RESET}")
    if "topic" in metadata:
        lines.append(f"{GRAY}Topic: {metadata['topic']}{RESET}")
    if "student_id" in metadata:
        lines.append(f"{GRAY}Student: {metadata['student_id']}{RESET}")

    lines.append(f"{BOLD}{'='*70}{RESET}")
    lines.append("")

    # Messages
    for msg in transcript.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            lines.append(f"{GRAY}[SYSTEM]{RESET}")
            lines.append(f"{GRAY}{content}{RESET}")
        elif role == "user":
            lines.append(f"{BLUE}[STUDENT]{RESET}")
            lines.append(f"{content}")
        elif role == "assistant":
            lines.append(f"{GREEN}[TEACHER]{RESET}")
            lines.append(f"{content}")
        else:
            lines.append(f"[{role.upper()}]")
            lines.append(content)

        lines.append("")

    # Footer
    if "duration_seconds" in metadata:
        duration = metadata["duration_seconds"]
        minutes = duration // 60
        seconds = duration % 60
        lines.append(f"{GRAY}Duration: {minutes}m {seconds}s{RESET}")

    lines.append(f"{BOLD}{'='*70}{RESET}")

    return "\n".join(lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="View transcripts in human-readable format")
    parser.add_argument("file", type=str, help="Path to JSONLines file containing transcripts")
    parser.add_argument("-n", "--number", type=int, default=1, help="Which transcript number to view (1-based index)")
    parser.add_argument("-a", "--all", action="store_true", help="View all transcripts in the file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Load transcripts
    transcripts = []
    try:
        with open(args.file) as f:
            for line in f:
                transcripts.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
        return 1

    if not transcripts:
        print(f"❌ No transcripts found in {args.file}")
        return 1

    # Display transcripts
    if args.all:
        for i, transcript in enumerate(transcripts, 1):
            if i > 1:
                print("\n" + "=" * 70 + "\n")
            print(f"Transcript {i} of {len(transcripts)}")
            print(format_transcript(transcript, colored=not args.no_color))
    else:
        if args.number < 1 or args.number > len(transcripts):
            print(f"❌ Invalid transcript number. File contains {len(transcripts)} transcripts.")
            return 1

        print(f"Transcript {args.number} of {len(transcripts)}")
        print(format_transcript(transcripts[args.number - 1], colored=not args.no_color))

    return 0


if __name__ == "__main__":
    exit(main())
