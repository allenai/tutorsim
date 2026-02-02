#!/usr/bin/env python
"""Generate sample transcript files for testing the tutor-bench pipeline."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def generate_message(role: str, content_type: str = "general") -> dict[str, str]:
    """Generate a mock message based on role and content type."""
    messages = {
        "system": [
            "You are a helpful tutor specializing in science and mathematics.",
            "You are an AI tutor helping students learn effectively.",
            "You are a patient educator focused on student understanding.",
        ],
        "user": {
            "general": [
                "Can you explain this concept to me?",
                "I don't understand this part.",
                "Can you give me an example?",
                "How does this work?",
                "What's the difference between these two things?",
            ],
            "math": [
                "How do I solve this equation?",
                "What's the derivative of x^2?",
                "Can you explain logarithms?",
                "How do I factor this polynomial?",
            ],
            "science": [
                "What is photosynthesis?",
                "How do chemical bonds work?",
                "Can you explain Newton's laws?",
                "What causes seasons on Earth?",
            ],
        },
        "assistant": {
            "general": [
                "Let me explain that concept step by step. First, we need to understand the fundamental principle...",
                "That's a great question! Here's a clear example to illustrate the concept...",
                "I understand your confusion. Let's break this down into simpler parts...",
                "The key difference is in how they approach the problem. Let me clarify...",
            ],
            "math": [
                "To solve this equation, we'll use the quadratic formula. The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a...",
                "The derivative of x^2 is 2x. We can find this using the power rule, which states that d/dx(x^n) = n*x^(n-1)...",
                "Logarithms are the inverse of exponential functions. If b^x = y, then log_b(y) = x...",
            ],
            "science": [
                "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in two stages: the light reactions and the Calvin cycle...",
                "Chemical bonds form when atoms share or transfer electrons. There are three main types: ionic, covalent, and metallic bonds...",
                "Newton's three laws of motion describe the relationship between forces and motion. The first law states that an object at rest stays at rest...",
            ],
        },
    }

    if role == "system":
        return {"role": role, "content": random.choice(messages["system"])}
    elif role == "user":
        content = random.choice(messages[role][content_type])
        return {"role": role, "content": content}
    else:  # assistant
        content = random.choice(messages[role].get(content_type, messages[role]["general"]))
        return {"role": role, "content": content}


def generate_transcript(session_id: str, num_exchanges: int = 3, subject: str = "general") -> dict[str, Any]:
    """Generate a mock tutoring transcript."""
    messages = [generate_message("system")]

    for _ in range(num_exchanges):
        messages.append(generate_message("user", subject))
        messages.append(generate_message("assistant", subject))

    base_time = datetime.now() - timedelta(days=random.randint(0, 30))

    return {
        "messages": messages,
        "metadata": {
            "session_id": session_id,
            "timestamp": base_time.isoformat() + "Z",
            "subject": subject,
            "student_id": f"student_{random.randint(100, 999)}",
            "duration_seconds": random.randint(300, 1800),
            "platform": random.choice(["web", "mobile", "desktop"]),
        },
    }


def main():
    """Generate sample transcript files."""
    # Create output directory
    output_dir = Path("data/sample_transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate different sets of transcripts
    subjects = ["general", "math", "science"]

    # Generate a mixed set
    mixed_file = output_dir / "mixed_transcripts.jsonl"
    print(f"Generating mixed transcripts to {mixed_file}")

    with open(mixed_file, "w") as f:
        for i in range(10):
            subject = random.choice(subjects)
            num_exchanges = random.randint(2, 5)
            transcript = generate_transcript(
                session_id=f"session_{i:03d}", num_exchanges=num_exchanges, subject=subject
            )
            f.write(json.dumps(transcript) + "\n")

    # Generate subject-specific sets
    for subject in subjects:
        subject_file = output_dir / f"{subject}_transcripts.jsonl"
        print(f"Generating {subject} transcripts to {subject_file}")

        with open(subject_file, "w") as f:
            for i in range(5):
                num_exchanges = random.randint(2, 4)
                transcript = generate_transcript(
                    session_id=f"{subject}_{i:03d}", num_exchanges=num_exchanges, subject=subject
                )
                f.write(json.dumps(transcript) + "\n")

    print(f"\n✅ Generated sample transcripts in {output_dir}")
    print("Files created:")
    for file in output_dir.glob("*.jsonl"):
        num_lines = sum(1 for _ in open(file))
        print(f"  - {file.name}: {num_lines} transcripts")


if __name__ == "__main__":
    main()
