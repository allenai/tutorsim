"""Mock annotator for processing tutoring transcripts."""

import json
import random
from pathlib import Path
from typing import Any


class AnnotationResult:
    """Container for annotation results."""

    def __init__(self, annotations: list[dict[str, Any]]):
        self.annotations = annotations

    def save(self, filepath: str) -> None:
        """Save annotations to a JSONLines file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for annotation in self.annotations:
                f.write(json.dumps(annotation) + "\n")
        print(f"✅ Saved {len(self.annotations)} annotations to {filepath}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __repr__(self) -> str:
        return f"AnnotationResult(n_annotations={len(self.annotations)})"


class Annotator:
    """Mock annotator for tutoring transcripts."""

    def __init__(self, model: str = "gpt-4", verbose: bool = True):
        """Initialize the annotator.

        Args:
            model: The model to use for annotation (mock parameter)
            verbose: Whether to print progress messages
        """
        self.model = model
        self.verbose = verbose

        # Mock annotation labels
        self.span_labels = ["definition", "example", "question", "explanation", "feedback"]
        self.effectiveness_levels = ["low", "medium", "high"]
        self.concepts = [
            "photosynthesis",
            "mitosis",
            "algebra",
            "grammar",
            "history",
            "geography",
            "physics",
            "chemistry",
        ]

    def process_transcripts(self, filepath: str) -> AnnotationResult:
        """Process transcripts and generate mock annotations.

        Args:
            filepath: Path to the JSONLines file containing transcripts

        Returns:
            AnnotationResult containing the generated annotations
        """
        if self.verbose:
            print(f"🔍 Processing transcripts from {filepath} using {self.model}...")

        annotations = []

        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        transcript = json.loads(line)
                        annotation = self._annotate_transcript(transcript)
                        annotations.append(annotation)
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"⚠️  Skipping invalid JSON at line {line_num}")
                        continue
        except FileNotFoundError:
            # Create mock data if file doesn't exist
            if self.verbose:
                print("⚠️  File not found, generating mock annotations...")
            for i in range(3):
                annotations.append(self._create_mock_annotation(f"mock_{i}"))

        if self.verbose:
            print(f"✅ Generated annotations for {len(annotations)} transcripts")

        return AnnotationResult(annotations)

    def _annotate_transcript(self, transcript: dict[str, Any]) -> dict[str, Any]:
        """Generate mock annotations for a single transcript."""
        session_id = transcript.get("metadata", {}).get("session_id", "unknown")
        messages = transcript.get("messages", [])

        # Generate span-level annotations
        span_annotations = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant" and len(msg.get("content", "")) > 20:
                # Mock: randomly annotate parts of assistant messages
                content_len = len(msg["content"])
                num_spans = random.randint(0, 3)
                for _ in range(num_spans):
                    start = random.randint(0, max(0, content_len - 20))
                    end = min(start + random.randint(10, 50), content_len)
                    span_annotations.append(
                        {
                            "message_index": idx,
                            "start": start,
                            "end": end,
                            "label": random.choice(self.span_labels),
                            "confidence": round(random.uniform(0.7, 0.99), 2),
                        }
                    )

        # Generate transcript-level annotations
        transcript_annotations = {
            "quality_score": round(random.uniform(0.6, 0.95), 2),
            "pedagogical_effectiveness": random.choice(self.effectiveness_levels),
            "concepts_covered": random.sample(self.concepts, k=random.randint(1, 4)),
            "num_interactions": len(messages),
            "avg_response_length": sum(len(m.get("content", "")) for m in messages) // max(1, len(messages)),
        }

        return {
            "session_id": session_id,
            "annotations": {"span_level": span_annotations, "transcript_level": transcript_annotations},
        }

    def _create_mock_annotation(self, session_id: str) -> dict[str, Any]:
        """Create a completely mock annotation."""
        return {
            "session_id": session_id,
            "annotations": {
                "span_level": [
                    {
                        "message_index": i,
                        "start": 0,
                        "end": 20,
                        "label": random.choice(self.span_labels),
                        "confidence": round(random.uniform(0.7, 0.99), 2),
                    }
                    for i in range(random.randint(1, 3))
                ],
                "transcript_level": {
                    "quality_score": round(random.uniform(0.6, 0.95), 2),
                    "pedagogical_effectiveness": random.choice(self.effectiveness_levels),
                    "concepts_covered": random.sample(self.concepts, k=random.randint(1, 4)),
                    "num_interactions": random.randint(3, 10),
                    "avg_response_length": random.randint(50, 200),
                },
            },
        }
