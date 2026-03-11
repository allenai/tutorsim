"""Mock evaluator for computing metrics from transcripts and annotations."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    num_transcripts: int
    num_annotations: int
    avg_quality_score: float
    effectiveness_distribution: dict[str, int]
    concept_coverage: dict[str, int]
    span_label_distribution: dict[str, int]
    avg_confidence: float
    missing_annotations: list[str]

    def summary(self) -> str:
        """Generate a human-readable summary of metrics."""
        summary_lines = [
            "=" * 60,
            "Evaluation Metrics Summary",
            "=" * 60,
            f"Total transcripts: {self.num_transcripts}",
            f"Total annotations: {self.num_annotations}",
            f"Missing annotations: {len(self.missing_annotations)}",
            "",
            "Quality Metrics:",
            f"  Average quality score: {self.avg_quality_score:.3f}",
            f"  Average annotation confidence: {self.avg_confidence:.3f}",
            "",
            "Effectiveness Distribution:",
        ]

        for level, count in self.effectiveness_distribution.items():
            percentage = (count / max(1, self.num_annotations)) * 100
            summary_lines.append(f"  {level}: {count} ({percentage:.1f}%)")

        summary_lines.extend(
            [
                "",
                "Top Concepts Covered:",
            ]
        )

        top_concepts = sorted(self.concept_coverage.items(), key=lambda x: x[1], reverse=True)[:5]
        for concept, count in top_concepts:
            summary_lines.append(f"  {concept}: {count}")

        summary_lines.extend(
            [
                "",
                "Span Label Distribution:",
            ]
        )

        for label, count in sorted(self.span_label_distribution.items()):
            summary_lines.append(f"  {label}: {count}")

        if self.missing_annotations:
            summary_lines.extend(
                [
                    "",
                    f"⚠️  Warning: {len(self.missing_annotations)} transcripts missing annotations:",
                ]
            )
            for session_id in self.missing_annotations[:5]:
                summary_lines.append(f"  - {session_id}")
            if len(self.missing_annotations) > 5:
                summary_lines.append(f"  ... and {len(self.missing_annotations) - 5} more")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "num_transcripts": self.num_transcripts,
            "num_annotations": self.num_annotations,
            "avg_quality_score": self.avg_quality_score,
            "effectiveness_distribution": self.effectiveness_distribution,
            "concept_coverage": self.concept_coverage,
            "span_label_distribution": self.span_label_distribution,
            "avg_confidence": self.avg_confidence,
            "missing_annotations": self.missing_annotations,
        }

    def save(self, filepath: str) -> None:
        """Save metrics to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Saved metrics to {filepath}")


class Evaluator:
    """Mock evaluator for tutoring benchmark."""

    def __init__(self, verbose: bool = True):
        """Initialize the evaluator.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose

    def evaluate(self, transcripts: str, annotations: str, output_path: str | None = None) -> EvaluationMetrics:
        """Evaluate transcripts with annotations to produce metrics.

        Args:
            transcripts: Path to JSONLines file containing transcripts
            annotations: Path to JSONLines file containing annotations
            output_path: Optional path to save metrics JSON

        Returns:
            EvaluationMetrics object containing computed metrics
        """
        if self.verbose:
            print("📊 Evaluating transcripts and annotations...")

        # Load transcripts
        transcript_data = self._load_jsonlines(transcripts)
        transcript_ids = {
            t.get("metadata", {}).get("session_id", f"transcript_{i}") for i, t in enumerate(transcript_data)
        }

        # Load annotations
        annotation_data = self._load_jsonlines(annotations)
        annotation_map: dict[str, dict[str, Any]] = {}
        for annotation in annotation_data:
            session_id = annotation.get("session_id")
            if isinstance(session_id, str) and session_id:
                annotation_map[session_id] = annotation

        # Compute metrics
        metrics = self._compute_metrics(transcript_ids, annotation_map)

        if output_path:
            metrics.save(output_path)

        if self.verbose:
            print("✅ Evaluation complete!")

        return metrics

    def _load_jsonlines(self, filepath: str) -> list[dict[str, Any]]:
        """Load a JSONLines file."""
        data = []
        try:
            with open(filepath) as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            if self.verbose:
                print(f"⚠️  File not found: {filepath}")
        return data

    def _compute_metrics(self, transcript_ids: set, annotation_map: dict[str, dict[str, Any]]) -> EvaluationMetrics:
        """Compute evaluation metrics."""
        # Basic counts
        num_transcripts = len(transcript_ids)
        num_annotations = len(annotation_map)

        # Find missing annotations
        annotated_ids = set(annotation_map.keys())
        missing_annotations = list(transcript_ids - annotated_ids)

        # Aggregate annotation metrics
        quality_scores = []
        effectiveness_dist = {"low": 0, "medium": 0, "high": 0}
        concept_counts = {}
        span_label_counts = {}
        confidence_scores = []

        for _session_id, annotation in annotation_map.items():
            annot = annotation.get("annotations", {})

            # Transcript-level metrics
            transcript_level = annot.get("transcript_level", {})
            if "quality_score" in transcript_level:
                quality_scores.append(transcript_level["quality_score"])

            effectiveness = transcript_level.get("pedagogical_effectiveness")
            if effectiveness in effectiveness_dist:
                effectiveness_dist[effectiveness] += 1

            concepts = transcript_level.get("concepts_covered", [])
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1

            # Span-level metrics
            span_level = annot.get("span_level", [])
            for span in span_level:
                label = span.get("label")
                if label:
                    span_label_counts[label] = span_label_counts.get(label, 0) + 1

                confidence = span.get("confidence")
                if confidence is not None:
                    confidence_scores.append(confidence)

        # Compute averages
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return EvaluationMetrics(
            num_transcripts=num_transcripts,
            num_annotations=num_annotations,
            avg_quality_score=avg_quality,
            effectiveness_distribution=effectiveness_dist,
            concept_coverage=concept_counts,
            span_label_distribution=span_label_counts,
            avg_confidence=avg_confidence,
            missing_annotations=missing_annotations,
        )
