"""Leaderboard: rank and display tutor model benchmark results."""

import json
from pathlib import Path

from ..core.aggregate import ModelSummary


def build_leaderboard(summaries: list[ModelSummary]) -> list[dict]:
    """Sort models by mean composite score, return ranked list."""
    ranked = sorted(summaries, key=lambda s: s.mean_score, reverse=True)
    leaderboard = []
    for rank, summary in enumerate(ranked, 1):
        entry = summary.to_dict()
        entry["rank"] = rank
        leaderboard.append(entry)
    return leaderboard


def print_leaderboard(leaderboard: list[dict]):
    """Print ASCII table of model rankings."""
    if not leaderboard:
        print("No results to display.")
        return

    # Header
    print()
    print(f"{'Rank':<6} {'Model':<30} {'Score':<8} {'N':<6} "
          f"{'Generous':<10} {'Balanced':<10} {'Demanding':<10}")
    print("-" * 90)

    for entry in leaderboard:
        style = entry.get("style_breakdown", {})
        print(
            f"{entry['rank']:<6} "
            f"{entry['tutor_model']:<30} "
            f"{entry['mean_score']:<8.3f} "
            f"{entry['n_scenarios']:<6} "
            f"{style.get('generous', 0):<10.3f} "
            f"{style.get('balanced', 0):<10.3f} "
            f"{style.get('demanding', 0):<10.3f}"
        )

    print()


def save_leaderboard(leaderboard: list[dict], path: Path):
    """Save leaderboard as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Leaderboard saved to {path}")
