"""Screenshot anchoring and loading helpers.

Screenshots are stored under `deidentified/screenshots/{uuid}/{timestamp}.jpg`
where `timestamp` is the video-timestamp in seconds (filename is the source of
truth). Anchoring a screenshot to a dialogue turn is deterministic: pick the
latest turn whose start_seconds <= screenshot timestamp.
"""

import os
from typing import Iterable

from . import storage


def timestamp_seconds_from_filename(filename: str) -> float:
    """Parse the video timestamp encoded in a screenshot filename.

    >>> timestamp_seconds_from_filename("603.834.jpg")
    603.834
    """
    stem, _ = os.path.splitext(filename)
    return float(stem)


def anchor_screenshots(filenames: Iterable[str], turns: list[dict]) -> list[dict]:
    """Anchor each screenshot to the latest turn with start_seconds <= its timestamp.

    Falls back to the first turn if the screenshot precedes all turns.
    Returns entries sorted by anchor_turn ascending:
        [{"filename", "timestamp_seconds", "anchor_turn"}, ...]
    """
    if not turns:
        return []

    # Turns sorted by start_seconds for deterministic anchoring
    sorted_turns = sorted(turns, key=lambda t: t.get("start_seconds", 0.0))

    out = []
    for fname in filenames:
        ts = timestamp_seconds_from_filename(fname)
        # Find latest turn whose start_seconds <= ts
        chosen = sorted_turns[0]
        for t in sorted_turns:
            if t.get("start_seconds", 0.0) <= ts:
                chosen = t
            else:
                break
        out.append({
            "filename": fname,
            "timestamp_seconds": ts,
            "anchor_turn": chosen["turn_number"],
        })

    out.sort(key=lambda r: (r["anchor_turn"], r["timestamp_seconds"]))
    return out


def load_anchored_screenshots(conv_id: str, turns: list[dict]) -> list[dict]:
    """Load, filter (flagged / eedi_ip), and anchor screenshots for a conv.

    Returns list of dicts with:
      - filename
      - timestamp_seconds
      - anchor_turn
      - storage_path (relative to storage backend root)
    """
    filenames = storage.list_screenshots(conv_id)
    if not filenames:
        return []

    verification = storage.load_screenshot_verification(conv_id)
    flagged = {
        f for f, m in verification.get("images", {}).items()
        if m.get("flagged") or m.get("eedi_ip")
    }
    usable = [f for f in filenames if f not in flagged]

    anchored = anchor_screenshots(usable, turns)
    for row in anchored:
        row["storage_path"] = storage._screenshot_rel_path(conv_id, row["filename"])

    return anchored
