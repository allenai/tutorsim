"""Per-model recommended thinking config for the benchmark tutor.

Each entry returns the kwargs that should be passed to ModelClient.generate()
(and to run_batch / run_sync_entries via the same path) so the model runs at
its provider-recommended sweet spot for the tutoring task.

The matching is prefix-based -- pass the full model id and the first matching
prefix wins. Order matters: longer / more specific prefixes go first.

Why these levels (sources in `docs/plans/2026-06-17-model-configs.md` if
you want the receipts):

- claude-opus-4-8: adaptive + effort=high. Anthropic docs flag `high` as the
  minimum for intelligence-sensitive work and `xhigh` as the coding/agentic
  default. Tutoring is intelligence-sensitive but not coding-heavy -> high.
- claude-sonnet-4-6: adaptive + effort=medium. Docs explicitly call this the
  "default sweet spot for quality vs. cost" for most applications. Sonnet 4.6
  defaults to high which the migration guide flags as "noticeably higher
  latency/tokens" -- set medium explicitly.
- claude-opus-4-6: adaptive + effort=high (annotator infrastructure).
- claude-haiku-4-5: adaptive only; effort is not supported (would 400).
- gpt-5.5*: reasoning_effort=medium. OpenAI's documented balanced default.
- gemini-*-pro / -pro-*: thinking_budget=16384 (pro tier).
- gemini-*-flash / -flash-*: thinking_budget=8192 (flash tier, smaller).

DeepSeek is intentionally absent until the user confirms which variant
(reasoner vs chat) and the exact model id.
"""
from __future__ import annotations

from typing import Any


# Per-model recommended config when the model is used as the TUTOR.
# Longer/more specific prefixes must appear first.
_TUTOR_RECOMMENDED: list[tuple[str, dict[str, Any]]] = [
    # Anthropic Opus 4.8 (newest opus)
    ("claude-opus-4-8", {
        "thinking": True,
        "effort": "high",
    }),
    # Anthropic Opus 4.7
    ("claude-opus-4-7", {
        "thinking": True,
        "effort": "high",
    }),
    # Anthropic Opus 4.6 (also used as annotator default)
    ("claude-opus-4-6", {
        "thinking": True,
        "effort": "high",
    }),
    # Anthropic Sonnet 4.6
    ("claude-sonnet-4-6", {
        "thinking": True,
        "effort": "medium",
    }),
    # Anthropic Haiku 4.5 -- effort is not supported on Haiku and will 400.
    ("claude-haiku-4-5", {
        "thinking": True,
    }),
    # Anthropic Fable 5 -- thinking is always on; no explicit param.
    ("claude-fable-5", {
        "thinking": True,
        "effort": "high",
    }),
    # OpenAI GPT-5.5 + mini variants
    ("gpt-5.5", {
        "thinking": True,
        "reasoning_effort": "medium",
    }),
    # Google Gemini pro tier
    ("gemini-2.5-pro", {
        "thinking": True,
        "thinking_budget": 16384,
    }),
    ("gemini-3.1-pro", {
        "thinking": True,
        "thinking_budget": 16384,
    }),
    ("gemini-3.5-pro", {
        "thinking": True,
        "thinking_budget": 16384,
    }),
    # Google Gemini flash tier
    ("gemini-2.5-flash", {
        "thinking": True,
        "thinking_budget": 8192,
    }),
    ("gemini-3.5-flash", {
        "thinking": True,
        "thinking_budget": 8192,
    }),
]


def tutor_kwargs_for(model_id: str) -> dict[str, Any]:
    """Return the recommended generate(...) kwargs for `model_id` as tutor.

    Falls back to {"thinking": True} (no extra knobs) if the model isn't in
    the table -- safest default for an unknown provider is "thinking on,
    let the model decide depth."
    """
    if not model_id:
        return {"thinking": True}
    for prefix, kwargs in _TUTOR_RECOMMENDED:
        if model_id.startswith(prefix):
            return dict(kwargs)
    return {"thinking": True}


# Per-role config for non-tutor roles. Held fixed across all tutor models so
# the comparison stays apples-to-apples.
STUDENT_KWARGS: dict[str, Any] = {
    "thinking": False,
}
