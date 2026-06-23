"""Adapter shim so synth-students' TraitGenerator can use our ModelClient
without pulling synth-students' src.models (which carries its own cost
tracker, multi-provider switch, batch infra, etc.).

TraitGenerator calls `self.model.call(non_system_messages, system_prompt, **kw)`
where non_system_messages is [{"role": "user", "content": "..."}] and **kw
includes temperature / max_tokens. It returns the model's response text.

The adapter wraps an `annotator.core.client.ModelClient` and exposes the same
`.call(...)` shape. Nothing else from ModelWrapper is needed by traits.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from annotator.core.client import ModelClient


class ModelWrapperAdapter:
    """Thin shim around our ModelClient mimicking synth-students.src.models.ModelWrapper."""

    def __init__(self, client: ModelClient):
        self._client = client
        # synth-students traits.py reads .model_name; expose it for compatibility.
        self.model_name = client.model

    def call(
        self,
        non_system_messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **_kwargs: Any,
    ) -> str:
        """Send a single user-message + system-prompt to the wrapped ModelClient.

        synth-students passes the conversation as a list of role/content dicts.
        Our ModelClient takes a flat prompt + an optional cacheable_prefix.
        We fold the system prompt into a cacheable_prefix and concatenate the
        user messages into the prompt body.
        """
        # Join user messages (typically just one in TraitGenerator's case).
        user_text = "\n\n".join(
            m.get("content", "") for m in (non_system_messages or [])
            if m.get("role") in (None, "user")
        )
        resp = self._client.generate(
            user_text,
            json_mode=False,
            max_tokens=max_tokens or 1024,
            cacheable_prefix=system_prompt or None,
        )
        return resp.text or ""

    def write_cache(self) -> None:
        """No-op: cost-tracking happens elsewhere in our stack."""
        return None


def get_history_str(messages: List[Dict[str, str]]) -> str:
    """Render a list of role/content message dicts as a single string.

    Mirror of synth-students.src.models.get_history_str (which traits.py
    imports but doesn't actually use on the trait-gen path -- imported for
    parity in case downstream code calls it).
    """
    lines: List[str] = []
    for m in messages or []:
        role = (m.get("role") or "").upper()
        content = m.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
