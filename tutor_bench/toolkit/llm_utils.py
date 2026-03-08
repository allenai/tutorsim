"""Shared LLM request parsing, retry, and cost utilities."""

from __future__ import annotations

import json
import time
from typing import Any


def robust_sleep(attempt: int) -> None:
    time.sleep(min(12.0, 1.5 * (2 ** max(0, attempt - 1))))


def extract_json_array(text: str) -> list[dict[str, Any]]:
    body = text.strip()
    if not body:
        return []
    try:
        obj = json.loads(body)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    start = body.find("[")
    end = body.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(body[start : end + 1])
        except Exception:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    return []


def extract_json_object(text: str) -> dict[str, Any]:
    body = text.strip()
    if not body:
        return {}
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(body[start : end + 1])
        except Exception:
            return {}
        if isinstance(obj, dict):
            return obj
    return {}


def provider_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def compute_cost_usd(usage: dict[str, float], provider: str, model: str, pricing: dict[str, Any]) -> dict[str, Any]:
    key = provider_key(provider, model)
    rate = pricing.get(key, {}) if isinstance(pricing, dict) else {}
    input_per_1m = float(rate.get("input_per_1m", 0.0))
    output_per_1m = float(rate.get("output_per_1m", 0.0))
    prompt_tokens = float(usage.get("prompt_tokens", 0.0))
    completion_tokens = float(usage.get("completion_tokens", 0.0))
    input_cost = prompt_tokens / 1_000_000.0 * input_per_1m
    output_cost = completion_tokens / 1_000_000.0 * output_per_1m
    return {
        "pricing_key": key,
        "has_pricing": bool(key in pricing),
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }
