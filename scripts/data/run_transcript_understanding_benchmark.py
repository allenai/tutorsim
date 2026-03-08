#!/usr/bin/env python3
"""Plan 012: Screenshot-conditioned transcript understanding benchmark.

Pipeline:
1) Build selected moments from 010 merged_union.
2) Build 1:1 matched control moments from non-selected transcript timepoints.
3) Generate image-only dense captions for selected screenshots with 3 providers.
4) Generate MCQs (2 comprehend + 2 predict, 4 choices each) with iteration tracking.
5) Run solver matrix:
   - selected_text
   - selected_caption_{caption_provider}
   - control_text
6) Score exact-match MCQ accuracy (no judge model).
7) Iterate difficult-question regeneration for easy selected moments.
8) Emit JSON + Markdown summary with rank-order pass rates:
   selected+caption > selected-text > control-text
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib import parse, request

from tutor_bench.toolkit.io_utils import read_stems_file, save_jsonl
from tutor_bench.toolkit.llm_utils import (
    compute_cost_usd,
    extract_json_array,
    extract_json_object,
    robust_sleep,
)
from tutor_bench.toolkit.prompts import (
    PROMPT_IDS,
    build_dense_caption_prompt,
    build_qa_author_prompt,
    build_qa_solver_prompt,
)
from tutor_bench.toolkit.time_utils import fmt_ts
from tutor_bench.toolkit.transcript_utils import Segment, context_prefix, parse_gold_transcript

PROVIDERS = ["openai", "anthropic", "gemini"]

# Hardcoded premium models with fixed alias fallback lists.
MODEL_PREFS = {
    "openai": ["gpt-5.3", "gpt-5.2", "gpt-5.1"],
    "anthropic": ["claude-opus-4.6", "claude-opus-4-1", "claude-sonnet-4-6"],
    "gemini": ["gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-3-flash-preview"],
}

AUTHOR_PROVIDER = "openai"
DEFAULT_PRICING_CONFIG = Path("configs/llm_pricing.json")


def http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def tokenize_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def read_stems(path: Path, max_stems: int) -> list[str]:
    return read_stems_file(path, max_stems=max_stems)


def parse_transcript(path: Path) -> list[Segment]:
    return parse_gold_transcript(path)


def adaptive_future_turns(segments: list[Segment], center_t: float) -> list[Segment]:
    future = [s for s in segments if s.start > center_t]
    if not future:
        return []
    picked: list[Segment] = []
    word_target = 45
    total_words = 0
    for s in future:
        picked.append(s)
        total_words += tokenize_count(s.text)
        if len(picked) >= 6:
            break
        if len(picked) >= 3 and total_words >= word_target:
            break
    return picked


def nearest_point_for_t(points: list[dict[str, Any]], t: float, max_delta: float = 1.5) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for p in points:
        ts = p.get("timestamp_seconds")
        if not isinstance(ts, int | float):
            continue
        d = abs(float(ts) - t)
        if best is None or d < best[0]:
            best = (d, p)
    if best is None or best[0] > max_delta:
        return None
    return best[1]


def load_selected_moments(stems: list[str], manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for stem in stems:
        p = manifest_dir / f"{stem}_moments.json"
        if not p.exists():
            out[stem] = []
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        moments = []
        for i, m in enumerate(data.get("moments", [])):
            t = m.get("t")
            if not isinstance(t, int | float) or not math.isfinite(float(t)):
                continue
            moments.append(
                {
                    "moment_idx": i,
                    "t": float(t),
                    "timestamp": m.get("timestamp", fmt_ts(float(t))),
                    "reason": str(m.get("reason", "")),
                    "tags": list(m.get("tags", [])) if isinstance(m.get("tags"), list) else [],
                }
            )
        out[stem] = sorted(moments, key=lambda x: x["t"])
    return out


def load_points(stem: str, diag_dir: Path) -> list[dict[str, Any]]:
    p = diag_dir / stem / "_points.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [x for x in data if isinstance(x, dict)]


def choose_control_timestamps(
    selected_ts: list[float], segments: list[Segment], min_sep_seconds: float, seed: int
) -> list[float]:
    rnd = random.Random(seed)
    mids = sorted(0.5 * (s.start + s.end) for s in segments)
    chosen: list[float] = []
    selected_sorted = sorted(selected_ts)

    def is_far_from_selected(x: float) -> bool:
        for t in selected_sorted:
            if abs(x - t) <= min_sep_seconds:
                return False
        return True

    for t in selected_sorted:
        cands = [x for x in mids if is_far_from_selected(x) and all(abs(x - y) > 8.0 for y in chosen)]
        if not cands:
            chosen.append(t + min_sep_seconds + 5.0)
            continue
        # Prefer nearby points to keep topical matching.
        cands.sort(key=lambda x: abs(x - t))
        head = cands[: min(30, len(cands))]
        picked = rnd.choice(head)
        chosen.append(picked)
    return chosen


def provider_api_key(provider: str) -> str:
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    return ""


def call_openai_text(model: str, prompt: str, timeout_seconds: int) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("openai")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0.0,
        "max_output_tokens": 2500,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = http_post_json("https://api.openai.com/v1/responses", headers, payload, timeout_seconds)
    if isinstance(resp.get("error"), dict):
        raise RuntimeError(resp["error"].get("message", "openai error"))
    text = ""
    for item in resp.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in {"output_text", "text"}:
                    text = c.get("text", "")
                    break
        if text:
            break
    usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
    usage_out = {
        "prompt_tokens": float(usage.get("input_tokens", 0.0)),
        "completion_tokens": float(usage.get("output_tokens", 0.0)),
        "total_tokens": float(usage.get("total_tokens", 0.0)),
    }
    return text, usage_out, resp


def call_openai_image(
    model: str, prompt: str, image_path: Path, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("openai")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ],
        "temperature": 0.0,
        "max_output_tokens": 1500,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = http_post_json("https://api.openai.com/v1/responses", headers, payload, timeout_seconds)
    if isinstance(resp.get("error"), dict):
        raise RuntimeError(resp["error"].get("message", "openai error"))
    text = ""
    for item in resp.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in {"output_text", "text"}:
                    text = c.get("text", "")
                    break
        if text:
            break
    usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
    usage_out = {
        "prompt_tokens": float(usage.get("input_tokens", 0.0)),
        "completion_tokens": float(usage.get("output_tokens", 0.0)),
        "total_tokens": float(usage.get("total_tokens", 0.0)),
    }
    return text, usage_out, resp


def call_anthropic_text(model: str, prompt: str, timeout_seconds: int) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("anthropic")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    payload = {
        "model": model,
        "max_tokens": 2500,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    resp = http_post_json("https://api.anthropic.com/v1/messages", headers, payload, timeout_seconds)
    if resp.get("type") == "error" or isinstance(resp.get("error"), dict):
        err = resp.get("error", {}) if isinstance(resp.get("error"), dict) else {}
        raise RuntimeError(err.get("message", "anthropic error"))
    text = ""
    for c in resp.get("content", []) or []:
        if isinstance(c, dict) and c.get("type") == "text":
            text = c.get("text", "")
            break
    usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
    in_toks = float(usage.get("input_tokens", 0.0))
    out_toks = float(usage.get("output_tokens", 0.0))
    usage_out = {
        "prompt_tokens": in_toks,
        "completion_tokens": out_toks,
        "total_tokens": in_toks + out_toks,
    }
    return text, usage_out, resp


def call_anthropic_image(
    model: str, prompt: str, image_path: Path, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("anthropic")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "max_tokens": 1800,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                ],
            }
        ],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    resp = http_post_json("https://api.anthropic.com/v1/messages", headers, payload, timeout_seconds)
    if resp.get("type") == "error" or isinstance(resp.get("error"), dict):
        err = resp.get("error", {}) if isinstance(resp.get("error"), dict) else {}
        raise RuntimeError(err.get("message", "anthropic error"))
    text = ""
    for c in resp.get("content", []) or []:
        if isinstance(c, dict) and c.get("type") == "text":
            text = c.get("text", "")
            break
    usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
    in_toks = float(usage.get("input_tokens", 0.0))
    out_toks = float(usage.get("output_tokens", 0.0))
    usage_out = {
        "prompt_tokens": in_toks,
        "completion_tokens": out_toks,
        "total_tokens": in_toks + out_toks,
    }
    return text, usage_out, resp


def call_gemini_text(model: str, prompt: str, timeout_seconds: int) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("gemini")
    if not key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2500,
            "responseMimeType": "application/json",
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{parse.quote(model)}"
        f":generateContent?key={parse.quote(key)}"
    )
    headers = {"Content-Type": "application/json"}
    resp = http_post_json(url, headers, payload, timeout_seconds)
    if isinstance(resp.get("error"), dict):
        raise RuntimeError(resp["error"].get("message", "gemini error"))
    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = ""
    usage = resp.get("usageMetadata", {}) if isinstance(resp.get("usageMetadata"), dict) else {}
    usage_out = {
        "prompt_tokens": float(usage.get("promptTokenCount", 0.0)),
        "completion_tokens": float(usage.get("candidatesTokenCount", 0.0)),
        "total_tokens": float(usage.get("totalTokenCount", 0.0)),
    }
    return text, usage_out, resp


def call_gemini_image(
    model: str, prompt: str, image_path: Path, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = provider_api_key("gemini")
    if not key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1800,
            "responseMimeType": "application/json",
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{parse.quote(model)}"
        f":generateContent?key={parse.quote(key)}"
    )
    headers = {"Content-Type": "application/json"}
    resp = http_post_json(url, headers, payload, timeout_seconds)
    if isinstance(resp.get("error"), dict):
        raise RuntimeError(resp["error"].get("message", "gemini error"))
    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = ""
    usage = resp.get("usageMetadata", {}) if isinstance(resp.get("usageMetadata"), dict) else {}
    usage_out = {
        "prompt_tokens": float(usage.get("promptTokenCount", 0.0)),
        "completion_tokens": float(usage.get("candidatesTokenCount", 0.0)),
        "total_tokens": float(usage.get("totalTokenCount", 0.0)),
    }
    return text, usage_out, resp


def call_with_model_fallback(
    provider: str,
    mode: str,
    prompt: str,
    timeout_seconds: int,
    image_path: Path | None = None,
) -> tuple[str, dict[str, float], dict[str, Any], str]:
    errs: list[str] = []
    models = MODEL_PREFS[provider]
    for mi, model in enumerate(models):
        try:
            if provider == "openai":
                if mode == "text":
                    text, usage, raw = call_openai_text(model, prompt, timeout_seconds)
                else:
                    assert image_path is not None
                    text, usage, raw = call_openai_image(model, prompt, image_path, timeout_seconds)
            elif provider == "anthropic":
                if mode == "text":
                    text, usage, raw = call_anthropic_text(model, prompt, timeout_seconds)
                else:
                    assert image_path is not None
                    text, usage, raw = call_anthropic_image(model, prompt, image_path, timeout_seconds)
            else:
                if mode == "text":
                    text, usage, raw = call_gemini_text(model, prompt, timeout_seconds)
                else:
                    assert image_path is not None
                    text, usage, raw = call_gemini_image(model, prompt, image_path, timeout_seconds)
            return text, usage, raw, model
        except Exception as e:  # noqa: BLE001
            errs.append(f"{model}: {e}")
            if mi < len(models) - 1:
                robust_sleep(mi + 1)
    raise RuntimeError(f"{provider} all model aliases failed: {' | '.join(errs)}")


def call_cached(
    cache_path: Path,
    provider: str,
    mode: str,
    prompt: str,
    timeout_seconds: int,
    image_path: Path | None = None,
) -> tuple[str, dict[str, float], dict[str, Any], str, bool]:
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        usage_obj = obj.get("usage", {}) if isinstance(obj.get("usage"), dict) else {}
        raw_obj = obj.get("raw", {}) if isinstance(obj.get("raw"), dict) else {}
        usage = {
            "prompt_tokens": float(usage_obj.get("prompt_tokens", 0.0)),
            "completion_tokens": float(usage_obj.get("completion_tokens", 0.0)),
            "total_tokens": float(usage_obj.get("total_tokens", 0.0)),
        }
        if usage["total_tokens"] <= 0.0:
            if provider == "openai":
                u = raw_obj.get("usage", {}) if isinstance(raw_obj.get("usage"), dict) else {}
                usage = {
                    "prompt_tokens": float(u.get("input_tokens", 0.0)),
                    "completion_tokens": float(u.get("output_tokens", 0.0)),
                    "total_tokens": float(u.get("total_tokens", 0.0)),
                }
            elif provider == "anthropic":
                u = raw_obj.get("usage", {}) if isinstance(raw_obj.get("usage"), dict) else {}
                in_toks = float(u.get("input_tokens", 0.0))
                out_toks = float(u.get("output_tokens", 0.0))
                usage = {
                    "prompt_tokens": in_toks,
                    "completion_tokens": out_toks,
                    "total_tokens": in_toks + out_toks,
                }
            elif provider == "gemini":
                u = raw_obj.get("usageMetadata", {}) if isinstance(raw_obj.get("usageMetadata"), dict) else {}
                usage = {
                    "prompt_tokens": float(u.get("promptTokenCount", 0.0)),
                    "completion_tokens": float(u.get("candidatesTokenCount", 0.0)),
                    "total_tokens": float(u.get("totalTokenCount", 0.0)),
                }
        return str(obj.get("text", "")), usage, raw_obj, str(obj.get("model", "")), True
    text, usage, raw, used_model = call_with_model_fallback(
        provider=provider,
        mode=mode,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        image_path=image_path,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"text": text, "usage": usage, "raw": raw, "model": used_model}, indent=2) + "\n",
        encoding="utf-8",
    )
    return text, usage, raw, used_model, False


def parse_answer_index(text: str) -> int | None:
    obj = extract_json_object(text)
    idx = obj.get("answer_index")
    if isinstance(idx, int) and 0 <= idx <= 3:
        return idx
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return int(m.group(1))
    return None


def validate_mcqs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        task = str(it.get("task_type", "")).strip().lower()
        if task not in {"comprehend", "predict"}:
            continue
        q = str(it.get("question", "")).strip()
        ch = it.get("choices", [])
        ai = it.get("answer_index")
        if not q or not isinstance(ch, list) or len(ch) != 4:
            continue
        if not isinstance(ai, int) or not (0 <= ai <= 3):
            continue
        out.append(
            {
                "task_type": task,
                "question": q,
                "choices": [str(x).strip() for x in ch],
                "answer_index": ai,
                "requires_visual": bool(it.get("requires_visual", False)),
                "difficulty_tag": str(it.get("difficulty_tag", "")).strip()[:80],
            }
        )
    return out


def md_escape(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Plan 012 transcript understanding benchmark.")
    p.add_argument("--stems-file", type=Path, default=Path("configs/visual_selection_fixed10.txt"))
    p.add_argument("--max-stems", type=int, default=0)
    p.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/stepup/transcripts/2_13/Transcripts"),
    )
    p.add_argument(
        "--selected-manifest-dir",
        type=Path,
        default=Path("output/llm_moment_selection_010/merged_union"),
    )
    p.add_argument(
        "--selected-diag-dir",
        type=Path,
        default=Path("output/screenshot_diagnostic_010_llm/merged_union"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output/transcript_understanding_012"))
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--pricing-config", type=Path, default=DEFAULT_PRICING_CONFIG)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--providers",
        type=str,
        default="openai,anthropic,gemini",
        help="Comma-separated provider subset from: openai,anthropic,gemini",
    )
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--easy-threshold", type=float, default=0.85)
    p.add_argument("--easy-stop-ratio", type=float, default=0.15)
    p.add_argument("--max-iterations", type=int, default=4)
    p.add_argument("--min-control-sep-seconds", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    active_providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    if not active_providers:
        raise SystemExit("no providers configured; pass --providers")
    bad = [p for p in active_providers if p not in PROVIDERS]
    if bad:
        raise SystemExit(f"invalid providers: {', '.join(bad)}")

    if not args.stems_file.exists():
        raise SystemExit(f"stems file not found: {args.stems_file}")
    if not args.transcripts_dir.is_dir():
        raise SystemExit(f"transcripts dir not found: {args.transcripts_dir}")
    if not args.selected_manifest_dir.is_dir():
        raise SystemExit(f"selected manifest dir not found: {args.selected_manifest_dir}")
    if not args.selected_diag_dir.is_dir():
        raise SystemExit(f"selected diagnostic dir not found: {args.selected_diag_dir}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (out / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pricing: dict[str, Any] = {}
    if args.pricing_config.is_file():
        pricing = json.loads(args.pricing_config.read_text(encoding="utf-8"))

    provider_cost_rows: dict[str, dict[str, Any]] = {
        p: {
            "provider": p,
            "calls": 0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "estimated_cost_usd": 0.0,
            "executed_cost_usd": 0.0,
            "priced_calls": 0,
            "unpriced_calls": 0,
            "models": set(),
            "missing_pricing_models": set(),
        }
        for p in active_providers
    }
    stem_provider_costs: dict[tuple[str, str], dict[str, Any]] = {}

    def record_call_cost(
        *,
        stem: str,
        provider: str,
        stage: str,
        usage: dict[str, float],
        model: str,
        from_cache: bool,
    ) -> dict[str, Any]:
        if not model:
            return {}
        cost = compute_cost_usd(usage, provider, model, pricing)
        row = provider_cost_rows[provider]
        row["calls"] += 1
        row["prompt_tokens"] += float(usage.get("prompt_tokens", 0.0))
        row["completion_tokens"] += float(usage.get("completion_tokens", 0.0))
        row["total_tokens"] += float(usage.get("total_tokens", 0.0))
        row["estimated_cost_usd"] += float(cost["total_cost_usd"])
        row["models"].add(model)
        if cost["has_pricing"]:
            row["priced_calls"] += 1
        else:
            row["unpriced_calls"] += 1
            row["missing_pricing_models"].add(model)
        if not from_cache:
            row["executed_cost_usd"] += float(cost["total_cost_usd"])

        key = (stem, provider)
        srow = stem_provider_costs.get(key)
        if srow is None:
            srow = {
                "stem": stem,
                "provider": provider,
                "calls": 0,
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "estimated_cost_usd": 0.0,
                "executed_cost_usd": 0.0,
                "stages": {},
            }
            stem_provider_costs[key] = srow
        srow["calls"] += 1
        srow["prompt_tokens"] += float(usage.get("prompt_tokens", 0.0))
        srow["completion_tokens"] += float(usage.get("completion_tokens", 0.0))
        srow["total_tokens"] += float(usage.get("total_tokens", 0.0))
        srow["estimated_cost_usd"] += float(cost["total_cost_usd"])
        if not from_cache:
            srow["executed_cost_usd"] += float(cost["total_cost_usd"])
        stage_row = srow["stages"].setdefault(
            stage,
            {
                "calls": 0,
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "estimated_cost_usd": 0.0,
            },
        )
        stage_row["calls"] += 1
        stage_row["prompt_tokens"] += float(usage.get("prompt_tokens", 0.0))
        stage_row["completion_tokens"] += float(usage.get("completion_tokens", 0.0))
        stage_row["total_tokens"] += float(usage.get("total_tokens", 0.0))
        stage_row["estimated_cost_usd"] += float(cost["total_cost_usd"])
        return cost

    stems = read_stems(args.stems_file, args.max_stems)
    selected_by_stem = load_selected_moments(stems, args.selected_manifest_dir)

    all_moments: list[dict[str, Any]] = []
    selected_moments: list[dict[str, Any]] = []
    control_moments: list[dict[str, Any]] = []
    selected_to_control: dict[str, str] = {}

    for si, stem in enumerate(stems):
        t_path = args.transcripts_dir / f"{stem}_transcript.txt"
        if not t_path.exists():
            print(f"SKIP stem missing transcript: {stem}")
            continue
        segs = parse_transcript(t_path)
        if not segs:
            print(f"SKIP stem empty transcript: {stem}")
            continue
        points = load_points(stem, args.selected_diag_dir)
        sel = selected_by_stem.get(stem, [])
        sel_ts = [m["t"] for m in sel]
        ctr_ts = choose_control_timestamps(sel_ts, segs, args.min_control_sep_seconds, args.seed + si)

        for mi, m in enumerate(sel):
            t = float(m["t"])
            point = nearest_point_for_t(points, t)
            fut = adaptive_future_turns(segs, t)
            moment_id = f"sel::{stem}::{mi:04d}"
            row = {
                "moment_id": moment_id,
                "stem": stem,
                "moment_type": "selected",
                "pair_idx": mi,
                "timestamp_seconds": round(t, 3),
                "timestamp": fmt_ts(t),
                "source_reason": m.get("reason", ""),
                "source_tags": m.get("tags", []),
                "screenshot_path": point.get("screenshot_path") if point else None,
                "clip_path": point.get("clip_path") if point else None,
                # Keep context strictly historical: [-90s, t].
                "transcript_window_text": context_prefix(segs, t, pre_s=90.0),
                "prediction_context_text": context_prefix(segs, t, pre_s=90.0),
                "future_turns_text": "\n".join(
                    f"[{fmt_ts(x.start)} - {fmt_ts(x.end)}] {x.role}: {x.text}" for x in fut
                ),
            }
            selected_moments.append(row)
            all_moments.append(row)

            # Paired control.
            if mi < len(ctr_ts):
                tc = float(ctr_ts[mi])
                futc = adaptive_future_turns(segs, tc)
                cid = f"ctl::{stem}::{mi:04d}"
                crow = {
                    "moment_id": cid,
                    "stem": stem,
                    "moment_type": "control",
                    "pair_idx": mi,
                    "timestamp_seconds": round(tc, 3),
                    "timestamp": fmt_ts(tc),
                    "source_reason": "",
                    "source_tags": [],
                    "screenshot_path": None,
                    "clip_path": None,
                    # Keep context strictly historical: [-90s, t].
                    "transcript_window_text": context_prefix(segs, tc, pre_s=90.0),
                    "prediction_context_text": context_prefix(segs, tc, pre_s=90.0),
                    "future_turns_text": "\n".join(
                        f"[{fmt_ts(x.start)} - {fmt_ts(x.end)}] {x.role}: {x.text}" for x in futc
                    ),
                }
                control_moments.append(crow)
                all_moments.append(crow)
                selected_to_control[moment_id] = cid

    save_jsonl(out / "moments" / "selected_moments.jsonl", selected_moments)
    save_jsonl(out / "moments" / "control_moments.jsonl", control_moments)

    # Caption generation (selected moments only, image-only).
    caption_rows: list[dict[str, Any]] = []
    for provider in active_providers:
        p_dir = out / "captions" / provider
        p_dir.mkdir(parents=True, exist_ok=True)
        for m in selected_moments:
            shot = m.get("screenshot_path")
            rec = {
                "moment_id": m["moment_id"],
                "provider": provider,
                "caption_text": "",
                "model": "",
                "caption_prompt_id": PROMPT_IDS["dense_caption"],
                "error": "",
            }
            if not shot:
                rec["error"] = "missing_screenshot_path"
                caption_rows.append(rec)
                continue
            image_path = Path(shot)
            if not image_path.exists():
                rec["error"] = "screenshot_not_found"
                caption_rows.append(rec)
                continue
            cp = p_dir / f"{m['moment_id'].replace(':', '__')}.json"
            try:
                text, usage, raw, used_model, from_cache = call_cached(
                    cache_path=cache_dir / "captions" / provider / cp.name,
                    provider=provider,
                    mode="image",
                    prompt=build_dense_caption_prompt(),
                    timeout_seconds=args.timeout_seconds,
                    image_path=image_path,
                )
                record_call_cost(
                    stem=str(m["stem"]),
                    provider=provider,
                    stage="captions_image",
                    usage=usage,
                    model=used_model,
                    from_cache=from_cache,
                )
                obj = extract_json_object(text)
                caption_text = str(obj.get("dense_caption", "")).strip() or str(text).strip()
                rec["caption_text"] = caption_text
                rec["model"] = used_model
                cp.write_text(
                    json.dumps(
                        {
                            "moment_id": m["moment_id"],
                            "provider": provider,
                            "model": used_model,
                            "caption_text": caption_text,
                            "raw": raw,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)
            caption_rows.append(rec)
    save_jsonl(out / "captions" / "captions.jsonl", caption_rows)
    cap_map: dict[tuple[str, str], str] = {}
    for r in caption_rows:
        if r.get("caption_text"):
            cap_map[(str(r["moment_id"]), str(r["provider"]))] = str(r["caption_text"])

    # Moment lookup.
    moment_by_id = {m["moment_id"]: m for m in all_moments}

    # Iterative question generation + solver runs.
    easy_ids_prev: set[str] = set()
    final_questions: list[dict[str, Any]] = []
    final_responses: list[dict[str, Any]] = []
    iteration_meta: list[dict[str, Any]] = []

    for it in range(args.max_iterations):
        iter_name = f"iter_{it:02d}"
        iter_dir = out / "questions" / iter_name
        iter_dir.mkdir(parents=True, exist_ok=True)
        prev_questions = final_questions if it > 0 else []
        prev_by_moment: dict[str, list[dict[str, Any]]] = {}
        for q in prev_questions:
            prev_by_moment.setdefault(str(q["moment_id"]), []).append(q)

        regen_ids: set[str]
        if it == 0:
            regen_ids = {m["moment_id"] for m in selected_moments + control_moments}
        else:
            regen_ids = set()
            for sid in easy_ids_prev:
                regen_ids.add(sid)
                cid = selected_to_control.get(sid)
                if cid:
                    regen_ids.add(cid)

        questions: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []

        # Carry forward unchanged moments.
        if it > 0:
            for mid, rows in prev_by_moment.items():
                if mid in regen_ids:
                    continue
                for r in rows:
                    questions.append(r)
                    lineage.append(
                        {
                            "question_id": r["question_id"],
                            "moment_id": mid,
                            "iteration": it,
                            "parent_question_id": r.get("question_id"),
                            "action": "carried_forward",
                        }
                    )

        # Generate/re-generate MCQs.
        for mid in sorted(regen_ids):
            m = moment_by_id[mid]
            caption_for_author = None
            if m["moment_type"] == "selected":
                caption_for_author = cap_map.get((mid, AUTHOR_PROVIDER))
            prompt = build_qa_author_prompt(m, caption_for_author)
            q_items: list[dict[str, Any]] = []
            used_model = ""
            err_msg = ""
            for attempt in range(1, args.max_retries + 1):
                try:
                    text, usage, raw, used_model, from_cache = call_cached(
                        cache_path=cache_dir / "questions" / iter_name / f"{mid.replace(':', '__')}.json",
                        provider=AUTHOR_PROVIDER,
                        mode="text",
                        prompt=prompt,
                        timeout_seconds=args.timeout_seconds,
                        image_path=None,
                    )
                    record_call_cost(
                        stem=str(m["stem"]),
                        provider=AUTHOR_PROVIDER,
                        stage="questions_authoring",
                        usage=usage,
                        model=used_model,
                        from_cache=from_cache,
                    )
                    parsed = validate_mcqs(extract_json_array(text))
                    c_count = sum(1 for x in parsed if x["task_type"] == "comprehend")
                    p_count = sum(1 for x in parsed if x["task_type"] == "predict")
                    if len(parsed) == 4 and c_count == 2 and p_count == 2:
                        q_items = parsed
                        break
                    err_msg = f"invalid_shape len={len(parsed)} c={c_count} p={p_count}"
                except Exception as e:  # noqa: BLE001
                    err_msg = str(e)
                if attempt < args.max_retries:
                    robust_sleep(attempt)

            if not q_items:
                # deterministic fallback trivial template to keep pipeline moving
                q_items = [
                    {
                        "task_type": "comprehend",
                        "question": "What is the tutor/student discussing in this context?",
                        "choices": ["math problem setup", "weather", "sports", "cooking"],
                        "answer_index": 0,
                        "requires_visual": False,
                        "difficulty_tag": "fallback",
                    },
                    {
                        "task_type": "comprehend",
                        "question": "What is the likely instructional focus here?",
                        "choices": ["solving or explaining a problem", "telling a joke", "music lesson", "travel plan"],
                        "answer_index": 0,
                        "requires_visual": False,
                        "difficulty_tag": "fallback",
                    },
                    {
                        "task_type": "predict",
                        "question": "What is most likely to happen in the next turns?",
                        "choices": [
                            "continued tutoring dialogue",
                            "session ends immediately",
                            "topic changes to sports",
                            "silence only",
                        ],
                        "answer_index": 0,
                        "requires_visual": False,
                        "difficulty_tag": "fallback",
                    },
                    {
                        "task_type": "predict",
                        "question": "What is the most plausible next interaction?",
                        "choices": [
                            "follow-up explanation or response",
                            "movie discussion",
                            "no response",
                            "switch to unrelated story",
                        ],
                        "answer_index": 0,
                        "requires_visual": False,
                        "difficulty_tag": "fallback",
                    },
                ]

            prev_rows = prev_by_moment.get(mid, [])
            parent_ids = [str(x["question_id"]) for x in prev_rows]
            for qi, q in enumerate(q_items):
                qid = f"{mid}::it{it:02d}::q{qi:02d}"
                row = {
                    "question_id": qid,
                    "moment_id": mid,
                    "iteration": it,
                    "task_type": q["task_type"],
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer_index": int(q["answer_index"]),
                    "requires_visual": bool(q["requires_visual"]),
                    "difficulty_tag": q.get("difficulty_tag", ""),
                    "author_provider": AUTHOR_PROVIDER,
                    "author_model": used_model,
                    "author_prompt_id": PROMPT_IDS["qa_author"],
                    "author_error": err_msg,
                }
                questions.append(row)
                lineage.append(
                    {
                        "question_id": qid,
                        "moment_id": mid,
                        "iteration": it,
                        "parent_question_id": parent_ids[qi] if qi < len(parent_ids) else None,
                        "action": "generated" if it == 0 else "regenerated",
                    }
                )

        questions.sort(key=lambda x: x["question_id"])
        save_jsonl(iter_dir / "questions.jsonl", questions)
        (iter_dir / "question_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

        # Solver runs for this iteration.
        responses: list[dict[str, Any]] = []
        selected_qs = [q for q in questions if moment_by_id[q["moment_id"]]["moment_type"] == "selected"]
        control_qs = [q for q in questions if moment_by_id[q["moment_id"]]["moment_type"] == "control"]

        for solver in active_providers:
            # selected_text
            for q in selected_qs:
                m = moment_by_id[q["moment_id"]]
                cond = "selected_text"
                prompt = build_qa_solver_prompt(q, m, caption_text=None)
                ck = cache_dir / "solve" / iter_name / solver / cond / f"{q['question_id'].replace(':', '__')}.json"
                pred = None
                used_model = ""
                err = ""
                try:
                    text, usage, raw, used_model, from_cache = call_cached(
                        cache_path=ck,
                        provider=solver,
                        mode="text",
                        prompt=prompt,
                        timeout_seconds=args.timeout_seconds,
                    )
                    record_call_cost(
                        stem=str(m["stem"]),
                        provider=solver,
                        stage="solve_selected_text",
                        usage=usage,
                        model=used_model,
                        from_cache=from_cache,
                    )
                    pred = parse_answer_index(text)
                except Exception as e:  # noqa: BLE001
                    raw = {}
                    err = str(e)
                responses.append(
                    {
                        "iteration": it,
                        "question_id": q["question_id"],
                        "moment_id": q["moment_id"],
                        "solver_provider": solver,
                        "solver_model": used_model,
                        "condition": cond,
                        "caption_provider": None,
                        "solver_prompt_id": PROMPT_IDS["qa_solver"],
                        "predicted_index": pred,
                        "answer_index": q["answer_index"],
                        "is_correct": bool(pred == q["answer_index"]) if pred is not None else False,
                        "task_type": q["task_type"],
                        "error": err,
                    }
                )

            # control_text
            for q in control_qs:
                m = moment_by_id[q["moment_id"]]
                cond = "control_text"
                prompt = build_qa_solver_prompt(q, m, caption_text=None)
                ck = cache_dir / "solve" / iter_name / solver / cond / f"{q['question_id'].replace(':', '__')}.json"
                pred = None
                used_model = ""
                err = ""
                try:
                    text, usage, raw, used_model, from_cache = call_cached(
                        cache_path=ck,
                        provider=solver,
                        mode="text",
                        prompt=prompt,
                        timeout_seconds=args.timeout_seconds,
                    )
                    record_call_cost(
                        stem=str(m["stem"]),
                        provider=solver,
                        stage="solve_control_text",
                        usage=usage,
                        model=used_model,
                        from_cache=from_cache,
                    )
                    pred = parse_answer_index(text)
                except Exception as e:  # noqa: BLE001
                    raw = {}
                    err = str(e)
                responses.append(
                    {
                        "iteration": it,
                        "question_id": q["question_id"],
                        "moment_id": q["moment_id"],
                        "solver_provider": solver,
                        "solver_model": used_model,
                        "condition": cond,
                        "caption_provider": None,
                        "solver_prompt_id": PROMPT_IDS["qa_solver"],
                        "predicted_index": pred,
                        "answer_index": q["answer_index"],
                        "is_correct": bool(pred == q["answer_index"]) if pred is not None else False,
                        "task_type": q["task_type"],
                        "error": err,
                    }
                )

            # selected_caption_<provider>
            for cap_provider in active_providers:
                cond = f"selected_caption_{cap_provider}"
                for q in selected_qs:
                    m = moment_by_id[q["moment_id"]]
                    cap_text = cap_map.get((q["moment_id"], cap_provider), "")
                    prompt = build_qa_solver_prompt(q, m, caption_text=cap_text)
                    ck = cache_dir / "solve" / iter_name / solver / cond / f"{q['question_id'].replace(':', '__')}.json"
                    pred = None
                    used_model = ""
                    err = ""
                    try:
                        text, usage, raw, used_model, from_cache = call_cached(
                            cache_path=ck,
                            provider=solver,
                            mode="text",
                            prompt=prompt,
                            timeout_seconds=args.timeout_seconds,
                        )
                        record_call_cost(
                            stem=str(m["stem"]),
                            provider=solver,
                            stage=f"solve_selected_caption_{cap_provider}",
                            usage=usage,
                            model=used_model,
                            from_cache=from_cache,
                        )
                        pred = parse_answer_index(text)
                    except Exception as e:  # noqa: BLE001
                        raw = {}
                        err = str(e)
                    responses.append(
                        {
                            "iteration": it,
                            "question_id": q["question_id"],
                            "moment_id": q["moment_id"],
                            "solver_provider": solver,
                            "solver_model": used_model,
                            "condition": cond,
                            "caption_provider": cap_provider,
                            "solver_prompt_id": PROMPT_IDS["qa_solver"],
                            "predicted_index": pred,
                            "answer_index": q["answer_index"],
                            "is_correct": bool(pred == q["answer_index"]) if pred is not None else False,
                            "task_type": q["task_type"],
                            "error": err,
                        }
                    )

        save_jsonl(out / "responses" / iter_name / "responses.jsonl", responses)

        # Compute easy moments from selected_text (avg across solvers).
        sel_text = [r for r in responses if r["condition"] == "selected_text"]
        by_mid: dict[str, list[float]] = {}
        for r in sel_text:
            by_mid.setdefault(str(r["moment_id"]), []).append(1.0 if r["is_correct"] else 0.0)
        easy_ids = {mid for mid, vals in by_mid.items() if vals and (sum(vals) / len(vals) > args.easy_threshold)}
        easy_ratio = (len(easy_ids) / max(1, len(by_mid))) if by_mid else 0.0

        iteration_meta.append(
            {
                "iteration": it,
                "n_questions": len(questions),
                "n_selected_moments": len(by_mid),
                "n_easy_selected_moments": len(easy_ids),
                "easy_ratio": round(easy_ratio, 4),
            }
        )

        final_questions = questions
        final_responses = responses
        easy_ids_prev = easy_ids
        print(
            f"iter={it} questions={len(questions)} selected_moments={len(by_mid)} "
            f"easy={len(easy_ids)} ratio={easy_ratio:.3f}"
        )

        if it + 1 >= args.max_iterations or easy_ratio <= args.easy_stop_ratio:
            break

    # Final summary.
    def acc(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(1.0 if r["is_correct"] else 0.0 for r in rows) / len(rows)

    rows_final = final_responses
    rank_rows = []
    for solver in active_providers:
        b = acc([r for r in rows_final if r["solver_provider"] == solver and r["condition"] == "selected_text"])
        c = acc([r for r in rows_final if r["solver_provider"] == solver and r["condition"] == "control_text"])
        for cap_provider in active_providers:
            cond = f"selected_caption_{cap_provider}"
            a = acc([r for r in rows_final if r["solver_provider"] == solver and r["condition"] == cond])
            rank_rows.append(
                {
                    "solver": solver,
                    "caption_provider": cap_provider,
                    "selected_caption_acc": round(a, 4),
                    "selected_text_acc": round(b, 4),
                    "control_text_acc": round(c, 4),
                    "rank_pass": bool(a > b > c),
                }
            )

    pass_rate = sum(1 for r in rank_rows if r["rank_pass"]) / max(1, len(rank_rows))

    provider_cost_summary = []
    for provider in active_providers:
        row = provider_cost_rows[provider]
        calls = int(row["calls"])
        provider_cost_summary.append(
            {
                "provider": provider,
                "calls": calls,
                "prompt_tokens": round(float(row["prompt_tokens"]), 2),
                "completion_tokens": round(float(row["completion_tokens"]), 2),
                "total_tokens": round(float(row["total_tokens"]), 2),
                "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 8),
                "executed_cost_usd": round(float(row["executed_cost_usd"]), 8),
                "priced_calls": int(row["priced_calls"]),
                "unpriced_calls": int(row["unpriced_calls"]),
                "pricing_coverage": round((float(row["priced_calls"]) / calls), 4) if calls else 0.0,
                "models": sorted(str(x) for x in row["models"]),
                "missing_pricing_models": sorted(str(x) for x in row["missing_pricing_models"]),
            }
        )

    stem_cost_rows = []
    for (stem, provider), row in sorted(stem_provider_costs.items(), key=lambda x: (x[0][0], x[0][1])):
        stage_rows = []
        for stage, s in sorted(row["stages"].items()):
            stage_rows.append(
                {
                    "stage": stage,
                    "calls": int(s["calls"]),
                    "prompt_tokens": round(float(s["prompt_tokens"]), 2),
                    "completion_tokens": round(float(s["completion_tokens"]), 2),
                    "total_tokens": round(float(s["total_tokens"]), 2),
                    "estimated_cost_usd": round(float(s["estimated_cost_usd"]), 8),
                }
            )
        stem_cost_rows.append(
            {
                "stem": stem,
                "provider": provider,
                "calls": int(row["calls"]),
                "prompt_tokens": round(float(row["prompt_tokens"]), 2),
                "completion_tokens": round(float(row["completion_tokens"]), 2),
                "total_tokens": round(float(row["total_tokens"]), 2),
                "estimated_cost_usd": round(float(row["estimated_cost_usd"]), 8),
                "executed_cost_usd": round(float(row["executed_cost_usd"]), 8),
                "stages": stage_rows,
            }
        )

    (out / "_costs_by_stem.json").write_text(json.dumps(stem_cost_rows, indent=2) + "\n", encoding="utf-8")

    # Task-type slices.
    task_slice = {}
    for cond in sorted({r["condition"] for r in rows_final}):
        for task in ["comprehend", "predict"]:
            k = f"{cond}::{task}"
            task_slice[k] = round(
                acc([r for r in rows_final if r["condition"] == cond and r["task_type"] == task]),
                4,
            )

    summary = {
        "n_stems": len({m["stem"] for m in selected_moments}),
        "n_selected_moments": len(selected_moments),
        "n_control_moments": len(control_moments),
        "providers": active_providers,
        "model_prefs": MODEL_PREFS,
        "prompt_ids": {
            "caption": PROMPT_IDS["dense_caption"],
            "question_author": PROMPT_IDS["qa_author"],
            "solver": PROMPT_IDS["qa_solver"],
        },
        "iteration_meta": iteration_meta,
        "final_iteration": iteration_meta[-1]["iteration"] if iteration_meta else None,
        "rank_rows": rank_rows,
        "rank_pass_rate": round(pass_rate, 4),
        "task_type_accuracy": task_slice,
        "pricing_config_path": str(args.pricing_config),
        "provider_cost_rows": provider_cost_summary,
        "easy_threshold": args.easy_threshold,
        "easy_stop_ratio": args.easy_stop_ratio,
    }
    (out / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# Plan 012 Summary")
    md.append("")
    md.append(f"- Stems: `{summary['n_stems']}`")
    md.append(f"- Selected moments: `{summary['n_selected_moments']}`")
    md.append(f"- Control moments: `{summary['n_control_moments']}`")
    md.append(f"- Final iteration: `{summary['final_iteration']}`")
    md.append(
        f"- Rank pass rate (`selected+caption > selected-text > control-text`): `{summary['rank_pass_rate']:.2%}`"
    )
    md.append(f"- Pricing config: `{summary['pricing_config_path']}`")
    md.append(f"- Prompt caption: `{summary['prompt_ids']['caption']}`")
    md.append(f"- Prompt question-author: `{summary['prompt_ids']['question_author']}`")
    md.append(f"- Prompt solver: `{summary['prompt_ids']['solver']}`")
    md.append("")
    md.append("## Cost Summary")
    md.append("")
    md.append(
        "| Provider | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Estimated Cost (USD) | Executed Cost (USD) | Pricing Coverage |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in provider_cost_summary:
        md.append(
            f"| `{row['provider']}` | {row['calls']} | {row['prompt_tokens']:.0f} | {row['completion_tokens']:.0f} | "
            f"{row['total_tokens']:.0f} | {row['estimated_cost_usd']:.6f} | {row['executed_cost_usd']:.6f} | {row['pricing_coverage']:.2%} |"
        )
    md.append("")
    for row in provider_cost_summary:
        if row["missing_pricing_models"]:
            md.append(
                f"- `{row['provider']}` missing pricing for models: {', '.join(f'`{m}`' for m in row['missing_pricing_models'])}"
            )
    md.append("")
    md.append("Per-stem/provider cost details: `_costs_by_stem.json`")
    md.append("")
    md.append("## Rank Rows")
    md.append("")
    md.append("| Solver | Captioner | Selected+Caption | Selected Text | Control Text | Pass |")
    md.append("|---|---|---:|---:|---:|---|")
    for r in rank_rows:
        md.append(
            f"| `{r['solver']}` | `{r['caption_provider']}` | {r['selected_caption_acc']:.4f} | "
            f"{r['selected_text_acc']:.4f} | {r['control_text_acc']:.4f} | {r['rank_pass']} |"
        )
    md.append("")
    md.append("## Iterations")
    md.append("")
    md.append("| Iteration | Questions | Selected Moments | Easy Selected | Easy Ratio |")
    md.append("|---:|---:|---:|---:|---:|")
    for it in iteration_meta:
        md.append(
            f"| {it['iteration']} | {it['n_questions']} | {it['n_selected_moments']} | "
            f"{it['n_easy_selected_moments']} | {it['easy_ratio']:.4f} |"
        )
    md.append("")
    md.append("## Task-Type Accuracy")
    md.append("")
    md.append("| Condition::Task | Accuracy |")
    md.append("|---|---:|")
    for k in sorted(task_slice):
        md.append(f"| `{md_escape(k)}` | {task_slice[k]:.4f} |")
    md.append("")
    (out / "_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "_prompts.json").write_text(
        json.dumps(
            {
                "prompt_ids": summary["prompt_ids"],
                "rendered_examples": {
                    "caption": build_dense_caption_prompt(),
                    "question_author": build_qa_author_prompt(
                        {
                            "moment_type": "selected",
                            "timestamp": "00:00:10.000",
                            "transcript_window_text": "[00:00:01.000 - 00:00:03.000] TUTOR: ...",
                            "prediction_context_text": "[00:00:01.000 - 00:00:03.000] TUTOR: ...",
                            "future_turns_text": "[00:00:04.000 - 00:00:06.000] STUDENT: ...",
                        },
                        caption_text="A whiteboard with a division problem.",
                    ),
                    "solver": build_qa_solver_prompt(
                        {
                            "task_type": "comprehend",
                            "question": "Example question?",
                            "choices": ["A", "B", "C", "D"],
                        },
                        {
                            "moment_type": "selected",
                            "transcript_window_text": "[00:00:01.000 - 00:00:03.000] TUTOR: ...",
                            "prediction_context_text": "[00:00:01.000 - 00:00:03.000] TUTOR: ...",
                        },
                        caption_text="A whiteboard with a division problem.",
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. Summary: {out / '_summary.md'}")


if __name__ == "__main__":
    main()
