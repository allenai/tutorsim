#!/usr/bin/env python3
"""Select semantically visual-dependent moments from transcripts using multiple LLMs.

Produces per-provider manifests compatible with test_screenshot_diagnostic.py,
plus merged manifests and cost summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from tutor_bench.toolkit.io_utils import read_stems_file
from tutor_bench.toolkit.llm_utils import compute_cost_usd, extract_json_array, robust_sleep
from tutor_bench.toolkit.prompts import (
    PROMPT_IDS,
    build_moment_selection_pass1_prompt,
    build_moment_selection_pass2_prompt,
)
from tutor_bench.toolkit.time_utils import fmt_ts, ts_to_seconds
from tutor_bench.toolkit.transcript_utils import Segment, parse_gold_transcript, transcript_lines

DEFAULT_OUTPUT = Path("output/llm_moment_selection_010")
DEFAULT_PROVIDERS = ["openai", "anthropic", "gemini"]
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
}
DEFAULT_CHUNK_MINUTES = 8.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 90.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SNAP_MAX_SECONDS = 20.0
DEFAULT_DEDUP_GAP_SECONDS = 8.0
DEFAULT_MERGE_CLUSTER_SECONDS = 20.0
DEFAULT_CONSENSUS_MIN_PROVIDERS = 2
DEFAULT_PRICING_CONFIG = Path("configs/llm_pricing.json")


@dataclass
class LLMMoment:
    t: float
    timestamp: str
    reason: str
    evidence_quote: str
    confidence: float
    tags: list[str]
    source: str
    chunk_id: int | None


def parse_transcript(path: Path) -> tuple[list[Segment], list[str]]:
    segments = parse_gold_transcript(path)
    return segments, transcript_lines(segments)


def chunk_segments(segments: list[Segment], chunk_minutes: float, overlap_seconds: float) -> list[dict[str, Any]]:
    if not segments:
        return []
    t_start = segments[0].start
    t_end = segments[-1].end
    chunk_len = max(60.0, chunk_minutes * 60.0)
    step = max(15.0, chunk_len - overlap_seconds)

    chunks: list[dict[str, Any]] = []
    i = 0
    cur = t_start
    while cur < t_end + 1e-6:
        c_start = cur
        c_end = min(t_end, cur + chunk_len)
        lines = []
        for seg in segments:
            if seg.end >= c_start and seg.start <= c_end:
                lines.append(f"[{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {seg.role}: {seg.text}")
        if lines:
            chunks.append(
                {
                    "chunk_id": i,
                    "start": c_start,
                    "end": c_end,
                    "lines": lines,
                }
            )
            i += 1
        if c_end >= t_end:
            break
        cur += step
    return chunks


def build_pass1_prompt(stem: str, chunk: dict[str, Any]) -> str:
    return build_moment_selection_pass1_prompt(
        stem=stem,
        chunk={
            "chunk_id": chunk["chunk_id"],
            "start_ts": fmt_ts(chunk["start"]),
            "end_ts": fmt_ts(chunk["end"]),
        },
        transcript_text="\n".join(chunk["lines"]),
    )


def build_pass2_prompt(stem: str, transcript_lines: list[str], pass1: list[LLMMoment]) -> str:
    candidates = [
        {
            "timestamp": m.timestamp,
            "reason": m.reason,
            "evidence_quote": m.evidence_quote,
            "confidence": m.confidence,
            "tags": m.tags,
            "source": m.source,
            "chunk_id": m.chunk_id,
        }
        for m in pass1
    ]
    return build_moment_selection_pass2_prompt(stem=stem, transcript_lines=transcript_lines, candidates=candidates)


def http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def parse_timestamp_flexible(val: Any) -> float | None:
    if isinstance(val, int | float):
        return float(val)
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if re.match(r"^\d+(?:\.\d+)?$", s):
        return float(s)
    if re.match(r"^\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?$", s):
        if "." not in s:
            s += ".000"
        else:
            base, frac = s.split(".")
            s = f"{base}.{frac.ljust(3, '0')[:3]}"
        return ts_to_seconds(s)
    return None


def nearest_segment_time(t: float, segments: list[Segment], max_delta: float) -> float | None:
    if not segments:
        return None
    mids = [0.5 * (s.start + s.end) for s in segments]
    best = min(mids, key=lambda x: abs(x - t))
    if abs(best - t) <= max_delta:
        return best
    return None


def normalize_moments(
    raw_items: list[dict[str, Any]],
    segments: list[Segment],
    source: str,
    chunk_id: int | None,
    snap_max_seconds: float,
) -> tuple[list[LLMMoment], dict[str, int]]:
    stats = {
        "raw": len(raw_items),
        "invalid_timestamp": 0,
        "out_of_range": 0,
        "accepted": 0,
    }
    out: list[LLMMoment] = []
    if not segments:
        return out, stats
    t_min = segments[0].start
    t_max = segments[-1].end
    for item in raw_items:
        t0 = parse_timestamp_flexible(item.get("timestamp"))
        if t0 is None:
            stats["invalid_timestamp"] += 1
            continue
        if t0 < t_min - 1.0 or t0 > t_max + 1.0:
            stats["out_of_range"] += 1
            continue
        t0 = min(max(t0, t_min), t_max)
        snapped = nearest_segment_time(t0, segments, snap_max_seconds)
        t = snapped if snapped is not None else t0
        reason = str(item.get("reason", "")).strip()[:400]
        evidence = str(item.get("evidence_quote", "")).strip()[:300]
        conf_raw = item.get("confidence", 0.6)
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.6
        conf = min(max(conf, 0.0), 1.0)
        tags_raw = item.get("tags", [])
        tags = [str(x).strip()[:40] for x in tags_raw if str(x).strip()] if isinstance(tags_raw, list) else []
        out.append(
            LLMMoment(
                t=t,
                timestamp=fmt_ts(t),
                reason=reason,
                evidence_quote=evidence,
                confidence=conf,
                tags=tags,
                source=source,
                chunk_id=chunk_id,
            )
        )
        stats["accepted"] += 1
    return out, stats


def dedupe_moments(moments: list[LLMMoment], gap_seconds: float) -> list[LLMMoment]:
    if not moments:
        return []
    moments = sorted(moments, key=lambda m: (m.t, -m.confidence))
    kept: list[LLMMoment] = []
    cluster: list[LLMMoment] = []

    def flush_cluster(c: list[LLMMoment]) -> None:
        if not c:
            return
        best = sorted(c, key=lambda m: (m.confidence, len(m.reason)), reverse=True)[0]
        tags = sorted({tag for m in c for tag in m.tags})
        best.tags = tags
        kept.append(best)

    for m in moments:
        if not cluster:
            cluster = [m]
            continue
        if abs(m.t - cluster[-1].t) <= gap_seconds:
            cluster.append(m)
        else:
            flush_cluster(cluster)
            cluster = [m]
    flush_cluster(cluster)
    return sorted(kept, key=lambda m: m.t)


def call_openai(
    prompt: str, model: str, temperature: float, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": 1800,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
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


def call_anthropic(
    prompt: str, model: str, temperature: float, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    payload = {
        "model": model,
        "max_tokens": 3200,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
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
    usage_out = {
        "prompt_tokens": float(usage.get("input_tokens", 0.0)),
        "completion_tokens": float(usage.get("output_tokens", 0.0)),
        "total_tokens": float(usage.get("input_tokens", 0.0)) + float(usage.get("output_tokens", 0.0)),
    }
    return text, usage_out, resp


def call_gemini(
    prompt: str, model: str, temperature: float, timeout_seconds: int
) -> tuple[str, dict[str, float], dict[str, Any]]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 3200,
            "responseMimeType": "application/json",
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{parse.quote(model)}"
        f":generateContent?key={parse.quote(key)}"
    )
    headers = {
        "Content-Type": "application/json",
    }
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


def call_provider(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    cache_path: Path,
) -> tuple[str, dict[str, float], dict[str, Any], bool]:
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            str(data.get("text", "")),
            data.get("usage", {}),
            data.get("raw", {}),
            True,
        )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openai":
                text, usage, raw = call_openai(prompt, model, temperature, timeout_seconds)
            elif provider == "anthropic":
                text, usage, raw = call_anthropic(prompt, model, temperature, timeout_seconds)
            elif provider == "gemini":
                text, usage, raw = call_gemini(prompt, model, temperature, timeout_seconds)
            else:
                raise RuntimeError(f"unknown provider: {provider}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"text": text, "usage": usage, "raw": raw}, indent=2) + "\n",
                encoding="utf-8",
            )
            return text, usage, raw, False
        except (RuntimeError, error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < max_retries:
                robust_sleep(attempt)
    raise RuntimeError(f"{provider} call failed after retries: {last_err}")


def merge_provider_manifests(
    manifests_by_provider: dict[str, list[LLMMoment]],
    cluster_seconds: float,
    consensus_min_providers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for provider, moments in manifests_by_provider.items():
        for m in moments:
            rows.append(
                {
                    "provider": provider,
                    "t": m.t,
                    "timestamp": m.timestamp,
                    "reason": m.reason,
                    "evidence_quote": m.evidence_quote,
                    "confidence": m.confidence,
                    "tags": m.tags,
                }
            )
    rows.sort(key=lambda x: x["t"])

    clusters: list[list[dict[str, Any]]] = []
    for r in rows:
        if not clusters:
            clusters.append([r])
            continue
        if abs(r["t"] - clusters[-1][-1]["t"]) <= cluster_seconds:
            clusters[-1].append(r)
        else:
            clusters.append([r])

    def cluster_to_moment(cluster: list[dict[str, Any]]) -> dict[str, Any]:
        providers = sorted({x["provider"] for x in cluster})
        confs = [max(1e-6, float(x.get("confidence", 0.5))) for x in cluster]
        ts = [float(x["t"]) for x in cluster]
        weighted = sum(t * c for t, c in zip(ts, confs, strict=False)) / sum(confs)
        all_tags = sorted({tag for x in cluster for tag in x.get("tags", [])})
        best = sorted(cluster, key=lambda x: (x.get("confidence", 0.0), len(x.get("reason", ""))), reverse=True)[0]
        t_min = min(ts)
        t_max = max(ts)
        return {
            "t": round(weighted, 3),
            "timestamp": fmt_ts(weighted),
            "reason": str(best.get("reason", ""))[:400],
            "evidence_quote": str(best.get("evidence_quote", ""))[:300],
            "combined_confidence": round(sum(confs) / len(confs), 4),
            "tags": all_tags,
            "provider_support_count": len(providers),
            "providers": providers,
            "cluster_span_seconds": round(t_max - t_min, 3),
        }

    union = [cluster_to_moment(c) for c in clusters]
    consensus = [m for m in union if m["provider_support_count"] >= consensus_min_providers]
    return union, consensus


def write_moments_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if not xs or len(xs) != len(ys):
        return 0.0, 0.0
    if len(xs) == 1:
        return ys[0], 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-9:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denom
    a = my - b * mx
    return a, b


def bootstrap_projection_ci(
    sample_x: list[float],
    sample_y: list[float],
    full_x: list[float],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if not sample_x or not sample_y:
        return 0.0, 0.0
    rng = random.Random(seed)
    vals = []
    n = len(sample_x)
    for _ in range(max(50, n_boot)):
        idxs = [rng.randrange(n) for _ in range(n)]
        xs = [sample_x[i] for i in idxs]
        ys = [sample_y[i] for i in idxs]
        a, b = linear_fit(xs, ys)
        total = 0.0
        for x in full_x:
            total += max(0.0, a + b * x)
        vals.append(total)
    vals.sort()
    lo = vals[int(0.1 * (len(vals) - 1))]
    hi = vals[int(0.9 * (len(vals) - 1))]
    return lo, hi


def collect_full_word_counts(transcripts_dir: Path, stems: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for stem in stems:
        p = transcripts_dir / f"{stem}_transcript.txt"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        out[stem] = len(re.findall(r"\b\w+\b", text))
    return out


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# LLM Transcript Moment Selection Summary",
        "",
        f"- Files requested: {summary['n_requested']}",
        f"- Files completed: {summary['n_completed']}",
        f"- Providers: {', '.join(summary['providers'])}",
        f"- Prompt pass1: `{summary['prompt_ids']['pass1']}`",
        f"- Prompt pass2: `{summary['prompt_ids']['pass2']}`",
        "",
        "| Provider | Model | Files OK | Files Failed | Total Cost (USD) | Avg Cost/File (USD) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["provider_rows"]:
        lines.append(
            f"| `{row['provider']}` | `{row['model']}` | {row['ok_files']} | {row['failed_files']} | "
            f"{row['total_cost_usd']:.6f} | {row['avg_cost_usd']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_cost_projection_md(cost_projection: dict[str, Any]) -> str:
    lines = [
        "# Cost Projection (Length-Adjusted)",
        "",
        f"- Total eligible transcripts in projection set: {cost_projection['n_full_transcripts']}",
        "",
        "| Provider | Model | Sample Total (USD) | Projected Full (USD) | CI Low | CI High |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cost_projection["rows"]:
        lines.append(
            f"| `{row['provider']}` | `{row['model']}` | {row['sample_total_cost_usd']:.6f} | "
            f"{row['projected_full_cost_usd']:.6f} | {row['ci_low_usd']:.6f} | {row['ci_high_usd']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select semantic visual moments from transcripts using LLMs.")
    parser.add_argument("transcripts_dir", type=Path)
    parser.add_argument("--stems-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--providers", type=str, default=",".join(DEFAULT_PROVIDERS))
    parser.add_argument("--openai-model", type=str, default=DEFAULT_MODELS["openai"])
    parser.add_argument("--anthropic-model", type=str, default=DEFAULT_MODELS["anthropic"])
    parser.add_argument("--gemini-model", type=str, default=DEFAULT_MODELS["gemini"])
    parser.add_argument("--chunk-minutes", type=float, default=DEFAULT_CHUNK_MINUTES)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=DEFAULT_CHUNK_OVERLAP_SECONDS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--request-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--snap-max-seconds", type=float, default=DEFAULT_SNAP_MAX_SECONDS)
    parser.add_argument("--dedup-gap-seconds", type=float, default=DEFAULT_DEDUP_GAP_SECONDS)
    parser.add_argument("--merge-cluster-seconds", type=float, default=DEFAULT_MERGE_CLUSTER_SECONDS)
    parser.add_argument("--consensus-min-providers", type=int, default=DEFAULT_CONSENSUS_MIN_PROVIDERS)
    parser.add_argument("--pricing-config", type=Path, default=DEFAULT_PRICING_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stems", type=int, default=0)
    args = parser.parse_args()

    if not args.transcripts_dir.is_dir():
        raise SystemExit(f"transcripts dir not found: {args.transcripts_dir}")
    if not args.stems_file.is_file():
        raise SystemExit(f"stems file not found: {args.stems_file}")

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    for p in providers:
        if p not in {"openai", "anthropic", "gemini"}:
            raise SystemExit(f"unsupported provider: {p}")

    model_map = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
        "gemini": args.gemini_model,
    }

    stems = read_stems_file(args.stems_file)
    if args.max_stems > 0:
        stems = stems[: args.max_stems]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.pricing_config.is_file():
        pricing = json.loads(args.pricing_config.read_text(encoding="utf-8"))
    else:
        pricing = {}

    per_provider_rows: dict[str, dict[str, Any]] = {}
    provider_sample_cost_pairs: dict[str, list[tuple[int, float]]] = {p: [] for p in providers}

    for provider in providers:
        per_provider_rows[provider] = {
            "provider": provider,
            "model": model_map[provider],
            "ok_files": 0,
            "failed_files": 0,
            "total_cost_usd": 0.0,
            "avg_cost_usd": 0.0,
        }

    all_manifest_stems: set[str] = set()
    full_word_counts = collect_full_word_counts(args.transcripts_dir, stems)

    for idx, stem in enumerate(stems):
        print(f"[{idx + 1}/{len(stems)}] {stem}")
        t_path = args.transcripts_dir / f"{stem}_transcript.txt"
        if not t_path.exists():
            print("  SKIP missing transcript")
            for provider in providers:
                per_provider_rows[provider]["failed_files"] += 1
            continue

        segments, transcript_lines = parse_transcript(t_path)
        if not segments:
            print("  SKIP empty transcript")
            for provider in providers:
                per_provider_rows[provider]["failed_files"] += 1
            continue

        chunks = chunk_segments(segments, args.chunk_minutes, args.chunk_overlap_seconds)
        word_count = full_word_counts.get(stem, 0)

        per_provider_moments: dict[str, list[LLMMoment]] = {}

        for provider in providers:
            start_t = time.time()
            model = model_map[provider]
            provider_out_dir = args.output_dir / provider
            provider_out_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = provider_out_dir / f"{stem}_moments.json"

            usage_total = {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
            }
            pass1_cost = 0.0
            pass2_cost = 0.0
            p1_moments: list[LLMMoment] = []
            stats_accum = {
                "raw": 0,
                "invalid_timestamp": 0,
                "out_of_range": 0,
                "accepted": 0,
            }
            ok = True
            err_msg = ""

            try:
                for chunk in chunks:
                    p1_prompt = build_pass1_prompt(stem, chunk)
                    cache_path = cache_dir / provider / stem / f"pass1_chunk_{chunk['chunk_id']}.json"
                    text, usage, _, _ = call_provider(
                        provider=provider,
                        model=model,
                        prompt=p1_prompt,
                        temperature=args.temperature,
                        timeout_seconds=args.request_timeout_seconds,
                        max_retries=args.max_retries,
                        cache_path=cache_path,
                    )
                    for k in usage_total:
                        usage_total[k] += float(usage.get(k, 0.0))
                    c = compute_cost_usd(usage, provider, model, pricing)
                    pass1_cost += c["total_cost_usd"]
                    raw_items = extract_json_array(text)
                    nrm, st = normalize_moments(
                        raw_items,
                        segments,
                        source="pass1",
                        chunk_id=chunk["chunk_id"],
                        snap_max_seconds=args.snap_max_seconds,
                    )
                    p1_moments.extend(nrm)
                    for k in stats_accum:
                        stats_accum[k] += st.get(k, 0)

                p1_moments = dedupe_moments(p1_moments, args.dedup_gap_seconds)
                p2_prompt = build_pass2_prompt(stem, transcript_lines, p1_moments)
                p2_cache = cache_dir / provider / stem / "pass2_refine.json"
                text2, usage2, _, _ = call_provider(
                    provider=provider,
                    model=model,
                    prompt=p2_prompt,
                    temperature=args.temperature,
                    timeout_seconds=args.request_timeout_seconds,
                    max_retries=args.max_retries,
                    cache_path=p2_cache,
                )
                for k in usage_total:
                    usage_total[k] += float(usage2.get(k, 0.0))
                c2 = compute_cost_usd(usage2, provider, model, pricing)
                pass2_cost += c2["total_cost_usd"]

                raw2 = extract_json_array(text2)
                p2_moments, st2 = normalize_moments(
                    raw2,
                    segments,
                    source="pass2_refined",
                    chunk_id=None,
                    snap_max_seconds=args.snap_max_seconds,
                )
                for k in stats_accum:
                    stats_accum[k] += st2.get(k, 0)

                final_moments = dedupe_moments(p2_moments if p2_moments else p1_moments, args.dedup_gap_seconds)
                per_provider_moments[provider] = final_moments

                total_cost_obj = compute_cost_usd(usage_total, provider, model, pricing)
                total_cost = total_cost_obj["total_cost_usd"]
                provider_sample_cost_pairs[provider].append((word_count, total_cost))

                payload = {
                    "stem": stem,
                    "provider": provider,
                    "model": model,
                    "prompt_ids": {
                        "pass1": PROMPT_IDS["moment_selection_pass1"],
                        "pass2": PROMPT_IDS["moment_selection_pass2"],
                    },
                    "status": "ok",
                    "runtime_seconds_total": round(time.time() - start_t, 3),
                    "n_chunks": len(chunks),
                    "n_candidates_pass1": len(p1_moments),
                    "n_moments_final": len(final_moments),
                    "word_count": word_count,
                    "normalization_stats": stats_accum,
                    "usage": {
                        "prompt_tokens": round(usage_total["prompt_tokens"], 2),
                        "completion_tokens": round(usage_total["completion_tokens"], 2),
                        "total_tokens": round(usage_total["total_tokens"], 2),
                    },
                    "cost": {
                        **total_cost_obj,
                        "pass1_total_cost_usd": round(pass1_cost, 8),
                        "pass2_total_cost_usd": round(pass2_cost, 8),
                    },
                    "moments": [
                        {
                            "t": round(m.t, 3),
                            "timestamp": m.timestamp,
                            "reason": m.reason,
                            "evidence_quote": m.evidence_quote,
                            "confidence": round(m.confidence, 4),
                            "tags": m.tags,
                            "source": m.source,
                            "chunk_id": m.chunk_id,
                        }
                        for m in final_moments
                    ],
                }
                write_moments_manifest(manifest_path, payload)
                per_provider_rows[provider]["ok_files"] += 1
                per_provider_rows[provider]["total_cost_usd"] += total_cost
                print(
                    f"  {provider}: moments={len(final_moments)} cost=${total_cost:.6f} tokens={int(usage_total['total_tokens'])}"
                )
            except Exception as e:
                ok = False
                err_msg = str(e)
                per_provider_rows[provider]["failed_files"] += 1
                payload = {
                    "stem": stem,
                    "provider": provider,
                    "model": model,
                    "prompt_ids": {
                        "pass1": PROMPT_IDS["moment_selection_pass1"],
                        "pass2": PROMPT_IDS["moment_selection_pass2"],
                    },
                    "status": "failed",
                    "error": err_msg,
                    "runtime_seconds_total": round(time.time() - start_t, 3),
                    "moments": [],
                }
                write_moments_manifest(manifest_path, payload)
                per_provider_moments[provider] = []
                print(f"  {provider}: FAILED {err_msg}")

            if ok:
                all_manifest_stems.add(stem)

        # merged manifests from successful provider moments
        union, consensus = merge_provider_manifests(
            per_provider_moments,
            cluster_seconds=args.merge_cluster_seconds,
            consensus_min_providers=args.consensus_min_providers,
        )
        write_moments_manifest(
            args.output_dir / "merged_union" / f"{stem}_moments.json",
            {
                "stem": stem,
                "status": "ok",
                "type": "merged_union",
                "prompt_ids": {
                    "source_pass1": PROMPT_IDS["moment_selection_pass1"],
                    "source_pass2": PROMPT_IDS["moment_selection_pass2"],
                },
                "n_moments_final": len(union),
                "moments": union,
            },
        )
        write_moments_manifest(
            args.output_dir / "merged_consensus" / f"{stem}_moments.json",
            {
                "stem": stem,
                "status": "ok",
                "type": "merged_consensus",
                "prompt_ids": {
                    "source_pass1": PROMPT_IDS["moment_selection_pass1"],
                    "source_pass2": PROMPT_IDS["moment_selection_pass2"],
                },
                "consensus_min_providers": args.consensus_min_providers,
                "n_moments_final": len(consensus),
                "moments": consensus,
            },
        )

    provider_rows = []
    for provider in providers:
        row = per_provider_rows[provider]
        if row["ok_files"] > 0:
            row["avg_cost_usd"] = row["total_cost_usd"] / row["ok_files"]
        provider_rows.append(row)

    summary = {
        "n_requested": len(stems),
        "n_completed": len(all_manifest_stems),
        "providers": providers,
        "provider_rows": provider_rows,
        "prompt_ids": {
            "pass1": PROMPT_IDS["moment_selection_pass1"],
            "pass2": PROMPT_IDS["moment_selection_pass2"],
        },
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "_summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")
    (args.output_dir / "_prompts.json").write_text(
        json.dumps(
            {
                "prompt_ids": {
                    "pass1": PROMPT_IDS["moment_selection_pass1"],
                    "pass2": PROMPT_IDS["moment_selection_pass2"],
                },
                "rendered_examples": {
                    "pass1": build_moment_selection_pass1_prompt(
                        stem="EXAMPLE_STEM",
                        chunk={"chunk_id": 0, "start_ts": "00:00:00.000", "end_ts": "00:05:00.000"},
                        transcript_text="[00:00:01.000 - 00:00:03.000] TUTOR: ...",
                    ),
                    "pass2": build_moment_selection_pass2_prompt(
                        stem="EXAMPLE_STEM",
                        transcript_lines=["[00:00:01.000 - 00:00:03.000] TUTOR: ..."],
                        candidates=[
                            {
                                "timestamp": "00:00:02.000",
                                "reason": "example",
                                "evidence_quote": "example",
                                "confidence": 0.8,
                                "tags": ["visual_ref"],
                                "source": "pass1",
                                "chunk_id": 0,
                            }
                        ],
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Cost projection across same stems-file scope (can be widened later by alternate stems file)
    full_x = [float(v) for _, v in sorted((k, v) for k, v in full_word_counts.items())]
    projection_rows = []
    for provider in providers:
        pairs = provider_sample_cost_pairs.get(provider, [])
        sample_x = [float(x) for x, _ in pairs]
        sample_y = [float(y) for _, y in pairs]
        if not sample_x:
            projection_rows.append(
                {
                    "provider": provider,
                    "model": model_map[provider],
                    "sample_total_cost_usd": 0.0,
                    "projected_full_cost_usd": 0.0,
                    "ci_low_usd": 0.0,
                    "ci_high_usd": 0.0,
                }
            )
            continue
        a, b = linear_fit(sample_x, sample_y)
        projected = 0.0
        for x in full_x:
            projected += max(0.0, a + b * x)
        ci_lo, ci_hi = bootstrap_projection_ci(sample_x, sample_y, full_x, n_boot=300, seed=args.seed + 19)
        projection_rows.append(
            {
                "provider": provider,
                "model": model_map[provider],
                "sample_total_cost_usd": float(sum(sample_y)),
                "fit_intercept": a,
                "fit_slope": b,
                "projected_full_cost_usd": projected,
                "ci_low_usd": ci_lo,
                "ci_high_usd": ci_hi,
            }
        )

    cost_projection = {
        "n_full_transcripts": len(full_x),
        "rows": projection_rows,
    }
    (args.output_dir / "_cost_projection.json").write_text(
        json.dumps(cost_projection, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_cost_projection.md").write_text(build_cost_projection_md(cost_projection), encoding="utf-8")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
