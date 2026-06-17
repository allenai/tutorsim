"""
Provider-agnostic model client and batch API utilities.

ModelClient provides a unified sync API with built-in retry logic, JSON mode
handling, and normalized usage tracking across Gemini, OpenAI, and Anthropic.

Batch utilities handle batch job submission, polling, and result downloading
for all three providers (Gemini, OpenAI, Anthropic).

Usage:
    from annotator.core.client import ModelClient, run_batch, run_sync_entries

    client = ModelClient("gpt-5.4")
    response = client.generate("Return a JSON object with key 'hello'")
    print(response.text)   # '{"hello": "world"}'
    print(response.usage)  # {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}
"""

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google.genai import types

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ===================================================================
# Provider routing
# ===================================================================

# Model name -> provider mapping.
# Prefixes are checked in order; first match wins.
PROVIDER_PREFIXES = [
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    # Together-hosted open-weight models use vendor/model slash IDs.
    # infer_provider lowercases the name before matching, so list lowercase.
    ("deepseek-ai/", "together"),
    ("moonshotai/", "together"),
    ("minimaxai/", "together"),
    ("google/gemma", "together"),
    ("meta-llama/", "together"),
    ("qwen/", "together"),
]

from .config import get_retry_config, get_batch_timeout

# Provider-specific max output token limits
MAX_OUTPUT_TOKENS = {
    "gemini": 65536,
    "openai": 128000,
    "anthropic": 128000,
    "together": 16384,  # open-weight reasoners (DeepSeek/Kimi) need room to think
}

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


VISION_CAPABLE_PREFIXES = (
    "claude-opus-4", "claude-sonnet-4",
    "gemini-2", "gemini-3",
    "gpt-4o", "gpt-4.1", "gpt-5", "o4",
)


def validate_vision_support(model: str) -> None:
    """Raise ValueError if the model is not known to support vision input."""
    m = model.lower()
    if not any(m.startswith(p) for p in VISION_CAPABLE_PREFIXES):
        raise ValueError(
            f"Model '{model}' is not in the vision-capable list. "
            f"Vision-capable prefixes: {', '.join(VISION_CAPABLE_PREFIXES)}."
        )


_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MIME_BY_EXT:
        raise ValueError(f"unknown image extension: {ext} (path: {path})")
    return _MIME_BY_EXT[ext]


def _should_use_presigned_url() -> bool:
    """True when the storage backend is S3 (pre-signed URLs available)."""
    from . import storage
    be = storage._get_backend()
    return not isinstance(be, storage.LocalBackend)


def _base64_bytes(rel_path: str) -> str:
    from . import storage
    raw = storage._get_backend().read_bytes(rel_path)
    return base64.b64encode(raw).decode("ascii")


def _presigned_url(rel_path: str, expires_seconds: int = 172800) -> str:
    from . import storage
    return storage._get_backend().get_presigned_url(rel_path, expires_seconds=expires_seconds)


# Models that use the new adaptive thinking API (no budget_tokens, no
# output_config in the current SDK). Older models still use enabled+budget.
# Haiku 4.5 + the Opus 4.x family are all on the modern API; legacy enabled+
# budget_tokens is rejected on these.
_ANTHROPIC_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",      # 4-6 accepts both adaptive and legacy enabled+budget_tokens;
                            # adaptive is the non-deprecated path.
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
)

# Per-model max output token caps (from the model catalog). Requests above
# the cap are rejected by the API. Keys are matched by startswith().
_ANTHROPIC_MAX_OUTPUT_CAP = {
    "claude-haiku-4-5": 64000,
    "claude-sonnet-4-6": 64000,
}


def _anthropic_thinking_param(model: str, thinking_budget: int) -> dict:
    """Return the right `thinking` kwarg shape for the given model.

    - Newer models (Opus 4.8+) require {"type": "adaptive"}; legacy
      enabled+budget_tokens is rejected.
    - Older models (Opus 4.6 / 4.7) accept the enabled+budget form.
    """
    if model and any(model.startswith(prefix) for prefix in _ANTHROPIC_ADAPTIVE_THINKING_MODELS):
        return {"type": "adaptive"}
    budget = thinking_budget if thinking_budget > 0 else 16384
    return {"type": "enabled", "budget_tokens": budget}


def _build_image_blocks_anthropic(
    image_paths: list[str], use_url: bool, enable_cache: bool,
) -> list[dict]:
    blocks = []
    for path in image_paths:
        media_type = _mime_from_path(path)
        if use_url:
            source = {"type": "url", "url": _presigned_url(path)}
        else:
            source = {
                "type": "base64",
                "media_type": media_type,
                "data": _base64_bytes(path),
            }
        block = {"type": "image", "source": source}
        if enable_cache:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _build_image_blocks_openai(
    image_paths: list[str], use_url: bool,
) -> list[dict]:
    blocks = []
    for path in image_paths:
        if use_url:
            url = _presigned_url(path)
        else:
            b64 = _base64_bytes(path)
            url = f"data:{_mime_from_path(path)};base64,{b64}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _build_image_blocks_gemini(image_paths: list[str]) -> list[dict]:
    # Gemini does not accept S3 URIs; always inline.
    blocks = []
    for path in image_paths:
        blocks.append({
            "inline_data": {
                "mime_type": _mime_from_path(path),
                "data": _base64_bytes(path),
            }
        })
    return blocks


# Marker emitted by format_transcript / format_excerpt at each anchored screenshot.
# Permissive on the content between SCREEN and `image N]` so future enrichments
# (e.g. timestamp) don't break interleaving.
_SCREEN_MARKER_RE = re.compile(
    r"^[ \t]*\[SCREEN[^\]]*?image (\d+)\][ \t]*$",
    re.MULTILINE,
)


def _interleave_text_and_images(
    prompt: str,
    image_blocks: list[dict],
    text_block: callable,
) -> list[dict]:
    """Split prompt at screenshot markers and insert image blocks at their referenced positions.

    Each `[SCREEN ... image K]` marker line in the prompt is followed in the output by
    `image_blocks[K-1]`. The marker line itself is preserved as text so the model still
    sees the explicit "turn N" label next to the image.

    `text_block` wraps a string into the provider's text-block shape:
      Anthropic / OpenAI: `lambda s: {"type": "text", "text": s}`
      Gemini:             `lambda s: {"text": s}`

    A marker referencing an out-of-range index is left as text (no image inserted).
    Image blocks not referenced by any marker are appended at the end so no image
    is ever silently dropped.
    """
    parts: list[dict] = []
    cursor = 0
    used: set[int] = set()

    for m in _SCREEN_MARKER_RE.finditer(prompt):
        chunk = prompt[cursor:m.end()]
        if chunk:
            parts.append(text_block(chunk))
        cursor = m.end()

        idx = int(m.group(1)) - 1  # 1-based markers, 0-based list
        if 0 <= idx < len(image_blocks):
            parts.append(image_blocks[idx])
            used.add(idx)

    if cursor < len(prompt):
        tail = prompt[cursor:]
        if tail:
            parts.append(text_block(tail))

    for i, block in enumerate(image_blocks):
        if i not in used:
            parts.append(block)

    return parts if parts else [text_block(prompt)]


def infer_provider(model: str) -> str:
    """Infer provider from model name string.

    Examples:
        'gemini-3.1-pro-preview' -> 'gemini'
        'gpt-4o'               -> 'openai'
        'o3-mini'              -> 'openai'
        'claude-sonnet-4-6'    -> 'anthropic'
    """
    model_lower = model.lower()
    for prefix, provider in PROVIDER_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Expected prefix: {', '.join(p for p, _ in PROVIDER_PREFIXES)}"
    )


@dataclass
class ModelResponse:
    """Unified response from any provider."""
    text: str
    usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    # Wall-clock seconds from the start of the successful generate() attempt
    # to the response landing. Stamped by ModelClient.generate(). Retries
    # are not included (they're bookkeeping, not model reasoning time).
    latency_seconds: float | None = None


class ModelClient:
    """Provider-agnostic synchronous model client.

    Instantiates the appropriate SDK client based on the model name,
    and provides a unified `generate()` method with retry logic.
    """

    def __init__(self, model: str):
        self.model = model
        self.provider = infer_provider(model)
        self._client = self._init_client()

    def _init_client(self):
        """Initialize the SDK client for the inferred provider."""
        if self.provider == "gemini":
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not found in environment")
            return genai.Client(api_key=api_key)

        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment")
            return OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not found in environment")
            return anthropic.Anthropic(api_key=api_key)

        elif self.provider == "together":
            # Together is OpenAI-compatible -- same SDK, different base_url + key.
            from openai import OpenAI
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise RuntimeError("TOGETHER_API_KEY not found in environment")
            return OpenAI(api_key=api_key, base_url=TOGETHER_BASE_URL)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str,
                 images: list[str] | None = None,
                 json_mode: bool = True,
                 max_tokens: int = 0, timeout: int = 120,
                 thinking: bool = False,
                 thinking_budget: int = 0,
                 reasoning_effort: str = "",
                 effort: str = "",
                 enable_cache: bool = False,
                 *,
                 cacheable_prefix: str | None = None) -> ModelResponse:
        if max_tokens <= 0:
            max_tokens = MAX_OUTPUT_TOKENS.get(self.provider, 8192)

        retry_cfg = get_retry_config()
        max_retries = retry_cfg.get("max_retries", 5)
        base_delay = retry_cfg.get("base_delay", 5)

        last_error = None
        call_t0 = time.monotonic()
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    resp = self._generate_gemini(prompt, json_mode, max_tokens, timeout,
                                                 thinking, thinking_budget, images,
                                                 cacheable_prefix=cacheable_prefix)
                elif self.provider == "openai":
                    resp = self._generate_openai(prompt, json_mode, max_tokens, timeout,
                                                  thinking, thinking_budget,
                                                  reasoning_effort=reasoning_effort,
                                                  images=images,
                                                  cacheable_prefix=cacheable_prefix)
                elif self.provider == "anthropic":
                    resp = self._generate_anthropic(prompt, json_mode, max_tokens, timeout,
                                                     thinking, thinking_budget,
                                                     reasoning_effort=reasoning_effort,
                                                     effort=effort,
                                                     images=images,
                                                     enable_cache=enable_cache,
                                                     cacheable_prefix=cacheable_prefix)
                elif self.provider == "together":
                    resp = self._generate_together(prompt, json_mode, max_tokens, timeout,
                                                    cacheable_prefix=cacheable_prefix)
                else:
                    raise RuntimeError(f"unknown provider {self.provider}")
                # Stamp wall-clock latency for the successful attempt only
                # (retries are bookkeeping, not the model's reasoning time).
                resp.latency_seconds = time.monotonic() - call_t0
                return resp
            except Exception as e:
                last_error = e
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    logger.warning("API error (attempt %d/%d): %s. Retrying in %ds...",
                                   attempt + 1, max_retries, e, delay)
                    time.sleep(delay)
                else:
                    logger.error("API failed after %d attempts: %s", max_retries, e)

        raise RuntimeError(
            f"API call failed after {max_retries} attempts: {last_error}"
        )

    def _generate_gemini(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0, images=None,
                         cacheable_prefix: str | None = None):
        """Gemini API call via google-genai SDK."""
        config = {
            "max_output_tokens": max_tokens,
            "http_options": {"timeout": timeout * 1000},
        }
        if json_mode:
            config["response_mime_type"] = "application/json"
        if thinking:
            # thinking_budget = -1 means "dynamic" (model self-paces).
            # 0 = no thinking. Positive = fixed budget. None/unset = default 16384.
            if thinking_budget is None or thinking_budget == 0:
                budget = 16384
            else:
                budget = thinking_budget  # may be -1 (dynamic) or positive
            config["thinking_config"] = {"include_thoughts": True, "thinking_budget": budget}

        # TODO(gemini-cache): wire CachedContent.create/refresh/delete here.
        # For now concatenate the cacheable head into the prompt so behavior
        # is semantically correct even without a real cache hit.
        effective_prompt = (cacheable_prefix or "") + prompt
        if images:
            image_blocks = _build_image_blocks_gemini(images)
            parts = _interleave_text_and_images(
                effective_prompt, image_blocks, lambda s: {"text": s},
            )
            contents = [{"role": "user", "parts": parts}]
        else:
            contents = effective_prompt

        response = self._client.models.generate_content(
            model=f"models/{self.model}",
            contents=contents,
            config=config,
        )

        text = response.text or ""
        usage_meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
        }
        return ModelResponse(text=text, usage=usage)

    def _generate_openai(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0,
                         reasoning_effort: str = "", images=None,
                         cacheable_prefix: str | None = None):
        """OpenAI API call via openai SDK."""
        if images:
            image_blocks = _build_image_blocks_openai(
                images, use_url=_should_use_presigned_url(),
            )
            # Prepend cacheable head so auto-cache sees the same prefix on repeats.
            head_text = cacheable_prefix or ""
            content = _interleave_text_and_images(
                head_text + prompt, image_blocks, lambda s: {"type": "text", "text": s},
            )
        else:
            content = (cacheable_prefix or "") + prompt

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        cached = 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        usage = {
            "input_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
            "cached_tokens": cached,
        }
        return ModelResponse(text=text, usage=usage)

    def _generate_together(self, prompt, json_mode, max_tokens, timeout,
                           cacheable_prefix: str | None = None):
        """Together (open-weight) call via OpenAI-compatible chat completions.

        Together uses `max_tokens` (not `max_completion_tokens`) and does not
        accept `reasoning_effort`. Open-weight reasoners (DeepSeek-V4, Kimi)
        produce their own chain-of-thought internally; there's no depth knob
        to pass. Caching isn't supported, so the cacheable head is just
        concatenated into the prompt (same as the Gemini path).
        """
        content = (cacheable_prefix or "") + prompt
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS["together"]),
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        if json_mode:
            text = _strip_json_fences(text)
        u = response.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
        return ModelResponse(text=text, usage=usage)

    def _generate_anthropic(self, prompt, json_mode, max_tokens, timeout,
                            thinking=False, thinking_budget=0,
                            reasoning_effort="", effort="",
                            images=None, enable_cache=False,
                            cacheable_prefix: str | None = None):
        """Anthropic API call via anthropic SDK."""
        system_parts = []
        if json_mode:
            system_parts.append(
                "You must respond with valid JSON only. "
                "Do not include markdown code fences, explanations, or any text "
                "outside the JSON object."
            )

        if images:
            image_blocks = _build_image_blocks_anthropic(
                images, use_url=_should_use_presigned_url(), enable_cache=enable_cache,
            )
            content = _interleave_text_and_images(
                prompt, image_blocks, lambda s: {"type": "text", "text": s},
            )
            if cacheable_prefix is not None:
                # Prepend the cacheable head as its own text block.
                content = [
                    {"type": "text", "text": cacheable_prefix,
                     "cache_control": {"type": "ephemeral"}},
                ] + content
        elif cacheable_prefix is not None:
            content = [
                {"type": "text", "text": cacheable_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        # Clamp max_tokens to the model's per-model output cap (Haiku 4.5 and
        # Sonnet 4.6 max out at 64K — requests above the cap are rejected).
        capped_max = max_tokens
        for prefix, cap in _ANTHROPIC_MAX_OUTPUT_CAP.items():
            if self.model and self.model.startswith(prefix):
                capped_max = min(capped_max, cap)
                break
        kwargs = {
            "model": self.model,
            "max_tokens": capped_max,
            "messages": [{"role": "user", "content": content}],
            "timeout": timeout,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if thinking:
            kwargs["thinking"] = _anthropic_thinking_param(self.model, thinking_budget)
            if kwargs["thinking"].get("type") == "enabled":
                # Legacy enabled mode needs max_tokens >= budget + headroom.
                budget = kwargs["thinking"]["budget_tokens"]
                if kwargs["max_tokens"] < budget + 64:
                    kwargs["max_tokens"] = budget + 64

        # effort goes inside output_config and is only valid on adaptive
        # thinking models that support the effort parameter. Haiku 4.5 will
        # 400 if effort is sent -- skip there.
        # SDK <= 0.71 doesn't expose output_config as a top-level kwarg, so
        # we forward it via extra_body. The server accepts it either way.
        if effort and self.model and not self.model.startswith("claude-haiku-4-5"):
            kwargs["extra_body"] = {"output_config": {"effort": effort}}

        response = self._client.messages.create(**kwargs)

        # Extract text from content blocks (skip thinking blocks)
        text = _extract_anthropic_text(response.content)
        # Strip markdown fences if present
        if json_mode:
            text = _strip_json_fences(text)

        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
            "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        return ModelResponse(text=text, usage=usage)

    def __repr__(self):
        return f"ModelClient(model='{self.model}', provider='{self.provider}')"


def _extract_anthropic_text(content) -> str:
    """Concatenate all text blocks from an Anthropic response/message.

    Anthropic returns a list of content blocks; non-text blocks (e.g. thinking)
    are skipped. There can be more than one text block -- when thinking is
    enabled, or when output is interleaved -- so all text blocks must be joined.
    Keeping only the first silently truncates the response (and can produce
    invalid JSON from a response split mid-string across two blocks).
    """
    return "".join(block.text for block in content if block.type == "text")


def _strip_json_fences(text: str) -> str:
    """Strip markdown JSON code fences from text.

    Handles: ```json ... ``` and ``` ... ```
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def _extract_entry(entry: dict) -> tuple[str, str, bool, int, list[str]]:
    """Extract key, prompt, json_mode, max_tokens, images from a batch entry."""
    key = entry["key"]
    parts = entry["request"]["contents"][0]["parts"]
    prompt_text = parts[0]["text"]
    gen_config = entry["request"].get("generation_config", {})
    json_mode = "application/json" in gen_config.get("response_mime_type", "")
    max_tokens = gen_config.get("max_output_tokens", 0)
    images = entry["request"].get("images", [])
    return key, prompt_text, json_mode, max_tokens, images


# ===================================================================
# Shared utilities
# ===================================================================

def build_batch_entry(key: str, prompt_text: str,
                      images: list[str] | None = None,
                      json_mode: bool = True,
                      max_tokens: int = 65536,
                      cacheable_prefix: str | None = None) -> dict:
    """Build a single batch entry from a key and prompt text.

    Uses a provider-neutral internal format. run_batch() and run_sync_entries()
    both consume these entries.

    cacheable_prefix: when set, the Anthropic batch path will emit the
    two-block structured content (prefix with cache_control + prompt text).
    For Gemini and OpenAI batch, the prefix is concatenated into the prompt
    text (auto-cache handles it; Gemini has no explicit batch cache API yet).
    """
    gen_config = {"max_output_tokens": max_tokens}
    if json_mode:
        gen_config["response_mime_type"] = "application/json"
    request = {
        "contents": [{
            "parts": [{"text": prompt_text}],
            "role": "user"
        }],
        "generation_config": gen_config,
    }
    if images:
        request["images"] = list(images)
    entry = {"key": key, "request": request}
    if cacheable_prefix is not None:
        entry["cacheable_prefix"] = cacheable_prefix
    return entry


def write_jsonl(entries: list[dict], jsonl_path: str) -> int:
    """Write a list of batch entry dicts to a JSONL file.

    Returns the number of entries written.
    """
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def run_sync_entries(client: 'ModelClient', entries: list[dict],
                     json_mode: bool = True, max_tokens: int = 0) -> dict:
    """Run entries synchronously one at a time.

    Returns {key: {text, usage}} dict (same shape as run_batch).
    """
    raw_entries = {}
    total = len(entries)
    for i, entry in enumerate(entries):
        key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
        if not entry_max_tokens:
            entry_max_tokens = max_tokens

        logger.debug("[%d/%d] %s...", i + 1, total, key[:60])
        try:
            response = client.generate(
                prompt_text,
                images=images or None,
                json_mode=entry_json_mode if json_mode else False,
                max_tokens=entry_max_tokens,
            )
            raw_entries[key] = {
                "text": response.text,
                "usage": response.usage,
            }
        except Exception as e:
            logger.error("ERROR on %s: %s", key, e)
            raw_entries[key] = {
                "text": "",
                "error": str(e),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
    return raw_entries


# ===================================================================
# Unified batch API
# ===================================================================

def run_batch(client: 'ModelClient', entries: list[dict],
              json_mode: bool = True, display_name: str = "batch",
              poll_interval: int = 60,
              thinking: bool = False, thinking_budget: int = 0,
              reasoning_effort: str = "",
              enable_cache: bool = False,
              existing_batch_id: str | None = None,
              on_batch_created=None) -> dict:
    """Run entries as a batch job via the provider's batch API.

    Resume support: if `existing_batch_id` is set, skip submission and resume
    polling on that batch (the entries list still drives result parsing, so
    the caller must pass the same entries that were submitted with the batch).
    `on_batch_created` is called with the provider's batch id immediately after
    a fresh submission succeeds, before the poll loop starts -- callers use
    this to persist the id for ctrl-C recovery.
    """
    provider = client.provider
    if existing_batch_id:
        logger.info("Resuming in-flight %s batch %s (%d entries)",
                    provider, existing_batch_id, len(entries))
    else:
        logger.info("Running batch (%s): %d entries, display_name=%s",
                    provider, len(entries), display_name)

    if provider == "gemini":
        return _run_batch_gemini(client, entries, json_mode, display_name, poll_interval,
                                 thinking, thinking_budget,
                                 existing_batch_id=existing_batch_id,
                                 on_batch_created=on_batch_created)
    elif provider == "openai":
        return _run_batch_openai(client, entries, json_mode, display_name, poll_interval,
                                 thinking, thinking_budget, reasoning_effort,
                                 existing_batch_id=existing_batch_id,
                                 on_batch_created=on_batch_created)
    elif provider == "anthropic":
        return _run_batch_anthropic(client, entries, json_mode, display_name, poll_interval,
                                    thinking, thinking_budget, reasoning_effort,
                                    enable_cache=enable_cache,
                                    existing_batch_id=existing_batch_id,
                                    on_batch_created=on_batch_created)
    else:
        raise ValueError(f"Batch API not supported for provider: {provider}")


# ===================================================================
# Gemini Batch API
# ===================================================================

def _run_batch_gemini(client, entries, json_mode, display_name, poll_interval,
                      thinking=False, thinking_budget=0,
                      existing_batch_id=None, on_batch_created=None):
    """Gemini batch: upload JSONL, submit, poll, download.

    If existing_batch_id is set, skip upload+submit and retrieve that job.
    """
    import tempfile
    gemini_client = client._client
    jsonl_path = None

    if existing_batch_id:
        batch_job = gemini_client.batches.get(name=existing_batch_id)
    else:
        # Write Gemini-format JSONL to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                          encoding="utf-8") as f:
            for entry in entries:
                key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
                # Gemini has no explicit batch cache API; concatenate prefix (auto-cache).
                cacheable_prefix = entry.get("cacheable_prefix")
                effective_prompt = (cacheable_prefix or "") + prompt_text
                if images:
                    image_blocks = _build_image_blocks_gemini(images)
                    parts = _interleave_text_and_images(
                        effective_prompt, image_blocks, lambda s: {"text": s},
                    )
                else:
                    parts = [{"text": effective_prompt}]
                gem_entry = {
                    "key": key,
                    "request": {
                        "contents": [{"parts": parts, "role": "user"}],
                        "generation_config": entry["request"].get("generation_config", {}),
                    },
                }
                f.write(json.dumps(gem_entry, ensure_ascii=False) + "\n")
            jsonl_path = f.name

        logger.info("Uploading batch request file...")
        uploaded_file = gemini_client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="jsonl"
            )
        )
        logger.info("Uploaded file: %s", uploaded_file.name)

        logger.info("Submitting batch job...")
        batch_job = gemini_client.batches.create(
            model=f"models/{client.model}",
            src=uploaded_file.name,
            config={"display_name": display_name},
        )
        logger.info("Batch job created: %s", batch_job.name)
        if on_batch_created:
            on_batch_created(batch_job.name)

    try:
        poll_start = time.monotonic()
        batch_timeout = get_batch_timeout()
        completed_states = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
        }
        while batch_job.state.name not in completed_states:
            if time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"Gemini batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.state.name})"
                )
            logger.debug("State: %s -- polling in %ds...", batch_job.state.name, poll_interval)
            time.sleep(poll_interval)
            batch_job = gemini_client.batches.get(name=batch_job.name)

        logger.info("Batch job finished: %s", batch_job.state.name)
        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch failed: {batch_job.state.name}")

        if not batch_job.dest or not batch_job.dest.file_name:
            raise RuntimeError("No output file in batch job result")

        logger.info("Downloading results from: %s", batch_job.dest.file_name)
        result_bytes = gemini_client.files.download(file=batch_job.dest.file_name)
        result_text = result_bytes.decode("utf-8")

        raw_entries = {}
        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            key = result.get("key")
            response = result.get("response")

            if response:
                candidates = response.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                usage = response.get("usageMetadata", {})
                raw_entries[key] = {
                    "text": text,
                    "usage": {
                        "input_tokens": usage.get("promptTokenCount", 0),
                        "output_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0),
                    },
                }
            else:
                error = result.get("error")
                raw_entries[key] = {
                    "text": "",
                    "error": str(error) if error else "No response",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
        return raw_entries
    finally:
        if jsonl_path:
            os.unlink(jsonl_path)


# ===================================================================
# OpenAI Batch API
# ===================================================================

def _run_batch_openai(client, entries, json_mode, display_name, poll_interval,
                      thinking=False, thinking_budget=0, reasoning_effort="",
                      existing_batch_id=None, on_batch_created=None):
    """OpenAI batch: upload JSONL, create batch, poll, download results.

    If existing_batch_id is set, skip upload+create and retrieve that batch.
    """
    import tempfile
    openai_client = client._client
    max_tokens = MAX_OUTPUT_TOKENS["openai"]
    jsonl_path = None

    if existing_batch_id:
        batch_job = openai_client.batches.retrieve(existing_batch_id)
    else:
        # Write OpenAI-format JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                          encoding="utf-8") as f:
            for entry in entries:
                key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
                if not entry_max_tokens or entry_max_tokens > max_tokens:
                    entry_max_tokens = max_tokens

                # OpenAI uses automatic prefix caching; concatenate the prefix so the
                # same static head is always at the front of the message.
                cacheable_prefix = entry.get("cacheable_prefix")
                effective_prompt = (cacheable_prefix or "") + prompt_text

                if images:
                    image_blocks = _build_image_blocks_openai(
                        images, use_url=_should_use_presigned_url(),
                    )
                    content = _interleave_text_and_images(
                        effective_prompt, image_blocks, lambda s: {"type": "text", "text": s},
                    )
                else:
                    content = effective_prompt

                body = {
                    "model": client.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_completion_tokens": entry_max_tokens,
                }
                if json_mode and entry_json_mode:
                    body["response_format"] = {"type": "json_object"}
                if reasoning_effort:
                    body["reasoning_effort"] = reasoning_effort
                line = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            jsonl_path = f.name

        logger.info("Uploading batch request file...")
        with open(jsonl_path, "rb") as f:
            uploaded_file = openai_client.files.create(file=f, purpose="batch")
        logger.info("Uploaded file: %s", uploaded_file.id)

        logger.info("Submitting batch job...")
        batch_job = openai_client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": display_name},
        )
        logger.info("Batch job created: %s", batch_job.id)
        if on_batch_created:
            on_batch_created(batch_job.id)

    try:
        poll_start = time.monotonic()
        batch_timeout = get_batch_timeout()
        terminal_states = {"completed", "failed", "expired", "cancelled"}
        while batch_job.status not in terminal_states:
            if time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"OpenAI batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.status})"
                )
            logger.debug("Status: %s -- polling in %ds...", batch_job.status, poll_interval)
            time.sleep(poll_interval)
            batch_job = openai_client.batches.retrieve(batch_job.id)

        logger.info("Batch job finished: %s", batch_job.status)
        if batch_job.status != "completed":
            raise RuntimeError(f"OpenAI batch failed: {batch_job.status}")

        if not batch_job.output_file_id:
            raise RuntimeError("No output file in batch job result")

        logger.info("Downloading results from: %s", batch_job.output_file_id)
        result_bytes = openai_client.files.content(batch_job.output_file_id).content
        result_text = result_bytes.decode("utf-8")

        raw_entries = {}
        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            key = result.get("custom_id")
            error = result.get("error")

            if error:
                raw_entries[key] = {
                    "text": "",
                    "error": str(error),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
                continue

            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            text = ""
            if choices:
                text = choices[0].get("message", {}).get("content", "")

            usage = response_body.get("usage", {})
            raw_entries[key] = {
                "text": text,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
        return raw_entries
    finally:
        if jsonl_path:
            os.unlink(jsonl_path)


# ===================================================================
# Anthropic Batch API
# ===================================================================

def _run_batch_anthropic(client, entries, json_mode, display_name, poll_interval,
                         thinking=False, thinking_budget=0, reasoning_effort="",
                         enable_cache=False,
                         existing_batch_id=None, on_batch_created=None):
    """Anthropic batch: create message batch, poll, stream results.

    If existing_batch_id is set, skip submission and retrieve that batch.
    The id_to_key mapping is rebuilt deterministically from `entries` order
    (so callers must pass the same entries that were originally submitted).
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    anthropic_client = client._client
    max_tokens = MAX_OUTPUT_TOKENS["anthropic"]
    # Apply per-model output cap (e.g. Haiku 4.5 / Sonnet 4.6 cap at 64K).
    for prefix, cap in _ANTHROPIC_MAX_OUTPUT_CAP.items():
        if client.model and client.model.startswith(prefix):
            max_tokens = min(max_tokens, cap)
            break

    # id_to_key mapping is deterministic in entries order, so it can be rebuilt
    # on resume without re-submitting. Anthropic custom_id has a 64-char limit,
    # so we use short indexed IDs.
    id_to_key = {f"r{i}": _extract_entry(e)[0] for i, e in enumerate(entries)}

    if existing_batch_id:
        message_batch = anthropic_client.messages.batches.retrieve(existing_batch_id)
    else:
        thinking_param = (
            _anthropic_thinking_param(client.model, thinking_budget) if thinking else None
        )
        thinking_min = 0
        if thinking_param is not None and thinking_param.get("type") == "enabled":
            # Legacy enabled mode needs max_tokens >= budget + headroom.
            thinking_min = thinking_param["budget_tokens"] + 64

        requests = []
        for i, entry in enumerate(entries):
            key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
            if not entry_max_tokens or entry_max_tokens > max_tokens:
                entry_max_tokens = max_tokens
            if thinking_min and entry_max_tokens < thinking_min:
                entry_max_tokens = thinking_min

            cacheable_prefix = entry.get("cacheable_prefix")

            if images:
                image_blocks = _build_image_blocks_anthropic(
                    images, use_url=_should_use_presigned_url(), enable_cache=enable_cache,
                )
                content = _interleave_text_and_images(
                    prompt_text, image_blocks, lambda s: {"type": "text", "text": s},
                )
                if cacheable_prefix is not None:
                    content = [
                        {"type": "text", "text": cacheable_prefix,
                         "cache_control": {"type": "ephemeral"}},
                    ] + content
            elif cacheable_prefix is not None:
                content = [
                    {"type": "text", "text": cacheable_prefix,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompt_text},
                ]
            else:
                content = prompt_text

            params = {
                "model": client.model,
                "max_tokens": entry_max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if json_mode and entry_json_mode:
                params["system"] = (
                    "You must respond with valid JSON only. "
                    "Do not include markdown code fences, explanations, or any text "
                    "outside the JSON object."
                )
            if thinking_param is not None:
                params["thinking"] = thinking_param

            requests.append(Request(
                custom_id=f"r{i}",
                params=MessageCreateParamsNonStreaming(**params),
            ))

        logger.info("Submitting batch (%d requests)...", len(requests))
        retry_cfg = get_retry_config()
        max_retries = retry_cfg.get("max_retries", 5)
        base_delay = retry_cfg.get("base_delay", 5)
        for attempt in range(max_retries):
            try:
                message_batch = anthropic_client.messages.batches.create(requests=requests)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Batch submit error (attempt %d/%d): %s. Retrying in %ds...",
                                   attempt + 1, max_retries, e, delay)
                    time.sleep(delay)
                else:
                    raise
        logger.info("Batch created: %s", message_batch.id)
        if on_batch_created:
            on_batch_created(message_batch.id)

    poll_start = time.monotonic()
    batch_timeout = get_batch_timeout()
    while message_batch.processing_status != "ended":
        if time.monotonic() - poll_start > batch_timeout:
            raise RuntimeError(
                f"Anthropic batch timed out after {batch_timeout}s "
                f"(state: {message_batch.processing_status})"
            )
        logger.debug("Status: %s -- polling in %ds...", message_batch.processing_status, poll_interval)
        time.sleep(poll_interval)
        message_batch = anthropic_client.messages.batches.retrieve(message_batch.id)

    logger.info("Batch finished: %s", message_batch.processing_status)
    logger.info("  Counts: %s", message_batch.request_counts)

    # Parse results -- map short IDs back to original keys
    raw_entries = {}
    for result in anthropic_client.messages.batches.results(message_batch.id):
        key = id_to_key.get(result.custom_id, result.custom_id)

        if result.result.type == "succeeded":
            message = result.result.message
            # Skip thinking blocks, extract (and concatenate) text blocks
            text = _extract_anthropic_text(message.content)
            if json_mode:
                text = _strip_json_fences(text)
            usage = {
                "input_tokens": message.usage.input_tokens or 0,
                "output_tokens": message.usage.output_tokens or 0,
                "total_tokens": (message.usage.input_tokens or 0) + (message.usage.output_tokens or 0),
            }
            raw_entries[key] = {"text": text, "usage": usage}
        else:
            error_msg = f"{result.result.type}"
            if hasattr(result.result, "error") and result.result.error:
                error_msg = f"{result.result.type}: {result.result.error}"
            raw_entries[key] = {
                "text": "",
                "error": error_msg,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }

    return raw_entries
