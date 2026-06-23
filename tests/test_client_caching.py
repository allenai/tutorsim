"""Tests for ModelClient.generate cacheable_prefix kwarg + per-provider plumbing."""
from unittest.mock import MagicMock, patch

import pytest

from annotator.core.client import ModelClient


def _make_anthropic_client():
    client = ModelClient.__new__(ModelClient)
    client.model = "claude-opus-4-8"
    client.provider = "anthropic"
    client._client = MagicMock()
    return client


def _make_openai_client():
    client = ModelClient.__new__(ModelClient)
    client.model = "gpt-5.5"
    client.provider = "openai"
    client._client = MagicMock()
    return client


def _anthropic_response(cache_creation=0, cache_read=0):
    resp = MagicMock()
    text_block = MagicMock(); text_block.type = "text"; text_block.text = "ok"
    resp.content = [text_block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 5
    resp.usage.cache_creation_input_tokens = cache_creation
    resp.usage.cache_read_input_tokens = cache_read
    return resp


def _openai_response(cached=0):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "ok"
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 105
    resp.usage.prompt_tokens_details = MagicMock()
    resp.usage.prompt_tokens_details.cached_tokens = cached
    return resp


def test_anthropic_cacheable_prefix_marks_head_block():
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response()

    client.generate("the tail", json_mode=False, max_tokens=64,
                    cacheable_prefix="the cacheable head")

    kwargs = client._client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "the cacheable head",
                          "cache_control": {"type": "ephemeral"}}
    assert content[1] == {"type": "text", "text": "the tail"}


def test_anthropic_cacheable_prefix_none_uses_single_string_content():
    """Back-compat: when cacheable_prefix is None, behavior is unchanged."""
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response()

    client.generate("just the tail", json_mode=False, max_tokens=64)

    kwargs = client._client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    # Legacy path: content is a plain string.
    assert content == "just the tail"


def test_anthropic_usage_captures_cache_creation_and_read():
    client = _make_anthropic_client()
    client._client.messages.create.return_value = _anthropic_response(
        cache_creation=80, cache_read=20,
    )

    resp = client.generate("tail", json_mode=False, max_tokens=64,
                           cacheable_prefix="head"*200)

    assert resp.usage["cache_creation_input_tokens"] == 80
    assert resp.usage["cache_read_input_tokens"] == 20


def test_openai_cacheable_prefix_concatenates_into_single_user_message():
    client = _make_openai_client()
    client._client.chat.completions.create.return_value = _openai_response()

    client.generate("the tail", json_mode=False, max_tokens=64,
                    cacheable_prefix="the head")

    kwargs = client._client.chat.completions.create.call_args.kwargs
    msgs = kwargs["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    # OpenAI gets the concatenated text; auto-cache handles the rest.
    assert msgs[0]["content"] == "the head" + "the tail"


def test_openai_usage_captures_cached_tokens():
    client = _make_openai_client()
    client._client.chat.completions.create.return_value = _openai_response(cached=42)

    resp = client.generate("tail", json_mode=False, max_tokens=64,
                           cacheable_prefix="head")

    assert resp.usage["cached_tokens"] == 42
