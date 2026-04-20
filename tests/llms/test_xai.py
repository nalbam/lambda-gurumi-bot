"""Tests for src.llms.xai."""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

from src.llms.xai import XAIProvider


def _openai_completion(content="", tool_calls=None, finish="stop"):
    choice = MagicMock()
    choice.finish_reason = finish
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 10
    completion.usage.completion_tokens = 20
    return completion


def _openai_tool_call(call_id, name, args_obj):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args_obj)
    return tc


def test_xai_provider_uses_xai_base_url_and_api_key():
    """XAIProvider must instantiate OpenAI client with the xAI base URL and
    the explicit api_key, so traffic goes to api.x.ai rather than OpenAI."""
    provider = XAIProvider(
        model="grok-4-1-fast-reasoning",
        image_model="grok-imagine-image",
        api_key="xai-test",
    )
    with patch("openai.OpenAI") as openai_ctor:
        openai_ctor.return_value = MagicMock()
        provider._get_client()
    kwargs = openai_ctor.call_args.kwargs
    assert kwargs.get("base_url") == "https://api.x.ai/v1"
    assert kwargs.get("api_key") == "xai-test"


def test_xai_chat_parses_tool_calls():
    """Grok returns the same wire shape as OpenAI for tool calls; the
    shared parser must turn them into ToolCall objects."""
    provider = XAIProvider(model="grok-4-1-fast-reasoning", image_model="grok-imagine-image", api_key="x")
    provider._client = MagicMock()
    tc = _openai_tool_call("call_g1", "search_web", {"query": "xai"})
    provider._client.chat.completions.create.return_value = _openai_completion(
        tool_calls=[tc], finish="tool_calls"
    )
    result = provider.chat(
        system="s",
        messages=[],
        tools=[{"name": "search_web", "description": "", "parameters": {"type": "object"}}],
    )
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "search_web"
    assert result.tool_calls[0].arguments == {"query": "xai"}


def test_xai_chat_uses_legacy_max_tokens_always():
    """All current grok chat models accept max_tokens + temperature;
    XAIProvider must not switch to max_completion_tokens (OpenAI-only split)."""
    provider = XAIProvider(model="grok-4.20-0309-reasoning", image_model="grok-imagine-image", api_key="x")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="hi")
    provider.chat(system="s", messages=[])
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert "max_tokens" in kwargs
    assert "temperature" in kwargs
    assert "max_completion_tokens" not in kwargs


def test_xai_generate_image_skips_size_and_requests_b64():
    """xAI images.generate rejects `size` (uses aspect_ratio/resolution).
    We must omit it and explicitly ask for b64_json so we can decode bytes
    into files_upload_v2."""
    provider = XAIProvider(model="grok-4-1-fast-reasoning", image_model="grok-imagine-image", api_key="x")
    provider._client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(b64_json=base64.b64encode(b"xai-bytes").decode())]
    provider._client.images.generate.return_value = response

    assert provider.generate_image("a cat") == b"xai-bytes"
    kwargs = provider._client.images.generate.call_args.kwargs
    assert kwargs["model"] == "grok-imagine-image"
    assert kwargs["prompt"] == "a cat"
    assert kwargs["response_format"] == "b64_json"
    assert "size" not in kwargs  # xAI rejects this


def test_xai_stream_chat_emits_deltas():
    provider = XAIProvider(model="grok-4-1-fast-reasoning", image_model="grok-imagine-image", api_key="x")
    provider._client = MagicMock()

    def _chunk(text):
        ch = MagicMock()
        ch.choices[0].delta.content = text
        return ch

    provider._client.chat.completions.create.return_value = iter([_chunk("gr"), _chunk("ok")])
    seen: list[str] = []
    result = provider.stream_chat(system="s", messages=[], on_delta=seen.append)
    assert result == "grok"
    assert seen == ["gr", "ok"]
