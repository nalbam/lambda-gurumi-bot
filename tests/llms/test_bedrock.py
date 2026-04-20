"""Tests for src.llms.bedrock."""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from src.llms.bedrock import BedrockProvider


def _bedrock_response(payload: dict):
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    return {"body": body}


def test_bedrock_inference_profile_routes_to_claude():
    """us.anthropic.claude-* inference profile IDs must still hit the Claude
    path (tool_use, messages API), not the unknown-family fallback."""
    provider = BedrockProvider(
        model="us.anthropic.claude-opus-4-6-v1",
        image_model="amazon.nova-canvas-v1:0",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )
    tools = [{"name": "search_web", "description": "", "parameters": {"type": "object"}}]
    result = provider.chat(system="s", messages=[], tools=tools)
    assert result.content == "hi"
    # Claude body carries tools (not Nova's toolConfig)
    body = provider._client.invoke_model.call_args.kwargs["body"]
    parsed = json.loads(body)
    assert "tools" in parsed  # routed into _claude_chat, not fallback
    assert parsed["tools"][0]["name"] == "search_web"


def test_bedrock_inference_profile_image_routing():
    """global./us. prefixed image model IDs still reach the Titan/Nova body."""
    provider = BedrockProvider(
        model="us.anthropic.claude-opus-4-6-v1",
        image_model="us.amazon.nova-canvas-v1:0",
        region="us-east-1",
    )
    body = provider._build_image_body("cat")
    assert body["taskType"] == "TEXT_IMAGE"


def test_bedrock_claude_chat_text():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {
            "content": [{"type": "text", "text": "안녕"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )
    result = provider.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "안녕"
    assert result.stop_reason == "end_turn"
    assert result.token_usage == {"input": 5, "output": 7}


def test_bedrock_claude_chat_with_tool_use():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {
            "content": [
                {"type": "text", "text": "I'll search."},
                {"type": "tool_use", "id": "tu_1", "name": "search_web", "input": {"query": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )
    tools = [{"name": "search_web", "description": "", "parameters": {"type": "object"}}]
    result = provider.chat(system="s", messages=[], tools=tools)
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "search_web"
    assert result.tool_calls[0].arguments == {"query": "x"}


def test_bedrock_message_translation_tool_role():
    messages = [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "foo", "arguments": {"a": 1}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{\"ok\":true}"},
    ]
    translated = BedrockProvider._to_anthropic_messages(messages)
    assert translated[0] == {"role": "user", "content": "ask"}
    assert translated[1]["role"] == "assistant"
    assert translated[1]["content"][0]["type"] == "tool_use"
    assert translated[1]["content"][0]["name"] == "foo"
    assert translated[2]["role"] == "user"
    assert translated[2]["content"][0]["type"] == "tool_result"


def test_bedrock_nova_chat_text():
    provider = BedrockProvider(model="amazon.nova-pro-v1:0", image_model="amazon.nova-canvas-v1:0", region="us-east-1")
    provider._client = MagicMock()
    provider._client.converse.return_value = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 2},
    }
    result = provider.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "hi"
    assert result.stop_reason == "end_turn"
    assert result.token_usage == {"input": 1, "output": 2}


def test_bedrock_nova_tool_use():
    provider = BedrockProvider(model="amazon.nova-pro-v1:0", image_model="amazon.nova-canvas-v1:0", region="us-east-1")
    provider._client = MagicMock()
    provider._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "let me search"},
                    {"toolUse": {"toolUseId": "tu1", "name": "search_web", "input": {"query": "q"}}},
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 2, "outputTokens": 3},
    }
    result = provider.chat(
        system="s",
        messages=[],
        tools=[{"name": "search_web", "description": "", "parameters": {"type": "object"}}],
    )
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "search_web"


def test_build_image_body_titan():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    body = provider._build_image_body("a cat")
    assert body["taskType"] == "TEXT_IMAGE"


def test_build_image_body_stability():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="stability.stable-diffusion-xl-v1",
        region="us-east-1",
    )
    body = provider._build_image_body("a cat")
    assert body["text_prompts"][0]["text"] == "a cat"


def test_build_image_body_unknown_raises():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="mystery.v1",
        region="us-east-1",
    )
    with pytest.raises(ValueError):
        provider._build_image_body("x")


def test_bedrock_describe_image_returns_text():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"content": [{"type": "text", "text": "a cat"}]}
    )
    out = provider.describe_image(b"fake", "image/png")
    assert out == "a cat"


def test_bedrock_generate_image_titan_returns_bytes():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"images": [base64.b64encode(b"imgdata").decode()]}
    )
    assert provider.generate_image("cat") == b"imgdata"


def test_bedrock_generate_image_stability_returns_bytes():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="stability.stable-diffusion-xl-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"artifacts": [{"base64": base64.b64encode(b"xyz").decode()}]}
    )
    assert provider.generate_image("cat") == b"xyz"
