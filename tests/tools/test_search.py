"""Tests for src.tools.search."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tests.tools._helpers import _ctx, _settings
from src.tools.registry import ToolContext
from src.tools.search import search_web


# --------------------------------------------------------------------------- #
# search_web
# --------------------------------------------------------------------------- #


def test_search_web_ddg_parses_results():
    ctx = _ctx()
    payload = {
        "AbstractURL": "https://example.com/a",
        "AbstractText": "abstract",
        "RelatedTopics": [{"Text": "t1", "FirstURL": "https://example.com/1"}],
    }
    with patch("src.tools.search.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        results = search_web(ctx, query="q", limit=5)
    assert results[0]["url"] == "https://example.com/a"
    assert results[1]["url"] == "https://example.com/1"


def test_search_web_uses_tavily_when_key_set():
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event={},
        settings=_settings(tavily_api_key="tvly-xyz"),
        llm=MagicMock(),
    )
    payload = {"results": [{"title": "t", "url": "https://x", "content": "c"}]}
    with patch("src.tools.search.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        out = search_web(ctx, query="q", limit=5)
    assert out == [{"title": "t", "url": "https://x", "content": "c"}]
