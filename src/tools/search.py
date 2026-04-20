"""Web search tool (DuckDuckGo Instant Answer + optional Tavily)."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from src.tools.registry import ToolContext, default_registry, tool

logger = logging.getLogger(__name__)

DUCKDUCKGO_HOST = "api.duckduckgo.com"
TAVILY_HOST = "api.tavily.com"


@tool(
    default_registry,
    name="search_web",
    description="Search the public web for up-to-date information. Uses Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo Instant Answer.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
    },
)
def search_web(ctx: ToolContext, query: str, limit: int = 5) -> list[dict[str, str]]:
    if ctx.settings.tavily_api_key:
        return _tavily_search(ctx.settings.tavily_api_key, query, limit)
    return _ddg_search(query, limit)


def _ddg_search(query: str, limit: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": 1, "no_html": 1})
    url = f"https://{DUCKDUCKGO_HOST}/?{params}"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != DUCKDUCKGO_HOST:
        raise ValueError("invalid web search URL")
    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    results: list[dict[str, str]] = []
    if payload.get("AbstractURL"):
        results.append({"title": payload.get("AbstractText", ""), "url": payload["AbstractURL"]})
    for item in payload.get("RelatedTopics", []):
        if "Text" in item and "FirstURL" in item:
            results.append({"title": item["Text"], "url": item["FirstURL"]})
            if len(results) >= limit:
                break
    return results[:limit]


def _tavily_search(api_key: str, query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://{TAVILY_HOST}/search"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != TAVILY_HOST:
        raise ValueError("invalid Tavily URL")
    body = json.dumps({"api_key": api_key, "query": query, "max_results": limit}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in payload.get("results", [])[:limit]
    ]
