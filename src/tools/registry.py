"""Tool registry + executor. Tool functions live in sibling submodules
(slack.py, search.py, web.py, image.py, time.py) and register themselves
via the @tool decorator on import."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from botocore.exceptions import BotoCoreError, ClientError
from slack_sdk.errors import SlackApiError

from src.config import Settings
from src.llms import LLMProvider, ToolCall

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Any]
    timeout: float | None = None  # None -> use executor default


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolDef] = field(default_factory=dict)

    def register(self, td: ToolDef) -> None:
        self._tools[td.name] = td

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]


def tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    parameters: dict[str, Any],
    timeout: float | None = None,
):
    def decorator(fn: Callable[..., Any]):
        registry.register(
            ToolDef(name=name, description=description, parameters=parameters, fn=fn, timeout=timeout)
        )
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class ToolContext:
    slack_client: Any
    channel: str
    thread_ts: str
    event: dict[str, Any]
    settings: Settings
    llm: LLMProvider


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class ToolExecutor:
    def __init__(self, context: ToolContext, registry: ToolRegistry, timeout: float = 20.0):
        self.context = context
        self.registry = registry
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=2)

    def execute(self, call: ToolCall) -> dict[str, Any]:
        td = self.registry.get(call.name)
        started = time.monotonic()
        if td is None:
            return {"ok": False, "error": f"unknown tool: {call.name}"}
        effective_timeout = td.timeout if td.timeout is not None else self.timeout
        try:
            future = self._pool.submit(td.fn, self.context, **(call.arguments or {}))
            result = future.result(timeout=effective_timeout)
            return {"ok": True, "result": result, "duration_ms": int((time.monotonic() - started) * 1000)}
        except FuturesTimeout:
            logger.warning("tool %s timed out after %.1fs", call.name, effective_timeout)
            return {"ok": False, "error": f"tool '{call.name}' timed out after {effective_timeout}s"}
        except (
            TypeError,
            ValueError,
            KeyError,
            urllib.error.URLError,
            json.JSONDecodeError,
            SlackApiError,
            BotoCoreError,
            ClientError,
        ) as exc:
            logger.exception("tool %s failed", call.name)
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Built-in tools
# --------------------------------------------------------------------------- #

default_registry = ToolRegistry()
