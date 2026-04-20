"""Tool package.

Public API: the registry plumbing plus the shared default_registry.
Tool functions themselves register via side-effect imports of the
sibling submodules below — importing this package is enough for the
agent to see every built-in tool.
"""
from src.tools.registry import (
    ToolContext,
    ToolDef,
    ToolExecutor,
    ToolRegistry,
    default_registry,
    tool,
)

# Side-effect imports: each submodule uses @tool(default_registry, ...) on
# one or more functions. Importing them registers those tools.
from src.tools import (  # noqa: F401  (imported for side effects)
    image,
    search,
    slack,
    time,
    web,
)

__all__ = [
    "ToolContext",
    "ToolDef",
    "ToolExecutor",
    "ToolRegistry",
    "default_registry",
    "tool",
]
