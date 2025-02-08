"""A format-neutral conversation: what every reader produces and every writer consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLES = ("system", "user", "assistant", "tool")


@dataclass
class ToolCall:
    name: str
    arguments: str  # raw JSON text as it appears in the data; checking it parses is a lint rule
    id: str | None = None


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # on tool results: the call being answered
    name: str | None = None  # on tool results: the tool's name, where the format records it


@dataclass
class Tool:
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None  # JSON Schema for the arguments


@dataclass
class Conversation:
    messages: list[Message]
    tools: list[Tool] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # other keys on the record, carried through

    def tool(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)
