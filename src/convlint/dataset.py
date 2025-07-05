"""Checks that look across the whole dataset rather than one conversation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .model import Conversation
from .rules import Problem

Location = tuple[str, int]  # (path, line)


def essence(conversation: Conversation) -> list[list[object]]:
    """What a trainer sees: roles, whitespace-normalized text, tool calls with canonical JSON."""
    out = []
    for m in conversation.messages:
        calls = []
        for c in m.tool_calls:
            try:
                args = json.dumps(json.loads(c.arguments or "{}"), sort_keys=True)
            except json.JSONDecodeError:
                args = c.arguments
            calls.append([c.name, args])
        out.append([m.role, " ".join((m.content or "").split()), calls])
    return out


def fingerprint(conversation: Conversation) -> str:
    return hashlib.sha1(json.dumps(essence(conversation), ensure_ascii=False).encode()).hexdigest()


class ExactDuplicates:
    """Repeated conversations over-weight whatever they teach and leak between train/eval splits."""

    def __init__(self) -> None:
        self.first: dict[str, Location] = {}

    def observe(self, where: Location, conversation: Conversation) -> Iterable[Problem]:
        key = fingerprint(conversation)
        if key in self.first:
            path, line = self.first[key]
            return [Problem("duplicate", "warning", f"same conversation as {path}:{line}")]
        self.first[key] = where
        return []

    def finish(self) -> Iterable[tuple[Location, Problem]]:
        return []


@dataclass(frozen=True)
class DatasetRule:
    name: str
    severity: str
    summary: str
    factory: Callable[[], ExactDuplicates]


DATASET_RULES = {
    "duplicate": DatasetRule("duplicate", "warning", "No conversation appears twice", ExactDuplicates),
}
