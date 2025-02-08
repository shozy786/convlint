"""Lint rules. Each rule looks at one conversation and yields the problems it finds."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .model import ROLES, Conversation


@dataclass(frozen=True, slots=True)
class Problem:
    rule: str
    severity: str  # error | warning
    message: str
    turn: int | None = None  # index into conversation.messages


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    summary: str
    check: Callable[[Conversation], Iterable[tuple[str, int | None]]]  # (message, turn)


RULES: dict[str, Rule] = {}


def rule(name: str, severity: str, summary: str):
    def register(fn: Callable[[Conversation], Iterable[tuple[str, int | None]]]):
        RULES[name] = Rule(name, severity, summary, fn)
        return fn

    return register


def lint(conversation: Conversation, select: Iterable[str] | None = None, ignore: Iterable[str] = ()) -> list[Problem]:
    names = set(select) if select is not None else set(RULES)
    names -= set(ignore)
    return [
        Problem(r.name, r.severity, message, turn)
        for r in RULES.values()
        if r.name in names
        for message, turn in r.check(conversation)
    ]


@rule("unknown-role", "error", "Every message has a role of system, user, assistant or tool")
def _unknown_role(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages):
        if m.role not in ROLES:
            yield f"unknown role {m.role!r}", i


@rule("system-not-first", "error", "A system message can only be the first message")
def _system_not_first(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages[1:], 1):
        if m.role == "system":
            yield "system message after the first turn; most chat templates drop or reject it", i


@rule("empty-message", "error", "Messages have content (assistant messages may have only tool calls)")
def _empty_message(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages):
        empty = not (m.content or "").strip()
        if empty and not (m.role == "assistant" and m.tool_calls):
            yield f"empty {m.role} message", i


@rule("consecutive-turns", "warning", "User and assistant turns alternate")
def _consecutive(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i in range(1, len(conv.messages)):
        prev, cur = conv.messages[i - 1], conv.messages[i]
        # an assistant tool call followed by more assistant text is legitimate only via tool results
        if cur.role == prev.role and cur.role in {"user", "assistant"}:
            yield f"two {cur.role} messages in a row; merge them or the template may misattribute turns", i


@rule("no-assistant-turn", "error", "There is at least one assistant message to train on")
def _no_assistant(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    if not any(m.role == "assistant" for m in conv.messages):
        yield "no assistant message; nothing for the model to learn from", None


@rule("bad-last-turn", "error", "The conversation ends with an assistant message")
def _bad_last_turn(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    if conv.messages and conv.messages[-1].role != "assistant":
        last = len(conv.messages) - 1
        yield f"ends with a {conv.messages[-1].role} message, which is never trained on", last
    elif conv.messages and conv.messages[-1].tool_calls and not (conv.messages[-1].content or "").strip():
        yield "ends on a tool call with no result or final answer", len(conv.messages) - 1


@rule("first-turn-not-user", "warning", "The first non-system message comes from the user")
def _first_turn(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages):
        if m.role == "system":
            continue
        if m.role != "user":
            yield f"first turn is {m.role}, not user", i
        return
