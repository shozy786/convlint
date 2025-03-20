"""Lint rules. Each rule looks at one conversation and yields the problems it finds."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from jsonschema import exceptions as schema_errors
from jsonschema import validators

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


@rule("tool-call-pairing", "error", "Every tool call gets exactly one result before the conversation moves on")
def _tool_call_pairing(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    # call id -> turn of the assistant message that made it; reset at every assistant turn
    pending: dict[str, int] = {}
    unnamed: list[int] = []  # calls without ids (ShareGPT-style), answered in order

    def unanswered() -> Iterator[tuple[str, int | None]]:
        for call_id, turn in pending.items():
            yield f"tool call {call_id!r} never gets a result", turn
        for turn in unnamed:
            yield "tool call never gets a result", turn
        pending.clear()
        unnamed.clear()

    for i, m in enumerate(conv.messages):
        if m.role == "tool":
            if m.tool_call_id is not None and m.tool_call_id in pending:
                del pending[m.tool_call_id]
            elif m.tool_call_id is None and (unnamed or pending):
                if unnamed:
                    unnamed.pop(0)
                else:
                    pending.pop(next(iter(pending)))
            elif m.tool_call_id is not None:
                yield f"tool result for {m.tool_call_id!r}, which is not an open call", i
            else:
                yield "tool result with no tool call before it", i
            continue
        yield from unanswered()
        for call in m.tool_calls if m.role == "assistant" else ():
            if call.id is None:
                unnamed.append(i)
            else:
                pending[call.id] = i
    yield from unanswered()


@rule("duplicate-tool-call-id", "error", "Tool call ids are unique within a conversation")
def _duplicate_ids(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    seen: dict[str, int] = {}
    for i, m in enumerate(conv.messages):
        for call in m.tool_calls:
            if call.id is None:
                continue
            if call.id in seen:
                yield f"tool call id {call.id!r} already used in turn {seen[call.id]}", i
            seen.setdefault(call.id, i)


@rule("tool-calls-outside-assistant", "error", "Only assistant messages make tool calls")
def _calls_outside_assistant(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages):
        if m.tool_calls and m.role != "assistant":
            yield f"{m.role} message has tool calls", i


@rule("invalid-tool-arguments", "error", "Tool call arguments are a JSON object")
def _invalid_arguments(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for i, m in enumerate(conv.messages):
        for call in m.tool_calls:
            parsed = _arguments(call.arguments)
            if parsed is _UNPARSEABLE:
                yield f"{call.name}: arguments are not valid JSON: {call.arguments[:60]!r}", i
            elif not isinstance(parsed, dict):
                yield f"{call.name}: arguments must be a JSON object, got {type(parsed).__name__}", i


@rule("unknown-tool", "error", "Tool calls only use tools declared on the conversation")
def _unknown_tool(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    declared = {t.name for t in conv.tools}
    for i, m in enumerate(conv.messages):
        for call in m.tool_calls:
            if not declared:
                yield f"calls {call.name!r} but no tools are declared; the model learns to call tools it was never shown", i
            elif call.name not in declared:
                yield f"calls {call.name!r}, which is not among the declared tools {sorted(declared)}", i


@rule("invalid-tool-schema", "error", "Declared tool parameters are a valid JSON Schema")
def _invalid_schema(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    for tool in conv.tools:
        if tool.parameters is None:
            continue
        try:
            validators.validator_for(tool.parameters).check_schema(tool.parameters)
        except schema_errors.SchemaError as e:
            yield f"tool {tool.name!r} has an invalid parameters schema: {e.message}", None


@rule("arguments-schema-mismatch", "error", "Tool call arguments validate against the tool's schema")
def _schema_mismatch(conv: Conversation) -> Iterator[tuple[str, int | None]]:
    checkers: dict[str, Any] = {}
    for tool in conv.tools:
        if tool.parameters is None:
            continue
        cls = validators.validator_for(tool.parameters)
        try:
            cls.check_schema(tool.parameters)
        except schema_errors.SchemaError:
            continue  # reported by invalid-tool-schema
        checkers[tool.name] = cls(tool.parameters)
    for i, m in enumerate(conv.messages):
        for call in m.tool_calls:
            checker = checkers.get(call.name)
            parsed = _arguments(call.arguments)
            if checker is None or not isinstance(parsed, dict):
                continue
            for error in sorted(checker.iter_errors(parsed), key=lambda e: list(e.path)):
                where = "/".join(str(p) for p in error.path)
                yield f"{call.name}: {where + ': ' if where else ''}{error.message}", i


_UNPARSEABLE = object()


def _arguments(text: str) -> Any:
    if not text.strip():
        return {}  # "no arguments" is commonly written as an empty string
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _UNPARSEABLE
