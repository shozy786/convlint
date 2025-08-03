"""Readers and writers for the dataset formats that fine-tuning APIs and trainers expect."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .model import Conversation, Message, Tool, ToolCall


class FormatError(ValueError):
    pass


def read_openai(record: dict[str, Any]) -> Conversation:
    """OpenAI chat fine-tuning format, including the legacy ``functions``/``function_call`` keys."""
    if not isinstance(record.get("messages"), list):
        raise FormatError("expected a 'messages' list")
    messages = []
    for i, m in enumerate(record["messages"]):
        if not isinstance(m, dict) or "role" not in m:
            raise FormatError(f"messages[{i}]: expected an object with a 'role'")
        calls = [_openai_call(c, f"messages[{i}].tool_calls[{k}]") for k, c in enumerate(m.get("tool_calls") or [])]
        if m.get("function_call"):
            calls.append(_function(m["function_call"], f"messages[{i}].function_call"))
        messages.append(
            Message(
                role="tool" if m["role"] == "function" else m["role"],
                content=_text(m.get("content")),
                tool_calls=calls,
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
            )
        )

    tools = []
    for k, t in enumerate(record.get("tools") or []):
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or "name" not in fn:
            raise FormatError(f"tools[{k}]: expected {{'type': 'function', 'function': {{'name': ...}}}}")
        tools.append(Tool(fn["name"], fn.get("description"), fn.get("parameters")))
    for fn in record.get("functions") or []:
        tools.append(Tool(fn["name"], fn.get("description"), fn.get("parameters")))

    meta = {k: v for k, v in record.items() if k not in {"messages", "tools", "functions"}}
    return Conversation(messages, tools, meta)


def read_anthropic(record: dict[str, Any]) -> Conversation:
    """Anthropic Messages API shape: top-level ``system``, content blocks, ``tool_use``/``tool_result``.

    Tool results arrive inside user messages; they become separate ``tool`` messages here, in
    order, followed by whatever text the user message also carried.
    """
    if not isinstance(record.get("messages"), list):
        raise FormatError("expected a 'messages' list")
    messages = []
    if record.get("system"):
        messages.append(Message("system", _text(record["system"])))
    for i, m in enumerate(record["messages"]):
        if not isinstance(m, dict) or m.get("role") not in {"user", "assistant"}:
            raise FormatError(f"messages[{i}]: expected role 'user' or 'assistant'")
        blocks = m.get("content")
        if isinstance(blocks, str) or blocks is None:
            messages.append(Message(m["role"], blocks))
            continue
        texts, calls = [], []
        for k, block in enumerate(blocks):
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "text":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                if "name" not in block:
                    raise FormatError(f"messages[{i}].content[{k}]: tool_use without a name")
                calls.append(
                    ToolCall(block["name"], json.dumps(block.get("input", {}), ensure_ascii=False), block.get("id"))
                )
            elif kind == "tool_result":
                # content is a string or, as often, a list of text/image blocks
                messages.append(Message("tool", _text(block.get("content")), tool_call_id=block.get("tool_use_id")))
        if m["role"] == "assistant":
            messages.append(Message("assistant", "\n".join(texts) or None, calls))
        elif texts or not any(b.get("type") == "tool_result" for b in blocks if isinstance(b, dict)):
            messages.append(Message("user", "\n".join(texts)))

    tools = []
    for k, t in enumerate(record.get("tools") or []):
        if not isinstance(t, dict) or "name" not in t:
            raise FormatError(f"tools[{k}]: expected an object with a 'name'")
        tools.append(Tool(t["name"], t.get("description"), t.get("input_schema")))
    meta = {k: v for k, v in record.items() if k not in {"system", "messages", "tools"}}
    return Conversation(messages, tools, meta)


_SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "function_call": "assistant",
    "observation": "tool",
    "tool": "tool",
}


def read_sharegpt(record: dict[str, Any]) -> Conversation:
    """ShareGPT as used by LLaMA-Factory and Axolotl: ``conversations`` of ``{"from", "value"}``.

    Tool calls are ``function_call`` turns whose value is ``{"name", "arguments"}`` JSON (or a
    list of them for parallel calls) and results are ``observation`` turns. There are no call
    ids; results answer calls in order.
    """
    turns = record.get("conversations")
    if not isinstance(turns, list):
        raise FormatError("expected a 'conversations' list")
    messages = []
    if record.get("system"):
        messages.append(Message("system", record["system"]))
    for i, turn in enumerate(turns):
        sender = turn.get("from") if isinstance(turn, dict) else None
        if sender not in _SHAREGPT_ROLES:
            raise FormatError(f"conversations[{i}]: unknown 'from' {sender!r}")
        value = turn.get("value")
        if sender != "function_call":
            messages.append(Message(_SHAREGPT_ROLES[sender], value))
            continue
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            raise FormatError(f"conversations[{i}]: function_call value is not JSON") from None
        calls = [_function(c, f"conversations[{i}]") for c in (payload if isinstance(payload, list) else [payload])]
        if messages and messages[-1].role == "assistant" and not messages[-1].tool_calls:
            messages[-1].tool_calls = calls  # "gpt" thought followed by its call is one turn
        else:
            messages.append(Message("assistant", None, calls))

    tools = [Tool(t["name"], t.get("description"), t.get("parameters")) for t in record.get("tools") or []]
    meta = {k: v for k, v in record.items() if k not in {"conversations", "system", "tools"}}
    return Conversation(messages, tools, meta)


def _openai_call(call: Any, where: str) -> ToolCall:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise FormatError(f"{where}: expected {{'id': ..., 'function': {{'name': ..., 'arguments': ...}}}}")
    tool_call = _function(call["function"], where)
    tool_call.id = call.get("id")
    return tool_call


def _function(fn: dict[str, Any], where: str) -> ToolCall:
    if "name" not in fn:
        raise FormatError(f"{where}: missing function name")
    arguments = fn.get("arguments", "")
    if not isinstance(arguments, str):
        # some exporters write the object instead of its JSON text; keep it checkable
        arguments = json.dumps(arguments, ensure_ascii=False)
    return ToolCall(fn["name"], arguments)


def _text(content: Any) -> str | None:
    """Plain text from a string or a list of content parts; non-text parts are dropped."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(texts)
    return str(content)


def detect(record: dict[str, Any]) -> str:
    """Which format a record is in. Text-only OpenAI and Anthropic records look identical, and
    read identically, so the tie goes to OpenAI."""
    if "conversations" in record:
        return "sharegpt"
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise FormatError("can't tell the format: no 'messages' or 'conversations' list")
    if "system" in record or any("input_schema" in t for t in record.get("tools") or [] if isinstance(t, dict)):
        return "anthropic"
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("tool_calls") or m.get("role") in {"system", "tool", "function"}:
            return "openai"
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in {"tool_use", "tool_result"} for b in content
        ):
            return "anthropic"
    return "openai"


READERS = {"openai": read_openai, "anthropic": read_anthropic, "sharegpt": read_sharegpt}


def read_auto(record: dict[str, Any]) -> Conversation:
    return READERS[detect(record)](record)


def with_ids(conversation: Conversation) -> Conversation:
    """A copy where every tool call has an id and every result names the call it answers.

    ShareGPT has no ids at all; results answer calls in order. Formats that need ids get
    stable generated ones (``call_<turn>_<n>``) so conversions are deterministic.
    """
    messages, open_calls = [], []
    for i, m in enumerate(conversation.messages):
        calls = [replace(c, id=c.id or f"call_{i}_{k}") for k, c in enumerate(m.tool_calls)]
        if m.role == "assistant":
            open_calls = [c.id for c in calls]
        tool_call_id = m.tool_call_id
        if m.role == "tool":
            if tool_call_id is None and open_calls:
                tool_call_id = open_calls[0]
            if tool_call_id in open_calls:
                open_calls.remove(tool_call_id)
        messages.append(replace(m, tool_calls=calls, tool_call_id=tool_call_id))
    return replace(conversation, messages=messages)


def write_openai(conversation: Conversation) -> dict[str, Any]:
    conv = with_ids(conversation)
    messages = []
    for m in conv.messages:
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in m.tool_calls
            ]
        if m.role == "tool":
            out["tool_call_id"] = m.tool_call_id
        messages.append(out)
    record: dict[str, Any] = {"messages": messages}
    if conv.tools:
        record["tools"] = [
            {
                "type": "function",
                "function": _drop_none(name=t.name, description=t.description, parameters=t.parameters),
            }
            for t in conv.tools
        ]
    return {**conv.meta, **record}


def write_anthropic(conversation: Conversation) -> dict[str, Any]:
    """Tool results and any user text right after them merge into one user message, since the
    Messages API requires user and assistant turns to alternate."""
    conv = with_ids(conversation)
    record: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    for m in conv.messages:
        if m.role == "system":
            record["system"] = m.content
            continue
        if m.role == "assistant":
            blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
            blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": _parse_arguments(c)} for c in m.tool_calls
            ]
            messages.append({"role": "assistant", "content": blocks})
            continue
        block = (
            {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content or ""}
            if m.role == "tool"
            else {"type": "text", "text": m.content or ""}
        )
        previous = messages[-1] if messages else None
        if previous and previous["role"] == "user" and isinstance(previous["content"], list):
            previous["content"].append(block)
        else:
            messages.append({"role": "user", "content": [block]})
    for m in messages:  # plain text messages read better as strings
        if m["role"] == "user" and len(m["content"]) == 1 and m["content"][0]["type"] == "text":
            m["content"] = m["content"][0]["text"]
    record["messages"] = messages
    if conv.tools:
        record["tools"] = [
            _drop_none(name=t.name, description=t.description, input_schema=t.parameters or {"type": "object"})
            for t in conv.tools
        ]
    return {**conv.meta, **record}


def write_sharegpt(conversation: Conversation) -> dict[str, Any]:
    record: dict[str, Any] = {}
    turns = []
    names = {"user": "human", "assistant": "gpt", "tool": "observation"}
    for m in conversation.messages:
        if m.role == "system":
            record["system"] = m.content
            continue
        if m.content or not m.tool_calls:
            turns.append({"from": names[m.role], "value": m.content or ""})
        if m.tool_calls:
            calls = [{"name": c.name, "arguments": _parse_arguments(c)} for c in m.tool_calls]
            payload = calls[0] if len(calls) == 1 else calls
            turns.append({"from": "function_call", "value": json.dumps(payload, ensure_ascii=False)})
    record["conversations"] = turns
    if conversation.tools:
        record["tools"] = [
            _drop_none(name=t.name, description=t.description, parameters=t.parameters) for t in conversation.tools
        ]
    return {**conversation.meta, **record}


WRITERS = {"openai": write_openai, "anthropic": write_anthropic, "sharegpt": write_sharegpt}


def _parse_arguments(call: ToolCall) -> Any:
    try:
        return json.loads(call.arguments) if call.arguments.strip() else {}
    except json.JSONDecodeError:
        raise FormatError(f"can't convert call to {call.name!r}: arguments are not valid JSON") from None


def _drop_none(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}
