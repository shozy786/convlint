"""Readers for the dataset formats that fine-tuning APIs and trainers expect."""

from __future__ import annotations

import json
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
                calls.append(ToolCall(block["name"], json.dumps(block.get("input", {}), ensure_ascii=False), block.get("id")))
            elif kind == "tool_result":
                messages.append(Message("tool", block.get("content"), tool_call_id=block.get("tool_use_id")))
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
