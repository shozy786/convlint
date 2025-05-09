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
