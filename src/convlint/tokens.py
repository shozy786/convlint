"""Token counts: exact for OpenAI encodings when tiktoken is installed, estimated otherwise.

Trainers use their own tokenizer (Llama, Qwen, Mistral...), so even the "exact" count is a
proxy. It's close enough to catch conversations that will be truncated, which is the point.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from .model import Conversation

# chat templates wrap each message in role markers and separators
PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=4)
def _encoder(name: str) -> Any:
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken.get_encoding(name)


def count(text: str, encoding: str = "o200k_base") -> int:
    enc = _encoder(encoding)
    if enc is None:
        return math.ceil(len(text) / 4)  # ~4 characters per token for English prose
    return len(enc.encode(text, disallowed_special=()))


def conversation_tokens(conversation: Conversation, encoding: str = "o200k_base") -> int:
    parts = [json.dumps([t.name, t.description, t.parameters], ensure_ascii=False) for t in conversation.tools]
    for m in conversation.messages:
        parts.append(m.content or "")
        parts += [f"{c.name}{c.arguments}" for c in m.tool_calls]
    return sum(count(p, encoding) for p in parts if p) + PER_MESSAGE_OVERHEAD * len(conversation.messages)
