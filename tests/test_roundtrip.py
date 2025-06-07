"""Converting between any two formats keeps what a trainer would see."""

import json
from itertools import product

import pytest

from convlint.formats import READERS, WRITERS
from convlint.rules import lint

from .test_formats import ANTHROPIC, OPENAI, SHAREGPT

SOURCES = {"openai": OPENAI, "anthropic": ANTHROPIC, "sharegpt": SHAREGPT}


def essence(conv):
    """Roles, text, tool names and parsed arguments: the parts that survive any format."""
    return [
        (m.role, (m.content or "").strip(), [(c.name, json.loads(c.arguments or "{}")) for c in m.tool_calls])
        for m in conv.messages
    ]


@pytest.mark.parametrize("source, target", list(product(SOURCES, WRITERS)))
def test_round_trip(source, target):
    original = READERS[source](SOURCES[source])
    converted = READERS[target](json.loads(json.dumps(WRITERS[target](original))))
    assert essence(converted) == essence(original)
    assert [t.name for t in converted.tools] == [t.name for t in original.tools]
    assert lint(converted) == lint(original) == []


def test_anthropic_tool_result_content_blocks():
    record = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "31C"}, {"type": "text", "text": "sunny"}],
                    }
                ],
            },
            {"role": "assistant", "content": "31C and sunny."},
        ],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    }
    conv = READERS["anthropic"](record)
    assert conv.messages[2].content == "31C\nsunny"
    assert lint(conv) == []
