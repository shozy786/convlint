import pytest

from convlint.formats import FormatError, read_openai

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}

OPENAI = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Weather in Lahore?"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lahore"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 31}'},
        {"role": "assistant", "content": "It's 31°C in Lahore."},
    ],
    "tools": [WEATHER_TOOL],
    "source": "batch-7",
}


def test_read_openai():
    conv = read_openai(OPENAI)
    assert [m.role for m in conv.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert conv.messages[1].content == "Weather in Lahore?"
    (call,) = conv.messages[2].tool_calls
    assert (call.id, call.name, call.arguments) == ("call_1", "get_weather", '{"city": "Lahore"}')
    assert conv.messages[3].tool_call_id == "call_1"
    assert conv.tool("get_weather").parameters["required"] == ["city"]
    assert conv.meta == {"source": "batch-7"}


def test_legacy_functions():
    conv = read_openai(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "function_call": {"name": "f", "arguments": {"x": 1}}},
                {"role": "function", "name": "f", "content": "ok"},
            ],
            "functions": [{"name": "f", "parameters": {"type": "object"}}],
        }
    )
    assert conv.messages[1].tool_calls[0].arguments == '{"x": 1}'
    assert conv.messages[2].role == "tool" and conv.messages[2].name == "f"
    assert conv.tool("f") is not None


@pytest.mark.parametrize(
    "record, message",
    [
        ({"conversations": []}, "expected a 'messages' list"),
        ({"messages": [{"content": "x"}]}, r"messages\[0\]: expected an object with a 'role'"),
        ({"messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}]}, r"messages\[0\].tool_calls\[0\]"),
        ({"messages": [], "tools": [{"name": "f"}]}, r"tools\[0\]"),
    ],
)
def test_malformed(record, message):
    with pytest.raises(FormatError, match=message):
        read_openai(record)
