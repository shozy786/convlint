import json

import pytest

from convlint.formats import (
    FormatError,
    detect,
    read_anthropic,
    read_openai,
    read_sharegpt,
    write_anthropic,
    write_openai,
    write_sharegpt,
)

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
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Lahore"}'},
                }
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


ANTHROPIC = {
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "Weather in Lahore?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Lahore"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"temp_c": 31}'}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It's 31°C in Lahore."}]},
    ],
    "tools": [{"name": "get_weather", "description": "Current weather", "input_schema": {"type": "object"}}],
}


def test_read_anthropic():
    conv = read_anthropic(ANTHROPIC)
    assert [m.role for m in conv.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert conv.messages[2].content == "Let me check."
    (call,) = conv.messages[2].tool_calls
    assert (call.id, call.name, call.arguments) == ("toolu_1", "get_weather", '{"city": "Lahore"}')
    assert conv.messages[3].tool_call_id == "toolu_1"
    assert conv.messages[3].content == '{"temp_c": 31}'
    assert conv.tool("get_weather").parameters == {"type": "object"}


def test_anthropic_user_text_alongside_tool_results():
    record = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
                    {"type": "text", "text": "thanks, now summarise"},
                ],
            },
            {"role": "assistant", "content": "Summary."},
        ]
    }
    roles = [(m.role, m.content) for m in read_anthropic(record).messages]
    assert roles[2:] == [("tool", "done"), ("user", "thanks, now summarise"), ("assistant", "Summary.")]


def test_anthropic_rejects_openai_roles():
    with pytest.raises(FormatError, match="expected role 'user' or 'assistant'"):
        read_anthropic({"messages": [{"role": "system", "content": "x"}]})


SHAREGPT = {
    "system": "You are a helpful assistant.",
    "conversations": [
        {"from": "human", "value": "Weather in Lahore and Karachi?"},
        {"from": "gpt", "value": "Checking both."},
        {
            "from": "function_call",
            "value": '[{"name": "get_weather", "arguments": {"city": "Lahore"}}, '
            '{"name": "get_weather", "arguments": {"city": "Karachi"}}]',
        },
        {"from": "observation", "value": '{"temp_c": 31}'},
        {"from": "observation", "value": '{"temp_c": 34}'},
        {"from": "gpt", "value": "Lahore 31°C, Karachi 34°C."},
    ],
    "tools": [{"name": "get_weather", "description": "Current weather", "parameters": {"type": "object"}}],
}


def test_read_sharegpt():
    conv = read_sharegpt(SHAREGPT)
    assert [m.role for m in conv.messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    thought = conv.messages[2]
    assert thought.content == "Checking both."
    assert [c.arguments for c in thought.tool_calls] == ['{"city": "Lahore"}', '{"city": "Karachi"}']
    assert all(c.id is None for c in thought.tool_calls)
    assert conv.tool("get_weather") is not None


def test_sharegpt_errors():
    with pytest.raises(FormatError, match="unknown 'from' 'bot'"):
        read_sharegpt({"conversations": [{"from": "bot", "value": "hi"}]})
    with pytest.raises(FormatError, match="function_call value is not JSON"):
        read_sharegpt({"conversations": [{"from": "function_call", "value": "get_weather(Lahore)"}]})


@pytest.mark.parametrize(
    "record, expected",
    [
        (OPENAI, "openai"),
        (ANTHROPIC, "anthropic"),
        (SHAREGPT, "sharegpt"),
        ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}, "openai"),
        ({"messages": [], "tools": [{"name": "f", "input_schema": {}}]}, "anthropic"),
        ({"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x"}]}]}, "anthropic"),
    ],
)
def test_detect(record, expected):
    assert detect(record) == expected


def test_detect_unknown():
    with pytest.raises(FormatError, match="can't tell the format"):
        detect({"prompt": "hi", "completion": "hello"})


def test_sharegpt_to_openai_assigns_ids_in_order():
    record = write_openai(read_sharegpt(SHAREGPT))
    calls = record["messages"][2]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_2_0", "call_2_1"]
    assert [m.get("tool_call_id") for m in record["messages"][3:5]] == ["call_2_0", "call_2_1"]


def test_openai_to_anthropic_merges_tool_results():
    record = write_anthropic(read_openai(OPENAI))
    assert record["system"] == "You are a helpful assistant."
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert record["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "Lahore"},
    }
    assert record["tools"][0]["input_schema"]["required"] == ["city"]
    assert record["source"] == "batch-7"


def test_anthropic_to_sharegpt():
    record = write_sharegpt(read_anthropic(ANTHROPIC))
    assert [t["from"] for t in record["conversations"]] == ["human", "gpt", "function_call", "observation", "gpt"]
    assert record["conversations"][2]["value"] == '{"name": "get_weather", "arguments": {"city": "Lahore"}}'


def test_unconvertible_arguments():
    conv = read_openai(
        {
            "messages": [
                {"role": "user", "content": "?"},
                {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{oops"}}]},
            ]
        }
    )
    with pytest.raises(FormatError, match="arguments are not valid JSON"):
        write_anthropic(conv)


def test_sharegpt_tools_as_json_string():
    record = {**SHAREGPT, "tools": json.dumps(SHAREGPT["tools"])}
    assert read_sharegpt(record).tool("get_weather") is not None
    assert read_sharegpt({**SHAREGPT, "tools": ""}).tools == []
    with pytest.raises(FormatError, match="'tools' is a string but not valid JSON"):
        read_sharegpt({**SHAREGPT, "tools": "[{"})
    with pytest.raises(FormatError, match="'tools' must be a list"):
        read_sharegpt({**SHAREGPT, "tools": {"name": "x"}})
