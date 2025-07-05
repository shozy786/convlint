from convlint import tokens
from convlint.model import Conversation, Message, Tool, ToolCall


def test_count_with_tiktoken():
    assert tokens.count("Hello, world!") == 4


def test_fallback_estimate(monkeypatch):
    monkeypatch.setattr(tokens, "_encoder", lambda name: None)
    assert tokens.count("x" * 41) == 11


def test_conversation_tokens_include_calls_and_schemas():
    bare = Conversation([Message("user", "Weather?"), Message("assistant", "Sunny")])
    with_tools = Conversation(
        [Message("user", "Weather?"), Message("assistant", None, [ToolCall("get_weather", '{"city": "Lahore"}', "1")])],
        [Tool("get_weather", "Current weather", {"type": "object", "properties": {"city": {"type": "string"}}})],
    )
    assert tokens.conversation_tokens(bare) == tokens.count("Weather?") + tokens.count("Sunny") + 8
    assert tokens.conversation_tokens(with_tools) > tokens.conversation_tokens(bare) + 10
