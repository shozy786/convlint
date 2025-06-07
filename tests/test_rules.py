import pytest

from convlint.model import Conversation, Message, Tool, ToolCall
from convlint.rules import RULES, lint


def conv(*messages, tools=()):
    return Conversation([m if isinstance(m, Message) else Message(*m) for m in messages], list(tools))


def found(c, **kwargs):
    return {(p.rule, p.turn) for p in lint(c, **kwargs)}


GOOD = conv(("system", "Be brief."), ("user", "Hi"), ("assistant", "Hello!"))


def test_clean_conversation():
    assert lint(GOOD) == []


def test_unknown_role_and_misplaced_system():
    c = conv(("user", "Hi"), ("system", "late"), ("bot", "Hello"))
    assert {("unknown-role", 2), ("system-not-first", 1)} <= found(c)


def test_empty_messages():
    c = conv(("user", "  "), ("assistant", None, [ToolCall("f", "{}", "1")]), ("tool", "ok", [], "1"), ("assistant", ""))
    assert {r for r in found(c) if r[0] == "empty-message"} == {("empty-message", 0), ("empty-message", 3)}


def test_consecutive_turns():
    c = conv(("user", "Hi"), ("user", "Are you there?"), ("assistant", "Yes"))
    assert ("consecutive-turns", 1) in found(c)


def test_last_turn():
    assert ("bad-last-turn", 1) in found(conv(("user", "Hi"), ("user", "?")))
    dangling = conv(("user", "Hi"), ("assistant", None, [ToolCall("f", "{}", "1")]))
    assert ("bad-last-turn", 1) in found(dangling)


def test_no_assistant_turn():
    assert ("no-assistant-turn", None) in found(conv(("user", "Hi")))


def test_first_turn():
    assert ("first-turn-not-user", 1) in found(conv(("system", "x"), ("assistant", "Hi"), ("user", "Hey"), ("assistant", "?")))


def test_select_and_ignore():
    c = conv(("user", "Hi"), ("user", "?"))
    assert {p.rule for p in lint(c, select=["consecutive-turns"])} == {"consecutive-turns"}
    assert "consecutive-turns" not in {p.rule for p in lint(c, ignore=["consecutive-turns"])}


def test_every_rule_is_documented():
    for r in RULES.values():
        assert r.severity in {"error", "warning"} and r.summary


def call(name="get_weather", args='{"city": "Lahore"}', id="c1"):
    return ToolCall(name, args, id)


def assistant_calls(*calls, content=None):
    return Message("assistant", content, list(calls))


def result(id="c1", content="31C"):
    return Message("tool", content, [], id)


def pairing(c):
    return sorted(p.turn for p in lint(c, select=["tool-call-pairing"]))


def test_answered_calls_are_fine():
    c = conv(("user", "?"), assistant_calls(call(id="a"), call(id="b")), result("b"), result("a"), ("assistant", "Done"))
    assert pairing(c) == []


def test_unanswered_call():
    c = conv(("user", "?"), assistant_calls(call(id="a"), call(id="b")), result("a"), ("assistant", "Done"))
    (problem,) = lint(c, select=["tool-call-pairing"])
    assert problem.turn == 1 and "'b' never gets a result" in problem.message


def test_result_for_unknown_or_closed_call():
    c = conv(("user", "?"), assistant_calls(call(id="a")), result("a"), result("a"), ("assistant", "ok"))
    assert pairing(c) == [3]
    c = conv(("user", "?"), assistant_calls(call(id="a")), result("x"), ("assistant", "ok"))
    assert pairing(c) == [1, 2]  # 'a' unanswered, 'x' not open


def test_result_with_no_call():
    assert pairing(conv(("user", "?"), result(None), ("assistant", "ok"))) == [1]


def test_calls_without_ids_are_answered_in_order():
    c = conv(("user", "?"), assistant_calls(call(id=None), call(id=None)), result(None), result(None), ("assistant", "ok"))
    assert pairing(c) == []
    c = conv(("user", "?"), assistant_calls(call(id=None), call(id=None)), result(None), ("assistant", "ok"))
    assert pairing(c) == [1]


def test_duplicate_ids():
    c = conv(("user", "?"), assistant_calls(call(id="a")), result("a"), assistant_calls(call(id="a")), result("a"), ("assistant", "ok"))
    assert ("duplicate-tool-call-id", 3) in found(c)


def test_tool_calls_outside_assistant():
    c = conv(Message("user", "?", [call()]), ("assistant", "ok"))
    assert ("tool-calls-outside-assistant", 0) in found(c)


WEATHER = Tool(
    "get_weather",
    "Current weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


def with_call(args, name="get_weather", tools=(WEATHER,)):
    return conv(("user", "?"), assistant_calls(call(name, args, "a")), result("a"), ("assistant", "ok"), tools=tools)


def messages(c, rule_name):
    return [p.message for p in lint(c, select=[rule_name])]


def test_valid_call():
    assert lint(with_call('{"city": "Lahore", "unit": "c"}')) == []


def test_invalid_json_arguments():
    assert messages(with_call('{"city": "Lahore"'), "invalid-tool-arguments") == [
        "get_weather: arguments are not valid JSON: '{\"city\": \"Lahore\"'"
    ]
    assert "got list" in messages(with_call("[1, 2]"), "invalid-tool-arguments")[0]


def test_unknown_tool():
    assert "not among the declared tools ['get_weather']" in messages(with_call("{}", name="get_time"), "unknown-tool")[0]
    assert "no tools are declared" in messages(with_call("{}", tools=()), "unknown-tool")[0]


def test_schema_mismatch():
    found_ = messages(with_call('{"unit": "k", "country": "PK"}'), "arguments-schema-mismatch")
    assert "get_weather: 'city' is a required property" in found_
    assert any(m.startswith("get_weather: unit: 'k' is not one of") for m in found_)
    assert any("Additional properties are not allowed ('country'" in m for m in found_)


def test_empty_arguments_count_as_an_empty_object():
    no_args = Tool("get_time", None, {"type": "object", "properties": {}})
    assert lint(with_call("", name="get_time", tools=(no_args,))) == []


def test_invalid_schema():
    broken = Tool("get_weather", None, {"type": "objekt"})
    assert "invalid parameters schema" in messages(with_call("{}", tools=(broken,)), "invalid-tool-schema")[0]


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Write to ayesha.khan@meridian.pk for details", "email address ayes…pk"),
        ("Call +92 300 1234567 after 5pm", "phone number +92 …67"),
        ("card 4111 1111 1111 1111 exp 12/27", "card number 4111…11"),
        ("use sk-proj-abcdefghijklmnopqrstuvwxyz012345 in the header", "API key or token sk-p…45"),
        ("AKIAIOSFODNN7EXAMPLE", "API key or token AKIA…LE"),
    ],
)
def test_pii_found(text, kind):
    assert [p.message for p in lint(conv(("user", text), ("assistant", "ok")), select=["pii"])] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "Contact jane@example.com",
        "Order 12345 shipped",
        "The year 2025 had 365 days",
        "Pi is 3.14159265358979",
    ],
)
def test_pii_not_found(text):
    assert lint(conv(("user", text), ("assistant", "ok")), select=["pii"]) == []


def test_phone_numbers_passing_luhn_are_still_phones():
    # +1 415 555 0131 happens to be Luhn-valid; it's 11 digits, so it can't be a card
    found_ = [p.message for p in lint(conv(("user", "ring +1 415 555 0131"), ("assistant", "ok")), select=["pii"])]
    assert found_ == ["phone number +1 4…31"]


@pytest.mark.parametrize(
    "text, placeholder",
    [
        ("Dear {{customer_name}}, your order shipped.", "{{customer_name}}"),
        ("It will arrive on [INSERT DATE].", "[INSERT DATE]"),
        ("TODO: write the answer", "TODO"),
        ("Lorem ipsum dolor sit amet", "Lorem ipsum"),
    ],
)
def test_template_leftovers(text, placeholder):
    (problem,) = lint(conv(("user", "?"), ("assistant", text)), select=["template-leftover"])
    assert problem.message == f"unfilled placeholder {placeholder!r}" and problem.turn == 1


def test_code_braces_are_not_placeholders():
    text = "Use f'{name}' or {'a': 1} in Python, and a list like [1, 2]."
    assert lint(conv(("user", "?"), ("assistant", text)), select=["template-leftover"]) == []


def test_boilerplate_only_in_assistant_turns():
    c = conv(("user", "As an AI language model, what do you think?"), ("assistant", "As an AI language model, I can't."))
    assert [p.turn for p in lint(c, select=["assistant-boilerplate"])] == [1]
