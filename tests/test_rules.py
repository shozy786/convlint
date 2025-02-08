from convlint.model import Conversation, Message, ToolCall
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
