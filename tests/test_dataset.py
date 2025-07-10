import json

from convlint.check import check_files
from convlint.formats import read_auto, read_openai, write_anthropic

from .test_formats import OPENAI


def write(path, *records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_exact_duplicates_across_files_and_formats(tmp_path):
    reformatted = json.loads(json.dumps(OPENAI))
    reformatted["messages"][4]["content"] = "  It's 31°C   in Lahore. "  # whitespace doesn't make it new
    a = write(tmp_path / "a.jsonl", OPENAI, write_anthropic(read_openai(OPENAI)))
    b = write(tmp_path / "b.jsonl", reformatted)
    found = [(f.path, f.line, f.problem.message) for f in check_files([a, b], read_auto, select=["duplicate"])]
    assert found == [
        (str(a), 2, f"same conversation as {a}:1"),  # same content, other format
        (str(b), 1, f"same conversation as {a}:1"),
    ]


def test_duplicate_rule_can_be_ignored(tmp_path):
    a = write(tmp_path / "a.jsonl", OPENAI, OPENAI)
    assert [f.problem.rule for f in check_files([a], read_auto, ignore=["duplicate"])] == []


def chat(question, answer):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


TEMPLATE = (
    "Our {city} branch is open from nine to five on weekdays and from ten to two on Saturdays. "
    "It is closed on Sundays and public holidays. You can book an appointment online up to two weeks "
    "ahead, and walk-in customers are served in the order they arrive. Bring a photo ID and your "
    "account number, and allow about twenty minutes for most requests."
)


def test_near_duplicates(tmp_path):
    question = "When is the {city} branch open?"
    path = write(
        tmp_path / "chats.jsonl",
        chat(question.format(city="Lahore"), TEMPLATE.format(city="Lahore")),
        chat(question.format(city="Multan"), TEMPLATE.format(city="Multan")),  # a templated generation
        chat("How do I reset my password?", "Open settings, choose security and press reset."),
        chat(question.format(city="Lahore"), TEMPLATE.format(city="Lahore")),  # exact: the other rule's job
    )
    found = [(f.line, f.problem.rule, f.problem.message) for f in check_files([path], read_auto)]
    near = [(line, msg) for line, rule, msg in found if rule == "near-duplicate"]
    assert len(near) == 1 and near[0][0] == 2 and near[0][1].endswith(f"similar to {path}:1")
    assert (4, "duplicate", f"same conversation as {path}:1") in found
