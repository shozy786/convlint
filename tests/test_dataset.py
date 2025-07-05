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
