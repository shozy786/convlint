from convlint.io import read_jsonl


def test_read_jsonl_keeps_going_after_bad_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages": []}\n\n{"messages": [}\n[1, 2]\n{"messages": [{"role": "user"}]}\n')
    lines = list(read_jsonl(path))
    assert [line.number for line in lines] == [1, 3, 4, 5]
    assert lines[0].record == {"messages": []}
    assert lines[1].record is None and lines[1].error.startswith("invalid JSON: Expecting value (column 15)")
    assert lines[2].error == "expected a JSON object, got list"
    assert lines[3].record["messages"][0]["role"] == "user"
