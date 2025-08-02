import json

from convlint.stats import collect, percentiles, render

from .test_formats import ANTHROPIC, OPENAI, SHAREGPT


def test_percentiles():
    assert percentiles(list(range(1, 101))) == [50, 90, 99, 100]
    assert percentiles([7]) == [7, 7, 7, 7]
    assert percentiles([]) == [0, 0, 0, 0]


def test_collect_and_render(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (OPENAI, ANTHROPIC, SHAREGPT)) + "\n{oops\n")
    stats = collect([path])
    assert stats.conversations == 3 and stats.unreadable == 1
    assert stats.formats == {"openai": 1, "anthropic": 1, "sharegpt": 1}
    assert stats.calls_by_tool == {"get_weather": 4}
    assert stats.tool_calls == [1, 1, 2]
    text = "\n".join(render(stats))
    assert "conversations      3 (1 unreadable lines skipped)" in text
    assert "get_weather" in text
