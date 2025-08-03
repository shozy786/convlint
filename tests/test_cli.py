from pathlib import Path

from click.testing import CliRunner

from convlint.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def invoke(*args):
    try:
        runner = CliRunner(mix_stderr=False)  # click < 8.2 mixes stderr into output by default
    except TypeError:
        runner = CliRunner()  # click 8.2 dropped the flag; stderr is always separate
    return runner.invoke(main, [str(a) for a in args])


def test_check_reports_every_problem_with_location():
    result = invoke("check", FIXTURES / "openai.jsonl")
    assert result.exit_code == 1
    out = result.stdout.splitlines()
    path = str(FIXTURES / "openai.jsonl")
    assert f"{path}:2: error tool-call-pairing [turn 1] tool call 'call_2' never gets a result" in out
    assert any(line.startswith(f"{path}:3: error invalid-json") for line in out)
    assert any(line.startswith(f"{path}:4: error arguments-schema-mismatch [turn 1]") for line in out)
    assert f"{path}:5: warning consecutive-turns [turn 1]" in "\n".join(out)
    assert not any(line.startswith(f"{path}:1:") for line in out)
    assert "errors" in result.stderr


def test_select_and_strict(tmp_path):
    path = tmp_path / "warn.jsonl"
    path.write_text((FIXTURES / "openai.jsonl").read_text().splitlines()[4] + "\n")
    assert invoke("check", path, "--select", "consecutive-turns").exit_code == 0
    assert invoke("check", path, "--select", "consecutive-turns", "--strict").exit_code == 1


def test_parse_errors_ignore_select():
    result = invoke("check", FIXTURES / "openai.jsonl", "--select", "consecutive-turns")
    assert result.exit_code == 1
    assert ":3: error invalid-json" in result.stdout


def test_unknown_rule_name():
    result = invoke("check", FIXTURES / "openai.jsonl", "--ignore", "nope")
    assert result.exit_code == 2
    assert "unknown rules ['nope']" in result.stderr


def test_rules_listing():
    result = invoke("rules")
    assert "tool-call-pairing" in result.stdout and "error" in result.stdout


def test_mixed_formats_in_one_file(tmp_path):
    import json

    from .test_formats import ANTHROPIC, OPENAI, SHAREGPT

    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (OPENAI, ANTHROPIC, SHAREGPT)) + "\n")
    result = invoke("check", path)
    # the fixtures declare tools whose schemas the calls satisfy; only formats differ
    assert result.exit_code == 0, result.stdout
    forced = invoke("check", path, "--format", "openai")
    assert ":3: error invalid-format expected a 'messages' list" in forced.stdout


def test_convert(tmp_path):
    import json

    from .test_formats import SHAREGPT

    source = tmp_path / "sharegpt.jsonl"
    source.write_text(json.dumps(SHAREGPT) + "\n" + '{"conversations": "nope"}\n')
    out = tmp_path / "anthropic.jsonl"
    result = invoke("convert", source, "--to", "anthropic", "-o", out)
    assert result.exit_code == 1
    assert ":2: skipped, expected a 'conversations' list" in result.stderr
    assert "converted 1 conversations to anthropic, skipped 1" in result.stderr
    (record,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert record["system"] == "You are a helpful assistant."
    assert [m["role"] for m in record["messages"]] == ["user", "assistant", "user", "assistant"]
    assert invoke("check", out).exit_code == 0


def test_dedupe(tmp_path):
    import json

    from .test_dataset import TEMPLATE, chat

    source = tmp_path / "in.jsonl"
    lines = [
        json.dumps(chat("Lahore hours?", TEMPLATE.format(city="Lahore"))),
        json.dumps(chat("Multan hours?", TEMPLATE.format(city="Multan"))),
        "{broken",
        json.dumps(chat("Lahore hours?", TEMPLATE.format(city="Lahore")), indent=None, separators=(",", ":")),
        json.dumps(chat("Reset password?", "Open settings and press reset.")),
    ]
    source.write_text("\n".join(lines) + "\n")

    out = tmp_path / "out.jsonl"
    result = invoke("dedupe", source, "-o", out)
    assert result.exit_code == 0
    assert "kept 3, dropped 2 duplicates, 1 unreadable lines kept as-is" in result.stderr
    assert out.read_text().splitlines() == [lines[0], lines[2], lines[4]]

    invoke("dedupe", source, "-o", out, "--exact-only")
    assert out.read_text().splitlines() == [lines[0], lines[1], lines[2], lines[4]]


def test_check_uses_config(tmp_path):
    path = tmp_path / "warn.jsonl"
    path.write_text((FIXTURES / "openai.jsonl").read_text().splitlines()[4] + "\n")
    cfg = tmp_path / "convlint.toml"
    cfg.write_text('ignore = ["consecutive-turns"]\n')
    assert invoke("check", path, "--strict").exit_code == 1
    assert invoke("check", path, "--strict", "--config", cfg).exit_code == 0


def test_json_output():
    import json

    result = invoke("check", FIXTURES / "openai.jsonl", "--output-format", "json")
    findings = json.loads(result.stdout)
    assert {"path", "line", "rule", "severity", "turn", "message"} == set(findings[0])
    assert {(f["line"], f["rule"]) for f in findings} >= {(2, "tool-call-pairing"), (3, "invalid-json")}


def test_github_annotations():
    result = invoke("check", FIXTURES / "openai.jsonl", "--output-format", "github")
    path = str(FIXTURES / "openai.jsonl").replace(":", "%3A")
    assert (
        f"::error file={path},line=2,title=tool-call-pairing::turn 1%3A tool call 'call_2' never gets a result"
        not in result.stdout
    )  # messages keep their colons; only properties are escaped
    assert (
        f"::error file={path},line=2,title=tool-call-pairing::turn 1: tool call 'call_2' never gets a result"
        in result.stdout
    )
    assert f"::warning file={path},line=5,title=consecutive-turns::" in result.stdout
