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
