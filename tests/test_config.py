import pytest

from convlint import config


def test_pyproject_table(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[tool.convlint]\nignore = ["pii"]\nmax_tokens = 4096\nformat = "sharegpt"\n'
    )
    nested = tmp_path / "data" / "v2"
    nested.mkdir(parents=True)
    path = config.find(nested)
    assert path == tmp_path / "pyproject.toml"
    settings = config.load(path)
    assert settings.ignore == ["pii"] and settings.select is None
    assert settings.options.max_tokens == 4096 and settings.format == "sharegpt"


def test_pyproject_without_table_is_skipped(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (tmp_path / "convlint.toml").write_text("near_duplicate_threshold = 0.9\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "pyproject.toml").write_text('[project]\nname = "y"\n')
    assert config.find(sub) == tmp_path / "convlint.toml"
    assert config.load(tmp_path / "convlint.toml").options.near_duplicate_threshold == 0.9


@pytest.mark.parametrize(
    "toml, message",
    [
        ("max_tokns = 10\n", "unknown settings \\['max_tokns'\\]"),
        ('ignore = ["piii"]\n', "ignore names unknown rules \\['piii'\\]"),
        ("max_tokens = \n", "Invalid value"),
    ],
)
def test_bad_settings(tmp_path, toml, message):
    path = tmp_path / "convlint.toml"
    path.write_text(toml)
    with pytest.raises(config.ConfigError, match=message):
        config.load(path)
