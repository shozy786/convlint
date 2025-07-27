"""Settings from ``[tool.convlint]`` in pyproject.toml, or a convlint.toml, found upward from cwd."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .dataset import DATASET_RULES
from .rules import RULES, Options


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    select: list[str] | None = None
    ignore: list[str] = field(default_factory=list)
    format: str = "auto"
    options: Options = field(default_factory=Options)
    source: Path | None = None


_KEYS = {"select", "ignore", "format", "max_tokens", "encoding", "near_duplicate_threshold"}


def find(start: Path | None = None) -> Path | None:
    directory = (start or Path.cwd()).resolve()
    for d in (directory, *directory.parents):
        if (d / "convlint.toml").is_file():
            return d / "convlint.toml"
        pyproject = d / "pyproject.toml"
        if pyproject.is_file() and "convlint" in _load(pyproject).get("tool", {}):
            return pyproject
    return None


def load(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    data = _load(path)
    table: dict[str, Any] = data.get("tool", {}).get("convlint", {}) if path.name == "pyproject.toml" else data
    unknown = set(table) - _KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown settings {sorted(unknown)}")
    for key in ("select", "ignore"):
        bad = [r for r in table.get(key) or [] if r not in RULES and r not in DATASET_RULES]
        if bad:
            raise ConfigError(f"{path}: {key} names unknown rules {bad}")
    defaults = Options()
    options = Options(
        max_tokens=int(table.get("max_tokens", defaults.max_tokens)),
        encoding=table.get("encoding", defaults.encoding),
        near_duplicate_threshold=float(table.get("near_duplicate_threshold", defaults.near_duplicate_threshold)),
    )
    return Settings(table.get("select"), list(table.get("ignore") or []), table.get("format", "auto"), options, path)


def _load(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
