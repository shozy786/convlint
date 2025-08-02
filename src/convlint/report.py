"""Formatting findings for people, programs and CI."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from .check import Finding


def text(findings: Iterable[Finding]) -> Iterator[str]:
    for f in findings:
        turn = f" [turn {f.problem.turn}]" if f.problem.turn is not None else ""
        yield f"{f.path}:{f.line}: {f.problem.severity} {f.problem.rule}{turn} {f.problem.message}"


def github(findings: Iterable[Finding]) -> Iterator[str]:
    """GitHub Actions workflow commands; they show up as annotations on the PR diff."""
    for f in findings:
        message = f.problem.message if f.problem.turn is None else f"turn {f.problem.turn}: {f.problem.message}"
        yield f"::{f.problem.severity} file={_escape(f.path, True)},line={f.line},title={f.problem.rule}::{_escape(message)}"


def as_json(findings: Iterable[Finding]) -> str:
    return json.dumps(
        [
            {
                "path": f.path,
                "line": f.line,
                "rule": f.problem.rule,
                "severity": f.problem.severity,
                "turn": f.problem.turn,
                "message": f.problem.message,
            }
            for f in findings
        ],
        indent=2,
        ensure_ascii=False,
    )


def _escape(value: str, property_: bool = False) -> str:
    # https://github.com/actions/toolkit/blob/main/packages/core/src/command.ts
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return value.replace(":", "%3A").replace(",", "%2C") if property_ else value
