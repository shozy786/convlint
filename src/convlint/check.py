"""Run the rules over a whole dataset file."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formats import FormatError
from .io import read_jsonl
from .model import Conversation
from .rules import Problem, lint

Reader = Callable[[dict[str, Any]], Conversation]


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    problem: Problem


def check_file(
    path: str | Path, reader: Reader, select: Iterable[str] | None = None, ignore: Iterable[str] = ()
) -> Iterator[Finding]:
    for line in read_jsonl(path):
        if line.error is not None:
            yield Finding(str(path), line.number, Problem("invalid-json", "error", line.error))
            continue
        try:
            conversation = reader(line.record)
        except FormatError as e:
            yield Finding(str(path), line.number, Problem("invalid-format", "error", str(e)))
            continue
        for problem in lint(conversation, select, ignore):
            yield Finding(str(path), line.number, problem)
