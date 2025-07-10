"""Run the rules over dataset files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import DATASET_RULES
from .formats import FormatError
from .io import read_jsonl
from .model import Conversation
from .rules import Options, Problem, lint

Reader = Callable[[dict[str, Any]], Conversation]


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    problem: Problem


def check_files(
    paths: Iterable[str | Path],
    reader: Reader,
    select: Iterable[str] | None = None,
    ignore: Iterable[str] = (),
    options: Options | None = None,
) -> Iterator[Finding]:
    """Findings for every line of every file, then dataset-wide findings (duplicates span files)."""
    wanted = set(select) if select is not None else None
    ignore = set(ignore)
    dataset_checks = [
        r.factory(options or Options())
        for r in DATASET_RULES.values()
        if (wanted is None or r.name in wanted) and r.name not in ignore
    ]
    for path in paths:
        for line in read_jsonl(path):
            if line.error is not None:
                yield Finding(str(path), line.number, Problem("invalid-json", "error", line.error))
                continue
            try:
                conversation = reader(line.record)
            except FormatError as e:
                yield Finding(str(path), line.number, Problem("invalid-format", "error", str(e)))
                continue
            for problem in lint(conversation, wanted, ignore, options):
                yield Finding(str(path), line.number, problem)
            for check in dataset_checks:
                for problem in check.observe((str(path), line.number), conversation):
                    yield Finding(str(path), line.number, problem)
    for check in dataset_checks:
        for (path, line), problem in check.finish():
            yield Finding(path, line, problem)
