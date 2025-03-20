"""Read datasets line by line, keeping line numbers for error reports."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Line:
    number: int
    record: dict[str, Any] | None
    error: str | None = None


def read_jsonl(path: str | Path) -> Iterator[Line]:
    """Every non-blank line of a JSONL file, parsed, or with the reason it couldn't be.

    A broken line doesn't stop the read: in a 50k-line export you want all the bad lines in one
    pass, not one per run.
    """
    with Path(path).open(encoding="utf-8") as fh:
        for number, text in enumerate(fh, 1):
            if not text.strip():
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as e:
                yield Line(number, None, f"invalid JSON: {e.msg} (column {e.colno})")
                continue
            if not isinstance(record, dict):
                yield Line(number, None, f"expected a JSON object, got {type(record).__name__}")
                continue
            yield Line(number, record)
