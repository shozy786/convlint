"""Dataset shape: how long conversations are, which tools get called, how many tokens."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .formats import READERS, FormatError, detect
from .io import read_jsonl
from .tokens import conversation_tokens


@dataclass
class Stats:
    conversations: int = 0
    unreadable: int = 0
    formats: Counter[str] = field(default_factory=Counter)
    turns: list[int] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    calls_by_tool: Counter[str] = field(default_factory=Counter)
    with_system: int = 0
    with_tools: int = 0


def collect(paths: Iterable[str | Path], encoding: str = "o200k_base") -> Stats:
    stats = Stats()
    for path in paths:
        for line in read_jsonl(path):
            try:
                if line.record is None:
                    raise FormatError(line.error or "")
                fmt = detect(line.record)
                conv = READERS[fmt](line.record)
            except FormatError:
                stats.unreadable += 1
                continue
            stats.conversations += 1
            stats.formats[fmt] += 1
            stats.turns.append(sum(m.role in {"user", "assistant"} for m in conv.messages))
            calls = [c for m in conv.messages for c in m.tool_calls]
            stats.tool_calls.append(len(calls))
            stats.calls_by_tool.update(c.name for c in calls)
            stats.tokens.append(conversation_tokens(conv, encoding))
            stats.with_system += any(m.role == "system" for m in conv.messages)
            stats.with_tools += bool(conv.tools)
    return stats


def percentiles(values: Sequence[int], points: Sequence[int] = (50, 90, 99, 100)) -> list[int]:
    """Nearest-rank percentiles; exact values from the data rather than interpolated ones."""
    if not values:
        return [0] * len(points)
    ordered = sorted(values)
    return [ordered[max(0, -(-p * len(ordered) // 100) - 1)] for p in points]


def render(stats: Stats) -> list[str]:
    n = stats.conversations
    lines = [
        f"conversations      {n:,}" + (f" ({stats.unreadable:,} unreadable lines skipped)" if stats.unreadable else ""),
        "formats            " + ", ".join(f"{k} {v:,}" for k, v in stats.formats.most_common()),
        f"with system prompt {stats.with_system / max(n, 1):.0%}",
        f"with tools         {stats.with_tools / max(n, 1):.0%}",
        "",
        "                   p50      p90      p99      max",
    ]
    for label, values in (("turns", stats.turns), ("tool calls", stats.tool_calls), ("tokens", stats.tokens)):
        lines.append(f"{label:<14}" + "".join(f"{v:>9,}" for v in percentiles(values)))
    if stats.calls_by_tool:
        lines += ["", "calls by tool"]
        lines += [f"  {name:<24} {count:>8,}" for name, count in stats.calls_by_tool.most_common(20)]
    return lines
