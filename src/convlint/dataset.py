"""Checks that look across the whole dataset rather than one conversation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from .minhash import MinHasher, bands_for, shingles, similarity
from .model import Conversation
from .rules import Options, Problem

Location = tuple[str, int]  # (path, line)


def essence(conversation: Conversation) -> list[list[object]]:
    """What a trainer sees: roles, whitespace-normalized text, tool calls with canonical JSON."""
    out = []
    for m in conversation.messages:
        calls = []
        for c in m.tool_calls:
            try:
                args = json.dumps(json.loads(c.arguments or "{}"), sort_keys=True)
            except json.JSONDecodeError:
                args = c.arguments
            calls.append([c.name, args])
        out.append([m.role, " ".join((m.content or "").split()), calls])
    return out


def fingerprint(conversation: Conversation) -> str:
    return hashlib.sha1(json.dumps(essence(conversation), ensure_ascii=False).encode()).hexdigest()


class DatasetCheck(Protocol):
    def observe(self, where: Location, conversation: Conversation) -> Iterable[Problem]: ...

    def finish(self) -> Iterable[tuple[Location, Problem]]: ...


class ExactDuplicates:
    """Repeated conversations over-weight whatever they teach and leak between train/eval splits."""

    def __init__(self, options: Options) -> None:
        self.first: dict[str, Location] = {}

    def observe(self, where: Location, conversation: Conversation) -> Iterable[Problem]:
        key = fingerprint(conversation)
        if key in self.first:
            path, line = self.first[key]
            return [Problem("duplicate", "warning", f"same conversation as {path}:{line}")]
        self.first[key] = where
        return []

    def finish(self) -> Iterable[tuple[Location, Problem]]:
        return []


class NearDuplicates:
    """Conversations that differ by a few words: templated generations, light paraphrases.

    Only user and assistant text is compared. System prompts and tool schemas are usually shared
    by the whole dataset and would make everything look alike.
    """

    MIN_SHINGLES = 8  # "Hi" / "Hello!" exchanges are too short to call near-duplicates

    def __init__(self, options: Options, num_perm: int = 128) -> None:
        self.threshold = options.near_duplicate_threshold
        self.hasher = MinHasher(num_perm)
        self.rows = num_perm // bands_for(num_perm, self.threshold)
        self.buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self.signatures: list = []
        self.locations: list[Location] = []
        self.exact: set[str] = set()

    def observe(self, where: Location, conversation: Conversation) -> Iterable[Problem]:
        key = fingerprint(conversation)
        if key in self.exact:
            return []  # the duplicate rule reports these
        self.exact.add(key)
        text = "\n".join(m.content or "" for m in conversation.messages if m.role in {"user", "assistant"})
        tokens = shingles(text)
        if len(tokens) < self.MIN_SHINGLES:
            return []
        signature = self.hasher.signature(tokens)
        best, best_score = None, 0.0
        index = len(self.signatures)
        for start in range(0, self.hasher.num_perm, self.rows):
            bucket = self.buckets[(start, signature[start : start + self.rows].tobytes())]
            for other in bucket:
                score = similarity(signature, self.signatures[other])
                if score >= self.threshold and score > best_score:
                    best, best_score = other, score
            bucket.append(index)
        self.signatures.append(signature)
        self.locations.append(where)
        if best is None:
            return []
        path, line = self.locations[best]
        return [Problem("near-duplicate", "warning", f"~{best_score:.0%} similar to {path}:{line}")]

    def finish(self) -> Iterable[tuple[Location, Problem]]:
        return []


@dataclass(frozen=True)
class DatasetRule:
    name: str
    severity: str
    summary: str
    factory: Callable[[Options], DatasetCheck]


DATASET_RULES = {
    "duplicate": DatasetRule("duplicate", "warning", "No conversation appears twice", ExactDuplicates),
    "near-duplicate": DatasetRule(
        "near-duplicate", "warning", "No two conversations are near_duplicate_threshold similar", NearDuplicates
    ),
}
