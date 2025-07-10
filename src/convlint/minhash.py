"""MinHash signatures and LSH banding, for finding near-duplicate conversations in one pass."""

from __future__ import annotations

import re
import zlib

import numpy as np

_PRIME = np.uint64((1 << 61) - 1)
_MASK = np.uint64((1 << 32) - 1)
_WORD = re.compile(r"\w+")


def shingles(text: str, size: int = 3) -> set[str]:
    words = _WORD.findall(text.lower())
    return {" ".join(words[i : i + size]) for i in range(max(len(words) - size + 1, 1))} if words else set()


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, (1 << 61) - 1, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 61) - 1, num_perm, dtype=np.uint64)

    def signature(self, tokens: set[str]) -> np.ndarray:
        hashes = np.fromiter((zlib.crc32(t.encode()) for t in tokens), dtype=np.uint64, count=len(tokens))
        # universal hashing (a*x + b) mod p; uint64 overflow wraps, which keeps it a good hash
        with np.errstate(over="ignore"):
            permuted = ((hashes[:, None] * self.a + self.b) % _PRIME) & _MASK
        return permuted.min(axis=0).astype(np.uint32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Estimated Jaccard similarity of the shingle sets behind two signatures."""
    return float(np.mean(a == b))


def bands_for(num_perm: int, threshold: float, recall: float = 0.95) -> int:
    """How many LSH bands to split a signature into for a given similarity threshold.

    Two signatures with similarity s share at least one band with probability
    1 - (1 - s^r)^b for b bands of r rows. More bands catch more pairs but also more chaff to
    verify, so take the fewest bands that still catch a pair right at the threshold with
    probability ``recall``.
    """
    for bands in (b for b in range(1, num_perm + 1) if num_perm % b == 0):
        rows = num_perm // bands
        if 1 - (1 - threshold**rows) ** bands >= recall:
            return bands
    return num_perm
