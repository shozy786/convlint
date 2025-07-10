import pytest

from convlint.minhash import MinHasher, bands_for, shingles, similarity

A = "the quick brown fox jumps over the lazy dog while the cat watches from the warm windowsill"
B = "the quick brown fox jumps over the lazy dog while the cat watches from the cold windowsill"
C = "refund requests must be filed within thirty days of purchase with the original receipt attached"


def jaccard(x, y):
    return len(x & y) / len(x | y)


def test_shingles():
    assert shingles("One two three four") == {"one two three", "two three four"}
    assert shingles("Hi") == {"hi"}
    assert shingles("") == set()


def test_signature_estimates_jaccard():
    hasher = MinHasher(256)
    sa, sb, sc = (hasher.signature(shingles(t)) for t in (A, B, C))
    assert similarity(sa, sb) == pytest.approx(jaccard(shingles(A), shingles(B)), abs=0.1)
    assert similarity(sa, sc) < 0.1
    assert similarity(sa, hasher.signature(shingles(A))) == 1.0


@pytest.mark.parametrize("threshold, bands", [(0.95, 8), (0.85, 16), (0.7, 32)])
def test_bands_for(threshold, bands):
    assert bands_for(128, threshold) == bands
    rows = 128 // bands
    # a pair right at the threshold is caught with high probability
    assert 1 - (1 - threshold**rows) ** bands > 0.95
