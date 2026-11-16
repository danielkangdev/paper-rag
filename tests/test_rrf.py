import pytest

from scirag.retrieval.rrf import rrf_fuse


def test_rrf_scores_and_order():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)
    order = [d for d, _ in rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])]
    assert order[0] == "b" and order[1] == "c"


def test_rrf_single_list_preserves_order():
    assert [d for d, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_empty_and_one_empty_list():
    assert rrf_fuse([]) == []
    assert [d for d, _ in rrf_fuse([[], ["q", "r"]])] == ["q", "r"]


def test_rrf_duplicates_within_list_count_once():
    fused = dict(rrf_fuse([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)


def test_rrf_ties_broken_by_first_appearance():
    assert [d for d, _ in rrf_fuse([["a", "b"], ["b", "a"]])] == ["a", "b"]
