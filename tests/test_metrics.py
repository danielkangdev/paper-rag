import math

import pytest

from scirag.eval.dataset import Evidence, chunk_hits, evidence_hit_vector, normalize
from scirag.eval.metrics import mrr, ndcg_at_k, paired_bootstrap_ci, percentile, recall_at_k

# 3 evidence items; rank 2 covers e0, rank 4 covers e0+e1 (overlapping chunk), e2 never found
HITS = [
    [False, False, False],
    [True, False, False],
    [False, False, False],
    [True, True, False],
]


def test_recall_at_k():
    assert recall_at_k(HITS, 3, 1) == 0
    assert recall_at_k(HITS, 3, 2) == pytest.approx(1 / 3)
    assert recall_at_k(HITS, 3, 10) == pytest.approx(2 / 3)
    assert recall_at_k([], 0, 5) == 0


def test_mrr():
    assert mrr(HITS) == 0.5
    assert mrr([[False]]) == 0.0


def test_ndcg_counts_only_new_coverage():
    # gains at rank 2 (e0) and rank 4 (e1 new); ideal = 3 gains at ranks 1..3
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(HITS, 3, 10) == pytest.approx(dcg / idcg)
    perfect = [[True, False], [False, True]]
    assert ndcg_at_k(perfect, 2, 10) == pytest.approx(1.0)
    dup = [[True], [True]]  # duplicate chunk of the same quote adds nothing
    assert ndcg_at_k(dup, 1, 10) == pytest.approx(1.0)


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    a = [0.0, 0.5, 0.5, 1.0, 0.0, 0.5] * 5
    b = [0.5, 0.5, 1.0, 1.0, 0.5, 0.5] * 5
    r1 = paired_bootstrap_ci(a, b, n_resamples=1000, seed=1)
    r2 = paired_bootstrap_ci(a, b, n_resamples=1000, seed=1)
    assert r1 == r2
    assert r1["ci_low"] <= r1["mean_diff"] <= r1["ci_high"]
    assert r1["mean_diff"] == pytest.approx(0.25)
    assert r1["ci_low"] > 0


def test_bootstrap_zero_difference():
    r = paired_bootstrap_ci([0.3] * 10, [0.3] * 10)
    assert r["ci_low"] == r["ci_high"] == 0


def test_normalize_and_matching():
    assert normalize("Retrie-\nval  Augmented\tGeneration") == "retrieval augmented generation"
    ev = Evidence(paper_id="arxiv:1", quote="reflection   tokens")
    assert chunk_hits("arxiv:1", "Self-RAG uses Reflection\ntokens to critique.", ev)
    assert not chunk_hits("arxiv:2", "reflection tokens", ev)
    assert evidence_hit_vector("arxiv:1", "reflection tokens", [ev, Evidence(paper_id="arxiv:1", quote="xyz")]) == [True, False]
