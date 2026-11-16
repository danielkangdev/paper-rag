from scirag.chunk.models import Hit
from scirag.rag import abstention as ab


def hits(rerank=None, dense=None):
    return [Hit(chunk_id="c", paper_id="p", score_rerank=rerank, score_dense=dense)]


KW = dict(threshold_low=0.1, threshold_high=0.5, dense_threshold_low=0.6, dense_threshold_high=0.7)


def test_three_tiers_on_rerank_score():
    assert ab.decide(hits(rerank=0.05), **KW).decision == ab.NO_EVIDENCE
    assert ab.decide(hits(rerank=0.3), **KW).decision == ab.WEAK
    assert ab.decide(hits(rerank=0.9), **KW).decision == ab.CONFIDENT


def test_boundaries_are_inclusive_on_the_upper_side():
    assert ab.decide(hits(rerank=0.1), **KW).decision == ab.WEAK
    assert ab.decide(hits(rerank=0.5), **KW).decision == ab.CONFIDENT


def test_dense_fallback_uses_its_own_thresholds():
    d = ab.decide(hits(dense=0.65), **KW)
    assert d.signal == "dense" and d.decision == ab.WEAK
    assert ab.decide(hits(dense=0.5), **KW).abstain


def test_rerank_preferred_over_dense():
    assert ab.decide(hits(rerank=0.9, dense=0.1), **KW).signal == "rerank"


def test_empty_hits_always_abstain_even_when_disabled():
    assert ab.decide([], enabled=False, **KW).abstain


def test_disabled_and_unscored_fail_open():
    assert ab.decide(hits(rerank=0.0), enabled=False, **KW).decision == ab.CONFIDENT
    d = ab.decide([Hit(chunk_id="c", paper_id="p", score_sparse=3.0)], **KW)
    assert d.decision == ab.CONFIDENT and d.signal == "none"


def test_uses_max_score():
    hs = [Hit(chunk_id="a", paper_id="p", score_rerank=0.01),
          Hit(chunk_id="b", paper_id="p", score_rerank=0.8)]
    assert ab.decide(hs, **KW).score == 0.8


def test_mean_top3_dampens_one_spurious_hit():
    hs = [Hit(chunk_id=str(i), paper_id="p", score_rerank=s) for i, s in enumerate([0.98, 0.02, 0.01, 0.9])]
    assert ab.aggregate_scores([0.98, 0.02, 0.01, 0.9], "mean_top3") == (0.98 + 0.9 + 0.02) / 3
    assert ab.decide(hs, aggregate="mean_top3", **KW).decision == ab.CONFIDENT
    spurious = [Hit(chunk_id=str(i), paper_id="p", score_rerank=s) for i, s in enumerate([0.3, 0.0, 0.0])]
    assert ab.decide(spurious, aggregate="mean_top3", **KW).abstain
    assert not ab.decide(spurious, aggregate="max", **KW).abstain
