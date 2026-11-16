"""Three-tier abstention decision (pure function).

    no_evidence    top score < low   -> refuse, the LLM is not called
    weak_evidence  low <= top < high -> answer, flagged ``confidence: low``
    confident      top >= high       -> answer normally

The signal is the cross-encoder score of the best hits (top-1, or the mean of
the top 3, chosen by calibration). If the reranker is unavailable we fall back
to the dense cosine with its own calibrated thresholds. With
neither (sparse-only degraded mode) the decision fails open to ``confident``,
because raw bm25 scores cannot tell "in corpus" from "out of corpus".
Thresholds come from ``scripts/calibrate_abstain.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..chunk.models import Hit

NO_EVIDENCE = "no_evidence"
WEAK = "weak_evidence"
CONFIDENT = "confident"


@dataclass
class Decision:
    decision: str
    score: float | None
    signal: str  # rerank | dense | none | disabled | empty
    threshold_low: float | None = None
    threshold_high: float | None = None

    @property
    def abstain(self) -> bool:
        return self.decision == NO_EVIDENCE

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_scores(scores: list[float], how: str = "max") -> float:
    """``max`` = top-1 score; ``mean_top3`` = mean of the 3 best (robust to one spurious hit)."""
    s = sorted(scores, reverse=True)
    if how == "mean_top3":
        top = s[:3]
        return sum(top) / len(top)
    return s[0]


def evidence_signal(hits: list[Hit], how: str = "max") -> tuple[str, float | None]:
    rerank = [h.score_rerank for h in hits if h.score_rerank is not None]
    if rerank:
        return "rerank", aggregate_scores(rerank, how)
    dense = [h.score_dense for h in hits if h.score_dense is not None]
    if dense:
        return "dense", aggregate_scores(dense, how)
    return "none", None


def decide(
    hits: list[Hit],
    *,
    enabled: bool = True,
    threshold_low: float = 0.05,
    threshold_high: float = 0.3,
    dense_threshold_low: float = 0.55,
    dense_threshold_high: float = 0.65,
    aggregate: str = "max",
) -> Decision:
    if not hits:
        return Decision(NO_EVIDENCE, None, "empty")
    signal, score = evidence_signal(hits, aggregate)
    if not enabled:
        return Decision(CONFIDENT, score, "disabled")
    if signal == "none" or score is None:
        return Decision(CONFIDENT, None, "none")
    low, high = (
        (threshold_low, threshold_high) if signal == "rerank" else (dense_threshold_low, dense_threshold_high)
    )
    if score < low:
        label = NO_EVIDENCE
    elif score < high:
        label = WEAK
    else:
        label = CONFIDENT
    return Decision(label, round(score, 4), signal, low, high)
