"""Retrieval metrics over evidence-hit matrices, plus a paired bootstrap CI.

For one question, ``hits[r][e]`` is True when the chunk at rank r (0-based)
contains evidence item e. All metrics are pure functions of that matrix.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

HitMatrix = Sequence[Sequence[bool]]


def recall_at_k(hits: HitMatrix, n_evidence: int, k: int) -> float:
    """Fraction of evidence items found in the top-k chunks."""
    if n_evidence == 0:
        return 0.0
    covered = {e for row in hits[:k] for e, h in enumerate(row) if h}
    return len(covered) / n_evidence


def mrr(hits: HitMatrix) -> float:
    """Reciprocal rank of the first chunk that contains any evidence item."""
    for r, row in enumerate(hits, start=1):
        if any(row):
            return 1.0 / r
    return 0.0


def ndcg_at_k(hits: HitMatrix, n_evidence: int, k: int) -> float:
    """Binary nDCG@k. A chunk has gain 1 if it covers an evidence item not
    already covered higher up (so near-duplicate overlapping chunks are not
    double-counted); ideal DCG places min(n_evidence, k) gains at the top."""
    if n_evidence == 0:
        return 0.0
    covered: set[int] = set()
    dcg = 0.0
    for r, row in enumerate(hits[:k], start=1):
        new = {e for e, h in enumerate(row) if h} - covered
        if new:
            dcg += 1.0 / math.log2(r + 1)
            covered |= new
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_evidence, k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def paired_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI for mean(b - a) by resampling questions with replacement."""
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and the same length")
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return {
        "mean_diff": mean(diffs),
        "ci_low": percentile(boots, 100 * alpha / 2),
        "ci_high": percentile(boots, 100 * (1 - alpha / 2)),
        "n_resamples": n_resamples,
        "n_questions": n,
        "p_le_zero": sum(1 for d in boots if d <= 0) / n_resamples,
    }
