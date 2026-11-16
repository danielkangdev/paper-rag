"""Prompts for grounded QA."""

from __future__ import annotations

from ..chunk.models import Hit

SYSTEM = (
    "You are a careful research assistant. Answer the question using ONLY the numbered "
    "evidence passages provided. After every sentence that states a fact, cite the passage(s) "
    "it comes from with tags like [E1] or [E2][E3]. Use only the tags given; never use "
    "[1]-style numbers or (Author, Year) citations. Do not add facts from outside the evidence. "
    "If the evidence does not answer the question, say that the indexed papers do not contain "
    "the answer. Be concise: 2-6 sentences."
)

def format_evidence(evidence: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(evidence, start=1):
        blocks.append(f"[E{i}] {h.title} — §{h.section}, p.{h.page}\n{h.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, evidence: list[Hit]) -> list[dict]:
    user = (
        f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}\n\n"
        "Answer with [E#] citations:"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
