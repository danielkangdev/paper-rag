# LLM judge: sentence-level groundedness

Used by `scirag/eval/run_answer_eval.py`. The text between the `PROMPT` markers is sent
verbatim (with `{question}`, `{evidence}`, `{sentences}`, `{n}` filled in). Sentences the
judge does not label after 3 attempts are reported as `unjudged` and never counted as
unsupported (an earlier version did, which inflated the ungrounded rate).

## Rules

1. The answer is split into sentences; each sentence is judged on its own.
2. A sentence is **supported** if every factual claim in it is stated in, or directly
   entailed by, the passages it cites (`[E2]` etc.). If it cites nothing, judge it
   against all passages given to the model.
3. A sentence is **unsupported** if any claim in it is missing from, contradicts, or goes
   beyond the relevant passages: added numbers, names, mechanisms, or general knowledge
   that the passages do not state. Citing the wrong passage also counts as unsupported.
4. A sentence is **no_claim** if it makes no factual claim about the world, e.g. "The
   indexed papers do not contain this information." or a pure restatement of the question.
5. Paraphrase is fine; judge meaning, not wording. Do not use your own knowledge to fill
   gaps: a true statement that the passages do not support is still unsupported.
6. An answer is **ungrounded** if at least one sentence is unsupported. An answer whose
   sentences are all no_claim counts as a **refusal**.
7. Post-check (in code, not in the prompt): an uncited sentence matching the refusal
   pattern in `scirag/rag/citations.py::is_refusal_sentence` ("the papers do not contain
   …") is forced to no_claim. In the first run the judge had labelled such sentences
   "supported", which undercounted the baseline's refusals.
8. Judge model: `gpt-4o` (the generator is `gpt-4o-mini`). In the first run a
   `gpt-4o-mini` judge marked 167/177 sentences supported and looked too lenient.

## Prompt

<!-- PROMPT:start -->
You are checking whether an answer is grounded in its evidence.

Question: {question}

Evidence passages given to the answering model:
{evidence}

Answer sentences:
{sentences}

For each numbered sentence, output one label:
- "supported": every factual claim is stated in or directly entailed by the passages it cites (or, if it cites none, by any passage above).
- "unsupported": some claim is missing from, contradicts, or goes beyond those passages (including correct general knowledge the passages do not state, or citing the wrong passage).
- "no_claim": no factual claim (e.g. saying the papers do not contain the answer).

Judge meaning, not wording. Do not use outside knowledge to fill gaps.

There are {n} sentences. Return JSON with exactly {n} entries, one per sentence, with i = 1 to {n}:
{"sentences": [{"i": 1, "label": "supported|unsupported|no_claim", "reason": "<short>"}, {"i": 2, ...}, ...]}
<!-- PROMPT:end -->
