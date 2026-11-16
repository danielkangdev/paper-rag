# Answer-quality eval (2026-11-16T14:55:16)

| Metric | A: baseline (no abstention, no citation check) | B: full system |
| --- | ---: | ---: |
| Citation existence rate (cited ids exist in the evidence given) | 100.0% | 100.0% |
| Citation precision (cited chunk contains a gold quote) | 38.4% | 41.9% |
| Abstention success (unanswerable questions refused) | 66.7% | 100.0% |
| False refusal (answerable questions refused) | 0.0% | 15.6% |
| Ungrounded answer rate (LLM judge: ≥1 unsupported sentence) | 0.0% | 1.7% |

45 answerable + 15 unanswerable questions; generator `gpt-4o-mini`, judge `gpt-4o-mini`. Source: `eval/results/answer_eval_2026-11-16.json`.
