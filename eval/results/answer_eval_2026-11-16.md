# Answer-quality eval (2026-11-16T15:14:20)

| Metric | A: baseline (no abstention, no citation check) | B: full system |
| --- | ---: | ---: |
| Citation existence rate (cited ids exist in the evidence given) | 100.0% | 100.0% |
| Citation precision (cited chunk contains a gold quote) | 38.6% | 38.4% |
| Abstention success (unanswerable questions refused) | 100.0% | 100.0% |
| False refusal (answerable questions refused) | 0.0% | 0.0% |
| Ungrounded answer rate (LLM judge: ≥1 unsupported sentence) | 8.3% | 11.7% |
| LLM calls made for the 15 unanswerable questions | 15 | 0 |

Sentences the judge left unlabelled (excluded, never counted as unsupported): A 0, B 0. 45 answerable + 15 unanswerable questions; generator `gpt-4o-mini`, judge `gpt-4o`. Source: `eval/results/answer_eval_2026-11-16.json`.
