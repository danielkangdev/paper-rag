# Retrieval ablation (2026-11-16T14:36:59)

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.345 | 0.489 | 0.315 | 0.317 | 27 ms | 30 ms |
| `sparse` | 0.371 | 0.470 | 0.475 | 0.369 | 5 ms | 10 ms |
| `hybrid` | 0.216 | 0.292 | 0.353 | 0.245 | 34 ms | 38 ms |
| `hybrid_rerank` | 0.311 | 0.402 | 0.568 | 0.375 | 2519 ms | 2683 ms |

Recall@10, `hybrid_rerank` − `dense`: **-0.087** (paired bootstrap 95% CI [-0.174, +0.004], 1000 resamples over 44 questions).

44 answerable questions / 121 evidence quotes; corpus 31 papers / 797 chunks; embedding `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU. Source: `eval/results/ablation_*.json`.
