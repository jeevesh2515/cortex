# Cortex Retrieval Evaluation Report — baseline
**Date (UTC):** 2026-08-29T01:29:07.697346+00:00  ·  **Cortex:** 0.1.0  ·  **Mode:** Dense (live model)

## 1. Summary Metrics

| Metric | Value |
|---|---:|
| **Cases Evaluated** | 25 |
| **Recall@1** | 0.9000 |
| **Recall@5** | 1.0000 |
| **Recall@10** | 1.0000 |
| **MRR** | 1.0000 |
| **MAP** | 0.9933 |
| **nDCG@10** | 0.9968 |
| **p50 Latency** | 1664.6 ms |
| **p95 Latency** | 9687.0 ms |
| **p99 Latency** | 45599.4 ms |
| **Total Misses** | 0 |

## 2. Metrics by Category

| Category | Count | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `dense_semantic` | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `distractor_resistance` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hybrid_core` | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `lexical_exact` | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `multi_topic` | 4 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| `temporal_query` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `wikilink_graph` | 2 | 0.750 | 1.000 | 1.000 | 1.000 | 0.960 |

## 3. Component Ablation

| Disabled | Recall@5 | Δ Recall@5 | Δ nDCG@10 | Saved |
|---|---:|---:|---:|---:|
| **graph** | 1.0000 | +0.0000 | +0.0000 | +180.7 ms |
| **reranker** | 1.0000 | +0.0000 | +0.2545 | +1357.7 ms |
| **expansion** | 1.0000 | +0.0000 | +0.0000 | -54.1 ms |

## 5. Environment & Corpus Provenance

| Property | Value |
|---|---|
| **Corpus Vault** | `eval/vault` |
| **Notes / Chunks** | 11 notes / 11 chunks |
| **Store / Embedder** | LanceStore / qwen3-embedding:0.6b (OllamaEmbedder) |
| **Embedding Dims** | 1024 |
| **Reranker** | bge-reranker-v2-m3 (CrossEncoderReranker) |
| **Python / OS** | Python 3.12.13 (macOS-26.5.2-arm64-arm-64bit, 10 CPUs) |
| **Command** | `cortex eval` |