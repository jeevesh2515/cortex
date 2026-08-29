# Cortex Retrieval Quality Evaluation Benchmark

This directory contains the version-controlled evaluation dataset and fixture vault for measuring retrieval quality across embeddings, keyword search, graph expansion, and reranking models.

## Dataset Structure

- `cases.yaml`: 25 curated benchmark cases across 11 domain-specific notes.
- `vault/`: A realistic Obsidian vault fixture containing technical notes, wikilinks, temporal logs, and distractors.

## Case Schema

Each evaluation case defines:
- `id`: Stable identifier (e.g. `retrieval_01_hybrid_rrf`).
- `query`: The user query to evaluate.
- `expect`: List of relevant note identifiers (e.g. `["Retrieval.md"]`).
- `category`: Query classification (`hybrid_core`, `dense_semantic`, `lexical_exact`, `wikilink_graph`, `temporal_query`, `distractor_resistance`, `multi_topic`).
- `rationale`: Why the expected notes are relevant and what aspect of the retrieval engine is tested.

## Metrics Recorded

- **Recall@1**: Fraction of relevant notes retrieved at top rank.
- **Recall@5**: Fraction of relevant notes retrieved within top 5 candidates.
- **Recall@10**: Fraction of relevant notes retrieved within top 10 candidates.
- **MRR (Mean Reciprocal Rank)**: Position penalty $1/\text{rank}$ for the first relevant hit.
- **MAP (Mean Average Precision)**: Order-sensitive precision across all relevant documents.
- **nDCG@10 (Normalized Discounted Cumulative Gain)**: Logarithmic position discounting distinguishing top ranks.
- **Latency**: Mean, p50, p95, p99, min, and max query latencies in milliseconds.

## Running Evaluations

### 1. Offline Deterministic Smoke Evaluation (CI-Safe)

Runs with `HashEmbedder` and zero network or external model dependencies. Fast and fully reproducible across macOS and Linux:

```bash
cortex eval --cases eval/cases.yaml --vault eval/vault --offline --ignore-thermal
```

Or with JSON/Markdown export:

```bash
cortex eval --cases eval/cases.yaml --vault eval/vault --offline --ignore-thermal --output-json eval/results_offline.json --output-md eval/results_offline.md
```

### 2. Dense Quality Baseline (Live Model on AC Power)

Requires Ollama running locally with `qwen3-embedding:0.6b` (and optionally a reranker):

```bash
# Ensure Ollama is running and model is loaded
ollama pull qwen3-embedding:0.6b

# Run dense quality evaluation on AC power
cortex eval --cases eval/cases.yaml --vault eval/vault --output-json eval/baseline_dense.json --output-md eval/baseline_dense.md
```

If the dense embedding provider is unreachable, the evaluation halts with a clear error without generating fabricated fallback scores.

## Limitations

- The dataset contains 25 queries across 11 synthetic notes. It is calibrated for fast regression detection, ablation analysis, and ranking model validation rather than web-scale IR evaluation.
- Document-level binary relevance is used for recall, MRR, MAP, and nDCG calculations.
