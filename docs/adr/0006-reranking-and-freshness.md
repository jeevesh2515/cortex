# ADR 0006: Cross-encoder reranking as an opt-in extra

**Status:** Accepted · **Date:** 2026-08-04

## Context

`rerank_enabled = true` shipped as a default with no implementation behind it --
the config claimed something that never happened, and the README advertised a
recall improvement the code could not deliver. That is the worst kind of defect:
silent, and flattering.

Reranking is genuinely the highest-leverage addition to hybrid retrieval.
Bi-encoders embed query and document *independently*, so they can only measure
"same region of vector space". A cross-encoder sees both together in one forward
pass and scores actual relevance. Published figures put BGE-reranker-v2-m3 at
roughly +18% recall@5 for about 84ms on 100 candidates, and every comparable
project surveyed uses that model.

## Decision

Implement it properly, but as an **opt-in extra** (`pip install
'cortex-brain[rerank]'`), because it pulls torch.

Two concessions to a fanless 16 GB machine:

- **Lazy load.** Weights are untouched until the first rerank call, so
  `cortex index` pays nothing for a reranker it never invokes.
- **Idle unload.** After `rerank_idle_ttl` (default 600s) the weights are
  released. Embedder + reranker + chat model do not co-reside comfortably in
  the memory budget.

When the dependency is absent, `build_reranker` returns `None` and logs how to
install it. It deliberately does **not** substitute a no-op stub: returning
`None` lets `cortex status` report honestly that reranking is inactive, whereas
a silent stub would leave the config still claiming something untrue.

`bge-reranker-v2-m3` is the default; `minilm` is aliased as a ~22M-parameter
alternative that is roughly 4x faster with a meaningfully lower ceiling -- the
right trade on a thermally limited machine if bge proves too slow.

## Consequences

- The default install stays light. Someone who only wants search never downloads
  torch.
- On macOS the torch wheel is the MPS build (a few hundred MB), not the
  multi-gigabyte CUDA build seen on Linux CI.
- Device selection prefers MPS, then CUDA, then CPU.
- A model load is guarded by a lock: the MCP server and a background indexer can
  both reach the reranker, and loading the same weights twice would double peak
  memory.
- Reranker failure is caught by the engine and degrades to the fused order,
  which is already good.
