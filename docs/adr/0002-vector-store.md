# ADR 0002: LanceDB as the vector store, with an in-memory reference implementation

**Status:** Accepted · **Date:** 2026-08-04

## Context

The store must run embedded on a laptop with no daemon, hold 50k–500k chunks,
survive restarts, support real deletes, and ideally provide keyword search
alongside vectors.

| Option | Verdict |
|---|---|
| **LanceDB** | Disk-backed (not RAM-bound), native FTS + vector with RRF, real deletes, stable columnar format, no daemon. |
| sqlite-vec | Excellent below ~50k. Brute-force only, no hybrid search — FTS5 must be wired up by hand. |
| ChromaDB | Index must fit in RAM. No hybrid search. Format stability has been a concern across majors. |
| Qdrant (embedded) | Capable, has sparse-vector hybrid. Heavier than needed for single-user. |
| Milvus Lite | Prototyping only. No hybrid search, no HNSW, migration complexity. |
| DuckDB VSS | Experimental. HNSW index must fit in RAM; deletes only *mark* rows and the index goes stale. |

## Decision

LanceDB for production. Ship a pure-Python `MemoryStore` implementing the same
`VectorStore` protocol.

`MemoryStore` is not a toy: it does exact cosine and a real Okapi BM25. It is
the correct choice for a small vault, and it keeps the entire retrieval stack
testable in any CI image without native wheels.

## Consequences

- Swapping stores is a config change; nothing above the protocol knows which is
  active.
- `lancedb` is an optional extra. If it is missing, Cortex warns loudly and
  falls back — silently degrading to a non-persistent index would be worse than
  failing, because the user would get a working system that forgets everything
  on restart.
- An ANN index is only built past 10k vectors. Below that, LanceDB's exact scan
  is faster than an approximate lookup and has no recall loss.
- LanceDB's API is moving (`table_names` → `list_tables`, `create_fts_index` →
  `create_index(config=FTS())`). Both spellings are supported. The migration
  is not cosmetic: `list_tables()` returns a paginated response object whose
  names live on `.tables`, and iterating it directly fails silently.
