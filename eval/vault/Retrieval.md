---
title: Retrieval Systems
tags: [rag, search, retrieval]
sensitivity: private
---

# Retrieval Systems

Hybrid retrieval combines dense vector search with BM25 keyword matching.
The fusion step uses reciprocal rank fusion (RRF) to merge the two candidate rankings into a single sorted list.

Wikilink graph expansion traverses links to pull in second-order context:
- [[Embeddings]] provide the vector representations
- [[Vector Databases]] store and query dense vectors
- [[Thermal Management]] regulates indexing load on fanless hardware

Key invariant: Retrieval is always strictly local and private.
