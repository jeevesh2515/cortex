---
title: Embeddings
tags: [ml, vectors, ai]
sensitivity: private
---

# Embeddings

Embedding models map arbitrary text passages into a continuous dense vector space.
In this geometric space, semantic similarity between concepts is represented by cosine similarity or inner product.

Characteristics:
- Dense representations capture paraphrases, synonyms, and conceptual relationships that keyword matching misses.
- Modern small embedding models such as Qwen3-Embedding (0.6B parameters) achieve state-of-the-art MTEB scores with low latency.
- In Cortex, dense embeddings are combined with BM25 via [[Retrieval Systems]].
