---
title: Vector Databases
tags: [infra, databases, lancedb]
sensitivity: private
---

# Vector Databases

Vector databases index and store dense vector embeddings alongside chunk metadata.
Cortex uses LanceDB as its primary on-disk store due to its zero-server embedded architecture and columnar Apache Arrow foundation.

Key properties:
- Fast Approximate Nearest Neighbor (ANN) search on disk without requiring persistent background daemon processes.
- Memory-mapped columnar queries allow fast scans of metadata and scalar filtering.
- LanceDB integrates with the [[Retrieval Systems]] pipeline to provide candidate vector chunks.
