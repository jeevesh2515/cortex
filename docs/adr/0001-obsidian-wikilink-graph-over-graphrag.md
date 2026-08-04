# ADR 0001: Use the vault's wikilink graph instead of LLM graph extraction

**Status:** Accepted · **Date:** 2026-08-04

## Context

Graph-augmented retrieval (GraphRAG, LightRAG) promises better multi-hop
reasoning by extracting an entity-relationship graph over the corpus. The
question was whether to build one for Cortex.

The 2026 evidence is not flattering:

- A controlled evaluation (1,045 chunks, 23 queries, 12 technical books)
  found graph retrieval **wins** on single-fact lookup (+13–17%) and global
  synthesis (+17–63%), but **loses** to flat hybrid retrieval on multi-hop
  narrative queries. The loss is structural: relational fragmentation destroys
  the prose continuity dense retrieval preserves.
- Indexing cost was $34.09 and 200 minutes for 1,045 chunks using Claude Haiku
  as the extractor. Extrapolated to a 50k-chunk vault: roughly $1,600 and
  several days.
- Extraction quality dominates outcomes. A cheap extractor produced 93
  entities/chunk including bibliographic noise and actively degraded results.

## Decision

Do not extract a graph. Read the one the user already wrote.

An Obsidian vault contains an explicit, hand-curated link graph in the form of
`[[wikilinks]]`, tags and frontmatter. Cortex parses it directly and uses
1-hop neighbour expansion as a third retrieval signal fused via RRF alongside
dense and BM25.

## Consequences

**Positive**

- Zero extraction cost, zero extraction latency. The objection that killed
  GraphRAG for this use case simply does not apply.
- Higher precision than any extractor. A human decided these edges were
  meaningful.
- Updates the instant a note is saved; no re-extraction pass.
- Backlinks come free and matter: in a Zettelkasten, notes pointing *at* a hub
  are often better answers than the hub.

**Negative**

- Only works for link-carrying formats. Extending Cortex to plain PDFs would
  need a different strategy for those documents.
- A sparsely-linked vault gets little benefit. Degrades gracefully to
  dense+BM25 rather than failing.
- Unresolved links are dropped rather than becoming phantom nodes, so a vault
  full of aspirational links to non-existent notes contributes nothing.

## Notes

Expansion is seeded from the top *3* hits, not the top 10. On any vault a broad
seed set eventually covers everything the graph would have reached, and the
signal degenerates to noise. Verified by test: a note sharing zero vocabulary
with the query is retrieved via the graph and is absent without it.
