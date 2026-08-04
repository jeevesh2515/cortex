"""Query pipeline.

    query
      -> embed (local, always)
      -> dense search  ┐
      -> BM25 search   ├─ RRF fusion ─> rerank ─> synthesis ─> cited answer
      -> graph expand  ┘

Two invariants hold throughout:

* **Retrieval is always local.** Embedding and reranking never touch the
  network. Only the final synthesis step can escalate, and then only the
  top-k chunks actually needed to answer travel -- never the vault.
* **Every returned claim carries provenance.** The synthesis prompt numbers its
  sources and the answer cites them, so an answer can always be traced back to
  the note that produced it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from cortex.index.store import MemoryStore, VectorStore
from cortex.llm.protocol import ChatMessage, EmbeddingProvider, RerankProvider
from cortex.llm.router import Router
from cortex.models import Answer, Chunk, ScoredChunk, Sensitivity
from cortex.retrieve.fusion import reciprocal_rank_fusion
from cortex.retrieve.graph import LinkGraph, expand_by_links

logger = logging.getLogger(__name__)

__all__ = ["SYNTHESIS_SYSTEM_PROMPT", "RetrievalEngine", "RetrievalResult"]

SYNTHESIS_SYSTEM_PROMPT = """You are the user's second brain, answering from their own notes.

Rules:
- Answer only from the numbered sources provided. Do not add outside knowledge.
- Cite the sources you use inline as [1], [2], and so on.
- If the sources do not contain the answer, say so plainly. Do not speculate.
- The user wrote these notes. Refer to them as their notes, not as documents.
- Be concise and direct. Match the register of a knowledgeable colleague."""


@dataclass(slots=True)
class RetrievalResult:
    """Retrieved evidence plus a breakdown of how it was found."""

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    per_retriever: dict[str, int] = field(default_factory=dict)
    reranked: bool = False
    elapsed_ms: float = 0.0

    @property
    def notes(self) -> list[str]:
        seen: list[str] = []
        for scored in self.chunks:
            if scored.chunk.note_id not in seen:
                seen.append(scored.chunk.note_id)
        return seen


class RetrievalEngine:
    """Orchestrates hybrid retrieval and grounded synthesis."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        *,
        graph: LinkGraph | None = None,
        reranker: RerankProvider | None = None,
        router: Router | None = None,
        top_k: int = 8,
        dense_k: int = 30,
        fts_k: int = 30,
        graph_hops: int = 1,
        graph_seeds: int = 3,
        fusion_weights: dict[str, float] | None = None,
        rerank_candidates: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.graph = graph
        self.reranker = reranker
        self.router = router
        self.top_k = top_k
        self.dense_k = dense_k
        self.fts_k = fts_k
        self.graph_hops = graph_hops
        # Seeded from the strongest hits only. Expanding from a broad seed set
        # is self-defeating: on any vault the seeds eventually cover everything
        # the graph would have reached, and the signal degenerates to noise.
        self.graph_seeds = graph_seeds
        self.rerank_candidates = rerank_candidates
        self.fusion_weights = fusion_weights or {"dense": 1.0, "fts": 0.8, "graph": 0.5}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_graph: bool = True,
        rerank: bool = True,
        sensitivity: Sensitivity | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval. Entirely local."""
        started = time.monotonic()
        limit = top_k or self.top_k
        rankings: dict[str, Sequence[ScoredChunk]] = {}

        try:
            vectors = self.embedder.embed([query], is_query=True)
            if vectors:
                rankings["dense"] = self.store.search_dense(
                    vectors[0], top_k=self.dense_k, sensitivity=sensitivity
                )
        except Exception as exc:
            logger.warning("dense retrieval unavailable, continuing with BM25: %s", exc)

        rankings["fts"] = self.store.search_text(query, top_k=self.fts_k, sensitivity=sensitivity)

        # Graph expansion seeds from what the other retrievers already found,
        # so it needs them first.
        if use_graph and self.graph is not None:
            n = self.graph_seeds
            seeds = list(rankings.get("dense") or [])[:n] or list(rankings.get("fts") or [])[:n]
            if seeds:
                chunks_by_note = (
                    self.store.chunks_by_note()
                    if isinstance(self.store, MemoryStore)
                    else {
                        note_id: self.store.chunks_for_note(note_id)
                        for note_id in {s.chunk.note_id for s in seeds}
                    }
                )
                if not isinstance(self.store, MemoryStore):
                    # Expansion targets are not in the seed set, so fetch them.
                    expanded_ids = self.graph.expand(
                        [s.chunk.note_id for s in seeds], hops=self.graph_hops
                    )
                    for note_id in expanded_ids:
                        chunks_by_note.setdefault(note_id, self.store.chunks_for_note(note_id))
                graph_hits = expand_by_links(
                    seeds, self.graph, chunks_by_note, hops=self.graph_hops
                )
                if graph_hits:
                    rankings["graph"] = graph_hits

        per_retriever = {name: len(results) for name, results in rankings.items()}
        fused = reciprocal_rank_fusion(
            rankings,
            weights=self.fusion_weights,
            top_k=max(limit, self.rerank_candidates if rerank and self.reranker else limit),
        )

        did_rerank = False
        if rerank and self.reranker is not None and fused:
            candidates = fused[: self.rerank_candidates]
            try:
                order = self.reranker.rerank(
                    query, [c.chunk.with_context_header() for c in candidates], top_k=limit
                )
                fused = [
                    ScoredChunk(
                        chunk=candidates[idx].chunk,
                        score=score,
                        source="rerank",
                        rank=position,
                        components=dict(candidates[idx].components),
                    )
                    for position, (idx, score) in enumerate(order, start=1)
                    if 0 <= idx < len(candidates)
                ]
                did_rerank = True
            except Exception as exc:
                logger.warning("reranking failed, using fused order: %s", exc)

        return RetrievalResult(
            query=query,
            chunks=fused[:limit],
            per_retriever=per_retriever,
            reranked=did_rerank,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    @staticmethod
    def build_prompt(query: str, chunks: Sequence[Chunk]) -> list[ChatMessage]:
        """Assemble a numbered, citable context block."""
        blocks = [
            f"[{i}] {chunk.citation}\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
        ]
        context = "\n\n".join(blocks) if blocks else "(no matching notes found)"
        return [
            ChatMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Sources from my notes:\n\n{context}\n\nQuestion: {query}",
            ),
        ]

    @staticmethod
    def effective_sensitivity(chunks: Sequence[Chunk]) -> Sensitivity:
        """Sensitivity of a set of chunks.

        The whole set is only PUBLIC when every chunk is. One private note in
        the context makes the entire request private -- there is no partial
        redaction, because a summary of private notes is itself private.
        """
        if not chunks:
            return Sensitivity.PRIVATE
        return (
            Sensitivity.PUBLIC
            if all(c.sensitivity is Sensitivity.PUBLIC for c in chunks)
            else Sensitivity.PRIVATE
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        local_only: bool = False,
        use_graph: bool = True,
        rerank: bool = True,
        max_tokens: int = 800,
    ) -> Answer:
        """Retrieve, then synthesise a cited answer."""
        started = time.monotonic()
        result = self.retrieve(query, top_k=top_k, use_graph=use_graph, rerank=rerank)
        chunks = [scored.chunk for scored in result.chunks]

        if self.router is None:
            return Answer(
                question=query,
                text="No model provider is configured. Retrieval succeeded; "
                "see the cited notes below.",
                citations=chunks,
                provider="none",
                elapsed_ms=(time.monotonic() - started) * 1000,
            )

        if not chunks:
            return Answer(
                question=query,
                text="Nothing in your notes matches that.",
                citations=[],
                provider="none",
                elapsed_ms=(time.monotonic() - started) * 1000,
            )

        messages = self.build_prompt(query, chunks)
        completion, decision = self.router.complete(
            messages,
            sensitivity=self.effective_sensitivity(chunks),
            max_tokens=max_tokens,
            local_only=local_only,
        )
        return Answer(
            question=query,
            text=completion.text,
            citations=chunks,
            provider=decision.provider,
            escalated=decision.escalated,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
