"""Rank fusion.

Combining a dense score with a BM25 score by weighted averaging is the classic
production mistake: BM25 is an unbounded positive number, cosine similarity is
bounded in [-1, 1], so any linear blend is dominated by BM25 regardless of the
alpha you pick.

Reciprocal Rank Fusion sidesteps the problem by discarding magnitudes and using
only rank position. It needs no normalisation and no per-corpus tuning, which
is why it is the 2026 default in Elasticsearch, LanceDB and Qdrant alike.

    RRF(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

``k=60`` is the constant from Cormack et al. (SIGIR 2009) and is what every
mainstream implementation ships.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cortex.models import Chunk, ScoredChunk

__all__ = ["RRF_K", "normalise_scores", "reciprocal_rank_fusion"]

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Fuse several ranked lists into one.

    ``rankings`` maps a retriever name to its ordered results, best first.
    ``weights`` lets a retriever count for more without reintroducing the scale
    problem -- it multiplies the reciprocal-rank contribution, not a raw score.

    Per-retriever contributions are preserved on the result so that a
    surprising ranking can be explained rather than guessed at.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    chunks: dict[str, Chunk] = {}

    for retriever, results in rankings.items():
        weight = weights.get(retriever, 1.0)
        if weight == 0:
            continue
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            contribution = weight / (k + rank)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + contribution
            components.setdefault(chunk_id, {})[retriever] = contribution
            chunks.setdefault(chunk_id, scored.chunk)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=score,
            source="fused",
            rank=position,
            components=components.get(chunk_id, {}),
        )
        for position, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def normalise_scores(results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Min-max normalise scores into [0, 1] for display.

    Presentation only. Fusion never uses this -- that is the whole point of RRF.
    """
    if not results:
        return []
    scores = [r.score for r in results]
    low, high = min(scores), max(scores)
    span = high - low
    return [
        ScoredChunk(
            chunk=r.chunk,
            score=1.0 if span == 0 else (r.score - low) / span,
            source=r.source,
            rank=r.rank,
            components=dict(r.components),
        )
        for r in results
    ]
