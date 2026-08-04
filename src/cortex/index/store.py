"""Vector store abstraction plus a dependency-free reference implementation.

Two implementations ship:

* :class:`MemoryStore` -- pure Python, exact cosine plus a real BM25. No native
  wheels, so the whole retrieval stack is testable in any CI image, and it is
  genuinely the right choice for a vault under a few thousand notes where an
  ANN index buys nothing but complexity.
* :class:`~cortex.index.lance.LanceStore` -- LanceDB, disk-backed and
  daemon-free, for vaults where the index no longer wants to live in RAM.

Both satisfy :class:`VectorStore`, so swapping is a config change.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cortex.models import Chunk, ScoredChunk, Sensitivity

__all__ = ["BM25", "MemoryStore", "StoredChunk", "VectorStore", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation.

    Deliberately simple and shared between indexing and querying -- the two
    must never disagree, which is the usual source of silent BM25 misses.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class StoredChunk:
    """A chunk plus its vector, as held by a store."""

    chunk: Chunk
    vector: list[float] = field(default_factory=list)


@runtime_checkable
class VectorStore(Protocol):
    """Minimum surface a store must provide."""

    def upsert(self, items: Sequence[StoredChunk]) -> None: ...
    def delete(self, chunk_ids: Iterable[str]) -> int: ...
    def search_dense(
        self, vector: Sequence[float], *, top_k: int = 20, sensitivity: Sensitivity | None = None
    ) -> list[ScoredChunk]: ...
    def search_text(
        self, query: str, *, top_k: int = 20, sensitivity: Sensitivity | None = None
    ) -> list[ScoredChunk]: ...
    def chunks_for_note(self, note_id: str) -> list[Chunk]: ...
    def count(self) -> int: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 for a zero or mismatched vector."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class BM25:
    """Okapi BM25 over an in-memory corpus.

    Standard parameters: ``k1=1.5`` controls term-frequency saturation, ``b=0.75``
    controls length normalisation. Both are the values the literature settled on
    and neither is worth tuning for a personal vault.
    """

    __slots__ = ("_avg_len", "_df", "_docs", "_lengths", "b", "k1")

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._df: Counter[str] = Counter()
        self._docs: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._avg_len = 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._docs[doc_id] = counts
        self._lengths[doc_id] = len(tokens)
        for term in counts:
            self._df[term] += 1
        self._recompute_avg()

    def remove(self, doc_id: str) -> None:
        counts = self._docs.pop(doc_id, None)
        if counts is None:
            return
        self._lengths.pop(doc_id, None)
        for term in counts:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._recompute_avg()

    def _recompute_avg(self) -> None:
        self._avg_len = sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        total_docs = len(self._docs)
        scores: dict[str, float] = defaultdict(float)

        for term in set(terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            # BM25+ style IDF, kept non-negative so a term appearing in most
            # documents cannot subtract from a document's score.
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_id, counts in self._docs.items():
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                length = self._lengths[doc_id]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (length / self._avg_len if self._avg_len else 1.0)
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


class MemoryStore:
    """Exact in-memory store. No approximation, no native dependencies."""

    def __init__(self) -> None:
        self._items: dict[str, StoredChunk] = {}
        self._by_note: dict[str, list[str]] = defaultdict(list)
        self._bm25 = BM25()

    def upsert(self, items: Sequence[StoredChunk]) -> None:
        for item in items:
            chunk_id = item.chunk.chunk_id
            if chunk_id in self._items:
                self._detach(chunk_id)
            self._items[chunk_id] = item
            self._by_note[item.chunk.note_id].append(chunk_id)
            # Index the context header, not the bare text: a section titled
            # "Notes" is unsearchable without knowing which note it belongs to.
            self._bm25.add(chunk_id, item.chunk.with_context_header())

    def _detach(self, chunk_id: str) -> None:
        item = self._items.pop(chunk_id, None)
        if item is None:
            return
        note_chunks = self._by_note.get(item.chunk.note_id)
        if note_chunks and chunk_id in note_chunks:
            note_chunks.remove(chunk_id)
            if not note_chunks:
                del self._by_note[item.chunk.note_id]
        self._bm25.remove(chunk_id)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        removed = 0
        for chunk_id in list(chunk_ids):
            if chunk_id in self._items:
                self._detach(chunk_id)
                removed += 1
        return removed

    def _eligible(self, item: StoredChunk, sensitivity: Sensitivity | None) -> bool:
        if sensitivity is None:
            return True
        if sensitivity is Sensitivity.PUBLIC:
            return item.chunk.sensitivity is Sensitivity.PUBLIC
        return True

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 20,
        sensitivity: Sensitivity | None = None,
    ) -> list[ScoredChunk]:
        scored = [
            (cosine(vector, item.vector), item)
            for item in self._items.values()
            if item.vector and self._eligible(item, sensitivity)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))
        return [
            ScoredChunk(chunk=item.chunk, score=score, source="dense", rank=i)
            for i, (score, item) in enumerate(scored[:top_k], start=1)
        ]

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 20,
        sensitivity: Sensitivity | None = None,
    ) -> list[ScoredChunk]:
        results: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(
            self._bm25.search(query, top_k=top_k * 2), start=1
        ):
            item = self._items.get(chunk_id)
            if item is None or not self._eligible(item, sensitivity):
                continue
            results.append(ScoredChunk(chunk=item.chunk, score=score, source="fts", rank=rank))
            if len(results) >= top_k:
                break
        return results

    def chunks_for_note(self, note_id: str) -> list[Chunk]:
        return [
            self._items[cid].chunk for cid in self._by_note.get(note_id, []) if cid in self._items
        ]

    def all_chunks(self) -> list[Chunk]:
        return [item.chunk for item in self._items.values()]

    def chunks_by_note(self) -> dict[str, list[Chunk]]:
        """Grouping used by graph expansion."""
        out: dict[str, list[Chunk]] = {}
        for note_id, ids in self._by_note.items():
            out[note_id] = [self._items[cid].chunk for cid in ids if cid in self._items]
        return out

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._by_note.clear()
        self._bm25 = BM25()
