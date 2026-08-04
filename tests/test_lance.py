"""LanceDB store tests.

Marked ``integration`` because they need native wheels. They assert that
LanceStore is substitutable for MemoryStore -- same behaviour, different
scaling characteristics -- so the rest of the system never has to care which
one is configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.index.lance import lancedb_available
from cortex.index.store import StoredChunk
from cortex.models import Chunk, Sensitivity

pytestmark = pytest.mark.skipif(
    not lancedb_available(), reason="lancedb not installed (optional extra)"
)

DIMS = 8


def vec(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIMS - len(values))
    return padded[:DIMS]


def chunk(cid: str, note: str = "n.md", text: str = "hello world", **kw: object) -> Chunk:
    return Chunk(chunk_id=cid, note_id=note, text=text, **kw)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    from cortex.index.lance import LanceStore

    return LanceStore(tmp_path / "index", dimensions=DIMS)


class TestLanceStore:
    def test_starts_empty(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.count() == 0
        assert store.search_dense(vec(1.0)) == []

    def test_upsert_and_count(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert([StoredChunk(chunk("a"), vec(1.0))])
        assert store.count() == 1

    def test_upsert_is_idempotent(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert([StoredChunk(chunk("a"), vec(1.0))])
        store.upsert([StoredChunk(chunk("a"), vec(0.0, 1.0))])
        assert store.count() == 1

    def test_dense_search_ranks_by_similarity(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            [
                StoredChunk(chunk("near"), vec(1.0, 0.0)),
                StoredChunk(chunk("far"), vec(0.0, 1.0)),
            ]
        )
        results = store.search_dense(vec(1.0, 0.0), top_k=2)
        assert results[0].chunk.chunk_id == "near"

    def test_delete(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert([StoredChunk(chunk("a"), vec(1.0))])
        store.delete(["a"])
        assert store.count() == 0

    def test_delete_missing_is_safe(self, store) -> None:  # type: ignore[no-untyped-def]
        assert store.delete(["nope"]) >= 0

    def test_roundtrip_preserves_metadata(self, store) -> None:  # type: ignore[no-untyped-def]
        original = chunk(
            "a",
            note="dir/note.md",
            text="body text",
            heading_path=["Top", "Sub"],
            ordinal=3,
            tags={"alpha", "beta"},
            links={"other"},
            sensitivity=Sensitivity.PUBLIC,
        )
        store.upsert([StoredChunk(original, vec(1.0))])
        restored = store.chunks_for_note("dir/note.md")[0]
        assert restored.heading_path == ["Top", "Sub"]
        assert restored.ordinal == 3
        assert restored.tags == {"alpha", "beta"}
        assert restored.links == {"other"}
        assert restored.sensitivity is Sensitivity.PUBLIC
        assert restored.citation == "dir/note.md > Top > Sub"

    def test_chunks_for_note_ordered(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            [
                StoredChunk(chunk("a2", ordinal=2), vec(1.0)),
                StoredChunk(chunk("a0", ordinal=0), vec(1.0)),
                StoredChunk(chunk("a1", ordinal=1), vec(1.0)),
            ]
        )
        assert [c.ordinal for c in store.chunks_for_note("n.md")] == [0, 1, 2]

    def test_public_filter(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            [
                StoredChunk(chunk("priv"), vec(1.0)),
                StoredChunk(chunk("pub", sensitivity=Sensitivity.PUBLIC), vec(1.0)),
            ]
        )
        results = store.search_dense(vec(1.0), sensitivity=Sensitivity.PUBLIC)
        assert [r.chunk.chunk_id for r in results] == ["pub"]

    def test_full_text_search(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            [
                StoredChunk(chunk("a", text="vector databases and embeddings"), vec(1.0)),
                StoredChunk(chunk("b", note="b.md", text="sourdough bread baking"), vec(0.0, 1.0)),
            ]
        )
        results = store.search_text("sourdough")
        assert results
        assert results[0].chunk.chunk_id == "b"

    def test_sql_injection_in_id_is_escaped(self, store) -> None:  # type: ignore[no-untyped-def]
        # Chunk ids derive from note paths, and a filename can contain a quote.
        nasty = "note's.md#abc"
        store.upsert([StoredChunk(chunk(nasty, note="note's.md"), vec(1.0))])
        assert store.count() == 1
        store.delete([nasty])
        assert store.count() == 0

    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        from cortex.index.lance import LanceStore

        path = tmp_path / "index"
        first = LanceStore(path, dimensions=DIMS)
        first.upsert([StoredChunk(chunk("a"), vec(1.0))])
        reopened = LanceStore(path, dimensions=DIMS)
        assert reopened.count() == 1

    def test_clear(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert([StoredChunk(chunk("a"), vec(1.0))])
        store.clear()
        assert store.count() == 0

    def test_chunks_by_note_grouping(self, store) -> None:  # type: ignore[no-untyped-def]
        store.upsert(
            [
                StoredChunk(chunk("a1", note="a.md"), vec(1.0)),
                StoredChunk(chunk("b1", note="b.md"), vec(1.0)),
            ]
        )
        assert set(store.chunks_by_note()) == {"a.md", "b.md"}


class TestSubstitutability:
    """LanceStore must be a drop-in for MemoryStore."""

    def test_satisfies_protocol(self, store) -> None:  # type: ignore[no-untyped-def]
        from cortex.index.store import VectorStore

        assert isinstance(store, VectorStore)

    def test_drives_the_retrieval_engine(self, store) -> None:  # type: ignore[no-untyped-def]
        from cortex.llm.providers import HashEmbedder
        from cortex.retrieve.engine import RetrievalEngine

        embedder = HashEmbedder(dimensions=DIMS)
        texts = {
            "Sourdough.md": "feed the starter and bulk ferment the dough",
            "Retrieval.md": "hybrid search fuses dense vectors with bm25",
        }
        items = []
        for note_id, text in texts.items():
            vector = embedder.embed([text])[0]
            items.append(StoredChunk(chunk(f"{note_id}#1", note=note_id, text=text), vector))
        store.upsert(items)

        engine = RetrievalEngine(store, embedder, top_k=2)
        result = engine.retrieve("bulk ferment the starter")
        assert result.chunks
        assert result.chunks[0].chunk.note_id == "Sourdough.md"
