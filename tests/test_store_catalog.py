"""Store and catalog tests.

The catalog's job is to make re-indexing cheap and deletes precise, so the
tests focus on the diff logic and chunk lineage rather than CRUD mechanics.
"""

from __future__ import annotations

from cortex.catalog import Catalog
from cortex.index.store import BM25, MemoryStore, StoredChunk, cosine, tokenize
from cortex.models import Chunk, Sensitivity


def chunk(cid: str, note: str = "n.md", text: str = "hello world", **kw: object) -> Chunk:
    return Chunk(chunk_id=cid, note_id=note, text=text, **kw)  # type: ignore[arg-type]


class TestCosine:
    def test_identical_vectors(self) -> None:
        assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_magnitude_invariant(self) -> None:
        assert abs(cosine([1.0, 1.0], [5.0, 5.0]) - 1.0) < 1e-9

    def test_zero_vector_safe(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_length_safe(self) -> None:
        assert cosine([1.0], [1.0, 2.0]) == 0.0


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("Hello, World! 42") == ["hello", "world", "42"]

    def test_drops_punctuation(self) -> None:
        assert tokenize("a-b_c") == ["a", "b", "c"]


class TestBM25:
    def test_finds_matching_document(self) -> None:
        bm = BM25()
        bm.add("a", "retrieval augmented generation")
        bm.add("b", "cooking pasta at home")
        results = bm.search("retrieval")
        assert results[0][0] == "a"

    def test_rare_term_outranks_common(self) -> None:
        bm = BM25()
        for i in range(10):
            bm.add(f"common{i}", "the quick brown fox")
        bm.add("rare", "the quick brown zygote")
        results = dict(bm.search("zygote"))
        assert "rare" in results

    def test_removal_excludes_document(self) -> None:
        bm = BM25()
        bm.add("a", "unique term here")
        bm.remove("a")
        assert bm.search("unique") == []

    def test_readd_replaces(self) -> None:
        bm = BM25()
        bm.add("a", "original content")
        bm.add("a", "replacement content")
        assert bm.search("original") == []
        assert bm.search("replacement")

    def test_empty_query(self) -> None:
        bm = BM25()
        bm.add("a", "content")
        assert bm.search("") == []

    def test_unknown_term(self) -> None:
        bm = BM25()
        bm.add("a", "content")
        assert bm.search("nonexistent") == []


class TestMemoryStore:
    def test_upsert_and_count(self) -> None:
        store = MemoryStore()
        store.upsert([StoredChunk(chunk("a"), [1.0, 0.0])])
        assert store.count() == 1

    def test_upsert_is_idempotent(self) -> None:
        store = MemoryStore()
        store.upsert([StoredChunk(chunk("a"), [1.0, 0.0])])
        store.upsert([StoredChunk(chunk("a"), [0.0, 1.0])])
        assert store.count() == 1

    def test_dense_search_ranks_by_similarity(self) -> None:
        store = MemoryStore()
        store.upsert(
            [
                StoredChunk(chunk("near"), [1.0, 0.0]),
                StoredChunk(chunk("far"), [0.0, 1.0]),
            ]
        )
        results = store.search_dense([1.0, 0.0])
        assert results[0].chunk.chunk_id == "near"
        assert results[0].source == "dense"

    def test_dense_search_respects_top_k(self) -> None:
        store = MemoryStore()
        store.upsert([StoredChunk(chunk(f"c{i}"), [1.0, 0.0]) for i in range(10)])
        assert len(store.search_dense([1.0, 0.0], top_k=3)) == 3

    def test_text_search(self) -> None:
        store = MemoryStore()
        store.upsert(
            [
                StoredChunk(chunk("a", text="vector databases and embeddings"), [1.0]),
                StoredChunk(chunk("b", text="making sourdough bread"), [1.0]),
            ]
        )
        results = store.search_text("embeddings")
        assert results[0].chunk.chunk_id == "a"
        assert results[0].source == "fts"

    def test_text_search_indexes_breadcrumb(self) -> None:
        # A section called "Notes" is only findable via its note context.
        store = MemoryStore()
        store.upsert([StoredChunk(chunk("a", note="Kubernetes.md", text="see the docs"), [1.0])])
        assert store.search_text("kubernetes")

    def test_delete(self) -> None:
        store = MemoryStore()
        store.upsert([StoredChunk(chunk("a"), [1.0])])
        assert store.delete(["a"]) == 1
        assert store.count() == 0
        assert store.search_text("hello") == []

    def test_delete_missing_is_noop(self) -> None:
        assert MemoryStore().delete(["nope"]) == 0

    def test_chunks_for_note(self) -> None:
        store = MemoryStore()
        store.upsert(
            [
                StoredChunk(chunk("a1", note="a.md"), [1.0]),
                StoredChunk(chunk("a2", note="a.md"), [1.0]),
                StoredChunk(chunk("b1", note="b.md"), [1.0]),
            ]
        )
        assert len(store.chunks_for_note("a.md")) == 2

    def test_public_filter(self) -> None:
        store = MemoryStore()
        store.upsert(
            [
                StoredChunk(chunk("priv", text="secret"), [1.0]),
                StoredChunk(chunk("pub", text="secret", sensitivity=Sensitivity.PUBLIC), [1.0]),
            ]
        )
        results = store.search_dense([1.0], sensitivity=Sensitivity.PUBLIC)
        assert [r.chunk.chunk_id for r in results] == ["pub"]

    def test_chunks_by_note_grouping(self) -> None:
        store = MemoryStore()
        store.upsert(
            [
                StoredChunk(chunk("a1", note="a.md"), [1.0]),
                StoredChunk(chunk("b1", note="b.md"), [1.0]),
            ]
        )
        grouped = store.chunks_by_note()
        assert set(grouped) == {"a.md", "b.md"}


class TestCatalog:
    def test_roundtrip(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1", "a#2"])
            record = cat.get("a.md")
            assert record is not None
            assert record.content_hash == "h1"
            assert record.chunk_count == 2

    def test_chunk_lineage(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1", "a#2"])
            assert cat.chunk_ids("a.md") == ["a#1", "a#2"]

    def test_replace_returns_stale_chunks(self) -> None:
        # The property that makes deletes precise on re-index.
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1", "a#2", "a#3"])
            stale = cat.replace_note("a.md", content_hash="h2", chunk_ids=["a#1", "a#9"])
            assert stale == ["a#2", "a#3"]

    def test_replace_with_identical_chunks_yields_no_stale(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
            assert cat.replace_note("a.md", content_hash="h2", chunk_ids=["a#1"]) == []

    def test_forget_returns_chunks(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1", "a#2"])
            assert cat.forget("a.md") == ["a#1", "a#2"]
            assert cat.get("a.md") is None

    def test_diff_detects_new(self) -> None:
        with Catalog() as cat:
            changes = cat.diff({"a.md": "h1"})
            assert changes.new == ["a.md"]
            assert changes.needs_indexing == ["a.md"]

    def test_diff_detects_changed(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
            changes = cat.diff({"a.md": "h2"})
            assert changes.changed == ["a.md"]

    def test_diff_detects_unchanged(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
            changes = cat.diff({"a.md": "h1"})
            assert changes.unchanged == ["a.md"]
            assert changes.needs_indexing == []

    def test_diff_detects_deleted(self) -> None:
        # Ghost chunks from notes removed while the daemon was down.
        with Catalog() as cat:
            cat.replace_note("gone.md", content_hash="h1", chunk_ids=["g#1"])
            changes = cat.diff({})
            assert changes.deleted == ["gone.md"]

    def test_diff_is_empty_when_nothing_moved(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
            assert cat.diff({"a.md": "h1"}).is_empty

    def test_diff_summary_readable(self) -> None:
        with Catalog() as cat:
            assert "1 new" in cat.diff({"a.md": "h1"}).summary()

    def test_stats(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1", "a#2"])
            assert cat.stats() == {"notes": 1, "chunks": 2}

    def test_persists_to_disk(self, tmp_path: object) -> None:
        db = tmp_path / "cortex.db"  # type: ignore[operator]
        with Catalog(db) as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
        with Catalog(db) as reopened:
            assert reopened.get("a.md") is not None

    def test_meta_roundtrip(self) -> None:
        with Catalog() as cat:
            cat.set_meta("last_scan", "123")
            assert cat.get_meta("last_scan") == "123"
            assert cat.get_meta("missing", "fallback") == "fallback"

    def test_clear(self) -> None:
        with Catalog() as cat:
            cat.replace_note("a.md", content_hash="h1", chunk_ids=["a#1"])
            cat.clear()
            assert cat.stats() == {"notes": 0, "chunks": 0}
