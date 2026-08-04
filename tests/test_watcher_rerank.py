"""Watcher and reranker tests.

Both wrap machinery that is awkward to test directly -- a kernel event stream and
a neural model -- so the tests target the logic wrapped *around* them: debounce
and coalescing for the watcher, lifecycle and ordering for the reranker. The
model itself is exercised by an integration test that only runs when the optional
dependency is present.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from cortex.ingest.watcher import MARKDOWN_PATTERNS, VaultWatcher, watchdog_available
from cortex.retrieve.rerank import (
    CrossEncoderReranker,
    NoopReranker,
    build_reranker,
    rerank_available,
    resolve_model_id,
)


class TestWatcherDebounce:
    """The debounce is the whole point: editors emit several events per save."""

    def test_single_change_flushes_after_debounce(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.05)
        watcher._record(tmp_path / "a.md")
        assert seen == [], "must not fire before the debounce elapses"
        time.sleep(0.15)
        assert seen == [{tmp_path / "a.md"}]

    def test_burst_coalesces_into_one_batch(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.08)
        for name in ("a.md", "b.md", "c.md"):
            watcher._record(tmp_path / name)
            time.sleep(0.01)
        time.sleep(0.2)
        assert len(seen) == 1, "a burst of saves is one indexing pass"
        assert len(seen[0]) == 3

    def test_repeated_event_for_same_file_is_deduped(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.05)
        for _ in range(5):
            watcher._record(tmp_path / "a.md")
        time.sleep(0.15)
        assert seen == [{tmp_path / "a.md"}]

    def test_max_batch_wait_forces_a_flush(self, tmp_path: Path) -> None:
        """Continuous editing must not defer indexing indefinitely."""
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.05, max_batch_wait=0.12)
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            watcher._record(tmp_path / "a.md")
            time.sleep(0.02)
        time.sleep(0.15)
        assert seen, "ceiling must fire even while edits keep arriving"

    def test_handler_exception_does_not_kill_the_watcher(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def explode(_: set[Path]) -> None:
            calls.append(1)
            raise RuntimeError("indexing blew up")

        watcher = VaultWatcher(tmp_path, explode, debounce=0.03)
        watcher._record(tmp_path / "a.md")
        time.sleep(0.1)
        watcher._record(tmp_path / "b.md")
        time.sleep(0.1)
        assert len(calls) == 2, "a failed pass must not stop later ones"

    def test_pending_count(self, tmp_path: Path) -> None:
        watcher = VaultWatcher(tmp_path, lambda _: None, debounce=5.0)
        watcher._record(tmp_path / "a.md")
        watcher._record(tmp_path / "b.md")
        assert watcher.pending_count == 2

    def test_stop_cancels_pending_flush(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.05)
        watcher._record(tmp_path / "a.md")
        watcher.stop()
        time.sleep(0.15)
        assert seen == []

    def test_not_running_before_start(self, tmp_path: Path) -> None:
        assert not VaultWatcher(tmp_path, lambda _: None).running

    def test_patterns_cover_markdown_variants(self) -> None:
        assert "*.md" in MARKDOWN_PATTERNS
        assert "*.markdown" in MARKDOWN_PATTERNS


@pytest.mark.skipif(not watchdog_available(), reason="watchdog not installed")
class TestWatcherIntegration:
    def test_detects_a_real_file_write(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.2)
        watcher.start()
        try:
            assert watcher.running
            (tmp_path / "note.md").write_text("# Hello\n", encoding="utf-8")
            deadline = time.monotonic() + 5.0
            while not seen and time.monotonic() < deadline:
                time.sleep(0.05)
            assert seen, "a real write should surface as an event"
            assert any(p.name == "note.md" for p in seen[0])
        finally:
            watcher.stop()
        assert not watcher.running

    def test_ignores_non_markdown(self, tmp_path: Path) -> None:
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.2)
        watcher.start()
        try:
            (tmp_path / "image.png").write_bytes(b"\x89PNG")
            time.sleep(0.8)
            assert seen == []
        finally:
            watcher.stop()

    def test_ignores_dot_directories(self, tmp_path: Path) -> None:
        """`.obsidian` churns constantly and holds no knowledge."""
        seen: list[set[Path]] = []
        watcher = VaultWatcher(tmp_path, seen.append, debounce=0.2)
        watcher.start()
        try:
            hidden = tmp_path / ".obsidian"
            hidden.mkdir()
            (hidden / "workspace.md").write_text("x", encoding="utf-8")
            time.sleep(0.8)
            assert seen == []
        finally:
            watcher.stop()


class TestModelAliases:
    def test_shorthand_expands(self) -> None:
        assert resolve_model_id("bge-reranker-v2-m3") == "BAAI/bge-reranker-v2-m3"
        assert resolve_model_id("minilm") == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_case_and_whitespace_tolerated(self) -> None:
        assert resolve_model_id("  BGE-Reranker-V2-M3 ") == "BAAI/bge-reranker-v2-m3"

    def test_full_id_passes_through(self) -> None:
        assert resolve_model_id("org/some-model") == "org/some-model"


class TestBuildReranker:
    def test_disabled_returns_none(self) -> None:
        assert build_reranker(enabled=False) is None

    def test_returns_none_rather_than_a_stub_when_unavailable(self, monkeypatch: Any) -> None:
        # Returning None lets `cortex status` say honestly that reranking is
        # inactive, instead of the config claiming something untrue.
        monkeypatch.setattr("cortex.retrieve.rerank.rerank_available", lambda: False)
        assert build_reranker(enabled=True) is None

    def test_warns_when_enabled_but_missing(self, monkeypatch: Any, caplog: Any) -> None:
        import logging

        monkeypatch.setattr("cortex.retrieve.rerank.rerank_available", lambda: False)
        with caplog.at_level(logging.WARNING):
            build_reranker(enabled=True)
        assert "cortex-brain[rerank]" in caplog.text


class TestNoopReranker:
    def test_preserves_input_order(self) -> None:
        out = NoopReranker().rerank("q", ["a", "b", "c"], top_k=3)
        assert [i for i, _ in out] == [0, 1, 2]

    def test_respects_top_k(self) -> None:
        assert len(NoopReranker().rerank("q", ["a", "b", "c"], top_k=2)) == 2

    def test_empty_documents(self) -> None:
        assert NoopReranker().rerank("q", [], top_k=5) == []


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    """Loaded once for the module -- the download and warm-up are the slow part."""
    return CrossEncoderReranker("minilm", idle_ttl=3600.0)


@pytest.mark.skipif(not rerank_available(), reason="sentence-transformers not installed")
@pytest.mark.integration
class TestCrossEncoderIntegration:
    """Exercises a real cross-encoder. Downloads a ~90MB model on first run."""

    def test_lazy_until_first_use(self) -> None:
        fresh = CrossEncoderReranker("minilm")
        assert not fresh.loaded, "constructing must not load weights"

    def test_ranks_the_relevant_passage_first(self, reranker: CrossEncoderReranker) -> None:
        docs = [
            "LanceDB stores vectors on disk and needs no daemon.",
            "Feed the sourdough starter twice daily with flour and water.",
            "Reciprocal rank fusion merges rankings by position.",
        ]
        out = reranker.rerank("how do I feed my sourdough starter?", docs, top_k=3)
        assert out[0][0] == 1, "the starter passage must rank first"

    def test_scores_are_ordered_descending(self, reranker: CrossEncoderReranker) -> None:
        docs = ["totally unrelated text about cars", "python type annotations guide"]
        out = reranker.rerank("how do type hints work in python?", docs, top_k=2)
        assert out[0][1] >= out[1][1]

    def test_top_k_truncates(self, reranker: CrossEncoderReranker) -> None:
        docs = [f"document number {i}" for i in range(6)]
        assert len(reranker.rerank("document", docs, top_k=2)) == 2

    def test_empty_documents_short_circuits(self) -> None:
        fresh = CrossEncoderReranker("minilm")
        assert fresh.rerank("q", [], top_k=3) == []
        assert not fresh.loaded, "an empty call must not load the model"

    def test_unload_frees_and_reloads(self, reranker: CrossEncoderReranker) -> None:
        reranker.rerank("q", ["a"], top_k=1)
        assert reranker.loaded
        assert reranker.unload()
        assert not reranker.loaded
        assert not reranker.unload(), "unloading twice is a no-op"
        reranker.rerank("q", ["a"], top_k=1)
        assert reranker.loaded

    def test_idle_unload_respects_ttl(self) -> None:
        short = CrossEncoderReranker("minilm", idle_ttl=300.0)
        short.rerank("q", ["a"], top_k=1)
        assert not short.maybe_unload(), "still fresh, must stay loaded"
        assert short.maybe_unload(now=time.monotonic() + 600)
        assert not short.loaded

    def test_zero_ttl_disables_idle_unload(self) -> None:
        never = CrossEncoderReranker("minilm", idle_ttl=0.0)
        never.rerank("q", ["a"], top_k=1)
        assert not never.maybe_unload(now=time.monotonic() + 10_000)
        assert never.loaded

    def test_satisfies_the_protocol(self, reranker: CrossEncoderReranker) -> None:
        from cortex.llm.protocol import RerankProvider

        assert isinstance(reranker, RerankProvider)

    def test_drives_the_engine(self, tmp_path: Path) -> None:
        """End to end: reranking actually reorders engine output."""
        from cortex.catalog import Catalog
        from cortex.index.store import MemoryStore, StoredChunk
        from cortex.llm.providers import HashEmbedder
        from cortex.models import Chunk
        from cortex.retrieve.engine import RetrievalEngine

        embedder = HashEmbedder(dimensions=64)
        store = MemoryStore()
        texts = {
            "Sourdough.md": "Feed the starter twice daily with flour and water.",
            "Lance.md": "LanceDB stores vectors on disk with no daemon.",
            "Fusion.md": "Reciprocal rank fusion merges rankings by position.",
        }
        store.upsert(
            [
                StoredChunk(
                    Chunk(chunk_id=f"{n}#1", note_id=n, text=t),
                    embedder.embed([t])[0],
                )
                for n, t in texts.items()
            ]
        )
        engine = RetrievalEngine(
            store,
            embedder,
            reranker=CrossEncoderReranker("minilm"),
            top_k=3,
            rerank_candidates=3,
        )
        result = engine.retrieve("how do I feed my sourdough starter?")
        assert result.reranked, "reranker must have run"
        assert result.chunks[0].chunk.note_id == "Sourdough.md"
        assert result.chunks[0].source == "rerank"
        Catalog().close()
