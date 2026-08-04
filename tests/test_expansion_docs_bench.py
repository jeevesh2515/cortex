"""Tests for query expansion, document ingestion and the benchmark harness."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from cortex.bench import (
    BenchCase,
    IndexBenchmark,
    QualityReport,
    ablate,
    average_precision,
    evaluate,
    load_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from cortex.catalog import Catalog
from cortex.index.store import MemoryStore
from cortex.ingest.parsers import (
    available_formats,
    extract_document,
    html_to_markdown,
    supported_suffixes,
)
from cortex.ingest.pipeline import IndexPipeline
from cortex.llm.providers import HashEmbedder
from cortex.retrieve.engine import RetrievalEngine
from cortex.retrieve.expansion import STOPWORDS, expand_query


class TestQueryExpansion:
    def test_strips_stopwords(self) -> None:
        expanded = expand_query("what did I decide about the chunking strategy?")
        assert "chunking" in expanded.salient_terms
        assert "what" not in expanded.salient_terms
        assert "the" not in expanded.salient_terms

    def test_interrogatives_are_stopwords(self) -> None:
        # In a personal notes system "what" and "my" match everything.
        for word in ("what", "my", "i", "me", "show", "tell"):
            assert word in STOPWORDS

    def test_no_cyrillic_lookalikes(self) -> None:
        # Regression: a stray Cyrillic lookalike crept into the stopword list.
        for word in STOPWORDS:
            assert word.isascii(), f"non-ascii stopword: {word!r}"

    def test_quoted_phrase_extracted(self) -> None:
        expanded = expand_query('find "reciprocal rank fusion" in my notes')
        assert expanded.phrases == ["reciprocal rank fusion"]

    def test_quoted_phrase_leads_the_variants(self) -> None:
        # An explicit instruction outranks anything inferred.
        expanded = expand_query('the "exact phrase" matters here')
        assert expanded.lexical_variants()[0] == "exact phrase"

    def test_smart_quotes_supported(self) -> None:
        assert expand_query("find “smart quoted” text").phrases == ["smart quoted"]

    def test_tags_extracted(self) -> None:
        expanded = expand_query("notes tagged #project/cortex about retrieval")
        assert expanded.tags == ["project/cortex"]

    def test_tag_not_duplicated_as_a_term(self) -> None:
        expanded = expand_query("#rag notes")
        assert "rag" not in expanded.salient_terms

    def test_distinctive_terms_lead(self) -> None:
        expanded = expand_query("notes on LanceDB and k8s")
        assert expanded.salient_terms[0] in {"lancedb", "k8s"}

    def test_digits_are_distinctive(self) -> None:
        assert "qwen3" in expand_query("about qwen3 models").salient_terms

    def test_deduplicates(self) -> None:
        terms = expand_query("chunking chunking chunking strategy").salient_terms
        assert terms.count("chunking") == 1

    def test_all_stopwords_has_no_signal(self) -> None:
        # Nothing to gain from a second identical query.
        assert not expand_query("what do I know about it").has_signal

    def test_empty_query(self) -> None:
        expanded = expand_query("")
        assert expanded.salient_terms == []
        assert expanded.lexical_variants() == []

    def test_original_always_included(self) -> None:
        query = "what did I decide about chunking strategy?"
        assert query in expand_query(query).lexical_variants()

    def test_max_terms_respected(self) -> None:
        long_query = " ".join(f"distinctive{i}" for i in range(50))
        assert len(expand_query(long_query, max_terms=5).salient_terms) == 5


class TestExpansionInRetrieval:
    def _pipeline(self, root: Path) -> IndexPipeline:
        return IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )

    def test_verbose_question_still_finds_the_note(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Chunking.md").write_text(
            "# Chunking\n\nRecursive splitting at 512 tokens beats semantic chunking.\n",
            encoding="utf-8",
        )
        for i in range(8):
            (root / f"Filler {i}.md").write_text(
                f"# Filler {i}\n\nWhat did I do about the thing for my notes number {i}.\n",
                encoding="utf-8",
            )
        pipeline = self._pipeline(root)
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder, top_k=3)
        result = engine.retrieve("what did I decide about the chunking strategy for my notes?")
        assert result.chunks
        assert result.chunks[0].chunk.note_id == "Chunking.md"

    def test_can_be_disabled(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "A.md").write_text("# A\n\nchunking strategy content\n", encoding="utf-8")
        pipeline = self._pipeline(root)
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder, expansion_enabled=False)
        result = engine.retrieve("what about the chunking strategy for notes")
        assert not any(name.startswith("fts:") for name in result.per_retriever)

    def test_does_not_mutate_shared_weights(self, tmp_path: Path) -> None:
        """Regression: setdefault on a shared Settings dict leaked variants."""
        root = tmp_path / "vault"
        root.mkdir()
        (root / "A.md").write_text("# A\n\nchunking strategy content\n", encoding="utf-8")
        pipeline = self._pipeline(root)
        pipeline.run()

        shared = {"dense": 1.0, "fts": 0.8, "graph": 0.5}
        snapshot = dict(shared)
        for _ in range(3):
            engine = RetrievalEngine(pipeline.store, pipeline.embedder, fusion_weights=shared)
            engine.retrieve("what did I decide about the chunking strategy for my notes?")
        assert shared == snapshot


class TestHtmlExtraction:
    def test_strips_chrome(self) -> None:
        doc = html_to_markdown(
            "<html><body><nav>Menu</nav><p>Real content.</p>"
            "<footer>Legal</footer><script>x=1</script></body></html>"
        )
        assert "Real content." in doc.text
        assert "Menu" not in doc.text
        assert "Legal" not in doc.text
        assert "x=1" not in doc.text

    def test_preserves_headings(self) -> None:
        doc = html_to_markdown("<h1>Title</h1><h2>Sub</h2><p>Body</p>")
        assert "# Title" in doc.text
        assert "## Sub" in doc.text

    def test_extracts_title(self) -> None:
        doc = html_to_markdown("<html><head><title>Page Title</title></head><body>x</body></html>")
        assert doc.title == "Page Title"

    def test_list_items_become_bullets(self) -> None:
        doc = html_to_markdown("<ul><li>one</li><li>two</li></ul>")
        assert "- one" in doc.text

    def test_external_links_preserved(self) -> None:
        doc = html_to_markdown('<p>See <a href="https://example.com">this</a>.</p>')
        assert "[this](https://example.com)" in doc.text

    def test_entities_unescaped(self) -> None:
        assert "R&D" in html_to_markdown("<p>R&amp;D</p>").text

    def test_malformed_html_does_not_raise(self) -> None:
        assert html_to_markdown("<p>unclosed <b>bold").text


class TestDocumentDispatch:
    def test_unsupported_suffix_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(b"\x89PNG")
        assert extract_document(path) is None

    def test_plain_text(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("plain content", encoding="utf-8")
        doc = extract_document(path)
        assert doc is not None
        assert doc.text == "plain content"

    def test_epub(self, tmp_path: Path) -> None:
        book = tmp_path / "guide.epub"
        with zipfile.ZipFile(book, "w") as archive:
            archive.writestr(
                "ch1.xhtml",
                "<html><head><title>A Guide</title></head>"
                "<body><h1>Chapter One</h1><p>Content here.</p></body></html>",
            )
        doc = extract_document(book)
        assert doc is not None
        assert doc.title == "A Guide"
        assert "Content here." in doc.text

    def test_corrupt_epub_warns_rather_than_raising(self, tmp_path: Path) -> None:
        book = tmp_path / "broken.epub"
        book.write_bytes(b"not a zip file at all")
        doc = extract_document(book)
        assert doc is not None
        assert doc.is_empty
        assert doc.warning

    def test_available_formats_reports_html_and_epub(self) -> None:
        formats = available_formats()
        assert formats["html"] is True
        assert formats["epub"] is True

    def test_supported_suffixes_includes_html(self) -> None:
        assert ".html" in supported_suffixes()


class TestDocumentIngestion:
    def _vault(self, root: Path) -> None:
        root.mkdir(exist_ok=True)
        (root / "Note.md").write_text("# Note\n\nOrdinary markdown.\n", encoding="utf-8")
        (root / "clip.html").write_text(
            "<html><head><title>Clipped</title></head><body>"
            "<h1>Clipped</h1><p>Distinctive imported phrase.</p></body></html>",
            encoding="utf-8",
        )

    def test_documents_off_by_default(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        self._vault(root)
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )
        assert pipeline.run().indexed == 1

    def test_documents_indexed_when_enabled(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        self._vault(root)
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            documents=True,
        )
        assert pipeline.run().indexed == 2

    def test_imported_document_is_searchable_with_citation(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        self._vault(root)
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            documents=True,
        )
        pipeline.run()
        hits = pipeline.store.search_text("distinctive imported phrase")
        assert hits
        assert hits[0].chunk.note_id == "clip.html"

    def test_imported_documents_are_tagged(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        self._vault(root)
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            documents=True,
        )
        pipeline.run()
        chunks = pipeline.store.chunks_for_note("clip.html")
        assert chunks
        assert "cortex/imported" in chunks[0].tags

    def test_empty_extraction_is_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "empty.html").write_text("<html><body></body></html>", encoding="utf-8")
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            documents=True,
        )
        assert pipeline.run().indexed == 0


class TestMetrics:
    def test_recall_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], frozenset({"a", "b"}), 5) == 1.0
        assert recall_at_k(["a", "x", "y"], frozenset({"a", "b"}), 5) == 0.5
        assert recall_at_k(["x"], frozenset({"a"}), 5) == 0.0

    def test_recall_respects_k(self) -> None:
        assert recall_at_k(["x", "x2", "x3", "x4", "x5", "a"], frozenset({"a"}), 5) == 0.0

    def test_duplicate_notes_count_once(self) -> None:
        # Several chunks from one note must not inflate recall.
        assert recall_at_k(["a", "a", "a"], frozenset({"a", "b"}), 5) == 0.5

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(["a"], frozenset({"a"})) == 1.0
        assert reciprocal_rank(["x", "a"], frozenset({"a"})) == 0.5
        assert reciprocal_rank(["x"], frozenset({"a"})) == 0.0

    def test_average_precision_rewards_ordering(self) -> None:
        good = average_precision(["a", "b", "x"], frozenset({"a", "b"}))
        bad = average_precision(["x", "a", "b"], frozenset({"a", "b"}))
        assert good > bad

    def test_ndcg_rewards_early_hits(self) -> None:
        early = ndcg_at_k(["a", "x", "y"], frozenset({"a"}), 10)
        late = ndcg_at_k(["x", "y", "a"], frozenset({"a"}), 10)
        assert early > late
        assert early == 1.0

    def test_ndcg_perfect_is_one(self) -> None:
        assert ndcg_at_k(["a", "b"], frozenset({"a", "b"}), 10) == pytest.approx(1.0)

    def test_empty_expected_is_zero_not_an_error(self) -> None:
        assert recall_at_k(["a"], frozenset(), 5) == 0.0
        assert ndcg_at_k(["a"], frozenset(), 5) == 0.0


class TestLoadCases:
    def test_yaml_list(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.yaml"
        path.write_text(
            "- query: what about chunking?\n  expect: [Chunking.md, Retrieval.md]\n",
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert len(cases) == 1
        assert cases[0].expect == frozenset({"Chunking.md", "Retrieval.md"})

    def test_mapping_shorthand(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.yaml"
        path.write_text("chunking question: [A.md]\n", encoding="utf-8")
        assert load_cases(path)[0].query == "chunking question"

    def test_json(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.json"
        path.write_text('[{"query": "q", "expect": ["A.md"]}]', encoding="utf-8")
        assert len(load_cases(path)) == 1

    def test_string_expect_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.yaml"
        path.write_text("- query: q\n  expect: A.md\n", encoding="utf-8")
        assert load_cases(path)[0].expect == frozenset({"A.md"})

    def test_unusable_entries_skipped(self, tmp_path: Path) -> None:
        # One typo must not stop the other cases being measured.
        path = tmp_path / "cases.yaml"
        path.write_text(
            "- query: good\n  expect: [A.md]\n- query: ''\n  expect: [B.md]\n- notaquery: x\n",
            encoding="utf-8",
        )
        assert len(load_cases(path)) == 1


class TestEvaluateAndAblate:
    def _engine(self, tmp_path: Path) -> tuple[RetrievalEngine, list[BenchCase]]:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Sourdough.md").write_text(
            "# Sourdough\n\nFeed the starter twice daily and bulk ferment.\nSee [[Flour]].\n",
            encoding="utf-8",
        )
        (root / "Flour.md").write_text(
            "# Flour\n\nStrong white bread flour with high protein.\n", encoding="utf-8"
        )
        (root / "Retrieval.md").write_text(
            "# Retrieval\n\nHybrid search fuses dense vectors with bm25.\n", encoding="utf-8"
        )
        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )
        pipeline.run()
        engine = RetrievalEngine(
            pipeline.store, pipeline.embedder, graph=pipeline.build_graph(), top_k=5
        )
        cases = [
            BenchCase(query="feed the starter bulk ferment", expect=frozenset({"Sourdough.md"})),
            BenchCase(query="hybrid search dense vectors bm25", expect=frozenset({"Retrieval.md"})),
        ]
        return engine, cases

    def test_evaluate_produces_metrics(self, tmp_path: Path) -> None:
        engine, cases = self._engine(tmp_path)
        report = evaluate(engine, cases)
        assert report.cases == 2
        assert report.recall_at_5 > 0
        assert report.latencies_ms
        assert report.p95_ms >= report.p50_ms

    def test_misses_are_reported(self, tmp_path: Path) -> None:
        engine, _ = self._engine(tmp_path)
        impossible = [BenchCase(query="zzz nothing", expect=frozenset({"Nope.md"}))]
        report = evaluate(engine, impossible)
        assert report.misses == ["zzz nothing"]
        assert report.recall_at_5 == 0.0

    def test_empty_cases(self, tmp_path: Path) -> None:
        engine, _ = self._engine(tmp_path)
        assert evaluate(engine, []).cases == 0

    def test_ablation_covers_configured_components(self, tmp_path: Path) -> None:
        engine, cases = self._engine(tmp_path)
        result = ablate(engine, cases)
        labels = {variant.label for variant in result.variants}
        assert "graph" in labels
        assert "expansion" in labels

    def test_ablation_restores_engine_state(self, tmp_path: Path) -> None:
        """A benchmark must not leave the engine reconfigured."""
        engine, cases = self._engine(tmp_path)
        assert engine.expansion_enabled
        ablate(engine, cases)
        assert engine.expansion_enabled

    def test_deltas_are_signed_so_helpful_is_positive(self, tmp_path: Path) -> None:
        engine, cases = self._engine(tmp_path)
        rows = ablate(engine, cases).deltas()
        assert rows
        for row in rows:
            assert "recall_delta" in row
            assert "p50_saved_ms" in row

    def test_report_row_is_serialisable(self, tmp_path: Path) -> None:
        import json

        engine, cases = self._engine(tmp_path)
        json.dumps(evaluate(engine, cases).as_row())


class TestIndexBenchmark:
    def test_rates(self) -> None:
        timing = IndexBenchmark(notes=100, chunks=400, elapsed_s=2.0, reindex_elapsed_s=0.1)
        assert timing.notes_per_second == 50.0
        assert timing.chunks_per_second == 200.0
        assert timing.speedup == 20.0

    def test_zero_elapsed_is_safe(self) -> None:
        timing = IndexBenchmark(notes=10)
        assert timing.notes_per_second == 0.0
        assert timing.speedup == 0.0

    def test_percentiles_on_single_sample(self) -> None:
        report = QualityReport(label="x", latencies_ms=[42.0])
        assert report.p50_ms == 42.0
        assert report.p95_ms == 42.0
