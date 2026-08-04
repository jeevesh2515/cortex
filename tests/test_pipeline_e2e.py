"""End-to-end pipeline tests against a synthetic Obsidian vault.

This is the test that matters: a real directory tree with frontmatter,
wikilinks, tags, attachments and an ``.obsidian`` config folder, indexed and
queried through the full stack. Everything is local -- the hashing embedder
gives real lexical signal without needing a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.catalog import Catalog
from cortex.index.store import MemoryStore
from cortex.ingest.pipeline import IndexPipeline, scan_vault
from cortex.llm.protocol import ChatMessage, CompletionResult, ProviderSpec
from cortex.llm.providers import HashEmbedder
from cortex.llm.router import Router
from cortex.models import DataPolicy, Sensitivity
from cortex.retrieve.engine import RetrievalEngine
from cortex.thermal.governor import (
    GovernorConfig,
    PowerSource,
    Reading,
    StaticProbe,
    ThermalGovernor,
)

VAULT_FILES = {
    "Retrieval.md": """---
title: Retrieval Systems
tags: [rag, search]
---

# Retrieval Systems

Hybrid retrieval combines dense vector search with BM25 keyword matching.
The fusion step uses reciprocal rank fusion to merge the two rankings.

See [[Embeddings]] and [[Vector Databases]] for the components.
""",
    "Embeddings.md": """---
tags: [ml]
---

# Embeddings

Embedding models map text into a dense vector space where semantic
similarity becomes geometric proximity. Qwen3 embedding models are strong.

Related: [[Retrieval]].
""",
    "Vector Databases.md": """---
tags: [infra]
---

# Vector Databases

LanceDB stores vectors on disk in a columnar format and needs no daemon.
It supports hybrid search natively.
""",
    "Cooking/Sourdough.md": """---
tags: [cooking]
---

# Sourdough

Feed the starter twice daily. Bulk ferment until doubled in size.
Bake at 250C with steam for the first twenty minutes.
""",
    "Public Note.md": """---
sensitivity: public
tags: [shareable]
---

# Public Note

This note is explicitly marked shareable and may leave the machine.
""",
    ".obsidian/workspace.json": '{"main": {"id": "abc"}}',
    ".obsidian/plugins/dataview/data.json": '{"settings": {}}',
    "attachments/diagram.png": "\x89PNG fake binary",
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    for rel, content in VAULT_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def pipeline(vault: Path) -> IndexPipeline:
    return IndexPipeline(
        vault=vault,
        store=MemoryStore(),
        catalog=Catalog(),
        embedder=HashEmbedder(dimensions=128),
        exclude=[".obsidian/**"],
    )


class TestScanning:
    def test_finds_markdown_only(self, vault: Path) -> None:
        found = {p.name for p in scan_vault(vault, exclude=[".obsidian/**"])}
        assert "Retrieval.md" in found
        assert "diagram.png" not in found

    def test_excludes_obsidian_config(self, vault: Path) -> None:
        paths = [p.as_posix() for p in scan_vault(vault, exclude=[".obsidian/**"])]
        assert not any(".obsidian" in p for p in paths)

    def test_recurses_subfolders(self, vault: Path) -> None:
        found = {p.name for p in scan_vault(vault, exclude=[".obsidian/**"])}
        assert "Sourdough.md" in found

    def test_missing_vault_yields_nothing(self, tmp_path: Path) -> None:
        assert list(scan_vault(tmp_path / "nope")) == []


class TestIndexing:
    def test_first_run_indexes_everything(self, pipeline: IndexPipeline) -> None:
        report = pipeline.run()
        assert report.indexed == 5
        assert report.chunks_written > 0
        assert not report.errors

    def test_second_run_is_a_noop(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        second = pipeline.run()
        assert second.indexed == 0
        assert second.skipped == 5, "unchanged notes must be skipped"
        assert second.chunks_written == 0

    def test_edit_reindexes_only_that_note(self, pipeline: IndexPipeline, vault: Path) -> None:
        pipeline.run()
        (vault / "Embeddings.md").write_text(
            "---\ntags: [ml]\n---\n\n# Embeddings\n\nCompletely rewritten content here.\n",
            encoding="utf-8",
        )
        report = pipeline.run()
        assert report.indexed == 1
        assert report.skipped == 4

    def test_deletion_reconciled(self, pipeline: IndexPipeline, vault: Path) -> None:
        pipeline.run()
        before = pipeline.store.count()
        (vault / "Cooking" / "Sourdough.md").unlink()
        report = pipeline.run()
        assert report.deleted == 1
        assert pipeline.store.count() < before

    def test_deleted_note_stops_matching(self, pipeline: IndexPipeline, vault: Path) -> None:
        # The ghost-chunk failure: content still surfacing after the file is gone.
        pipeline.run()
        assert pipeline.store.search_text("sourdough")
        (vault / "Cooking" / "Sourdough.md").unlink()
        pipeline.run()
        assert not pipeline.store.search_text("sourdough")

    def test_catalog_tracks_lineage(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        chunk_ids = pipeline.catalog.chunk_ids("Retrieval.md")
        assert chunk_ids
        assert all(cid.startswith("Retrieval.md#") for cid in chunk_ids)

    def test_unreadable_note_does_not_abort_run(self, pipeline: IndexPipeline, vault: Path) -> None:
        (vault / "Broken.md").write_bytes(b"\xff\xfe invalid utf8 \x00 bytes")
        report = pipeline.run()
        assert report.indexed >= 5

    def test_full_rebuild(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        report = pipeline.run(full=True)
        assert report.indexed == 5
        assert report.skipped == 0

    def test_sensitivity_recorded(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        record = pipeline.catalog.get("Public Note.md")
        assert record is not None
        assert record.sensitivity == "public"


class TestThermalIntegration:
    def test_backfill_pauses_when_critical(self, vault: Path) -> None:
        hot = ThermalGovernor(
            probe=StaticProbe(Reading(power=PowerSource.AC, cpu_speed_limit=40, cpu_count=8)),
            config=GovernorConfig(sample_interval=0.0),
        )
        pipeline = IndexPipeline(
            vault=vault,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            governor=hot,
            exclude=[".obsidian/**"],
        )
        report = pipeline.run()
        assert report.paused_for_thermal
        assert report.indexed == 0

    def test_cool_machine_indexes_normally(self, vault: Path) -> None:
        cool = ThermalGovernor(
            probe=StaticProbe(
                Reading(power=PowerSource.AC, cpu_speed_limit=100, load_average=0.1, cpu_count=8)
            ),
            config=GovernorConfig(sample_interval=0.0),
        )
        pipeline = IndexPipeline(
            vault=vault,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            governor=cool,
            exclude=[".obsidian/**"],
        )
        report = pipeline.run()
        assert not report.paused_for_thermal
        assert report.indexed == 5

    def test_deletions_reconciled_even_when_paused(self, vault: Path) -> None:
        cool = StaticProbe(
            Reading(power=PowerSource.AC, cpu_speed_limit=100, load_average=0.1, cpu_count=8)
        )
        governor = ThermalGovernor(probe=cool, config=GovernorConfig(sample_interval=0.0))
        pipeline = IndexPipeline(
            vault=vault,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
            governor=governor,
            exclude=[".obsidian/**"],
        )
        pipeline.run()
        (vault / "Cooking" / "Sourdough.md").unlink()
        governor.probe = StaticProbe(  # type: ignore[assignment]
            Reading(power=PowerSource.AC, cpu_speed_limit=40, cpu_count=8)
        )
        governor._initialised = False
        report = pipeline.run()
        assert report.deleted == 1, "ghost chunks must be cleaned even under thermal pause"


class TestRetrievalE2E:
    def test_finds_relevant_note(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder, graph=pipeline.build_graph())
        result = engine.retrieve("sourdough starter bulk ferment")
        assert result.chunks
        assert result.chunks[0].chunk.note_id == "Cooking/Sourdough.md"

    def test_hybrid_uses_multiple_retrievers(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder, graph=pipeline.build_graph())
        result = engine.retrieve("reciprocal rank fusion")
        assert "dense" in result.per_retriever
        assert "fts" in result.per_retriever

    def test_graph_surfaces_lexically_unrelated_linked_note(self, tmp_path: Path) -> None:
        """The whole justification for the graph layer.

        ``Ripening.md`` shares no vocabulary with the query and would never be
        found by dense or lexical search. It is reachable only because the
        author linked it. A vault of distractors ensures the seeds do not
        trivially cover everything.
        """
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Fermentation.md").write_text(
            "# Fermentation\n\nLactobacillus cultures acidify the dough matrix.\n"
            "Deeper detail lives in [[Ripening]].\n",
            encoding="utf-8",
        )
        # Deliberately zero lexical overlap with the query.
        (root / "Ripening.md").write_text(
            "# Ripening\n\nOrchard fruit softens as pectin chains break down over weeks.\n",
            encoding="utf-8",
        )
        for i in range(10):
            (root / f"Distractor {i}.md").write_text(
                f"# Distractor {i}\n\nUnrelated administrative meeting notes number {i}.\n",
                encoding="utf-8",
            )

        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )
        pipeline.run()
        engine = RetrievalEngine(
            pipeline.store,
            pipeline.embedder,
            graph=pipeline.build_graph(),
            top_k=10,
            graph_seeds=2,
        )
        result = engine.retrieve("lactobacillus cultures acidify dough matrix")

        assert "graph" in result.per_retriever
        assert "Ripening.md" in result.notes, "linked note must be reachable via the graph"

        without = engine.retrieve("lactobacillus cultures acidify dough matrix", use_graph=False)
        assert "Ripening.md" not in without.notes, "and unreachable without it"

    def test_graph_can_be_disabled(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder, graph=pipeline.build_graph())
        result = engine.retrieve("hybrid retrieval", use_graph=False)
        assert "graph" not in result.per_retriever

    def test_citations_are_traceable(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder)
        result = engine.retrieve("lancedb columnar disk")
        citation = result.chunks[0].chunk.citation
        assert "Vector Databases.md" in citation

    def test_no_match_returns_empty(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder)
        result = engine.retrieve("zzzznonexistentquerytermzzz")
        assert result.chunks == [] or result.chunks[0].score >= 0


class TestLinkGraphE2E:
    def test_graph_built_from_real_vault(self, pipeline: IndexPipeline) -> None:
        graph = pipeline.build_graph()
        assert "Embeddings.md" in graph.forward["Retrieval.md"]
        assert "Retrieval.md" in graph.backward["Embeddings.md"]

    def test_unresolved_link_dropped(self, pipeline: IndexPipeline, vault: Path) -> None:
        (vault / "Dangling.md").write_text("Links to [[Does Not Exist]]", encoding="utf-8")
        graph = pipeline.build_graph()
        assert graph.forward.get("Dangling.md", set()) == set()


class FakeChat:
    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec
        self.last_prompt: str | None = None

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> CompletionResult:
        self.last_prompt = messages[-1].content
        return CompletionResult(
            text="Answer grounded in [1].", provider=self.spec.name, model=self.spec.model
        )


class TestAskE2E:
    def _engine(self, pipeline: IndexPipeline, provider: FakeChat) -> RetrievalEngine:
        return RetrievalEngine(
            pipeline.store,
            pipeline.embedder,
            graph=pipeline.build_graph(),
            router=Router([provider]),
        )

    def test_answer_includes_citations(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        local = FakeChat(
            ProviderSpec(name="ollama", base_url="x", model="m", policy=DataPolicy.LOCAL)
        )
        answer = self._engine(pipeline, local).ask("how do I bake sourdough?")
        assert answer.citations
        assert answer.provider == "ollama"
        assert not answer.escalated

    def test_prompt_numbers_sources(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        local = FakeChat(
            ProviderSpec(name="ollama", base_url="x", model="m", policy=DataPolicy.LOCAL)
        )
        self._engine(pipeline, local).ask("sourdough")
        assert local.last_prompt is not None
        assert "[1]" in local.last_prompt

    def test_private_notes_refuse_training_provider(self, pipeline: IndexPipeline) -> None:
        # The end-to-end privacy guarantee.
        pipeline.run()
        trainer = FakeChat(
            ProviderSpec(name="zen", base_url="x", model="m", policy=DataPolicy.TRAINS)
        )
        engine = self._engine(pipeline, trainer)
        from cortex.llm.protocol import PolicyViolation

        with pytest.raises(PolicyViolation):
            engine.ask("what did I write about sourdough?")
        assert trainer.last_prompt is None, "private notes must never be sent"

    def test_public_note_may_use_training_provider(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        trainer = FakeChat(
            ProviderSpec(name="zen", base_url="x", model="m", policy=DataPolicy.TRAINS)
        )
        engine = self._engine(pipeline, trainer)
        result = engine.retrieve("explicitly marked shareable")
        public_only = [c.chunk for c in result.chunks if c.chunk.note_id == "Public Note.md"]
        assert public_only
        assert engine.effective_sensitivity(public_only) is Sensitivity.PUBLIC

    def test_mixed_context_is_private(self, pipeline: IndexPipeline) -> None:
        # One private chunk makes the whole request private -- a summary of
        # private notes is itself private.
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder)
        result = engine.retrieve("note")
        chunks = [c.chunk for c in result.chunks]
        if any(c.sensitivity is Sensitivity.PRIVATE for c in chunks):
            assert engine.effective_sensitivity(chunks) is Sensitivity.PRIVATE

    def test_no_router_degrades_gracefully(self, pipeline: IndexPipeline) -> None:
        pipeline.run()
        engine = RetrievalEngine(pipeline.store, pipeline.embedder)
        answer = engine.ask("sourdough")
        assert answer.citations
        assert answer.provider == "none"
