"""Memory tests.

This is the only module that writes into the user's vault, so the tests are
weighted towards the safety invariants rather than the happy path: containment,
no-overwrite, atomicity, and the feedback-loop cap that stops memory notes
displacing the notes they were derived from.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cortex.ingest.obsidian import parse_text
from cortex.memory import (
    MEMORY_TAG,
    MemoryNote,
    MemoryWriter,
    cap_memory_results,
    is_memory_chunk,
    slugify,
)
from cortex.models import Answer, Chunk, ScoredChunk, Sensitivity

WHEN = datetime(2026, 8, 4, 14, 30)


def note(**kw: object) -> MemoryNote:
    base: dict[str, object] = {
        "question": "How do I feed my sourdough starter?",
        "answer": "Twice daily with equal parts flour and water [1].",
        "sources": ["Cooking/Sourdough.md"],
        "created": WHEN,
    }
    base.update(kw)
    return MemoryNote(**base)  # type: ignore[arg-type]


def scored(cid: str, *, memory: bool = False) -> ScoredChunk:
    tags = {MEMORY_TAG} if memory else set()
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, note_id=f"{cid}.md", text="body", tags=tags),
        score=1.0,
    )


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("How do I feed my starter") == "How-do-I-feed-my-starter"

    def test_strips_punctuation(self) -> None:
        assert slugify("What's this?! (really)") == "Whats-this-really"

    def test_removes_path_separators(self) -> None:
        # A title must never be able to contribute a path component.
        assert "/" not in slugify("a/b/c")
        assert "\\" not in slugify("a\\b")

    def test_removes_traversal(self) -> None:
        result = slugify("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_truncates(self) -> None:
        assert len(slugify("x" * 200, max_length=20)) <= 20

    def test_empty_falls_back(self) -> None:
        assert slugify("") == "untitled"
        assert slugify("!!!") == "untitled"


class TestMemoryNoteRendering:
    def test_frontmatter_is_valid_yaml(self) -> None:
        import yaml

        body = note().render()
        _, _, rest = body.partition("---\n")
        front, _, _ = rest.partition("\n---")
        parsed = yaml.safe_load(front)
        assert parsed["cortex_memory"] is True
        assert MEMORY_TAG in parsed["tags"]

    def test_always_private(self) -> None:
        # A summary of private notes is itself private, never inherited public.
        assert "sensitivity: private" in note().render()

    def test_sources_become_wikilinks(self) -> None:
        body = note(sources=["Cooking/Sourdough.md", "Baking.md"]).render()
        assert "[[Sourdough]]" in body
        assert "[[Baking]]" in body

    def test_duplicate_sources_deduped(self) -> None:
        body = note(sources=["A.md", "A.md", "B.md"]).render()
        assert body.count("[[A]]") == 1

    def test_no_sources_omits_section(self) -> None:
        assert "Drawn from" not in note(sources=[]).render()

    def test_question_and_answer_present(self) -> None:
        body = note().render()
        assert "sourdough starter" in body
        assert "equal parts flour" in body

    def test_quotes_in_title_do_not_break_frontmatter(self) -> None:
        import yaml

        body = note(question='What is a "starter"?').render()
        _, _, rest = body.partition("---\n")
        front, _, _ = rest.partition("\n---")
        assert yaml.safe_load(front)["title"]

    def test_round_trips_through_the_parser(self) -> None:
        """The real integration: a memory note must parse as an ordinary note."""
        body = note().render()
        parsed = parse_text(body, rel_path="Memory/2026-08-04 1430 test.md")
        assert MEMORY_TAG in parsed.tags
        assert parsed.sensitivity is Sensitivity.PRIVATE
        assert "sourdough" in parsed.outgoing
        assert parsed.note_date is not None

    def test_filename_is_sortable_and_safe(self) -> None:
        name = note().filename()
        assert name.startswith("2026-08-04 1430 ")
        assert name.endswith(".md")
        assert "/" not in name


class TestWriterSafety:
    def test_writes_into_the_memory_folder(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path)
        written = writer.write(note())
        assert written is not None
        assert written.parent == tmp_path / "Memory"
        assert written.exists()

    def test_never_overwrites(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path)
        first = writer.write(note())
        second = writer.write(note())
        assert first is not None and second is not None
        assert first != second, "a collision must not clobber the earlier note"
        assert first.exists() and second.exists()
        assert "(2)" in second.name

    def test_malicious_title_stays_contained(self, tmp_path: Path) -> None:
        """The invariant that matters most: no escaping the memory folder."""
        writer = MemoryWriter(tmp_path)
        written = writer.write(note(question="../../../../etc/passwd"))
        assert written is not None
        assert written.parent.resolve() == (tmp_path / "Memory").resolve()
        assert not (tmp_path.parent / "passwd").exists()

    def test_leaves_no_temp_files_behind(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path)
        writer.write(note())
        leftovers = list((tmp_path / "Memory").glob(".cortex-tmp-*"))
        assert leftovers == []

    def test_disabled_writes_nothing(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path, enabled=False)
        assert writer.write(note()) is None
        assert not (tmp_path / "Memory").exists()

    def test_empty_answer_is_skipped(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path)
        assert writer.write(note(answer="   ")) is None

    def test_custom_folder(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path, folder="Brain/Sessions")
        written = writer.write(note())
        assert written is not None
        assert written.parent == tmp_path / "Brain" / "Sessions"

    def test_folder_slashes_normalised(self, tmp_path: Path) -> None:
        assert MemoryWriter(tmp_path, folder="/Memory/").root == tmp_path / "Memory"

    def test_count(self, tmp_path: Path) -> None:
        writer = MemoryWriter(tmp_path)
        assert writer.count() == 0
        writer.write(note())
        writer.write(note(question="another thing"))
        assert writer.count() == 2

    def test_does_not_touch_existing_notes(self, tmp_path: Path) -> None:
        existing = tmp_path / "Important.md"
        existing.write_text("# Do not lose me\n", encoding="utf-8")
        MemoryWriter(tmp_path).write(note())
        assert existing.read_text(encoding="utf-8") == "# Do not lose me\n"

    def test_from_answer(self, tmp_path: Path) -> None:
        answer = Answer(
            question="what did I decide about chunking?",
            text="512 tokens with 10% overlap.",
            citations=[Chunk(chunk_id="c1", note_id="Chunking.md", text="...")],
            provider="groq",
        )
        written = MemoryWriter(tmp_path).from_answer(answer)
        assert written is not None
        body = written.read_text(encoding="utf-8")
        assert "512 tokens" in body
        assert "[[Chunking]]" in body
        assert "cortex_provider: groq" in body


class TestFeedbackLoopCap:
    """Memory summarises primary notes, so it competes with its own sources."""

    def test_caps_memory_results(self) -> None:
        results = [
            scored("m1", memory=True),
            scored("m2", memory=True),
            scored("m3", memory=True),
            scored("primary"),
        ]
        kept = cap_memory_results(results, limit=1)
        assert [r.chunk.chunk_id for r in kept] == ["m1", "primary"]

    def test_primary_sources_are_never_dropped(self) -> None:
        results = [scored(f"p{i}") for i in range(10)]
        assert len(cap_memory_results(results, limit=0)) == 10

    def test_zero_limit_excludes_all_memory(self) -> None:
        results = [scored("m1", memory=True), scored("p1")]
        kept = cap_memory_results(results, limit=0)
        assert [r.chunk.chunk_id for r in kept] == ["p1"]

    def test_negative_limit_disables_the_cap(self) -> None:
        results = [scored(f"m{i}", memory=True) for i in range(5)]
        assert len(cap_memory_results(results, limit=-1)) == 5

    def test_order_preserved(self) -> None:
        results = [scored("a"), scored("m", memory=True), scored("b")]
        kept = cap_memory_results(results, limit=5)
        assert [r.chunk.chunk_id for r in kept] == ["a", "m", "b"]

    def test_identifies_memory_chunks(self) -> None:
        assert is_memory_chunk(Chunk(chunk_id="x", note_id="y", text="z", tags={MEMORY_TAG}))
        assert not is_memory_chunk(Chunk(chunk_id="x", note_id="y", text="z"))


class TestMemoryInRetrieval:
    """End to end: a written memory note is indexed, retrievable and capped."""

    def _pipeline(self, root: Path):  # type: ignore[no-untyped-def]
        from cortex.catalog import Catalog
        from cortex.index.store import MemoryStore
        from cortex.ingest.pipeline import IndexPipeline
        from cortex.llm.providers import HashEmbedder

        return IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )

    def test_memory_note_becomes_searchable(self, tmp_path: Path) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Sourdough.md").write_text(
            "# Sourdough\n\nFeed the starter twice daily.\n", encoding="utf-8"
        )
        MemoryWriter(root).write(
            note(
                question="starter feeding schedule",
                answer="Distinctive conclusion: feed at dawn and dusk.",
            )
        )
        pipeline = self._pipeline(root)
        report = pipeline.run()
        assert report.indexed == 2, "the memory note must be indexed like any other"

        hits = pipeline.store.search_text("dawn and dusk")
        assert hits
        assert is_memory_chunk(hits[0].chunk)

    def test_memory_links_join_the_graph(self, tmp_path: Path) -> None:
        """The elegant part: memory becomes a hub in the real link graph."""
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Sourdough.md").write_text("# Sourdough\n\nbody\n", encoding="utf-8")
        MemoryWriter(root).write(note(sources=["Sourdough.md"]))

        pipeline = self._pipeline(root)
        pipeline.run()
        graph = pipeline.build_graph()

        backlinks = graph.backward.get("Sourdough.md", set())
        assert any("Memory/" in source for source in backlinks)

    def test_cap_applies_through_the_engine(self, tmp_path: Path) -> None:
        from cortex.retrieve.engine import RetrievalEngine

        root = tmp_path / "vault"
        root.mkdir()
        (root / "Topic.md").write_text(
            "# Topic\n\nShared vocabulary about widgets and gadgets.\n", encoding="utf-8"
        )
        writer = MemoryWriter(root)
        for i in range(5):
            writer.write(
                MemoryNote(
                    question=f"widgets and gadgets question {i}",
                    answer="Shared vocabulary about widgets and gadgets.",
                    created=datetime(2026, 8, 4, 10, i),
                )
            )
        pipeline = self._pipeline(root)
        pipeline.run()

        engine = RetrievalEngine(pipeline.store, pipeline.embedder, top_k=10, memory_max_results=2)
        result = engine.retrieve("widgets and gadgets")
        memory_hits = [c for c in result.chunks if is_memory_chunk(c.chunk)]
        assert len(memory_hits) <= 2, "memory must not crowd out primary sources"
        assert any(not is_memory_chunk(c.chunk) for c in result.chunks)


class TestMcpRememberTool:
    def test_writes_and_reports(self, tmp_path: Path) -> None:
        from cortex.config import Settings
        from cortex.mcp_server import CortexTools
        from cortex.runtime import build_runtime

        root = tmp_path / "vault"
        root.mkdir()
        (root / "A.md").write_text("# A\n\nbody\n", encoding="utf-8")

        settings = Settings()
        settings.vault_path = root
        settings.data_dir = tmp_path / "data"
        settings.embed_dimensions = 128
        rt = build_runtime(settings=settings, offline=True, in_memory=True)
        tools = CortexTools(rt)

        out = tools.remember(
            question="what did I decide?",
            answer="Use LanceDB.",
            sources=["A.md"],
        )
        assert "saved" in out
        assert out["saved"].startswith("Memory/")
        assert out["obsidian_uri"].startswith("obsidian://")
        assert out["total_memory_notes"] == 1
        assert (root / out["saved"]).exists()

    def test_reports_when_disabled(self, tmp_path: Path) -> None:
        from cortex.config import Settings
        from cortex.mcp_server import CortexTools
        from cortex.runtime import build_runtime

        root = tmp_path / "vault"
        root.mkdir()
        settings = Settings()
        settings.vault_path = root
        settings.data_dir = tmp_path / "data"
        settings.memory_enabled = False
        rt = build_runtime(settings=settings, offline=True, in_memory=True)
        out = CortexTools(rt).remember(question="q", answer="a")
        assert out["error"] == "memory_disabled"

    def test_registered_in_tool_definitions(self) -> None:
        from cortex.mcp_server import TOOL_DEFINITIONS

        names = {spec["name"] for spec in TOOL_DEFINITIONS}
        assert "remember" in names


class TestEmbedWeighting:
    """Transclusions are a stronger claim than ordinary links."""

    def _graph(self):  # type: ignore[no-untyped-def]
        from cortex.retrieve.graph import LinkGraph

        return LinkGraph.from_notes(
            [
                parse_text(
                    "Ordinary [[Reference]] and transcluded ![[Included]].", rel_path="Hub.md"
                ),
                parse_text("reference body", rel_path="Reference.md"),
                parse_text("included body", rel_path="Included.md"),
            ]
        )

    def test_embeds_tracked_separately(self) -> None:
        graph = self._graph()
        assert graph.embeds["Hub.md"] == {"Included.md"}
        assert graph.forward["Hub.md"] == {"Included.md", "Reference.md"}

    def test_transclusion_outranks_ordinary_link(self) -> None:
        from cortex.retrieve.graph import expand_by_links

        graph = self._graph()
        chunks_by_note = {
            n: [Chunk(chunk_id=f"{n}#1", note_id=n, text="body")]
            for n in ("Reference.md", "Included.md")
        }
        results = expand_by_links(
            [ScoredChunk(chunk=Chunk(chunk_id="h", note_id="Hub.md", text="x"), score=1.0)],
            graph,
            chunks_by_note,
        )
        by_note = {r.chunk.note_id: r.score for r in results}
        assert by_note["Included.md"] > by_note["Reference.md"]

    def test_embed_flag_exposed_in_components(self) -> None:
        from cortex.retrieve.graph import expand_by_links

        graph = self._graph()
        chunks_by_note = {
            n: [Chunk(chunk_id=f"{n}#1", note_id=n, text="body")]
            for n in ("Reference.md", "Included.md")
        }
        results = expand_by_links(
            [ScoredChunk(chunk=Chunk(chunk_id="h", note_id="Hub.md", text="x"), score=1.0)],
            graph,
            chunks_by_note,
        )
        flags = {r.chunk.note_id: r.components["embed"] for r in results}
        assert flags["Included.md"] == 1.0
        assert flags["Reference.md"] == 0.0

    def test_stats_report_embeds(self) -> None:
        assert self._graph().stats["embeds"] == 1


@pytest.mark.parametrize("folder", ["Memory", "Brain", "cortex/sessions"])
def test_any_folder_stays_contained(tmp_path: Path, folder: str) -> None:
    writer = MemoryWriter(tmp_path, folder=folder)
    written = writer.write(note(question="../escape attempt"))
    assert written is not None
    assert written.resolve().is_relative_to(tmp_path.resolve())
