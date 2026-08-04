"""Chunker tests.

The invariants that matter: no content is lost, no chunk exceeds the hard
ceiling, breadcrumbs are correct, and ids are content-derived so an edit to one
paragraph does not invalidate its neighbours.
"""

from __future__ import annotations

from cortex.ingest.chunker import ChunkConfig, chunk_note, estimate_tokens
from cortex.ingest.obsidian import parse_text


def _note(body: str, rel_path: str = "test.md") -> object:
    return parse_text(body, rel_path=rel_path)


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_monotonic(self) -> None:
        assert estimate_tokens("a" * 100) < estimate_tokens("a" * 1000)


class TestHeadingStructure:
    def test_splits_on_headings(self) -> None:
        note = _note(
            "# Alpha\n" + "Content about alpha. " * 30 + "\n\n"
            "# Beta\n" + "Content about beta. " * 30
        )
        chunks = chunk_note(note)
        crumbs = {tuple(c.heading_path) for c in chunks}
        assert ("Alpha",) in crumbs
        assert ("Beta",) in crumbs

    def test_nested_breadcrumb(self) -> None:
        note = _note(
            "# Top\n" + "top text. " * 30 + "\n\n"
            "## Middle\n" + "middle text. " * 30 + "\n\n"
            "### Deep\n" + "deep text. " * 30
        )
        chunks = chunk_note(note)
        crumbs = {tuple(c.heading_path) for c in chunks}
        assert ("Top", "Middle", "Deep") in crumbs

    def test_sibling_pops_stack(self) -> None:
        note = _note("# A\n## A1\n" + "a1 body. " * 30 + "\n\n## A2\n" + "a2 body. " * 30)
        crumbs = {tuple(c.heading_path) for c in chunk_note(note)}
        assert ("A", "A1") in crumbs
        assert ("A", "A2") in crumbs

    def test_hash_in_code_fence_is_not_a_heading(self) -> None:
        note = _note(
            "# Real\n" + "body text. " * 30 + "\n\n"
            "```bash\n# not a heading\necho hi\n```\n" + "more text. " * 30
        )
        crumbs = {tuple(c.heading_path) for c in chunk_note(note)}
        assert crumbs == {("Real",)}


class TestSizing:
    def test_respects_max_tokens(self) -> None:
        note = _note("# Long\n\n" + ("This is a sentence with several words in it. " * 400))
        config = ChunkConfig()
        for chunk in chunk_note(note, config):
            assert chunk.token_estimate <= config.max_tokens

    def test_no_headings_still_chunks(self) -> None:
        note = _note("Just a flat wall of prose. " * 200)
        chunks = chunk_note(note)
        assert chunks
        assert all(c.heading_path == [] for c in chunks)

    def test_short_note_yields_one_chunk(self) -> None:
        note = _note("A tiny note.")
        chunks = chunk_note(note)
        assert len(chunks) == 1
        assert chunks[0].text == "A tiny note."

    def test_empty_note_yields_nothing(self) -> None:
        assert chunk_note(_note("")) == []

    def test_frontmatter_only_yields_nothing(self) -> None:
        assert chunk_note(_note("---\ntitle: X\n---\n")) == []

    def test_content_is_preserved(self) -> None:
        body = "# Section\n\n" + ("Distinctive marker phrase alpha. " * 100)
        chunks = chunk_note(_note(body))
        joined = " ".join(c.text for c in chunks)
        assert "Distinctive marker phrase alpha." in joined


class TestChunkIdentity:
    def test_ids_are_content_derived(self) -> None:
        note = _note("# A\n\n" + "stable content. " * 40)
        first = chunk_note(note)
        second = chunk_note(note)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_edit_does_not_renumber_untouched_chunks(self) -> None:
        # The property that makes incremental re-indexing cheap.
        base = "# A\n\n" + "alpha content here. " * 60 + "\n\n# B\n\n" + "beta content here. " * 60
        edited = base.replace("beta content here. ", "beta content CHANGED. ")
        before = {c.chunk_id for c in chunk_note(_note(base))}
        after = {c.chunk_id for c in chunk_note(_note(edited))}
        assert before & after, "unchanged sections should keep their ids"

    def test_custom_id_factory(self) -> None:
        note = _note("# A\n\n" + "content. " * 40)
        chunks = chunk_note(note, id_factory=lambda nid, i: f"{nid}::{i}")
        assert chunks[0].chunk_id == "test.md::0"


class TestMetadataPropagation:
    def test_tags_and_links_flow_to_chunks(self) -> None:
        note = _note(
            "---\ntags: [alpha]\n---\n# H\n\nSee [[Other Note]]. Tagged #beta.\n" + "filler. " * 40
        )
        for chunk in chunk_note(note):
            assert "alpha" in chunk.tags
            assert "beta" in chunk.tags
            assert "other note" in chunk.links

    def test_sensitivity_propagates(self) -> None:
        note = _note("---\npublic: true\n---\n# H\n\n" + "body. " * 40)
        assert all(c.sensitivity.value == "public" for c in chunk_note(note))

    def test_citation_format(self) -> None:
        note = _note("# Top\n\n## Sub\n\n" + "body text. " * 40, rel_path="dir/note.md")
        chunk = next(c for c in chunk_note(note) if c.heading_path == ["Top", "Sub"])
        assert chunk.citation == "dir/note.md > Top > Sub"

    def test_context_header_includes_breadcrumb(self) -> None:
        note = _note("# Top\n\n" + "body text. " * 40, rel_path="n.md")
        chunk = chunk_note(note)[0]
        header = chunk.with_context_header()
        assert header.startswith("[n.md > Top]")
