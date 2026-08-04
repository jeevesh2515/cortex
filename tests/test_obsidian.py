"""Parser tests.

Weighted towards the cases that silently corrupt a link graph rather than the
happy path: code fences, heading-vs-tag ambiguity, and malformed frontmatter.
"""

from __future__ import annotations

from cortex.ingest.obsidian import (
    extract_dataview_fields,
    extract_links,
    extract_tags,
    parse_text,
    split_frontmatter,
)
from cortex.models import Sensitivity


class TestFrontmatter:
    def test_parses_yaml_block(self) -> None:
        fm, body = split_frontmatter("---\ntitle: Hello\ntags: [a, b]\n---\nBody here")
        assert fm == {"title": "Hello", "tags": ["a", "b"]}
        assert body == "Body here"

    def test_absent_frontmatter_returns_original(self) -> None:
        fm, body = split_frontmatter("# Just a heading\n")
        assert fm == {}
        assert body == "# Just a heading\n"

    def test_malformed_yaml_is_tolerated(self) -> None:
        # A single bad note must never abort a full vault scan.
        fm, body = split_frontmatter("---\n: : not: valid: yaml\n---\nBody")
        assert fm == {}
        assert body == "Body"

    def test_non_dict_frontmatter_ignored(self) -> None:
        fm, body = split_frontmatter("---\n- just\n- a\n- list\n---\nBody")
        assert fm == {}
        assert body == "Body"

    def test_horizontal_rule_is_not_frontmatter(self) -> None:
        text = "Some text\n\n---\n\nMore text"
        fm, body = split_frontmatter(text)
        assert fm == {}
        assert body == text


class TestWikiLinks:
    def test_plain_link(self) -> None:
        (link,) = extract_links("See [[Some Note]] for detail.")
        assert link.target == "Some Note"
        assert link.alias is None
        assert link.heading is None
        assert not link.is_embed
        assert link.display == "Some Note"

    def test_alias_and_heading(self) -> None:
        (link,) = extract_links("[[Target Note#Section Two|nice label]]")
        assert link.target == "Target Note"
        assert link.heading == "Section Two"
        assert link.alias == "nice label"
        assert link.display == "nice label"

    def test_embed_detected(self) -> None:
        (link,) = extract_links("![[Diagram.png]]")
        assert link.is_embed
        assert link.target == "Diagram.png"

    def test_links_inside_fenced_code_ignored(self) -> None:
        text = """Real [[Alpha]] link.

```python
# [[FakeLink]] should not count
x = "[[AlsoFake]]"
```

Another [[Beta]] link.
"""
        targets = [link.target for link in extract_links(text)]
        assert targets == ["Alpha", "Beta"]

    def test_links_inside_inline_code_ignored(self) -> None:
        targets = [link.target for link in extract_links("Use `[[literal]]` but [[Real]] counts")]
        assert targets == ["Real"]

    def test_tilde_fence_masked(self) -> None:
        text = "~~~\n[[Hidden]]\n~~~\n[[Shown]]"
        assert [link.target for link in extract_links(text)] == ["Shown"]

    def test_unterminated_fence_masks_to_end(self) -> None:
        text = "[[Before]]\n```\n[[Inside]]\n"
        assert [link.target for link in extract_links(text)] == ["Before"]

    def test_empty_target_skipped(self) -> None:
        assert extract_links("[[]] and [[   ]]") == []


class TestTags:
    def test_simple_tag(self) -> None:
        assert "project" in extract_tags("Working on #project today")

    def test_nested_tag_yields_ancestors(self) -> None:
        tags = extract_tags("#area/work/reports")
        assert tags == {"area", "area/work", "area/work/reports"}

    def test_heading_is_not_a_tag(self) -> None:
        # The single most common false positive.
        assert extract_tags("# Heading One\n## Heading Two") == set()

    def test_url_fragment_is_not_a_tag(self) -> None:
        assert extract_tags("See https://example.com/page#section for info") == set()

    def test_pure_numeric_is_not_a_tag(self) -> None:
        assert extract_tags("Issue #123 was closed") == set()

    def test_alphanumeric_tag_allowed(self) -> None:
        assert "v2rollout" in extract_tags("Shipping #v2rollout now")

    def test_tags_in_code_ignored(self) -> None:
        text = "```bash\ncurl host/#anchor\n# comment\n```\nReal #tag here"
        assert extract_tags(text) == {"tag"}


class TestDataview:
    def test_inline_fields(self) -> None:
        fields = extract_dataview_fields("status:: active\npriority:: high")
        assert fields == {"status": "active", "priority": "high"}

    def test_list_item_field(self) -> None:
        assert extract_dataview_fields("- due:: 2026-08-01") == {"due": "2026-08-01"}


class TestParseText:
    def test_full_note(self) -> None:
        text = """---
title: Retrieval Notes
tags:
  - rag
  - ml/embeddings
---

# Overview

Linked to [[Vector Databases]] and [[Chunking#Strategy|chunking]].

Tagged #experiment inline.
"""
        note = parse_text(text, rel_path="notes/retrieval.md")
        assert note.title == "Retrieval Notes"
        assert note.rel_path == "notes/retrieval.md"
        assert note.outgoing == {"vector databases", "chunking"}
        assert {"rag", "ml", "ml/embeddings", "experiment"} <= note.tags
        assert note.sensitivity is Sensitivity.PRIVATE
        assert note.content_hash

    def test_title_falls_back_to_filename(self) -> None:
        note = parse_text("no frontmatter", rel_path="daily/2026-08-04.md")
        assert note.title == "2026-08-04"

    def test_sensitivity_opt_in(self) -> None:
        note = parse_text("---\nsensitivity: public\n---\nx", rel_path="a.md")
        assert note.sensitivity is Sensitivity.PUBLIC

    def test_sensitivity_boolean_opt_in(self) -> None:
        note = parse_text("---\npublic: true\n---\nx", rel_path="a.md")
        assert note.sensitivity is Sensitivity.PUBLIC

    def test_ambiguous_value_stays_private(self) -> None:
        # "maybe" must not be read as an opt-in.
        note = parse_text("---\nsensitivity: maybe\n---\nx", rel_path="a.md")
        assert note.sensitivity is Sensitivity.PRIVATE

    def test_hash_is_stable_and_content_sensitive(self) -> None:
        a = parse_text("same", rel_path="a.md")
        b = parse_text("same", rel_path="b.md")
        c = parse_text("different", rel_path="a.md")
        assert a.content_hash == b.content_hash
        assert a.content_hash != c.content_hash
