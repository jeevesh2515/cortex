"""Fusion and graph-expansion tests."""

from __future__ import annotations

from cortex.ingest.obsidian import parse_text
from cortex.models import Chunk, ScoredChunk
from cortex.retrieve.fusion import RRF_K, normalise_scores, reciprocal_rank_fusion
from cortex.retrieve.graph import LinkGraph, expand_by_links


def chunk(cid: str, note: str = "n.md", text: str = "body") -> Chunk:
    return Chunk(chunk_id=cid, note_id=note, text=text)


def scored(cid: str, score: float = 1.0, note: str = "n.md") -> ScoredChunk:
    return ScoredChunk(chunk=chunk(cid, note), score=score)


class TestRRF:
    def test_single_list_preserves_order(self) -> None:
        results = reciprocal_rank_fusion({"dense": [scored("a"), scored("b"), scored("c")]})
        assert [r.chunk.chunk_id for r in results] == ["a", "b", "c"]

    def test_agreement_beats_single_list_top_hit(self) -> None:
        # 'b' is 2nd in both lists; 'a' is 1st in one and absent from the other.
        # Consensus should win -- this is the core RRF behaviour.
        fused = reciprocal_rank_fusion(
            {
                "dense": [scored("a"), scored("b")],
                "fts": [scored("c"), scored("b")],
            }
        )
        assert fused[0].chunk.chunk_id == "b"

    def test_score_matches_formula(self) -> None:
        fused = reciprocal_rank_fusion({"dense": [scored("a")]})
        assert fused[0].score == 1.0 / (RRF_K + 1)

    def test_magnitude_is_ignored(self) -> None:
        # An enormous raw score must not outrank a better position. This is the
        # exact failure of naive weighted averaging.
        fused = reciprocal_rank_fusion(
            {
                "dense": [scored("a", score=0.01), scored("b", score=999999.0)],
            }
        )
        assert fused[0].chunk.chunk_id == "a"

    def test_weights_applied(self) -> None:
        fused = reciprocal_rank_fusion(
            {"dense": [scored("a")], "fts": [scored("b")]},
            weights={"dense": 5.0, "fts": 1.0},
        )
        assert fused[0].chunk.chunk_id == "a"

    def test_zero_weight_excludes_retriever(self) -> None:
        fused = reciprocal_rank_fusion(
            {"dense": [scored("a")], "fts": [scored("b")]},
            weights={"fts": 0.0},
        )
        assert [r.chunk.chunk_id for r in fused] == ["a"]

    def test_components_recorded(self) -> None:
        fused = reciprocal_rank_fusion(
            {"dense": [scored("a")], "fts": [scored("a")]},
        )
        assert set(fused[0].components) == {"dense", "fts"}

    def test_top_k_truncates(self) -> None:
        fused = reciprocal_rank_fusion({"dense": [scored(c) for c in "abcdef"]}, top_k=3)
        assert len(fused) == 3

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"dense": []}) == []

    def test_ranks_are_sequential(self) -> None:
        fused = reciprocal_rank_fusion({"dense": [scored(c) for c in "abc"]})
        assert [r.rank for r in fused] == [1, 2, 3]

    def test_deterministic_tiebreak(self) -> None:
        first = reciprocal_rank_fusion({"a": [scored("x")], "b": [scored("y")]})
        second = reciprocal_rank_fusion({"a": [scored("x")], "b": [scored("y")]})
        assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]


class TestNormalise:
    def test_maps_to_unit_range(self) -> None:
        out = normalise_scores([scored("a", 10.0), scored("b", 5.0), scored("c", 0.0)])
        assert [round(r.score, 3) for r in out] == [1.0, 0.5, 0.0]

    def test_identical_scores(self) -> None:
        out = normalise_scores([scored("a", 3.0), scored("b", 3.0)])
        assert all(r.score == 1.0 for r in out)

    def test_empty(self) -> None:
        assert normalise_scores([]) == []


def build_vault() -> LinkGraph:
    notes = [
        parse_text("Links to [[Beta]] and [[Gamma]].\n#hub", rel_path="Alpha.md"),
        parse_text("Back to [[Alpha]].\n#leaf", rel_path="Beta.md"),
        parse_text("Standalone content.\n#leaf", rel_path="Gamma.md"),
        parse_text("Points at [[Gamma]].", rel_path="Delta.md"),
        parse_text("No links at all.", rel_path="Orphan.md"),
    ]
    return LinkGraph.from_notes(notes)


class TestLinkGraph:
    def test_forward_links_resolved(self) -> None:
        graph = build_vault()
        assert graph.forward["Alpha.md"] == {"Beta.md", "Gamma.md"}

    def test_backlinks_recorded(self) -> None:
        graph = build_vault()
        assert graph.backward["Gamma.md"] == {"Alpha.md", "Delta.md"}

    def test_case_insensitive_resolution(self) -> None:
        notes = [
            parse_text("See [[target note]]", rel_path="Source.md"),
            parse_text("content", rel_path="Target Note.md"),
        ]
        graph = LinkGraph.from_notes(notes)
        assert graph.forward["Source.md"] == {"Target Note.md"}

    def test_md_extension_in_link_resolved(self) -> None:
        notes = [
            parse_text("See [[Target.md]]", rel_path="Source.md"),
            parse_text("content", rel_path="Target.md"),
        ]
        graph = LinkGraph.from_notes(notes)
        assert graph.forward["Source.md"] == {"Target.md"}

    def test_folder_path_in_link_resolved(self) -> None:
        notes = [
            parse_text("See [[folder/Target]]", rel_path="Source.md"),
            parse_text("content", rel_path="folder/Target.md"),
        ]
        graph = LinkGraph.from_notes(notes)
        assert graph.forward["Source.md"] == {"folder/Target.md"}

    def test_unresolved_link_creates_no_phantom_node(self) -> None:
        notes = [parse_text("See [[Does Not Exist]]", rel_path="A.md")]
        graph = LinkGraph.from_notes(notes)
        assert graph.forward.get("A.md", set()) == set()

    def test_self_link_ignored(self) -> None:
        notes = [parse_text("See [[A]]", rel_path="A.md")]
        graph = LinkGraph.from_notes(notes)
        assert graph.forward.get("A.md", set()) == set()

    def test_neighbours_includes_both_directions(self) -> None:
        graph = build_vault()
        assert graph.neighbours("Gamma.md") == {"Alpha.md", "Delta.md"}

    def test_neighbours_forward_only(self) -> None:
        graph = build_vault()
        assert graph.neighbours("Gamma.md", include_backlinks=False) == set()

    def test_expand_one_hop(self) -> None:
        graph = build_vault()
        found = graph.expand(["Alpha.md"], hops=1)
        assert set(found) == {"Beta.md", "Gamma.md"}
        assert all(d == 1 for d in found.values())

    def test_expand_two_hops_reaches_further(self) -> None:
        graph = build_vault()
        found = graph.expand(["Beta.md"], hops=2)
        assert found["Alpha.md"] == 1
        assert found["Gamma.md"] == 2

    def test_expand_respects_limit(self) -> None:
        graph = build_vault()
        assert len(graph.expand(["Alpha.md"], hops=2, limit=1)) == 1

    def test_orphan_expands_to_nothing(self) -> None:
        graph = build_vault()
        assert graph.expand(["Orphan.md"], hops=2) == {}

    def test_tags_indexed(self) -> None:
        graph = build_vault()
        assert graph.tags["leaf"] == {"Beta.md", "Gamma.md"}

    def test_stats(self) -> None:
        # Alpha->Beta, Alpha->Gamma, Beta->Alpha, Delta->Gamma
        stats = build_vault().stats
        assert stats["edges"] == 4
        assert stats["linked_notes"] == 4  # Orphan is excluded
        assert stats["tags"] >= 2


class TestExpandByLinks:
    def test_surfaces_linked_notes(self) -> None:
        graph = build_vault()
        chunks_by_note = {
            "Beta.md": [chunk("beta-1", "Beta.md")],
            "Gamma.md": [chunk("gamma-1", "Gamma.md")],
        }
        results = expand_by_links([scored("alpha-1", note="Alpha.md")], graph, chunks_by_note)
        assert {r.chunk.note_id for r in results} == {"Beta.md", "Gamma.md"}
        assert all(r.source == "graph" for r in results)

    def test_score_decays_with_distance(self) -> None:
        graph = build_vault()
        chunks_by_note = {
            "Alpha.md": [chunk("alpha-1", "Alpha.md")],
            "Gamma.md": [chunk("gamma-1", "Gamma.md")],
        }
        results = expand_by_links([scored("beta-1", note="Beta.md")], graph, chunks_by_note, hops=2)
        by_note = {r.chunk.note_id: r.score for r in results}
        assert by_note["Alpha.md"] > by_note["Gamma.md"]

    def test_seed_note_excluded(self) -> None:
        graph = build_vault()
        chunks_by_note = {"Alpha.md": [chunk("alpha-1", "Alpha.md")]}
        results = expand_by_links([scored("alpha-1", note="Alpha.md")], graph, chunks_by_note)
        assert all(r.chunk.note_id != "Alpha.md" for r in results)

    def test_caps_chunks_per_neighbour(self) -> None:
        graph = build_vault()
        chunks_by_note = {
            "Beta.md": [chunk(f"beta-{i}", "Beta.md") for i in range(10)],
        }
        results = expand_by_links([scored("alpha-1", note="Alpha.md")], graph, chunks_by_note)
        assert len([r for r in results if r.chunk.note_id == "Beta.md"]) == 2

    def test_empty_seeds(self) -> None:
        assert expand_by_links([], build_vault(), {}) == []

    def test_feeds_rrf_cleanly(self) -> None:
        # The integration that matters: graph output is a ranking RRF can fuse.
        graph = build_vault()
        chunks_by_note = {"Beta.md": [chunk("beta-1", "Beta.md")]}
        graph_results = expand_by_links([scored("alpha-1", note="Alpha.md")], graph, chunks_by_note)
        fused = reciprocal_rank_fusion(
            {"dense": [scored("alpha-1", note="Alpha.md")], "graph": graph_results}
        )
        assert {r.chunk.chunk_id for r in fused} == {"alpha-1", "beta-1"}
