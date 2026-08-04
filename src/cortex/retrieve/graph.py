"""Wikilink graph expansion.

The case against GraphRAG for a personal knowledge base is strong: a controlled
2026 evaluation found it *loses* to flat hybrid retrieval on multi-hop prose
queries, and extracting a graph with an LLM costs roughly $34 per thousand
chunks -- which extrapolates to four figures and several days for a real vault.

None of that applies here, because an Obsidian vault already contains a graph.
The author built it by hand, one ``[[wikilink]]`` at a time. It is cleaner than
anything an extractor would infer, it costs nothing, and it updates the instant
a note is saved.

So this module does no extraction. It resolves the links that are already there
and uses them as a third retrieval signal: when a note scores well, the notes
its author deliberately connected to it are probably relevant too.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from cortex.models import Chunk, Note, ScoredChunk

__all__ = ["LinkGraph", "expand_by_links"]


def _normalise(name: str) -> str:
    """Canonical key for link matching.

    Obsidian resolves ``[[Some Note]]`` case-insensitively against the note's
    basename, ignoring any ``.md`` extension and any folder path the user
    happened to include.
    """
    cleaned = name.strip().replace("\\", "/")
    stem = PurePosixPath(cleaned).name
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    return stem.lower()


@dataclass(slots=True)
class LinkGraph:
    """Bidirectional index of the vault's wikilink structure.

    Backlinks matter as much as forward links: in a Zettelkasten the notes that
    *point at* a hub are often better answers than the hub itself.
    """

    forward: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    backward: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    embeds: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """Transclusion edges (``![[note]]``), tracked separately from ordinary
    links. An embed is a materially stronger claim: the author is saying this
    content *is part of* the note, not merely that it is worth a look. Treating
    the two identically discards that signal."""
    _alias: dict[str, str] = field(default_factory=dict)
    """Maps a normalised link target to the note_id that resolves it."""

    tags: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    """tag -> note_ids carrying it."""

    @classmethod
    def from_notes(cls, notes: Iterable[Note]) -> LinkGraph:
        graph = cls()
        notes = list(notes)

        # Pass 1: register every note under its resolvable names.
        for note in notes:
            graph._alias[_normalise(note.rel_path)] = note.note_id
            graph._alias.setdefault(_normalise(note.title), note.note_id)

        # Pass 2: resolve links. Unresolvable targets are dropped rather than
        # creating phantom nodes -- Obsidian calls these "unresolved links" and
        # they carry no retrievable content.
        for note in notes:
            for tag in note.tags:
                graph.tags[tag].add(note.note_id)
            for link in note.links:
                target = graph._alias.get(_normalise(link.target))
                if target is None or target == note.note_id:
                    continue
                graph.forward[note.note_id].add(target)
                graph.backward[target].add(note.note_id)
                if link.is_embed:
                    graph.embeds[note.note_id].add(target)
        return graph

    def neighbours(self, note_id: str, *, include_backlinks: bool = True) -> set[str]:
        out = set(self.forward.get(note_id, set()))
        if include_backlinks:
            out |= self.backward.get(note_id, set())
        return out

    def expand(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 1,
        include_backlinks: bool = True,
        limit: int = 50,
    ) -> dict[str, int]:
        """Breadth-first expansion from seed notes.

        Returns ``{note_id: hop_distance}`` excluding the seeds themselves.
        Depth is capped hard: link graphs in mature vaults are small-world, so
        two hops from a hub can reach most of the vault and stop being a signal.
        """
        seeds = list(seeds)
        seen: dict[str, int] = dict.fromkeys(seeds, 0)
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        found: dict[str, int] = {}

        while queue and len(found) < limit:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbour in sorted(self.neighbours(node, include_backlinks=include_backlinks)):
                if neighbour in seen:
                    continue
                seen[neighbour] = depth + 1
                found[neighbour] = depth + 1
                queue.append((neighbour, depth + 1))
                if len(found) >= limit:
                    break
        return found

    def shared_tags(self, note_id: str, *, note_tags: Mapping[str, set[str]]) -> set[str]:
        """Notes sharing at least one tag with the given note."""
        related: set[str] = set()
        for tag in note_tags.get(note_id, set()):
            related |= self.tags.get(tag, set())
        related.discard(note_id)
        return related

    @property
    def stats(self) -> dict[str, int]:
        edges = sum(len(targets) for targets in self.forward.values())
        return {
            "notes": len(self._alias),
            "linked_notes": len(set(self.forward) | set(self.backward)),
            "edges": edges,
            "embeds": sum(len(targets) for targets in self.embeds.values()),
            "tags": len(self.tags),
        }


def expand_by_links(
    seeds: Sequence[ScoredChunk],
    graph: LinkGraph,
    chunks_by_note: Mapping[str, Sequence[Chunk]],
    *,
    hops: int = 1,
    limit: int = 20,
    decay: float = 0.5,
    embed_boost: float = 1.6,
) -> list[ScoredChunk]:
    """Produce a graph-derived ranking from already-retrieved seeds.

    Score decays with hop distance so a directly-linked note outranks one two
    hops away. Output is a *ranking*, not a set of absolute scores -- it feeds
    RRF, which only reads position.
    """
    if not seeds:
        return []

    seed_notes = [s.chunk.note_id for s in seeds]
    seed_set = set(seed_notes)
    expanded = graph.expand(seed_notes, hops=hops, limit=limit)

    # Transclusions reachable in one hop from any seed. Only direct embeds earn
    # the boost -- an embed two hops away is not evidence about *this* query.
    boosted: set[str] = set()
    for seed in seed_set:
        boosted |= graph.embeds.get(seed, set())

    scored: list[ScoredChunk] = []
    for note_id, distance in expanded.items():
        if note_id in seed_set:
            continue
        weight = decay**distance
        if note_id in boosted:
            weight *= embed_boost
        for chunk in chunks_by_note.get(note_id, [])[:2]:
            # Cap at two chunks per neighbour so one long note cannot flood the
            # candidate pool.
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=weight,
                    source="graph",
                    components={
                        "hop_distance": float(distance),
                        "embed": 1.0 if note_id in boosted else 0.0,
                    },
                )
            )

    scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
    for position, item in enumerate(scored, start=1):
        item.rank = position
    return scored[:limit]
