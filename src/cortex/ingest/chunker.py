"""Structure-aware chunking.

Empirical finding from the 2026 literature that drives this design: pure
semantic chunking *underperforms* plain recursive splitting on real prose,
because it produces fragments averaging ~43 tokens that cannot answer anything.
Recursive splitting at ~512 tokens with 10-20% overlap remains the baseline to
beat.

For Obsidian specifically we can do better than blind recursion, because the
author already imposed structure with headings. We split on the heading
hierarchy first and only recurse inside a section that is genuinely too long.
Every chunk keeps its heading breadcrumb, which is what makes citations
readable and gives the embedding model context it would otherwise lack.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from cortex.models import Chunk, Note, content_hash

__all__ = ["ChunkConfig", "chunk_note", "estimate_tokens"]

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^(?P<fence>```+|~~~+)")

# Split points in descending order of semantic strength. We prefer to break at
# a paragraph boundary, then a line, then a sentence, and only mid-sentence as
# a last resort.
_SPLIT_PATTERNS = ["\n\n", "\n", ". ", " "]


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer dependency.

    ~3.6 chars/token is a good fit for English prose and markdown across the
    BPE tokenizers used by the Qwen3 and BGE families. This only needs to be
    accurate enough to pick split points; it is never used for billing or for
    sizing a model's context window, both of which use the real tokenizer.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 3.6))


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunking parameters.

    Defaults follow the 512-token / 10% overlap consensus. ``min_tokens`` exists
    to prevent the micro-fragment failure mode: a 20-token chunk is noise in the
    index and dilutes retrieval.
    """

    target_tokens: int = 512
    overlap_ratio: float = 0.10
    min_tokens: int = 64
    max_tokens: int = 900
    """Hard ceiling. A section longer than this is recursively split even if it
    is semantically coherent, so we never emit a chunk that blows the embedder's
    window."""

    @property
    def overlap_tokens(self) -> int:
        return int(self.target_tokens * self.overlap_ratio)


@dataclass(slots=True)
class _Section:
    """A run of body text under a heading breadcrumb."""

    heading_path: list[str]
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _iter_sections(body: str) -> Iterator[_Section]:
    """Walk the note, yielding one section per heading.

    Fenced code is tracked so that a ``#`` inside a shell snippet is never
    mistaken for a heading -- the same masking concern as link extraction, but
    it has to be line-aware here because we are preserving content.
    """
    current = _Section(heading_path=[], lines=[])
    stack: list[tuple[int, str]] = []
    fence: str | None = None

    for line in body.splitlines():
        fence_match = _FENCE_RE.match(line.strip())
        if fence_match:
            token = fence_match.group("fence")
            if fence is None:
                fence = token
            elif line.strip().startswith(fence):
                fence = None
            current.lines.append(line)
            continue

        if fence is not None:
            current.lines.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if not heading:
            current.lines.append(line)
            continue

        if current.text:
            yield current

        level = len(heading.group("hashes"))
        title = heading.group("title").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current = _Section(heading_path=[t for _, t in stack], lines=[])

    if current.text:
        yield current


def _split_recursive(text: str, config: ChunkConfig) -> list[str]:
    """Split text to fit ``target_tokens``, preferring strong boundaries."""
    if estimate_tokens(text) <= config.max_tokens:
        return [text]

    for pattern in _SPLIT_PATTERNS:
        if pattern not in text:
            continue
        parts = text.split(pattern)
        if len(parts) < 2:
            continue

        out: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for part in parts:
            part_tokens = estimate_tokens(part)
            if buf and buf_tokens + part_tokens > config.target_tokens:
                out.append(pattern.join(buf).strip())
                # Carry overlap from the tail of the emitted chunk.
                overlap: list[str] = []
                acc = 0
                for prev in reversed(buf):
                    prev_tokens = estimate_tokens(prev)
                    if acc + prev_tokens > config.overlap_tokens:
                        break
                    overlap.insert(0, prev)
                    acc += prev_tokens
                buf = [*overlap, part]
                buf_tokens = acc + part_tokens
            else:
                buf.append(part)
                buf_tokens += part_tokens
        if buf:
            out.append(pattern.join(buf).strip())

        result = [chunk for chunk in out if chunk]
        # Only accept this split level if it actually made progress.
        if result and all(estimate_tokens(chunk) <= config.max_tokens for chunk in result):
            return result
        if len(result) > 1:
            nested: list[str] = []
            for chunk in result:
                nested.extend(_split_recursive(chunk, config))
            return nested

    # No usable boundary: hard-cut on character count.
    limit = config.target_tokens * 4
    pieces = (text[i : i + limit].strip() for i in range(0, len(text), limit))
    return [piece for piece in pieces if piece]


def _merge_small(sections: list[_Section], config: ChunkConfig) -> list[_Section]:
    """Coalesce runs of tiny sections that share a parent heading.

    Vaults are full of notes shaped like a heading with two lines under it.
    Emitting those individually floods the index with fragments, so adjacent
    small siblings are merged while their common breadcrumb is preserved.
    """
    merged: list[_Section] = []
    for section in sections:
        if not merged:
            merged.append(section)
            continue
        prev = merged[-1]
        combined = estimate_tokens(prev.text) + estimate_tokens(section.text)
        share_parent = prev.heading_path[:-1] == section.heading_path[:-1]
        if (
            estimate_tokens(prev.text) < config.min_tokens
            and share_parent
            and combined <= config.target_tokens
        ):
            # Keep the shallower breadcrumb so the merged chunk is not
            # mis-attributed to only one of its sources.
            prev.lines.append("")
            prev.lines.append(section.text)
            shallower = prev.heading_path[: len(section.heading_path) - 1]
            prev.heading_path = shallower or prev.heading_path
        else:
            merged.append(section)
    return merged


def chunk_note(
    note: Note,
    config: ChunkConfig | None = None,
    *,
    id_factory: Callable[[str, int], str] | None = None,
) -> list[Chunk]:
    """Chunk a parsed note into retrievable units.

    Chunk ids are derived from the note id and the chunk's own content hash, so
    an edit to paragraph 9 does not renumber paragraphs 1-8. That keeps
    incremental re-indexing cheap and makes deletes precise.
    """
    config = config or ChunkConfig()
    sections = _merge_small(list(_iter_sections(note.body)), config)

    # A note with no headings at all still needs to produce something.
    if not sections:
        body = note.body.strip()
        if not body:
            return []
        sections = [_Section(heading_path=[], lines=[body])]

    chunks: list[Chunk] = []
    ordinal = 0
    for section in sections:
        text = section.text
        if not text:
            continue
        for piece in _split_recursive(text, config):
            piece = piece.strip()
            if not piece:
                continue
            # Drop trailing scraps, but never drop a note's only chunk.
            if estimate_tokens(piece) < config.min_tokens and (chunks or len(sections) > 1):
                continue
            chunk_id = (
                id_factory(note.note_id, ordinal)
                if id_factory
                else f"{note.note_id}#{content_hash(piece)[:12]}"
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    note_id=note.note_id,
                    text=piece,
                    heading_path=list(section.heading_path),
                    ordinal=ordinal,
                    token_estimate=estimate_tokens(piece),
                    tags=set(note.tags),
                    links=set(note.outgoing),
                    sensitivity=note.sensitivity,
                )
            )
            ordinal += 1

    # Guarantee at least one chunk for a note with real content.
    if not chunks and note.body.strip():
        text = note.body.strip()
        chunks.append(
            Chunk(
                chunk_id=f"{note.note_id}#{content_hash(text)[:12]}",
                note_id=note.note_id,
                text=text,
                heading_path=[],
                ordinal=0,
                token_estimate=estimate_tokens(text),
                tags=set(note.tags),
                links=set(note.outgoing),
                sensitivity=note.sensitivity,
            )
        )
    return chunks
