"""Core domain types for Cortex.

These are deliberately plain and serialisable. Everything that crosses a module
boundary (parser -> chunker -> index -> retrieval -> synthesis) is one of these.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class Sensitivity(StrEnum):
    """How freely a piece of content may be sent to a model provider.

    Vault content defaults to PRIVATE. Escalation is always explicit -- see
    :mod:`cortex.llm.router` for the gate that enforces this.
    """

    PRIVATE = "private"
    """Never leaves the machine except to a provider with a no-training policy."""

    PUBLIC = "public"
    """Explicitly marked shareable; any provider is eligible."""


class DataPolicy(StrEnum):
    """A provider's declared stance on training with submitted data."""

    NO_TRAIN = "no_train"
    """Contractually does not train on submitted data."""

    NO_TRAIN_IF_ZDR = "no_train_if_zdr"
    """Safe only when zero-data-retention is enabled on the account."""

    TRAINS = "trains"
    """May use submitted data for model improvement. Blocked for PRIVATE content."""

    LOCAL = "local"
    """Runs on this machine; data never leaves it."""


@dataclass(frozen=True, slots=True)
class WikiLink:
    """An Obsidian ``[[wikilink]]``.

    Obsidian link syntax is ``[[target#heading|alias]]`` where ``#heading`` and
    ``|alias`` are both optional. Embeds are the same syntax prefixed with ``!``.
    """

    target: str
    """The note name being linked to, without extension."""

    heading: str | None = None
    """Optional ``#heading`` or ``#^blockref`` anchor."""

    alias: str | None = None
    """Optional ``|display text``."""

    is_embed: bool = False
    """True for ``![[...]]`` transclusions."""

    @property
    def display(self) -> str:
        """The text a reader actually sees."""
        return self.alias or self.target


@dataclass(slots=True)
class Note:
    """A parsed Obsidian note, before chunking."""

    path: Path
    """Absolute path on disk."""

    rel_path: str
    """Path relative to the vault root. Stable identifier across machines."""

    title: str
    """Note title -- frontmatter ``title``, else the filename stem."""

    body: str
    """Markdown body with frontmatter stripped."""

    frontmatter: dict[str, object] = field(default_factory=dict)
    links: list[WikiLink] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    content_hash: str = ""
    mtime: float = 0.0
    sensitivity: Sensitivity = Sensitivity.PRIVATE

    @property
    def note_id(self) -> str:
        """Vault-relative identifier used for link resolution."""
        return self.rel_path

    @property
    def outgoing(self) -> set[str]:
        """Distinct link targets, lowercased for case-insensitive matching."""
        return {link.target.lower() for link in self.links}


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text with provenance back to its source note."""

    chunk_id: str
    note_id: str
    text: str

    heading_path: list[str] = field(default_factory=list)
    """Breadcrumb of enclosing headings, outermost first. Used for citations."""

    ordinal: int = 0
    """Position within the note, 0-based."""

    token_estimate: int = 0
    tags: set[str] = field(default_factory=set)
    links: set[str] = field(default_factory=set)
    sensitivity: Sensitivity = Sensitivity.PRIVATE

    @property
    def citation(self) -> str:
        """Human-readable source reference, e.g. ``Note Title > Section > Sub``."""
        parts = [self.note_id, *self.heading_path]
        return " > ".join(parts)

    def with_context_header(self) -> str:
        """Text prefixed with its heading breadcrumb.

        Embedding this rather than the bare chunk measurably improves retrieval
        on Obsidian vaults, where a section like "Notes" is meaningless without
        knowing which note it came from.
        """
        if not self.heading_path:
            return f"[{self.note_id}]\n{self.text}"
        crumb = " > ".join(self.heading_path)
        return f"[{self.note_id} > {crumb}]\n{self.text}"


@dataclass(slots=True)
class ScoredChunk:
    """A chunk with a retrieval score and an audit trail of how it was found."""

    chunk: Chunk
    score: float
    source: str = "dense"
    """Which retriever surfaced it: ``dense``, ``fts``, ``graph``, or ``fused``."""

    rank: int = 0
    components: dict[str, float] = field(default_factory=dict)
    """Per-retriever contributions, retained so fusion decisions are debuggable."""


@dataclass(slots=True)
class Answer:
    """A synthesised answer with its supporting evidence."""

    question: str
    text: str
    citations: list[Chunk] = field(default_factory=list)
    provider: str = "local"
    """Which provider served this, recorded so routing is auditable."""

    escalated: bool = False
    """True if this left the machine."""

    elapsed_ms: float = 0.0


def content_hash(data: str | bytes) -> str:
    """Stable content hash used to skip re-embedding unchanged files.

    BLAKE2b rather than SHA-256: roughly 2x faster on the same input and we need
    collision resistance, not cryptographic signing.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def utcnow() -> datetime:
    """Timezone-aware UTC now. Centralised so tests can monkeypatch one symbol."""

    return datetime.now(UTC)
