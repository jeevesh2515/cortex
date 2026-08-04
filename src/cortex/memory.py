"""Conversational memory as Markdown notes.

Every other part of Cortex only reads the vault. This module writes to it, which
makes it the one place where a bug could damage something irreplaceable. The
design follows from that.

**Why Markdown notes rather than a memory database.** A separate store would be
invisible: you could not read what the system believes about you, correct it, or
delete it. Writing plain notes into the vault means memory is auditable in
Obsidian like anything else, appears in the graph view, and is deleted with a
keystroke. It also means memory participates in retrieval for free -- no second
index, no second embedder.

**The elegant part.** A memory note wikilinks the sources it drew on. Those links
join the real link graph, so a memory about sourdough becomes a hub connecting
the notes that answered it. The graph gets denser in exactly the places you
actually think about.

**The trap.** Memory notes are summaries of primary notes, so they compete with
their own sources in retrieval. Left alone they would gradually crowd out the
real content -- a slow degradation where answers get more confident and less
grounded, each generation quoting the last. ``memory_max_results`` caps how much
memory can occupy any result set. See ``docs/adr/0007``.

Safety invariants, all tested:

* Writes only ever land inside the configured memory folder. The resolved path is
  checked against it, so a crafted title cannot escape via ``../``.
* Nothing is ever overwritten. A name collision gets a numeric suffix.
* Writes are atomic (temp file then rename), so an interrupted write cannot leave
  a half-written note in the vault.
* Nothing is ever deleted. Pruning is the user's job, in Obsidian.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from cortex.models import Answer, Chunk, ScoredChunk, utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "MEMORY_TAG",
    "MemoryNote",
    "MemoryWriter",
    "cap_memory_results",
    "is_memory_chunk",
    "slugify",
]

MEMORY_TAG = "cortex/memory"
"""Tag on every memory note. Lets you find them all in Obsidian with
``tag:#cortex/memory``, and lets retrieval identify them for capping."""

_UNSAFE = re.compile(r"[^\w\s-]", re.UNICODE)
_WHITESPACE = re.compile(r"[-\s]+")


def slugify(text: str, *, max_length: int = 60) -> str:
    """Make a filename-safe slug.

    Strips anything that is not a word character, whitespace or hyphen, which
    also removes path separators and dots -- so a title can never contribute a
    traversal component. The folder containment check is still enforced
    separately; defence in depth, because this is the one module that writes.
    """
    cleaned = _UNSAFE.sub("", text).strip()
    cleaned = _WHITESPACE.sub("-", cleaned)
    cleaned = cleaned.strip("-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-")
    return cleaned or "untitled"


def is_memory_chunk(chunk: Chunk) -> bool:
    """Whether a chunk came from a memory note."""
    return MEMORY_TAG in chunk.tags


@dataclass(slots=True)
class MemoryNote:
    """A session memory, before it is written."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    """Vault-relative note ids the answer drew on."""

    provider: str = ""
    tags: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=utcnow)

    @property
    def title(self) -> str:
        """A human title, derived from the question."""
        text = self.question.strip().rstrip("?").strip()
        return text[:80] if text else "Session"

    def filename(self) -> str:
        stamp = self.created.strftime("%Y-%m-%d %H%M")
        return f"{stamp} {slugify(self.title)}.md"

    def render(self) -> str:
        """Render as Obsidian-flavoured Markdown.

        Sources become wikilinks so the note joins the link graph rather than
        sitting inert beside it.
        """
        all_tags = sorted({MEMORY_TAG, *self.tags})
        lines = [
            "---",
            f'title: "{self.title.replace(chr(34), chr(39))}"',
            f"date: {self.created.date().isoformat()}",
            f"created: {self.created.isoformat(timespec='seconds')}",
            "tags:",
            *[f"  - {tag}" for tag in all_tags],
            # Derived from private notes, so private itself. Never inherit
            # PUBLIC: a summary of private notes is private.
            "sensitivity: private",
            "cortex_memory: true",
        ]
        if self.provider:
            lines.append(f"cortex_provider: {self.provider}")
        lines += ["---", "", f"# {self.title}", "", "## Asked", "", self.question.strip(), ""]
        lines += ["## Answered", "", self.answer.strip(), ""]

        if self.sources:
            lines += ["## Drawn from", ""]
            for source in dict.fromkeys(self.sources):
                # Wikilink by basename, which is how Obsidian resolves links.
                stem = PurePosixPath(source).name
                if stem.lower().endswith(".md"):
                    stem = stem[:-3]
                lines.append(f"- [[{stem}]]")
            lines.append("")

        return "\n".join(lines)


class MemoryWriter:
    """Writes memory notes into a folder inside the vault."""

    def __init__(
        self,
        vault: Path,
        *,
        folder: str = "Memory",
        enabled: bool = True,
    ) -> None:
        self.vault = Path(vault).expanduser()
        self.folder = folder.strip("/") or "Memory"
        self.enabled = enabled

    @property
    def root(self) -> Path:
        return self.vault / self.folder

    def _within_root(self, candidate: Path) -> bool:
        """Whether a path resolves inside the memory folder.

        Uses the resolved parent rather than the file, since the file does not
        exist yet. ``is_relative_to`` is the whole check: a title that somehow
        produced a traversal component would fail here even though slugify
        should already have removed it.
        """
        try:
            root = self.root.resolve()
            parent = candidate.parent.resolve()
        except OSError:
            return False
        return parent == root or parent.is_relative_to(root)

    def _unique_path(self, filename: str) -> Path:
        """A path that does not exist yet. Never overwrites."""
        target = self.root / filename
        if not target.exists():
            return target
        stem, suffix = target.stem, target.suffix
        for counter in range(2, 1000):
            candidate = self.root / f"{stem} ({counter}){suffix}"
            if not candidate.exists():
                return candidate
        # Astronomically unlikely; fall back to something certainly unique.
        return self.root / f"{stem} ({os.getpid()}-{int(utcnow().timestamp())}){suffix}"

    def write(self, note: MemoryNote) -> Path | None:
        """Write a memory note. Returns its path, or ``None`` when disabled."""
        if not self.enabled:
            return None
        if not note.answer.strip():
            logger.debug("skipping empty memory note")
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        target = self._unique_path(note.filename())

        if not self._within_root(target):
            # Should be unreachable given slugify, hence loud rather than silent.
            raise ValueError(f"refusing to write outside {self.root}: {target}")

        body = note.render()

        # Atomic: write a temp file in the same directory, fsync, then rename.
        # An interrupted write leaves a dot-prefixed temp file (which Obsidian
        # ignores), never a half-written note in the user's vault. Same
        # directory matters -- os.replace is only atomic within a filesystem.
        descriptor, temp_name = tempfile.mkstemp(dir=self.root, prefix=".cortex-tmp-", suffix=".md")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except OSError:
            Path(temp_name).unlink(missing_ok=True)
            raise

        logger.info("wrote memory note %s", target.name)
        return target

    def from_answer(
        self,
        answer: Answer,
        *,
        tags: Sequence[str] = (),
    ) -> Path | None:
        """Build and write a memory note from a synthesised answer."""
        return self.write(
            MemoryNote(
                question=answer.question,
                answer=answer.text,
                sources=[chunk.note_id for chunk in answer.citations],
                provider=answer.provider,
                tags=list(tags),
            )
        )

    def count(self) -> int:
        """How many memory notes exist."""
        if not self.root.exists():
            return 0
        return sum(1 for path in self.root.glob("*.md") if path.is_file())


def cap_memory_results(results: Sequence[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
    """Trim memory results to ``limit``, preserving order and primary sources.

    The feedback-loop guard. Without it, memory notes accumulate and gradually
    displace the notes they were derived from, so answers drift towards
    summarising previous answers -- more confident each round, less grounded.
    A negative limit disables the cap.
    """
    if limit < 0:
        return list(results)
    kept: list[ScoredChunk] = []
    memory_seen = 0
    for scored in results:
        if is_memory_chunk(scored.chunk):
            if memory_seen >= limit:
                continue
            memory_seen += 1
        kept.append(scored)
    return kept
