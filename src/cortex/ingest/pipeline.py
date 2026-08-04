"""Indexing pipeline.

    scan vault -> hash gate -> parse -> chunk -> embed -> store
                     |                                      |
                     +-- unchanged: skip (the big win) ------+

The hash gate is what makes this cheap. Editors emit several filesystem events
per save and almost none of them change content; without the gate a vault
re-embeds itself constantly. With it, a no-op scan of ten thousand notes costs
a directory walk and ten thousand hashes -- under a second.

Embedding batches are sized by the thermal governor, so a backfill on a fanless
Air politely gets out of the way when the machine is hot or on battery.
"""

from __future__ import annotations

import fnmatch
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cortex.catalog import Catalog, ChangeSet
from cortex.index.store import StoredChunk, VectorStore
from cortex.ingest.chunker import ChunkConfig, chunk_note
from cortex.ingest.obsidian import parse_note, parse_text
from cortex.ingest.parsers import extract_document, supported_suffixes
from cortex.llm.protocol import EmbeddingProvider
from cortex.models import Chunk, Note, content_hash
from cortex.retrieve.graph import LinkGraph
from cortex.thermal.governor import ThermalGovernor

logger = logging.getLogger(__name__)

__all__ = ["IndexPipeline", "IndexReport", "scan_vault"]

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}


@dataclass(slots=True)
class IndexReport:
    """Outcome of an indexing run."""

    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks_written: int = 0
    chunks_removed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    paused_for_thermal: bool = False

    def summary(self) -> str:
        parts = [
            f"{self.scanned} scanned",
            f"{self.indexed} indexed",
            f"{self.skipped} unchanged",
        ]
        if self.deleted:
            parts.append(f"{self.deleted} deleted")
        parts.append(f"{self.chunks_written} chunks")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.paused_for_thermal:
            parts.append("paused (thermal)")
        return ", ".join(parts)


def scan_vault(
    vault: Path, *, exclude: Sequence[str] = (), documents: bool = False
) -> Iterator[Path]:
    """Yield markdown files, skipping excluded paths.

    ``.obsidian`` in particular must be excluded: it holds the app's own JSON
    config and plugin data, none of which is knowledge.
    """
    if not vault.exists():
        return
    wanted = set(MARKDOWN_SUFFIXES)
    if documents:
        wanted |= supported_suffixes()
    for path in sorted(vault.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        try:
            rel = path.relative_to(vault).as_posix()
        except ValueError:
            continue
        if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
            continue
        if any(part.startswith(".") for part in Path(rel).parts[:-1]):
            continue
        yield path


class IndexPipeline:
    """Drives incremental indexing of a vault."""

    def __init__(
        self,
        *,
        vault: Path,
        store: VectorStore,
        catalog: Catalog,
        embedder: EmbeddingProvider,
        chunk_config: ChunkConfig | None = None,
        governor: ThermalGovernor | None = None,
        exclude: Sequence[str] = (),
        batch_size: int = 32,
        documents: bool = False,
    ) -> None:
        self.vault = Path(vault)
        self.store = store
        self.catalog = catalog
        self.embedder = embedder
        self.chunk_config = chunk_config or ChunkConfig()
        self.governor = governor
        self.exclude = list(exclude)
        self.batch_size = batch_size
        self.documents = documents
        self._notes: dict[str, Note] = {}

    # --- scanning ----------------------------------------------------------

    def observe(self) -> dict[str, tuple[Path, str]]:
        """Walk the vault, returning ``note_id -> (path, content_hash)``.

        Hashing raw bytes rather than parsed content means a whitespace-only
        edit still counts as a change, which is correct: we cannot know it was
        insignificant without parsing, and parsing is the expensive part we are
        trying to avoid.
        """
        observed: dict[str, tuple[Path, str]] = {}
        for path in scan_vault(self.vault, exclude=self.exclude, documents=self.documents):
            try:
                digest = content_hash(path.read_bytes())
            except OSError as exc:
                logger.warning("unreadable %s: %s", path, exc)
                continue
            rel = path.relative_to(self.vault).as_posix()
            observed[rel] = (path, digest)
        return observed

    def plan(self) -> tuple[ChangeSet, dict[str, tuple[Path, str]]]:
        observed = self.observe()
        changes = self.catalog.diff({nid: digest for nid, (_, digest) in observed.items()})
        return changes, observed

    # --- indexing ----------------------------------------------------------

    def _embed_batch(self, chunks: Sequence[Chunk]) -> list[StoredChunk]:
        texts = [chunk.with_context_header() for chunk in chunks]
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
        paired = zip(chunks, vectors, strict=True)
        return [StoredChunk(chunk=chunk, vector=vec) for chunk, vec in paired]

    def index_note(self, path: Path, digest: str) -> tuple[int, int] | None:
        """Index one note.

        Returns ``(chunks_written, chunks_removed)``, or ``None`` when the file
        yielded nothing indexable -- an image-only PDF, an empty HTML clipping.
        ``None`` is distinct from ``(0, 0)``: the latter means a real note whose
        chunks were replaced, and counting a skip as "indexed" made the report
        claim work that never happened.
        """
        note = self._parse_any(path)
        if note is None:
            return None
        self._notes[note.note_id] = note
        chunks = chunk_note(note, self.chunk_config)

        if not chunks:
            stale = self.catalog.forget(note.note_id)
            removed = self.store.delete(stale) if stale else 0
            return 0, removed

        written = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            self.store.upsert(self._embed_batch(batch))
            written += len(batch)

        # Catalog is updated only after the store has accepted the vectors, so
        # a crash mid-write leaves the catalog behind rather than ahead. Being
        # behind means we re-index a note; being ahead means we silently lose it.
        stale = self.catalog.replace_note(
            note.note_id,
            content_hash=digest,
            chunk_ids=[c.chunk_id for c in chunks],
            mtime=note.mtime,
            sensitivity=note.sensitivity.value,
        )
        removed = self.store.delete(stale) if stale else 0
        return written, removed

    def _parse_any(self, path: Path) -> Note | None:
        """Parse a vault file, converting non-markdown formats to markdown first.

        A PDF becomes markdown and then follows the identical path as a
        hand-written note -- same chunker, same linking, same citations. The rest
        of the system never learns that some notes were not markdown.
        """
        if path.suffix.lower() in MARKDOWN_SUFFIXES:
            return parse_note(path, vault_root=self.vault)

        extracted = extract_document(path)
        if extracted is None:
            return None
        if extracted.warning:
            logger.info("%s: %s", path.name, extracted.warning)
        if extracted.is_empty:
            return None

        try:
            rel = path.resolve().relative_to(self.vault.resolve()).as_posix()
        except ValueError:
            rel = path.name
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0

        title = extracted.title or path.stem
        # Synthesise frontmatter so the note carries its provenance and is
        # distinguishable from hand-written content in search results.
        body = (
            f"---\ntitle: {title!r}\nsource_format: {extracted.kind}\n"
            f"tags:\n  - cortex/imported\n---\n\n# {title}\n\n{extracted.text}"
        )
        return parse_text(body, rel_path=rel, path=path, mtime=mtime)

    def run(
        self,
        *,
        full: bool = False,
        progress: Callable[[str, int, int], None] | None = None,
        respect_thermal: bool = True,
    ) -> IndexReport:
        """Run an incremental (or full) index pass."""
        started = time.monotonic()
        report = IndexReport()

        if full:
            self.catalog.clear()

        changes, observed = self.plan()
        report.scanned = len(observed)
        report.skipped = len(changes.unchanged)

        targets = changes.needs_indexing
        total = len(targets)

        for position, note_id in enumerate(targets, start=1):
            if respect_thermal and self.governor is not None:
                self.governor.sample()
                if not self.governor.may_backfill():
                    logger.info("pausing backfill: thermal state %s", self.governor.state.value)
                    report.paused_for_thermal = True
                    break

            path, digest = observed[note_id]
            try:
                outcome = self.index_note(path, digest)
            except Exception as exc:
                logger.warning("failed to index %s: %s", note_id, exc)
                report.errors.append((note_id, str(exc)))
                continue

            if outcome is None:
                report.skipped += 1
                continue
            written, removed = outcome
            report.indexed += 1
            report.chunks_written += written
            report.chunks_removed += removed
            if progress:
                progress(note_id, position, total)

            if respect_thermal and self.governor is not None:
                pause = self.governor.cooldown_hint()
                if pause:
                    time.sleep(pause)

        # Reconcile deletions even if the backfill paused: ghost chunks are
        # cheap to remove and actively harmful to leave.
        for note_id in changes.deleted:
            stale = self.catalog.forget(note_id)
            report.chunks_removed += self.store.delete(stale)
            report.deleted += 1
            self._notes.pop(note_id, None)

        report.elapsed_ms = (time.monotonic() - started) * 1000
        return report

    # --- graph -------------------------------------------------------------

    def build_graph(self) -> LinkGraph:
        """Build the wikilink graph from the current vault state.

        Parses every note, because the graph needs vault-wide link resolution
        that a partial view cannot provide. Parsing is cheap next to embedding;
        this is a few hundred milliseconds on a large vault.
        """
        notes: list[Note] = []
        for path in scan_vault(self.vault, exclude=self.exclude, documents=self.documents):
            try:
                parsed = self._parse_any(path)
                if parsed is not None:
                    notes.append(parsed)
            except OSError as exc:
                logger.debug("skipping %s while building graph: %s", path, exc)
        self._notes = {note.note_id: note for note in notes}
        return LinkGraph.from_notes(notes)
