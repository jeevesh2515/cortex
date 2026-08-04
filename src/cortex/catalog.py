"""SQLite catalog: what has been indexed, and what it produced.

This is the bookkeeping that makes re-indexing cheap. The vector store knows
about vectors; it does not know that ``notes/rag.md`` hashed to ``abc123`` last
Tuesday and produced nine chunks. Without that, every scan re-embeds the whole
vault.

Three jobs:

1. **Hash gate.** Skip files whose content hash is unchanged. This is the
   single biggest cost saver -- editors fire many filesystem events per save
   and almost none of them change content.
2. **Chunk lineage.** Know exactly which chunk ids came from which note, so an
   edit deletes precisely the right vectors instead of orphaning them.
3. **Deletion reconciliation.** Notes deleted while the daemon was not running
   leave ghost chunks that keep surfacing in results. A sweep comparing the
   catalog against the filesystem catches them.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Catalog", "ChangeSet", "IndexedNote"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    note_id       TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    mtime         REAL NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    indexed_at    REAL NOT NULL,
    sensitivity   TEXT NOT NULL DEFAULT 'private'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    note_id   TEXT NOT NULL,
    ordinal   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (note_id) REFERENCES notes(note_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class IndexedNote:
    note_id: str
    content_hash: str
    mtime: float
    chunk_count: int
    indexed_at: float
    sensitivity: str = "private"


@dataclass(slots=True)
class ChangeSet:
    """The outcome of comparing the vault against the catalog."""

    new: list[str]
    changed: list[str]
    unchanged: list[str]
    deleted: list[str]

    @property
    def needs_indexing(self) -> list[str]:
        return [*self.new, *self.changed]

    @property
    def is_empty(self) -> bool:
        return not (self.new or self.changed or self.deleted)

    def summary(self) -> str:
        return (
            f"{len(self.new)} new, {len(self.changed)} changed, "
            f"{len(self.unchanged)} unchanged, {len(self.deleted)} deleted"
        )


class Catalog:
    """Thread-safe SQLite catalog.

    The watcher thread and any query thread share one instance, so access is
    serialised behind a lock. SQLite's own locking would mostly cover this, but
    the read-modify-write in :meth:`replace_note` must be atomic as a unit.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets a query read while the indexer writes -- essential when the
        # daemon is backfilling and the user runs a search.
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.set_meta("schema_version", SCHEMA_VERSION)

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # --- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # --- reads -------------------------------------------------------------

    def get(self, note_id: str) -> IndexedNote | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def hashes(self) -> dict[str, str]:
        """All known ``note_id -> content_hash``. One query, used by the diff."""
        with self._lock:
            rows = self._conn.execute("SELECT note_id, content_hash FROM notes").fetchall()
        return {row["note_id"]: row["content_hash"] for row in rows}

    def chunk_ids(self, note_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE note_id = ? ORDER BY ordinal", (note_id,)
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def all_note_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT note_id FROM notes").fetchall()
        return {row["note_id"] for row in rows}

    def stats(self) -> dict[str, int]:
        with self._lock:
            notes = self._conn.execute("SELECT COUNT(*) AS n FROM notes").fetchone()["n"]
            chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"notes": int(notes), "chunks": int(chunks)}

    # --- writes ------------------------------------------------------------

    def replace_note(
        self,
        note_id: str,
        *,
        content_hash: str,
        chunk_ids: Iterable[str],
        mtime: float = 0.0,
        indexed_at: float = 0.0,
        sensitivity: str = "private",
    ) -> list[str]:
        """Record an indexed note, replacing any prior chunk lineage.

        Returns the chunk ids that are now stale and must be deleted from the
        vector store. Returning them rather than deleting internally keeps the
        catalog ignorant of the store -- the caller owns that ordering, which
        matters because a crash between the two must leave the catalog behind,
        never ahead.
        """
        import time

        new_ids = list(chunk_ids)
        with self._write() as conn:
            previous = [
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE note_id = ?", (note_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
            conn.execute(
                "INSERT INTO notes(note_id, content_hash, mtime, chunk_count, indexed_at, "
                "sensitivity) VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(note_id) DO UPDATE SET "
                "content_hash = excluded.content_hash, mtime = excluded.mtime, "
                "chunk_count = excluded.chunk_count, indexed_at = excluded.indexed_at, "
                "sensitivity = excluded.sensitivity",
                (
                    note_id,
                    content_hash,
                    mtime,
                    len(new_ids),
                    indexed_at or time.time(),
                    sensitivity,
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, note_id, ordinal) VALUES(?, ?, ?)",
                [(cid, note_id, i) for i, cid in enumerate(new_ids)],
            )
        stale = set(previous) - set(new_ids)
        return sorted(stale)

    def forget(self, note_id: str) -> list[str]:
        """Remove a note. Returns its chunk ids for deletion from the store."""
        with self._write() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE note_id = ?", (note_id,)
            ).fetchall()
            ids = [row["chunk_id"] for row in rows]
            conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
            conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
        return ids

    def clear(self) -> None:
        with self._write() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM notes")

    # --- diffing -----------------------------------------------------------

    def diff(self, observed: dict[str, str]) -> ChangeSet:
        """Compare observed ``note_id -> content_hash`` against the catalog.

        ``observed`` must be the complete current state of the vault, because
        anything absent from it is treated as deleted.
        """
        known = self.hashes()
        new: list[str] = []
        changed: list[str] = []
        unchanged: list[str] = []

        for note_id, digest in observed.items():
            prior = known.get(note_id)
            if prior is None:
                new.append(note_id)
            elif prior != digest:
                changed.append(note_id)
            else:
                unchanged.append(note_id)

        deleted = sorted(set(known) - set(observed))
        return ChangeSet(
            new=sorted(new),
            changed=sorted(changed),
            unchanged=sorted(unchanged),
            deleted=deleted,
        )

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> IndexedNote:
        return IndexedNote(
            note_id=row["note_id"],
            content_hash=row["content_hash"],
            mtime=row["mtime"],
            chunk_count=row["chunk_count"],
            indexed_at=row["indexed_at"],
            sensitivity=row["sensitivity"],
        )
