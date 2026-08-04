"""LanceDB-backed store.

Chosen over sqlite-vec, Chroma, Milvus Lite and DuckDB VSS because it is the
only embedded option that satisfies every constraint at once: no daemon,
disk-backed rather than RAM-resident, native full-text search alongside
vectors, real deletes, and a stable on-disk format. See
``docs/adr/0002-vector-store.md``.

Import is lazy: ``lancedb`` pulls native wheels, and the rest of Cortex must
stay importable without them so the core test suite runs anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cortex.index.store import StoredChunk
from cortex.models import Chunk, ScoredChunk, Sensitivity


def _parse_iso_date(raw: object) -> date | None:
    """Read an ISO date back, tolerating the empty string and legacy rows.

    Indexes written before ``note_date`` existed have no such column, so a
    missing value must be ``None`` rather than an error -- otherwise upgrading
    would require a full rebuild before the store could even be read.
    """
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


if TYPE_CHECKING:
    import lancedb

logger = logging.getLogger(__name__)

__all__ = ["LanceStore", "lancedb_available"]

TABLE_NAME = "chunks"


def lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401
    except ImportError:
        return False
    return True


class LanceStore:
    """Disk-backed vector store with native hybrid search."""

    def __init__(self, path: Path | str, *, dimensions: int = 1024) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:  # pragma: no cover - exercised by install docs
            raise ImportError(
                "LanceDB is not installed. Install it with: pip install 'cortex-brain[index]'"
            ) from exc

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._pa = pa
        self._db: lancedb.DBConnection = lancedb.connect(str(self.path))
        self._schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("note_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("search_text", pa.string()),
                pa.field("heading_path", pa.string()),
                pa.field("ordinal", pa.int32()),
                pa.field("tags", pa.string()),
                pa.field("links", pa.string()),
                pa.field("sensitivity", pa.string()),
                # ISO date string rather than pa.date32(): nullable, trivially
                # readable in a Lance dump, and the filtering is done in Python
                # anyway. Absent from the schema entirely until now, which meant
                # temporal queries silently found nothing on the LanceDB path
                # while passing every test against the in-memory store.
                pa.field("note_date", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimensions)),
            ]
        )
        self._table = self._open_table()
        self._fts_ready = False

    def _list_tables(self) -> list[str]:
        """Table names, tolerating both LanceDB listing APIs.

        ``table_names()`` returned a plain list and is deprecated since 0.25.
        ``list_tables()`` returns a paginated ``ListTablesResponse`` whose names
        live on ``.tables`` -- iterating the response itself yields the wrong
        thing, silently, which manifests as the table appearing not to exist and
        a spurious "already exists" error on the subsequent create.
        """
        lister = getattr(self._db, "list_tables", None)
        if lister is not None:
            response = lister()
            tables = getattr(response, "tables", response)
            return [str(name) for name in tables]
        return [str(name) for name in self._db.table_names()]

    def _open_table(self) -> Any:
        if TABLE_NAME in self._list_tables():
            return self._db.open_table(TABLE_NAME)
        return self._db.create_table(TABLE_NAME, schema=self._schema)

    # --- serialisation -----------------------------------------------------

    @staticmethod
    def _to_row(item: StoredChunk) -> dict[str, Any]:
        chunk = item.chunk
        return {
            "chunk_id": chunk.chunk_id,
            "note_id": chunk.note_id,
            "text": chunk.text,
            # Indexed for FTS with its breadcrumb, so a section titled "Notes"
            # is still findable by its parent note's name.
            "search_text": chunk.with_context_header(),
            "heading_path": "\x1f".join(chunk.heading_path),
            "ordinal": chunk.ordinal,
            "tags": "\x1f".join(sorted(chunk.tags)),
            "links": "\x1f".join(sorted(chunk.links)),
            "sensitivity": chunk.sensitivity.value,
            "note_date": chunk.note_date.isoformat() if chunk.note_date else "",
            "vector": [float(x) for x in item.vector],
        }

    @staticmethod
    def _from_row(row: dict[str, Any]) -> Chunk:
        def unpack(value: Any) -> list[str]:
            return [part for part in str(value or "").split("\x1f") if part]

        return Chunk(
            chunk_id=row["chunk_id"],
            note_id=row["note_id"],
            text=row["text"],
            heading_path=unpack(row.get("heading_path")),
            ordinal=int(row.get("ordinal", 0)),
            tags=set(unpack(row.get("tags"))),
            links=set(unpack(row.get("links"))),
            sensitivity=Sensitivity(row.get("sensitivity", "private")),
            note_date=_parse_iso_date(row.get("note_date")),
        )

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a string for a LanceDB SQL predicate."""
        return value.replace("'", "''")

    # --- writes ------------------------------------------------------------

    def upsert(self, items: Sequence[StoredChunk]) -> None:
        if not items:
            return
        # Delete-then-insert: LanceDB's merge_insert exists but delete+add has
        # a simpler failure mode and these batches are small.
        self.delete([item.chunk.chunk_id for item in items])
        self._table.add([self._to_row(item) for item in items])
        self._fts_ready = False

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = [cid for cid in chunk_ids if cid]
        if not ids:
            return 0
        removed = 0
        # Chunked to keep the predicate under any statement-length limit.
        for start in range(0, len(ids), 200):
            batch = ids[start : start + 200]
            quoted = ", ".join(f"'{self._escape(cid)}'" for cid in batch)
            try:
                self._table.delete(f"chunk_id IN ({quoted})")
                removed += len(batch)
            except Exception as exc:
                logger.warning("delete failed for %d ids: %s", len(batch), exc)
        self._fts_ready = False
        return removed

    def ensure_fts_index(self) -> None:
        """Build the full-text index if it is missing or stale."""
        if self._fts_ready:
            return
        # create_index(config=FTS()) superseded create_fts_index in 0.25.
        try:
            from lancedb.index import FTS

            self._table.create_index("search_text", config=FTS(), replace=True)
            self._fts_ready = True
            return
        except (ImportError, TypeError, AttributeError):
            pass
        except Exception as exc:
            logger.debug("could not build FTS index: %s", exc)
            return
        try:
            self._table.create_fts_index("search_text", replace=True)
            self._fts_ready = True
        except Exception as exc:
            logger.debug("could not build FTS index: %s", exc)

    def create_vector_index(self) -> None:
        """Build an ANN index.

        Only worth it past a few tens of thousands of vectors -- below that,
        LanceDB's exact scan is faster than an approximate lookup and has no
        recall loss.
        """
        if self.count() < 10_000:
            logger.debug("skipping ANN index: corpus too small to benefit")
            return
        try:
            self._table.create_index(metric="cosine", replace=True)
        except Exception as exc:
            logger.warning("could not build vector index: %s", exc)

    # --- reads -------------------------------------------------------------

    def _predicate(self, sensitivity: Sensitivity | None) -> str | None:
        if sensitivity is Sensitivity.PUBLIC:
            return "sensitivity = 'public'"
        return None

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 20,
        sensitivity: Sensitivity | None = None,
    ) -> list[ScoredChunk]:
        if self.count() == 0:
            return []
        query = self._table.search([float(x) for x in vector]).limit(top_k)
        predicate = self._predicate(sensitivity)
        if predicate:
            query = query.where(predicate)
        try:
            rows = query.to_list()
        except Exception as exc:
            logger.warning("dense search failed: %s", exc)
            return []
        results = []
        for rank, row in enumerate(rows, start=1):
            # LanceDB returns cosine *distance*; convert to similarity.
            distance = float(row.get("_distance", 0.0))
            results.append(
                ScoredChunk(
                    chunk=self._from_row(row),
                    score=1.0 - distance,
                    source="dense",
                    rank=rank,
                )
            )
        return results

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 20,
        sensitivity: Sensitivity | None = None,
    ) -> list[ScoredChunk]:
        if self.count() == 0 or not query.strip():
            return []
        self.ensure_fts_index()
        try:
            search = self._table.search(query, query_type="fts").limit(top_k)
            predicate = self._predicate(sensitivity)
            if predicate:
                search = search.where(predicate)
            rows = search.to_list()
        except Exception as exc:
            logger.debug("fts search unavailable: %s", exc)
            return []
        return [
            ScoredChunk(
                chunk=self._from_row(row),
                score=float(row.get("_score", 0.0)),
                source="fts",
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def chunks_for_note(self, note_id: str) -> list[Chunk]:
        try:
            rows = (
                self._table.search()
                .where(f"note_id = '{self._escape(note_id)}'")
                .limit(1000)
                .to_list()
            )
        except Exception as exc:
            logger.debug("note lookup failed for %s: %s", note_id, exc)
            return []
        chunks = [self._from_row(row) for row in rows]
        chunks.sort(key=lambda c: c.ordinal)
        return chunks

    def chunks_by_note(self) -> dict[str, list[Chunk]]:
        out: dict[str, list[Chunk]] = {}
        try:
            rows = self._table.search().limit(100_000).to_list()
        except Exception as exc:
            logger.debug("full scan failed: %s", exc)
            return out
        for row in rows:
            chunk = self._from_row(row)
            out.setdefault(chunk.note_id, []).append(chunk)
        for chunks in out.values():
            chunks.sort(key=lambda c: c.ordinal)
        return out

    def count(self) -> int:
        try:
            return int(self._table.count_rows())
        except Exception:
            return 0

    def clear(self) -> None:
        self._db.drop_table(TABLE_NAME, ignore_missing=True)
        self._table = self._open_table()
        self._fts_ready = False
