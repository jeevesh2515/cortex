"""Filesystem watching.

``cortex watch`` previously rescanned the whole vault on a timer. That works, but
it is the wrong shape: a 30-second poll means a note you just saved is invisible
for up to 30 seconds, while a vault that has not changed all afternoon still gets
walked 500 times.

This uses watchdog, which selects FSEvents on macOS -- kernel-level change
notification with near-zero idle cost, and no per-file descriptors (unlike
kqueue, which does not scale to a large vault).

Two guards matter more than the watching itself:

* **Debounce.** Editors emit several events per save: a temp file, a rename, a
  metadata touch. Obsidian is no exception. Reacting to each one would re-index
  the same note three times.
* **Coalescing.** Changes are accumulated into a set and handed over as a batch,
  so saving ten notes in a burst is one indexing pass, not ten.

The content-hash gate in the pipeline is still the real backstop: even if a
spurious event gets through, an unchanged file costs a hash and nothing more.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["MARKDOWN_PATTERNS", "VaultWatcher", "watchdog_available"]

MARKDOWN_PATTERNS = ("*.md", "*.markdown", "*.mdx")


def watchdog_available() -> bool:
    try:
        import watchdog  # noqa: F401
    except ImportError:
        return False
    return True


class VaultWatcher:
    """Watches a vault and calls back with batches of changed paths.

    The callback runs on a dedicated timer thread, not the watchdog event
    thread, so a slow indexing pass cannot stall event delivery and cause the
    OS to drop notifications.
    """

    def __init__(
        self,
        vault: Path,
        on_change: Callable[[set[Path]], None],
        *,
        debounce: float = 2.0,
        max_batch_wait: float = 30.0,
    ) -> None:
        self.vault = Path(vault)
        self.on_change = on_change
        self.debounce = debounce
        # Ceiling on how long a *continuously* edited vault can defer indexing.
        # Without it, someone typing steadily for an hour would keep resetting
        # the debounce and never get indexed.
        self.max_batch_wait = max_batch_wait

        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._first_seen: float | None = None
        self._timer: threading.Timer | None = None
        self._observer: object | None = None
        self._stopped = threading.Event()

    # --- event intake ------------------------------------------------------

    def _record(self, path: Path) -> None:
        """Note a changed path and (re)arm the debounce timer."""
        with self._lock:
            self._pending.add(path)
            now = time.monotonic()
            if self._first_seen is None:
                self._first_seen = now

            waited = now - self._first_seen
            if waited >= self.max_batch_wait:
                # Held long enough; flush on the next tick rather than deferring
                # again behind continued typing.
                delay = 0.0
            else:
                delay = min(self.debounce, self.max_batch_wait - waited)

            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(delay, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch = set(self._pending)
            self._pending.clear()
            self._first_seen = None
            self._timer = None

        if not batch or self._stopped.is_set():
            return
        try:
            self.on_change(batch)
        except Exception:
            logger.exception("change handler failed for %d path(s)", len(batch))

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Begin watching. Raises ImportError if watchdog is missing."""
        from watchdog.events import PatternMatchingEventHandler
        from watchdog.observers import Observer

        watcher = self

        class _Handler(PatternMatchingEventHandler):
            def __init__(self) -> None:
                super().__init__(
                    patterns=list(MARKDOWN_PATTERNS),
                    ignore_directories=True,
                    case_sensitive=False,
                )

            def _handle(self, event: object) -> None:
                raw = getattr(event, "dest_path", None) or getattr(event, "src_path", None)
                if not raw:
                    return
                path = Path(str(raw))
                # Obsidian's own config directory churns constantly and holds no
                # knowledge; watching it would mean near-continuous wakeups.
                if any(part.startswith(".") for part in path.parts):
                    return
                watcher._record(path)

            on_created = _handle
            on_modified = _handle
            on_moved = _handle
            on_deleted = _handle

        observer = Observer()
        observer.schedule(_Handler(), str(self.vault), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("watching %s via %s", self.vault, type(observer).__name__)

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop watching and flush anything still pending."""
        self._stopped.set()
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        observer = self._observer
        if observer is not None:
            stop = getattr(observer, "stop", None)
            join = getattr(observer, "join", None)
            if callable(stop):
                stop()
            if callable(join):
                join(timeout)
            self._observer = None

    @property
    def running(self) -> bool:
        observer = self._observer
        return bool(observer is not None and getattr(observer, "is_alive", lambda: False)())

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def __enter__(self) -> VaultWatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
