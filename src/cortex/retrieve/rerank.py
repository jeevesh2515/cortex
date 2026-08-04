"""Cross-encoder reranking.

Reranking is the single highest-leverage addition to a hybrid retrieval
pipeline: published figures put BGE-reranker-v2-m3 at roughly +18% recall@5 over
fused retrieval alone, for about 84ms on 100 candidates.

The reason it is worth a second model is that bi-encoders and cross-encoders
answer different questions. A bi-encoder embeds the query and the document
*independently*, so it can only ever measure "are these in the same region of
vector space". A cross-encoder sees query and document *together* in one forward
pass and scores actual relevance. That is strictly more information, at the cost
of not being precomputable -- which is exactly why it runs on 30 candidates
rather than 30,000 chunks.

Two concessions to a fanless 16 GB machine:

* **Lazy load.** The model is not touched until the first rerank call, so
  ``cortex index`` never pays for it.
* **Idle unload.** After ``idle_ttl`` seconds without use the weights are
  released. Holding the embedder, the reranker and a chat model resident
  simultaneously does not fit in the memory budget.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CrossEncoderReranker", "NoopReranker", "rerank_available", "resolve_model_id"]

# Shorthand -> HuggingFace id, so config can stay readable.
_MODEL_ALIASES = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    # ~22M params. Meaningfully lower ceiling than bge-v2-m3, but ~4x faster and
    # a fraction of the memory -- the right default on a thermally limited
    # machine if bge proves too slow.
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def resolve_model_id(name: str) -> str:
    """Expand a shorthand model name to a HuggingFace id."""
    return _MODEL_ALIASES.get(name.strip().lower(), name)


def rerank_available() -> bool:
    """Whether the optional reranking dependency is installed."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _pick_device() -> str:
    """Prefer Apple's Metal backend, then CUDA, then CPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by rerank_available
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NoopReranker:
    """Passes the fused order through unchanged.

    Used when reranking is configured but unavailable. Returning the input order
    is the correct degradation: the fused ranking is already good, and silently
    reordering by something arbitrary would be worse than not reranking.
    """

    name = "noop"

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[tuple[int, float]]:
        return [(i, 0.0) for i in range(min(top_k, len(documents)))]


class CrossEncoderReranker:
    """Sentence-transformers cross-encoder with lazy load and idle unload."""

    def __init__(
        self,
        model: str = "bge-reranker-v2-m3",
        *,
        device: str | None = None,
        idle_ttl: float = 600.0,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_id = resolve_model_id(model)
        self.device = device or _pick_device()
        self.idle_ttl = idle_ttl
        self.max_length = max_length
        self.batch_size = batch_size

        self._model: Any = None
        self._last_used = 0.0
        # The MCP server and a background indexer can both reach this, and
        # loading the same model twice concurrently would double peak memory.
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self.model_id

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("loading reranker %s on %s", self.model_id, self.device)
                started = time.monotonic()
                self._model = CrossEncoder(
                    self.model_id,
                    device=self.device,
                    max_length=self.max_length,
                )
                logger.info("reranker ready in %.1fs", time.monotonic() - started)
            return self._model

    def unload(self) -> bool:
        """Release the weights. Returns True if something was freed."""
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            try:
                import gc

                import torch

                gc.collect()
                if self.device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.debug("reranker unloaded")
            return True

    def maybe_unload(self, *, now: float | None = None) -> bool:
        """Unload if idle beyond the TTL. Call from a maintenance loop."""
        if self._model is None or self.idle_ttl <= 0:
            return False
        clock = time.monotonic() if now is None else now
        if clock - self._last_used >= self.idle_ttl:
            return self.unload()
        return False

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[tuple[int, float]]:
        """Score documents against the query, best first.

        Returns ``(original_index, score)`` so the caller can map back to its own
        candidate objects without this module needing to know about them.
        """
        if not documents:
            return []

        model = self._load()
        self._last_used = time.monotonic()

        pairs = [(query, doc) for doc in documents]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:top_k]


def build_reranker(
    model: str = "bge-reranker-v2-m3",
    *,
    enabled: bool = True,
    device: str | None = None,
    idle_ttl: float = 600.0,
) -> CrossEncoderReranker | None:
    """Construct a reranker, or ``None`` when unavailable or disabled.

    Returns ``None`` rather than a stub so the engine's existing
    "no reranker configured" path handles it, and ``cortex status`` can report
    honestly that reranking is inactive. Quietly substituting a no-op would make
    the config claim something untrue.
    """
    if not enabled:
        return None
    if not rerank_available():
        logger.warning(
            "reranking is enabled in config but sentence-transformers is not "
            "installed -- continuing without it. Install with: "
            "pip install 'cortex-brain[rerank]'"
        )
        return None
    return CrossEncoderReranker(model, device=device, idle_ttl=idle_ttl)
