"""Composition root.

One place where settings become live objects. The CLI, the MCP server and the
HTTP API all build their world here, so there is exactly one definition of what
"a configured Cortex" means and no chance of the three drifting apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from cortex.catalog import Catalog
from cortex.config import Settings, load_settings
from cortex.index.store import MemoryStore, VectorStore
from cortex.ingest.pipeline import IndexPipeline
from cortex.llm.protocol import EmbeddingProvider, RerankProvider
from cortex.llm.providers import HashEmbedder, OllamaEmbedder, build_chat_providers
from cortex.llm.router import Router
from cortex.memory import MemoryWriter
from cortex.retrieve.engine import RetrievalEngine
from cortex.retrieve.graph import LinkGraph
from cortex.retrieve.rerank import build_reranker
from cortex.thermal.governor import StaticProbe, SystemProbe, ThermalGovernor, default_probe

logger = logging.getLogger(__name__)

__all__ = ["Runtime", "StaticProbe", "build_runtime"]


@dataclass(slots=True)
class Runtime:
    """A fully wired Cortex instance.

    ``governor`` is None only when the runtime was built in-memory (test/CI
    scaffolding). Production runtimes always carry a live ThermalGovernor so
    interactive queries stay fast while background indexing yields when the
    machine is hot or on low battery.
    """

    settings: Settings
    catalog: Catalog
    store: VectorStore
    embedder: EmbeddingProvider
    governor: ThermalGovernor | None
    router: Router
    pipeline: IndexPipeline
    graph: LinkGraph | None = None
    reranker: RerankProvider | None = None
    memory: MemoryWriter | None = None

    def engine(self) -> RetrievalEngine:
        return RetrievalEngine(
            self.store,
            self.embedder,
            graph=self.graph,
            reranker=self.reranker,
            router=self.router,
            top_k=self.settings.top_k,
            dense_k=self.settings.dense_k,
            fts_k=self.settings.fts_k,
            graph_hops=self.settings.graph_hops,
            fusion_weights=self.settings.fusion_weights,
            rerank_candidates=self.settings.rerank_candidates,
            temporal_enabled=self.settings.temporal_enabled,
            memory_max_results=self.settings.memory_max_results,
            expansion_enabled=self.settings.expansion_enabled,
        )

    def refresh_graph(self) -> LinkGraph:
        self.graph = self.pipeline.build_graph()
        return self.graph

    def close(self) -> None:
        self.catalog.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _build_store(settings: Settings) -> VectorStore:
    """Prefer LanceDB; fall back to the in-memory store with a clear warning.

    Falling back silently would be worse than failing: the user would get a
    working system that quietly forgets everything on restart.
    """
    from cortex.index.lance import lancedb_available

    if not lancedb_available():
        logger.warning(
            "LanceDB not installed -- using a non-persistent in-memory index. "
            "Install with: pip install 'cortex-brain[index]'"
        )
        return MemoryStore()

    from cortex.index.lance import LanceStore

    return LanceStore(settings.index_path, dimensions=settings.embed_dimensions)


def _build_embedder(settings: Settings, *, offline: bool = False) -> EmbeddingProvider:
    if offline:
        return HashEmbedder(dimensions=settings.embed_dimensions)
    return OllamaEmbedder(
        base_url=settings.ollama_url,
        model=settings.embed_model,
        dimensions=settings.embed_dimensions,
    )


def build_runtime(
    config_path: Path | None = None,
    *,
    settings: Settings | None = None,
    offline: bool = False,
    in_memory: bool = False,
    probe: SystemProbe | None = None,
) -> Runtime:
    """Assemble a Runtime from configuration.

    ``offline`` swaps the real embedder for the deterministic hashing one so
    the CLI is usable (and testable) before any model has been pulled.
    """
    settings = settings or load_settings(config_path)
    settings.ensure_dirs()

    catalog = Catalog(":memory:" if in_memory else settings.db_path)
    store = MemoryStore() if in_memory else _build_store(settings)
    embedder = _build_embedder(settings, offline=offline)
    # In-memory runtimes are test/CI scaffolding and must not depend on the
    # developer's real battery or chassis state. The pipeline-level gate would
    # otherwise refuse to index on a MacBook that ``pmset`` reports as
    # throttling or on battery below ``min_battery_for_backfill`` -- which is
    # exactly what happens on a developer laptop during a long test run. We
    # therefore skip the governor entirely in test mode (None propagates into
    # ``IndexPipeline``, whose ``respect_thermal`` gate becomes a no-op) and
    # leave status display to surface that fact explicitly.
    governor = (
        ThermalGovernor(
            probe=probe if probe is not None else default_probe(),
            config=settings.governor,
        )
        if not in_memory
        else None
    )

    providers = build_chat_providers(settings.providers)
    router = Router(providers)

    # Lazy inside: constructing this does not load weights, so `cortex index`
    # pays nothing for a reranker it never calls.
    reranker = build_reranker(
        settings.rerank_model,
        enabled=settings.rerank_enabled and not offline,
        idle_ttl=settings.rerank_idle_ttl,
    )

    pipeline = IndexPipeline(
        vault=settings.vault_path,
        store=store,
        catalog=catalog,
        embedder=embedder,
        chunk_config=settings.chunk,
        governor=governor,
        exclude=settings.exclude_globs,
        documents=settings.ingest_documents,
    )

    memory = MemoryWriter(
        settings.vault_path,
        folder=settings.memory_folder,
        enabled=settings.memory_enabled,
    )

    return Runtime(
        settings=settings,
        catalog=catalog,
        store=store,
        embedder=embedder,
        governor=governor,
        router=router,
        pipeline=pipeline,
        reranker=reranker,
        memory=memory,
    )
