"""MCP server exposing the vault to Antigravity and other MCP clients.

Register it in ``~/.gemini/antigravity-cli/mcp_config.json``:

    {
      "mcpServers": {
        "cortex": {
          "command": "/absolute/path/to/cortex/.venv/bin/python",
          "args": ["-m", "cortex.cli", "serve-mcp"]
        }
      }
    }

Use the venv's interpreter directly rather than a shell wrapper -- the MCP
client launches the process without a login shell, so anything relying on PATH
or shell activation will not be found.

Design note on write access: every tool here is read-only over the vault except
``reindex`` and ``remember``. ``remember`` only ever *creates* a new file inside
the memory folder -- it cannot edit or delete an existing note. An agent that
could silently rewrite your second brain is a liability, and Antigravity already
has general file tools if you genuinely want it editing notes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cortex.llm.protocol import PolicyViolation, ProviderError
from cortex.models import Sensitivity, obsidian_uri
from cortex.runtime import Runtime, build_runtime

logger = logging.getLogger(__name__)

__all__ = ["TOOL_DEFINITIONS", "build_server", "run_stdio"]

TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "description": (
            "Search the user's Obsidian vault using hybrid retrieval (dense vectors, "
            "BM25 keyword matching, and the vault's own wikilink graph). Returns "
            "matching passages with their source note and heading breadcrumb. Use this "
            "when you need evidence from the user's own notes rather than a synthesised "
            "answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return (default 8).",
                },
                "use_graph": {
                    "type": "boolean",
                    "description": "Follow wikilinks to surface related notes (default true).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_notes",
        "description": (
            "Ask a question and get a synthesised answer grounded in the user's notes, "
            "with inline citations. Routing respects the user's privacy policy: private "
            "notes are never sent to a provider that trains on submitted data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "local_only": {
                    "type": "boolean",
                    "description": "Force on-device synthesis only (default false).",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "list_links",
        "description": (
            "Show the wikilinks into and out of a specific note. Useful for exploring "
            "how an idea connects to the rest of the vault."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "Vault-relative path, e.g. 'notes/retrieval.md'.",
                }
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "vault_status",
        "description": (
            "Report index size, thermal state, and which model providers are eligible "
            "to receive private content."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "remember",
        "description": (
            "Save a short memory note into the user's vault so this exchange is "
            "recalled in future sessions. The note is plain Markdown in the Memory "
            "folder, wikilinked to the notes it drew on, so the user can read, edit "
            "or delete it in Obsidian. Use this when the user states a durable "
            "preference, a decision, or a conclusion worth keeping -- not for "
            "every passing question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What was asked or the topic being recorded.",
                },
                "answer": {
                    "type": "string",
                    "description": "The conclusion, decision or preference to remember.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Vault-relative note ids this drew on; becomes wikilinks.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra tags beyond cortex/memory.",
                },
            },
            "required": ["question", "answer"],
        },
    },
    {
        "name": "reindex",
        "description": (
            "Re-scan the vault and index new or changed notes. Incremental by default; "
            "unchanged notes are skipped via content hashing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "Rebuild the whole index from scratch (default false).",
                }
            },
        },
    },
]


class CortexTools:
    """Tool implementations over a Runtime.

    Separated from the MCP wire protocol so the behaviour is testable without
    the ``mcp`` package installed.
    """

    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime
        self._graph_loaded = False

    def _ensure_graph(self) -> None:
        if not self._graph_loaded:
            self.rt.refresh_graph()
            self._graph_loaded = True

    def search_notes(self, query: str, top_k: int = 8, use_graph: bool = True) -> dict[str, Any]:
        self._ensure_graph()
        result = self.rt.engine().retrieve(query, top_k=top_k, use_graph=use_graph)
        vault = self.rt.settings.display_vault_name
        return {
            "query": query,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "retrievers": result.per_retriever,
            "date_filter": str(result.date_range) if result.date_range else None,
            "full_coverage": not result.truncated,
            "results": [
                {
                    "rank": i,
                    "source": scored.chunk.citation,
                    "note_id": scored.chunk.note_id,
                    "obsidian_uri": obsidian_uri(scored.chunk.note_id, vault),
                    "note_date": (
                        scored.chunk.note_date.isoformat() if scored.chunk.note_date else None
                    ),
                    "text": scored.chunk.text,
                    "score": round(scored.score, 5),
                    "found_via": sorted(scored.components) or [scored.source],
                    "tags": sorted(scored.chunk.tags),
                }
                for i, scored in enumerate(result.chunks, start=1)
            ],
        }

    def ask_notes(self, question: str, local_only: bool = False) -> dict[str, Any]:
        self._ensure_graph()
        try:
            answer = self.rt.engine().ask(
                question, local_only=local_only or self.rt.settings.local_only
            )
        except PolicyViolation as exc:
            return {
                "error": "privacy_policy",
                "message": str(exc),
                "hint": (
                    "Cortex refused to send private notes to a provider that trains on "
                    "submitted data. Configure a no-training provider, or mark the "
                    "relevant notes public in frontmatter."
                ),
            }
        except ProviderError as exc:
            return {"error": "provider_unavailable", "message": str(exc)}

        return {
            "answer": answer.text,
            "provider": answer.provider,
            "left_this_machine": answer.escalated,
            "elapsed_ms": round(answer.elapsed_ms, 1),
            "citations": [
                {
                    "index": i,
                    "source": chunk.citation,
                    "note_id": chunk.note_id,
                    "obsidian_uri": obsidian_uri(
                        chunk.note_id, self.rt.settings.display_vault_name
                    ),
                }
                for i, chunk in enumerate(answer.citations, start=1)
            ],
        }

    def list_links(self, note_id: str) -> dict[str, Any]:
        self._ensure_graph()
        graph = self.rt.graph
        if graph is None:
            return {"error": "graph_unavailable"}
        return {
            "note_id": note_id,
            "links_to": sorted(graph.forward.get(note_id, set())),
            "linked_from": sorted(graph.backward.get(note_id, set())),
        }

    def vault_status(self) -> dict[str, Any]:
        # In-memory runtimes (test / CI) carry no live governor. Surface that
        # honestly rather than crashing on a None attr -- the schema stays
        # stable for callers regardless of runtime mode.
        if self.rt.governor is not None:
            self.rt.governor.sample(force=True)
            thermal = self.rt.governor.describe()
        else:
            thermal = {"state": "unavailable", "reason": "in_memory runtime"}
        counts = self.rt.catalog.stats()
        return {
            "vault_path": str(self.rt.settings.vault_path),
            "notes_indexed": counts["notes"],
            "chunks": counts["chunks"],
            "thermal": thermal,
            "provider_routing_private": self.rt.router.explain(Sensitivity.PRIVATE),
        }

    def reindex(self, full: bool = False) -> dict[str, Any]:
        report = self.rt.pipeline.run(full=full)
        self.rt.refresh_graph()
        self._graph_loaded = True
        return {
            "summary": report.summary(),
            "scanned": report.scanned,
            "indexed": report.indexed,
            "skipped": report.skipped,
            "deleted": report.deleted,
            "chunks_written": report.chunks_written,
            "paused_for_thermal": report.paused_for_thermal,
            "errors": [{"note": n, "error": e} for n, e in report.errors[:20]],
        }

    def remember(
        self,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write a memory note into the vault.

        The only tool in this server that mutates the vault, and it only ever
        creates a new file inside the memory folder -- never edits or deletes an
        existing note.
        """
        if self.rt.memory is None or not self.rt.memory.enabled:
            return {
                "error": "memory_disabled",
                "message": "Memory is disabled. Enable it under [memory] in cortex.toml.",
            }
        from cortex.memory import MemoryNote

        try:
            written = self.rt.memory.write(
                MemoryNote(
                    question=question,
                    answer=answer,
                    sources=list(sources or []),
                    tags=list(tags or []),
                    provider="mcp",
                )
            )
        except (OSError, ValueError) as exc:
            return {"error": "write_failed", "message": str(exc)}

        if written is None:
            return {"error": "not_written", "message": "Nothing to save."}
        return {
            "saved": written.relative_to(self.rt.settings.vault_path).as_posix(),
            "obsidian_uri": obsidian_uri(
                written.relative_to(self.rt.settings.vault_path).as_posix(),
                self.rt.settings.display_vault_name,
            ),
            "total_memory_notes": self.rt.memory.count(),
            "note": (
                "Indexed on the next reindex, after which it participates in "
                "retrieval like any other note."
            ),
        }

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "search_notes": self.search_notes,
            "ask_notes": self.ask_notes,
            "list_links": self.list_links,
            "vault_status": self.vault_status,
            "remember": self.remember,
            "reindex": self.reindex,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": "unknown_tool", "message": f"No such tool: {name}"}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": "bad_arguments", "message": str(exc)}
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return {"error": "tool_failed", "message": str(exc)}


def build_server(tools: CortexTools) -> Any:
    """Construct the MCP server. Requires the optional ``mcp`` extra.

    Uses ``MCPServer`` (the MCP 2.0 high-level API). A single ``dispatch``
    wrapper is registered for each tool name so the JSON-RPC routing stays in
    ``CortexTools.dispatch`` rather than being duplicated here.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import TextContent
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MCP support is not installed. Install with: pip install 'cortex-brain[mcp]'"
        ) from exc

    server = MCPServer("cortex")

    for spec in TOOL_DEFINITIONS:
        _name: str = spec["name"]  # type: ignore[assignment]
        _desc: str = spec["description"]  # type: ignore[assignment]

        def _make_handler(
            tool_name: str,
        ) -> Any:
            async def _handler(**kwargs: Any) -> list[TextContent]:
                payload = tools.dispatch(tool_name, kwargs)
                return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

            _handler.__name__ = tool_name
            return _handler

        server.add_tool(_make_handler(_name), name=_name, description=_desc)

    return server


def run_stdio(
    *,
    config_path: Path | None = None,
    vault: Path | None = None,
    offline: bool = False,
) -> None:
    """Run the MCP server over stdio.

    Logs go to stderr only: stdout is the JSON-RPC channel and anything written
    there corrupts the protocol.
    """
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from cortex.config import load_settings

    settings = load_settings(config_path)
    if vault is not None:
        settings.vault_path = vault.expanduser()

    runtime = build_runtime(settings=settings, offline=offline)
    tools = CortexTools(runtime)
    server = build_server(tools)

    try:
        asyncio.run(server.run_stdio_async())
    finally:
        runtime.close()
