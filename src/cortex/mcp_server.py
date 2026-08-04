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

Design note: the tools returned here are read-only over the vault plus an
explicit ``reindex``. Nothing mutates a note. An agent that can silently
rewrite your second brain is a liability, and Antigravity already has file
tools if you genuinely want it editing notes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cortex.llm.protocol import PolicyViolation, ProviderError
from cortex.models import Sensitivity
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
        return {
            "query": query,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "retrievers": result.per_retriever,
            "results": [
                {
                    "rank": i,
                    "source": scored.chunk.citation,
                    "note_id": scored.chunk.note_id,
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
                {"index": i, "source": chunk.citation, "note_id": chunk.note_id}
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
        self.rt.governor.sample(force=True)
        counts = self.rt.catalog.stats()
        return {
            "vault_path": str(self.rt.settings.vault_path),
            "notes_indexed": counts["notes"],
            "chunks": counts["chunks"],
            "thermal": self.rt.governor.describe(),
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

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "search_notes": self.search_notes,
            "ask_notes": self.ask_notes,
            "list_links": self.list_links,
            "vault_status": self.vault_status,
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
    """Construct the MCP server. Requires the optional ``mcp`` extra."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MCP support is not installed. Install with: pip install 'cortex-brain[mcp]'"
        ) from exc

    server = Server("cortex")

    @server.list_tools()  # type: ignore[untyped-decorator]
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["input_schema"],
            )
            for spec in TOOL_DEFINITIONS
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        payload = tools.dispatch(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

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

    async def _serve() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        asyncio.run(_serve())
    finally:
        runtime.close()
