"""HTTP surface for the Cortex local web app.

Mirror of :mod:`cortex.mcp_server` for a human-facing UI rather than an LLM
agent. Same Runtime, same privacy gate, same routing decisions -- the only
difference is the wire format (REST + SSE rather than JSON-RPC MCP) and the
shape of the responses (linear, ordered, friendly to a chat pane rather than
the formal tool call MCP expects).

The package is intentionally thin. Tool implementations live next to the
existing MCP tool handlers on :class:`cortex.runtime.Runtime`; the routers here
are adapters that translate HTTP requests into those calls and shape the
results for streaming.
"""

from __future__ import annotations

from cortex.server.app import create_app

__all__ = ["create_app"]
