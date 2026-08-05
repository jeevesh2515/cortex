"""Pydantic schemas for the HTTP wire format.

Streaming chat uses one ``SSEEvent`` discriminant, with the payload variant
encoded in ``type``. That keeps the schema flat on the wire (a mobile JS
client can decode each event with one switch) and trivial to evolve. New event
types add new unions; existing event types stay source-compatible.

The Minkowski convention: snappy names that survive type-narrowing on the
client without us re-stating fields in TypeScript.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cortex.models import Sensitivity

# -- Requests ---------------------------------------------------------------


class ChatMessageIn(BaseModel):
    """One message in a chat exchange.

    ``role`` matches the OpenAI spelling so the same shape can be forwarded
    verbatim into any provider that takes ``messages`` directly.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str
    ts: datetime | None = None


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``.

    The full message history is replayed each turn -- the server is stateless
    across requests, which matches how the MCP handlers behave. Conversation
    memory (across turns / sessions) is the user's vault, via :meth:`remember`.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessageIn]
    local_only: bool = False
    remember: bool = False
    top_k: int = Field(8, ge=1, le=50)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(8, ge=1, le=50)
    use_graph: bool = True
    sensitivity: Sensitivity | None = None


class RememberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    sources: list[str] = []
    tags: list[str] = []
    provider: str = ""


class ReindexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full: bool = False
    ignore_thermal: bool = False


# -- Responses --------------------------------------------------------------


class CitationOut(BaseModel):
    index: int
    note_id: str
    obsidian_uri: str
    title: str
    snippet: str
    score: float
    tags: list[str] = []


class ProviderStatus(BaseModel):
    name: str
    model: str
    policy: str
    priority: int
    eligible_private: bool
    config_issue: str | None = None
    reason: str | None = None


class ThermalStatus(BaseModel):
    state: str
    power: str | None = None
    cpu_speed_limit: int | None = None
    battery_percent: int | None = None
    workers: int
    may_backfill: bool
    available: bool = True
    reason: str | None = None


class MemoryNoteOut(BaseModel):
    path: str
    title: str
    created: datetime
    question: str
    answer: str
    sources: list[str]
    obsidian_uri: str


class StatusResponse(BaseModel):
    vault_path: str
    vault_name: str
    notes_indexed: int
    chunks: int
    providers: list[ProviderStatus]
    thermal: ThermalStatus
    memory_count: int
    memory_enabled: bool


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    date_filter: str | None = None
    retrievers: dict[str, int]
    results: list[CitationOut]


class GraphNode(BaseModel):
    """One node in the wikilink graph payload sent to the UI.

    ``degree`` is the total number of edges incident on this node (forward +
    backward + embeds to/from). The sidebar view sizes circles by degree so
    visual mass corresponds to importance in the vault.

    ``tag`` is the single top-level tag carried by this note (``# foo/bar``
    becomes ``foo``), used by the radial layout to colour the outer ring. A
    note with no tags gets ``tag=None`` and renders in the neutral palette.
    """

    id: str
    title: str
    degree: int
    tag: str | None = None
    is_hub: bool = False
    """True when this node was selected as the centre of the radial layout
    (top-degree note, or the ``center`` query parameter). The UI thickens its
    ring so it reads as the focal point at a glance."""


GraphEdgeKind = Literal["link", "embed"]


class GraphEdge(BaseModel):
    """A directed edge in the wikilink graph.

    ``kind == "embed"`` is a transclusion (``![[Note]]``) -- a materially
    stronger connection than a regular link, rendered differently on the UI.
    """

    source: str
    target: str
    kind: GraphEdgeKind = "link"


class GraphResponse(BaseModel):
    """Compact payload for the sidebar's graph panel.

    Constants from the larger graph (stats, all node ids) are returned once at
    the top; the visible subgraph is a curated subset of ``limit`` nodes by
    default, expanded to include the immediate neighbours of any surface hub.
    """

    stats: dict[str, int]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    # The note_id the UI should treat as the radial centre. Equal to ``center``
    # when the client requested one, else the top-degree node. ``None`` when
    # the graph has fewer than two nodes -- nothing meaningful to centre on.
    center: str | None = None
    truncated: bool = False
    """True when more nodes exist than were included in the response."""


# -- SSE event envelope -----------------------------------------------------


ChatEventType = Literal["provider", "retrieval", "citation", "token", "text", "error", "done"]


class SSEEvent(BaseModel):
    """Discriminated union member. The full event on the wire is::

        event: <type>
        data: {<json>}

    The ``type`` lives in the ``event:`` header, *not* inside the JSON
    payload. The browser's ``EventSource`` raises the natural ``type`` for
    free on the client (``es.addEventListener("text", ...)``), so the JSON
    body is just the payload of that event -- no nested envelope to unwrap.
    """

    type: ChatEventType | Literal["status", "memory"]
    data: dict[str, Any]

    def encode(self) -> bytes:
        import json

        # ``data`` is the inner dict, not the SSEEvent itself. If we dumped
        # the model, the JSON would carry a redundant ``type`` field that the
        # frontend then has to ignore.
        return f"event: {self.type}\ndata: {json.dumps(self.data)}\n\n".encode()
