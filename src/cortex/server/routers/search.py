"""Search and link-graph endpoints.

``/api/search`` is the same hybrid retrieval the CLI's ``search`` command
returns, but serialised as JSON. Used by the frontend's vault sidebar for raw
browse -- chat is the natural UI for grounded Q&A, but sometimes you'll want
to see a ranked list of the eight best passages without asking anything.

``/api/links/{note_id}`` exposes the wikilink graph forward/back edges. The
UI uses this for the "linked from" hint under each citation.

``/api/graph`` returns a curated subgraph sized for the sidebar's radial
view: top-``limit`` nodes by degree (or BFS-subgraph around ``center``),
plus the edges that connect them, plus enough metadata to centre the radial
layout and colour the outer ring by tag.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from fastapi import APIRouter, Query

from cortex.models import obsidian_uri
from cortex.server.dependencies import get_runtime
from cortex.server.schemas import (
    CitationOut,
    GraphEdge,
    GraphNode,
    GraphResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    rt = get_runtime()
    engine = rt.engine()
    rt.refresh_graph()

    result = engine.retrieve(
        req.query,
        top_k=req.top_k,
        use_graph=req.use_graph,
        sensitivity=req.sensitivity,
    )

    vault_name = rt.settings.display_vault_name
    citations = [
        CitationOut(
            index=i,
            note_id=scored.chunk.note_id,
            obsidian_uri=obsidian_uri(scored.chunk.note_id, vault_name),
            title=scored.chunk.citation,
            snippet=scored.chunk.text[:280].replace("\n", " "),
            score=round(scored.score, 5),
            tags=sorted(scored.chunk.tags),
        )
        for i, scored in enumerate(result.chunks, start=1)
    ]

    return SearchResponse(
        query=req.query,
        elapsed_ms=round(result.elapsed_ms, 1),
        date_filter=str(result.date_range) if result.date_range else None,
        retrievers=result.per_retriever,
        results=citations,
    )


@router.get("/links/{note_id:path}")
async def links(note_id: str) -> dict[str, object]:
    rt = get_runtime()
    rt.refresh_graph()
    graph = rt.graph
    if graph is None:
        return {"note_id": note_id, "links_to": [], "linked_from": []}

    return {
        "note_id": note_id,
        "links_to": sorted(graph.forward.get(note_id, set())),
        "linked_from": sorted(graph.backward.get(note_id, set())),
    }


@router.get("/graph/stats")
async def graph_stats() -> dict[str, object]:
    rt = get_runtime()
    return dict(rt.refresh_graph().stats)


@router.get("/graph", response_model=GraphResponse)
async def graph(
    limit: int = Query(60, ge=1, le=400),
    center: str | None = Query(
        None,
        description=(
            "Optional note_id to focus on. When provided, the response contains "
            "that note plus all of its one-hop neighbours (forward + backward) "
            "rather than the global top-deegree subset."
        ),
    ),
    include_embeds: bool = Query(
        True,
        description="Include transclusion edges (``![[Note]]``) separately from links.",
    ),
) -> GraphResponse:
    """Wikilink subgraph sized for the sidebar's radial view.

    The full graph on a real Obsidian vault can have thousands of nodes; sending
    them all over the wire would be cheap but unpaintable. We need a *curated*
    subgraph that shows the structure the user came for -- the hubs, the loops,
    the well-trodden paths -- without dragging in nodes that have one weakly
    connected link and would just clutter the canvas.

    The default strategy is ``top-N by degree``. ``center`` switches to a
    one-hop BFS so a user who clicks a search result can pivot from "this
    answer" to "what is connected to this answer" without re-rendering the
    whole layout.
    """
    rt = get_runtime()
    graph = rt.refresh_graph()

    # ----- degree table ---------------------------------------------------
    # Degree counts every kind of directed incident edge: outgoing links,
    # incoming backlinks, and embeds in either direction. Embeds count once
    # in degree even though they are also returned as separate edge kind, so
    # adding ``is_embed`` to an outgoing link does not inflate the node twice
    # visually -- a hub with transclusions would otherwise dwarf everything else.
    degree: dict[str, int] = defaultdict(int)
    for src, targets in graph.forward.items():
        degree[src] += len(targets)
        for tgt in targets:
            degree[tgt] += 1  # backlink weight
    for src, targets in graph.embeds.items():
        for tgt in targets:
            degree[src] += 1
            degree[tgt] += 1

    if not degree:
        # Empty vault (or only notes that are still unindexed). The UI renders
        # an empty-state, not a degenerate radial layout with one ring.
        return GraphResponse(stats=graph.stats, nodes=[], edges=[], center=None)

    # ----- node selection -------------------------------------------------
    selected: dict[str, int]  # note_id -> degree
    center_id: str | None

    if center is not None and center in degree:
        # Note present in the graph. Pull it plus all one-hop neighbours.
        centre_set: set[str] = {center}
        centre_set.update(graph.forward.get(center, set()))
        centre_set.update(graph.backward.get(center, set()))
        # Cap to ``limit`` by degree so a hive-mind hub does not drag in every
        # weakly-connected note in the vault. The hovered centre is always kept.
        if len(centre_set) > limit:
            ranked = sorted(centre_set, key=lambda nid: degree.get(nid, 0), reverse=True)
            centre_set = set(ranked[:limit])
            centre_set.add(center)
        selected = {nid: degree[nid] for nid in centre_set}
        center_id = center
    else:
        # Top-N by degree. Ties broken by note_id so two equally-connected
        # notes do not trade places between polls and visually jitter.
        ordered = sorted(
            degree.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:limit]
        selected = dict(ordered)
        center_id = ordered[0][0] if ordered else None

    # ----- node -> title resolution --------------------------------------
    titles = graph.titles or {}

    def _title(note_id: str) -> str:
        fallback = PurePosixPath(note_id).stem.replace("-", " ").replace("_", " ")
        return titles.get(note_id) or fallback

    # ----- edges within the selected subset -------------------------------
    edges: list[GraphEdge] = []
    for src in selected:
        for tgt in graph.forward.get(src, set()):
            if tgt not in selected:
                continue
            is_embed = tgt in graph.embeds.get(src, set())
            if is_embed:
                edges.append(GraphEdge(source=src, target=tgt, kind="embed"))
            elif include_embeds:
                # Only emit a plain link when embeds are wanted at all; if
                # ``include_embeds`` is False the caller wants a cleaner
                # wikilink-only picture.
                edges.append(GraphEdge(source=src, target=tgt, kind="link"))

    truncated = len(selected) >= limit and (center is None or center not in selected)

    nodes: list[GraphNode] = [
        GraphNode(
            id=nid,
            title=_title(nid),
            degree=d,
            tag=None,  # reserved for a follow-up that carries note-level tags
            is_hub=(nid == center_id),
        )
        for nid, d in sorted(
            selected.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    return GraphResponse(
        stats=graph.stats,
        nodes=nodes,
        edges=edges,
        center=center_id,
        truncated=truncated,
    )
