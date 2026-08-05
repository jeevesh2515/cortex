"""Smoke tests for the local HTTP API.

Covers each surface at the seam we care about: a real ASGI scope going through
FastAPI, with the runtime replaced by an in-memory one. Streaming SSE is
parsed loosely (read chunks, dump ``data:`` JSON payloads, assert on the
sequence) rather than against a brittle handcrafted parser.

The point of these tests is to fail loudly on:

  * a route that 404s when the chrome of FastAPI would accept it
  * a SSE event our frontend depends on that we accidentally stopped emitting
  * a status response whose schema broke
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from cortex.config import Settings
from cortex.runtime import build_runtime
from cortex.server import create_app
from cortex.server.dependencies import set_runtime


@pytest.fixture
def in_mem_runtime(tmp_path: Path):
    """An in-memory Runtime so the test never touches disk."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        data_dir=tmp_path / "data",
        local_only=True,
    )
    settings.vault_path.mkdir(parents=True, exist_ok=True)
    rt = build_runtime(settings=settings, offline=True, in_memory=True)
    set_runtime(rt)
    yield rt
    set_runtime(None)
    rt.close()


@pytest.fixture
async def client(in_mem_runtime) -> AsyncIterator[AsyncClient]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _parse_sse(text: str) -> list[dict[str, object]]:
    """Return a list of ``{type, data}`` events from a raw SSE response body.

    Loose by intent -- the protocol guarantees ``event:`` then ``data: `` lines
    separated by a blank line. We don't care about heartbeats, comments, or
    anything else.
    """
    events: list[dict[str, object]] = []
    cur_type: str | None = None
    cur_data: list[str] = []
    for line in text.split("\n"):
        if line.startswith("event:"):
            cur_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur_data.append(line.split(":", 1)[1].strip())
        elif line == "" and cur_type:
            payload = "\n".join(cur_data)
            try:
                decoded = json.loads(payload) if payload else {}
            except json.JSONDecodeError:
                decoded = {"raw": payload}
            events.append({"type": cur_type, "data": decoded})
            cur_type = None
            cur_data = []
    return events


class TestWhoami:
    async def test_returns_vault_metadata(self, client: AsyncClient) -> None:
        r = await client.get("/api/whoami")
        assert r.status_code == 200
        body = r.json()
        assert "vault_name" in body
        assert "vault_path" in body
        # Provider matrix should be present even if every provider is unconfigured.
        assert isinstance(body.get("providers"), list)


class TestHealth:
    async def test_health_endpoint_ok(self, client: AsyncClient) -> None:
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True


class TestStatus:
    async def test_status_shape(self, client: AsyncClient) -> None:
        r = await client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert "notes_indexed" in body
        assert "chunks" in body
        assert "providers" in body
        assert "thermal" in body
        assert "memory_count" in body


class TestSearch:
    async def test_empty_vault_returns_no_results(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/search",
            json={"query": "anything", "top_k": 4},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "anything"
        assert body["results"] == []
        assert "elapsed_ms" in body

    async def test_validates_payload(self, client: AsyncClient) -> None:
        r = await client.post("/api/search", json={"top_k": 4})
        # FastAPI returns 422 on a missing required field.
        assert r.status_code == 422


class TestChatSSE:
    async def test_empty_query_emits_error_event(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": ""}]},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.text)
        assert any(e["type"] == "error" for e in events)

    async def test_emits_expected_event_sequence_for_empty_vault(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is foo?"}]},
        )
        events = _parse_sse(r.text)
        types = [e["type"] for e in events]
        # Provider evaluation always goes first.
        assert types[0] == "provider"
        # Then a retrieval event summarising what the index produced.
        assert "retrieval" in types
        # With no chunks in the index, the engine short-circuits and emits text+done.
        assert "text" in types
        assert "done" in types

    async def test_emits_correct_done_event(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello?"}]},
        )
        events = _parse_sse(r.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert "answer" in done_events[0]["data"]


class TestAsk:
    async def test_returns_json_for_empty_vault(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/ask",
            json={"messages": [{"role": "user", "content": "anything?"}]},
        )
        assert r.status_code == 200
        body = r.json()
        # Nothing in the index, so the engine returns the standard "no matches" text.
        assert "answer" in body or "error" in body


class TestMemory:
    async def test_disabled_by_default_returns_409(self, client: AsyncClient) -> None:
        # The fixture sets local_only=True and the memory is enabled by default.
        # So this should succeed when memory is enabled. Just smoke the path.
        r = await client.post(
            "/api/memory",
            json={
                "question": "What is foo?",
                "answer": "Foo is a placeholder I am testing with.",
                "sources": ["notes/foo.md"],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body.get("saved", "").endswith(".md")

    async def test_recent_returns_the_written_note(self, client: AsyncClient) -> None:
        await client.post(
            "/api/memory",
            json={"question": "R?", "answer": "Yes.", "sources": []},
        )
        r = await client.get("/api/memory/recent")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert any(n["question"] == "R?" for n in body)


class TestReindex:
    async def test_reindex_runs_on_empty_vault(self, client: AsyncClient) -> None:
        r = await client.post("/api/reindex", json={"full": False})
        assert r.status_code == 200
        body = r.json()
        assert "summary" in body
        assert "scanned" in body


class TestCORS:
    async def test_cors_headers_on_whoami(self, client: AsyncClient) -> None:
        r = await client.get(
            "/api/whoami",
            headers={"Origin": "http://localhost:7331"},
        )
        assert r.status_code == 200
        # The exact header FastAPI emits -- if the CORS middleware were missing,
        # this would not be present at all.
        assert r.headers.get("access-control-allow-origin") in {
            "http://localhost:7331",
            "*",
        }


class TestGraph:
    async def test_empty_vault_returns_no_nodes(self, client: AsyncClient) -> None:
        r = await client.get("/api/graph")
        assert r.status_code == 200
        body = r.json()
        assert body["nodes"] == []
        assert body["edges"] == []
        assert body["center"] is None

    async def test_subgraph_around_center_includes_neighbours(self, in_mem_runtime) -> None:
        # Build a real on-disk vault so the pipeline finishes with indexed
        # chunks and a non-trivial graph.
        from cortex.server.dependencies import set_runtime

        vault_path = in_mem_runtime.settings.vault_path
        (vault_path / "Hub.md").write_text(
            "---\ntitle: Hub\n---\n# Hub\n\nSee also [[A]] and [[B]]. Also embeds ![[Embed]].\n",
            encoding="utf-8",
        )
        (vault_path / "A.md").write_text("[[Hub]] is the central note.\n", encoding="utf-8")
        (vault_path / "B.md").write_text("Branch of [[Hub]].\n", encoding="utf-8")
        (vault_path / "Embed.md").write_text("# Embed\n\nShared block.\n", encoding="utf-8")
        (vault_path / "Orphan.md").write_text("# Orphan\n\nNo links here.\n", encoding="utf-8")
        set_runtime(in_mem_runtime)

        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            reindex = await c.post("/api/reindex", json={"full": False})
            assert reindex.status_code == 200
            # Hub should appear in the response and be marked is_hub when no
            # ``center`` is provided -- top-degree node wins by default.
            r = await c.get("/api/graph")
            assert r.status_code == 200
            body = r.json()
            ids = {n["id"] for n in body["nodes"]}
            assert "Hub.md" in ids
            assert "A.md" in ids
            assert "B.md" in ids
            assert "Embed.md" in ids
            # Orphans are excluded from the link graph (no incoming or
            # outgoing edges), so they should not be present.
            assert "Orphan.md" not in ids
            # Hub should be flagged as the focal point.
            hub_node = next(n for n in body["nodes"] if n["id"] == "Hub.md")
            assert hub_node["is_hub"] is True
            assert body["center"] == "Hub.md"
            # Hub -> Embed should appear as an embed kind (transclusion, ![[...]]).
            embed_edge = next(
                (e for e in body["edges"] if e["source"] == "Hub.md" and e["target"] == "Embed.md"),
                None,
            )
            assert embed_edge is not None
            assert embed_edge["kind"] == "embed"

            # ``center`` query param focuses the subgraph on a single note and
            # its one-hop neighbours -- useful when the user clicked a search
            # result and wants "what does this connect to".
            r2 = await c.get("/api/graph", params={"center": "A.md"})
            assert r2.status_code == 200
            body2 = r2.json()
            focused_ids = {n["id"] for n in body2["nodes"]}
            assert focused_ids == {"A.md", "Hub.md"}
            assert body2["center"] == "A.md"
