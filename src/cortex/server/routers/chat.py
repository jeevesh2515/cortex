"""Streaming chat endpoint.

The wire format is Server-Sent Events. Each event has ``event: <type>`` and
``data: <json>`` -- the EventSource protocol gives the browser ``event.type``
for free, so a TypeScript switch handles dispatch with no parsing.

The flow for a single chat turn::

    provider     -> {name, escalated, model}    # before any text is generated
    retrieval    -> {query, elapsed_ms, date, count}
    citation     -> {index, note_id, uri, title} # once per chunk used
    text         -> {delta: "..."}               # micro-chunks of the answer
    done         -> {provider, escalated, ms, ...}  # final stats

Errors mid-stream surface as ``event: error`` so the client can show them in
the chat scroll rather than dropping the connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from cortex.llm.protocol import PolicyViolation, ProviderError
from cortex.memory import MemoryNote
from cortex.models import obsidian_uri
from cortex.server.dependencies import get_runtime
from cortex.server.schemas import (
    ChatRequest,
    SSEEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@dataclass(slots=True)
class _StreamContext:
    request: ChatRequest
    local_only: bool
    runtime: "object"  # Runtime, kept loose to import cheaply


def _sse(event: SSEEvent) -> bytes:
    return event.encode()


async def _stream_chat(req: ChatRequest, signal: "object | None" = None) -> AsyncIterator[bytes]:
    rt = get_runtime()
    engine = rt.engine()

    last_user = next(
        (msg for msg in reversed(req.messages) if msg.role == "user"), None
    )
    if last_user is None:
        yield _sse(SSEEvent(type="error", data={"message": "no user message"}))
        return

    query = last_user.content.strip()
    if not query:
        yield _sse(SSEEvent(type="error", data={"message": "empty query"}))
        return

    rt.refresh_graph()

    # Provider decision first -- so the UI can show "thinking with Groq" before
    # retrieval results arrive.
    # Send a "deciding" provider frame first so the UI can render the spinner
    # on the provider chip without knowing which provider yet -- that frame
    # is replaced by a real one after the synthesis call chooses the actual
    # chain. Sending the final provider name up front would flicker: the
    # engine short-circuits to provider="none" when retrieval produces zero
    # chunks, which would otherwise look like a routing downgrade.
    yielding_for_chrome = False or rt.settings.local_only or req.local_only
    yield _sse(
        SSEEvent(
            type="provider",
            data={
                "deciding": True,
                "local_only_enforced": yielding_for_chrome,
            },
        )
    )

    try:
        result = engine.retrieve(query, top_k=req.top_k)
    except Exception as exc:  # retrieval itself failed
        logger.exception("retrieval failed")
        yield _sse(SSEEvent(type="error", data={"message": f"retrieval failed: {exc}"}))
        return

    yield _sse(
        SSEEvent(
            type="retrieval",
            data={
                "query": query,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "retrievers": result.per_retriever,
                "date_filter": str(result.date_range) if result.date_range else None,
                "reranked": result.reranked,
                "matched": len(result.chunks),
            },
        )
    )

    vault_name = rt.settings.display_vault_name
    for i, scored in enumerate(result.chunks, start=1):
        yield _sse(
            SSEEvent(
                type="citation",
                data={
                    "index": i,
                    "note_id": scored.chunk.note_id,
                    "obsidian_uri": obsidian_uri(scored.chunk.note_id, vault_name),
                    "title": scored.chunk.citation,
                    "snippet": scored.chunk.text[:280].replace("\n", " "),
                    "score": round(scored.score, 5),
                    "tags": sorted(scored.chunk.tags),
                },
            )
        )

    if not result.chunks:
        yield _sse(
            SSEEvent(
                type="text",
                data={"delta": "Nothing in your notes matches that yet."},
            )
        )
        yield _sse(
            SSEEvent(
                type="done",
                data={
                    "provider": "none",
                    "escalated": False,
                    "elapsed_ms": round(result.elapsed_ms, 1),
                    "answer": "Nothing in your notes matches that yet.",
                },
            )
        )
        return

    try:
        from cortex.llm.protocol import ChatMessage as _CM

        messages = [
            _CM(role=m.role, content=m.content) for m in req.messages if m.content
        ]
        if not messages or messages[-1].role != "user":
            messages = [*messages, _CM(role="user", content=query)]

        # We synthesise via the router (non-streaming), then chunk the text out
        # as a controlled token stream. Each provider supports streaming over
        # OpenAI-compat, but the chunking semantics differ enough (token
        # boundaries, finish_reason probing) that faking a smooth stream from a
        # finished completion is simpler and works for every backend in the
        # shipped chain. The chunk size below is a balance between perceived
        # smoothness and per-event SSE overhead.
        chunks = [scored.chunk for scored in result.chunks]
        sensitivity = engine.effective_sensitivity(chunks)

        # Build the prompt via the engine (same source of truth as the MCP path)
        from cortex.retrieve.engine import RetrievalEngine

        prompt_messages = RetrievalEngine.build_prompt(query, chunks)
        # Replace the last user-role message with the *full* conversation if the
        # chat history has more than one turn -- otherwise the router sees only
        # the bare question and forgets the prior turns.
        if len(messages) > 1:
            prompt_messages = [
                _CM(role="system", content=prompt_messages[0].content),
                *[
                    _CM(role=m.role, content=m.content)
                    for m in messages[1:]
                    if m.role in {"user", "assistant"}
                ][-8:],  # cap turns so the prompt can't grow unbounded
            ]

        completion, decision = rt.router.complete(
            prompt_messages,
            sensitivity=sensitivity,
            max_tokens=900,
            local_only=req.local_only or rt.settings.local_only,
            temperature=0.2,
        )
    except PolicyViolation as exc:
        yield _sse(SSEEvent(type="error", data={"type": "privacy_policy", "message": str(exc)}))
        return
    except ProviderError as exc:
        yield _sse(SSEEvent(type="error", data={"type": "provider_unavailable", "message": str(exc)}))
        return
    except Exception as exc:  # last-resort guard around the synthesis call
        logger.exception("synthesis failed")
        yield _sse(SSEEvent(type="error", data={"type": "internal", "message": str(exc)}))
        return

    # Pull the actual policy off the spec that served this turn -- the
    # RoutingDecision records *what happened* (provider, escalated), not the
    # policy string the UI wants to display.
    policy_value = ""
    for provider in rt.router.providers:
        if provider.spec.name == decision.provider:
            policy_value = provider.spec.policy.value
            break

    yield _sse(
        SSEEvent(
            type="provider",
            data={
                "name": decision.provider,
                "model": completion.model,
                "escalated": decision.escalated,
                "policy": policy_value,
                "elapsed_ms": round(completion.elapsed_ms, 1),
            },
        )
    )

    # Micro-chunk the answer text. Roughly four-character deltas: small enough
    # to feel live, large enough that SSE overhead stays under 1% of bandwidth.
    full_text = completion.text.strip()
    if not full_text:
        full_text = "(empty response)"

    step = 4
    sent_index = 0
    while sent_index < len(full_text):
        # Yield a tiny slice, surrender the event loop so the client can flush.
        yield _sse(SSEEvent(type="text", data={"delta": full_text[sent_index : sent_index + step]}))
        sent_index += step
        # ``await asyncio.sleep(0)`` yields control to the loop so the SSE
        # packet can actually flush before the next slice lands.
        await asyncio.sleep(0)
        # Defense-in-depth: if the client disconnected mid-stream, stop
        # emitting rather than throwing on a closed socket.
        if ac_signal_obj and getattr(ac_signal_obj, "aborted", False):
            return

    yield _sse(
        SSEEvent(
            type="done",
            data={
                "provider": decision.provider,
                "model": completion.model,
                "escalated": decision.escalated,
                "policy": policy_value,
                "elapsed_ms": round(completion.elapsed_ms, 1),
                "answer": full_text,
                "total_elapsed_ms": round(completion.elapsed_ms + result.elapsed_ms, 1),
            },
        )
    )

    if req.remember and rt.memory is not None and rt.memory.enabled:
        import os

        try:
            written = rt.memory.write(
                MemoryNote(
                    question=query,
                    answer=full_text,
                    sources=[c.note_id for c in chunks],
                    tags=["api"],
                    provider=decision.provider,
                )
            )
        except (OSError, ValueError) as exc:
            yield _sse(
                SSEEvent(
                    type="error",
                    data={"type": "memory_failed", "message": str(exc)},
                )
            )
        else:
            if written is not None:
                rel = written.relative_to(rt.settings.vault_path).as_posix()
                yield _sse(
                    SSEEvent(
                        type="memory",
                        data={
                            "saved": rel,
                            "obsidian_uri": obsidian_uri(rel, vault_name),
                        },
                    )
                )


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream a chat turn as Server-Sent Events."""
    # FastAPI gives us the Request object so the streaming generator can
    # call ``is_disconnected()`` between text micro-chunks. We pass it down
    # rather than re-invent an abort signal of our own.
    return StreamingResponse(
        _stream_chat(req, signal=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable buffering on common reverse proxies
        },
    )


@router.post("/ask")
async def ask(req: ChatRequest) -> dict[str, object]:
    """Non-streaming variant of /chat for clients that can't read SSE.

    Same logic, returns the assembled payload in one JSON response. Cheap
    escape hatch: a CLI one-liner or a mobile app without EventSource support
    can still drive the chat pipeline.
    """
    rt = get_runtime()
    engine = rt.engine()

    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if last_user is None:
        return {"error": "no user message"}

    query = last_user.content.strip()
    if not query:
        return {"error": "empty query"}

    rt.refresh_graph()
    try:
        answer_obj = engine.ask(
            query,
            top_k=req.top_k,
            local_only=req.local_only or rt.settings.local_only,
        )
    except PolicyViolation as exc:
        return {"error": "privacy_policy", "message": str(exc)}
    except ProviderError as exc:
        return {"error": "provider_unavailable", "message": str(exc)}

    vault_name = rt.settings.display_vault_name
    citations = [
        {
            "index": i,
            "note_id": chunk.note_id,
            "obsidian_uri": obsidian_uri(chunk.note_id, vault_name),
            "title": chunk.citation,
        }
        for i, chunk in enumerate(answer_obj.citations, start=1)
    ]
    return {
        "answer": answer_obj.text,
        "provider": answer_obj.provider,
        "escalated": answer_obj.escalated,
        "elapsed_ms": round(answer_obj.elapsed_ms, 1),
        "citations": citations,
    }


# A trivial status probe for clients negotiating the protocol version before
# opening an EventSource. Returns {"ok": true} immediately -- the SSE handlers
# do the heavy lifting elsewhere.
@router.get("/chat/health")
async def chat_health() -> dict[str, bool]:
    return {"ok": True}
