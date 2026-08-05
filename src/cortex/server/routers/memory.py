"""Memory endpoints.

The memory folder is where "what the system has decided to remember" lives --
Markdown notes in the vault, wikilinked to the sources they drew on, auditable
in Obsidian. The MCP server already exposes ``remember``; the HTTP variant adds
a list endpoint so the frontend can render the recent-memory strip without
having to shell out to the CLI.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from cortex.memory import MemoryNote
from cortex.models import obsidian_uri
from cortex.server.dependencies import get_runtime
from cortex.server.schemas import MemoryNoteOut, RememberRequest

router = APIRouter(prefix="/api", tags=["memory"])


@router.post("/memory")
async def remember(req: RememberRequest) -> dict[str, object]:
    rt = get_runtime()
    if rt.memory is None or not rt.memory.enabled:
        raise HTTPException(status_code=409, detail="memory disabled in config")

    try:
        written = rt.memory.write(
            MemoryNote(
                question=req.question,
                answer=req.answer,
                sources=list(req.sources),
                tags=list(req.tags) or ["api"],
                provider=req.provider,
            )
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if written is None:
        return {"saved": None, "ok": True, "note": "nothing to write"}

    rel = written.relative_to(rt.settings.vault_path).as_posix()
    return {
        "saved": rel,
        "obsidian_uri": obsidian_uri(rel, rt.settings.display_vault_name),
        "total": rt.memory.count(),
        "ok": True,
    }


@router.get("/memory/recent", response_model=list[MemoryNoteOut])
async def memory_recent(limit: int = 12) -> list[MemoryNoteOut]:
    rt = get_runtime()
    if rt.memory is None or not rt.memory.enabled:
        return []

    root = rt.memory.root
    if not root.exists():
        return []

    notes = sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out: list[MemoryNoteOut] = []
    for path in notes:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Pull the title, the question, the answer out of the rendered memory
        # note without dragging a full Markdown parser into the request path.
        title = path.stem
        created = None
        question = ""
        answer = ""
        sources: list[str] = []
        for line in text.splitlines():
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
            elif line.startswith("created:"):
                try:
                    created = datetime.fromisoformat(line.split(":", 1)[1].strip())
                except ValueError:
                    continue
            elif line.startswith("- [[") and line.endswith("]]"):
                sources.append(line[4:-2].strip())

        sections = text.split("## ", 1)
        if len(sections) == 2:
            rest = sections[1]
            # Best-effort -- the schema is ours, so the two sections exist.
            for chunk in rest.split("## "):
                head, _, body = chunk.partition("\n")
                if head.strip() == "Asked":
                    question = body.strip()
                elif head.strip() == "Answered":
                    answer = body.strip()

        rel = path.relative_to(rt.settings.vault_path).as_posix()
        # Fall back to mtime when frontmatter didn't carry `created:`, which is
        # the older persistence shape.
        if created is None:
            created = datetime.fromtimestamp(path.stat().st_mtime)
        out.append(
            MemoryNoteOut(
                path=rel,
                title=title,
                created=created,
                question=question,
                answer=answer,
                sources=sources,
                obsidian_uri=obsidian_uri(rel, rt.settings.display_vault_name),
            )
        )
    return out


@router.get("/memory/count")
async def memory_count() -> dict[str, int]:
    rt = get_runtime()
    if rt.memory is None or not rt.memory.enabled:
        return {"count": 0, "enabled": False}
    return {"count": rt.memory.count(), "enabled": True}
