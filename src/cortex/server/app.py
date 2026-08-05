"""FastAPI application factory.

A factory (``create_app``) rather than a module-level ``app`` so tests can
build their own app with overrides -- the ``set_runtime`` hook in
:mod:`cortex.server.dependencies` is the seam.

The bundled Vite production build is mounted at ``/`` if it is present, so a
single ``cortex serve`` starts the API and the UI in one process. When the
build is missing (e.g. on first install with no frontend/ build yet), the API
runs alone and the SPA is reached via the Vite dev server.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from cortex.config import Settings
from cortex.server import routers

logger = logging.getLogger(__name__)

DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app, optionally mounting a built SPA on ``/``."""
    from cortex.server.dependencies import get_runtime

    app = FastAPI(
        title="Cortex",
        version="0.1.0",
        description="Local-first second brain. Privacy-tiered RAG over your Obsidian vault.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    _settings = settings or get_runtime().settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_settings.serve.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Routers. Each one owns its prefix; together they form the public API.
    app.include_router(routers.chat.router)
    app.include_router(routers.search.router)
    app.include_router(routers.system.router)
    app.include_router(routers.memory.router)

    @app.get("/api/whoami")
    async def whoami() -> dict[str, object]:
        """Single-shot snapshot of who I am talking to -- the UI's startup probe."""
        from cortex.server.dependencies import get_runtime

        rt = get_runtime()
        return {
            "vault_name": rt.settings.display_vault_name,
            "vault_path": str(rt.settings.vault_path),
            "providers": [
                {
                    "name": p.spec.name,
                    "model": p.spec.model,
                    "policy": p.spec.policy.value,
                    "configured": p.configured,
                }
                for p in rt.router.providers
            ],
            "memory_enabled": bool(rt.memory is not None and rt.memory.enabled),
            "local_only": rt.settings.local_only,
        }

    _mount_frontend(app, _settings)

    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Mount the Vite production build at ``/`` if the directory exists.

    The mount deliberately happens after the API routers are registered so
    /api/* is always served by the FastAPI app -- a missing SPA build must
    never 404 an API call.
    """
    dist = DEFAULT_FRONTEND_DIST
    if not dist.exists() or not (dist / "index.html").exists():
        logger.info(
            "no frontend/dist; the web UI is not bundled. Run it from frontend/ via npm run dev."
        )
        return

    assets = dist / "assets"
    if assets.exists():
        # ``html=False`` here so a request for /assets/foo.js doesn't fall
        # through to index.html with an SPA fallback -- we *want* a 404 when a
        # hashed bundle is actually missing.
        app.mount("/assets", StaticFiles(directory=str(assets), html=False), name="static")

    @app.get("/", include_in_schema=False, response_model=None)
    async def root_index() -> Response:
        return FileResponse(str(dist / "index.html"))

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def spa_fallback(path: str) -> Response:
        # Don't shadow the API: /api/* and /assets/* are already routed.
        if path.startswith("api/") or path.startswith("assets/"):
            return JSONResponse({"error": "not_found", "path": path}, status_code=404)
        file_path = dist / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(dist / "index.html"))
