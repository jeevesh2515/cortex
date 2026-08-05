"""System endpoints.

Status and control surfaces that are not about a single query:

* ``/api/status`` -- a single snapshot of everything the UI's sidebar wants to
  know on each refresh: index size, thermal state, eligible providers, memory
  count.
* ``/api/reindex`` -- trigger the indexer. Used by a manual refresh button;
  the watcher handles the automatic case.
* ``/api/health`` -- cheap liveness probe for orchestrators and the frontend's
  reconnect loop.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from cortex.models import Sensitivity
from cortex.server.dependencies import get_runtime
from cortex.server.schemas import (
    ProviderStatus,
    ReindexRequest,
    StatusResponse,
    ThermalStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    rt = get_runtime()

    # Router.explain returns a human-facing verdict per provider -- the cheapest
    # way to know "is this provider allowed to see PRIVATE traffic" is to ask
    # the gate directly and parse the verdict. A provider with no verdict is
    # either disabled or absent from the configured chain.
    verdict_map = rt.router.explain(Sensitivity.PRIVATE)
    providers: list[ProviderStatus] = []
    for provider in rt.router.providers:
        spec = provider.spec
        verdict = verdict_map.get(spec.name, "unknown")
        eligible = verdict.startswith("eligible")
        # Pull a friendly reason out of the verdict so the UI can show the
        # *why*, not just the verdict's first word.
        if "refused:" in verdict:
            reason = verdict.split("refused:", 1)[1].strip()
        elif verdict == "disabled":
            reason = "disabled"
        else:
            reason = None
        providers.append(
            ProviderStatus(
                name=spec.name,
                model=spec.model,
                policy=spec.policy.value,
                priority=spec.priority,
                eligible_private=eligible,
                config_issue=None if eligible else verdict,
                reason=reason,
            )
        )

    if rt.governor is not None:
        rt.governor.sample(force=True)
        described = rt.governor.describe()
        power_raw = described.get("power")
        cpu_raw = described.get("cpu_speed_limit")
        bat_raw = described.get("battery_percent")
        w_raw = described.get("workers", 0)
        thermal = ThermalStatus(
            state=str(described.get("state", "unknown")),
            power=str(power_raw) if power_raw is not None else None,
            cpu_speed_limit=int(cpu_raw) if isinstance(cpu_raw, (int, str)) else None,
            battery_percent=int(bat_raw) if isinstance(bat_raw, (int, str)) else None,
            workers=int(w_raw) if isinstance(w_raw, (int, str)) else 0,
            may_backfill=bool(described.get("may_backfill", False)),
            available=True,
        )
    else:
        thermal = ThermalStatus(
            state="unavailable",
            workers=0,
            may_backfill=False,
            available=False,
            reason="in-memory runtime",
        )

    counts = rt.catalog.stats()
    memory_count = rt.memory.count() if rt.memory is not None and rt.memory.enabled else 0

    return StatusResponse(
        vault_path=str(rt.settings.vault_path),
        vault_name=rt.settings.display_vault_name,
        notes_indexed=counts.get("notes", 0),
        chunks=counts.get("chunks", 0),
        providers=providers,
        thermal=thermal,
        memory_count=memory_count,
        memory_enabled=bool(rt.memory is not None and rt.memory.enabled),
    )


@router.post("/reindex")
async def reindex(req: ReindexRequest) -> dict[str, object]:
    from cortex.ingest.pipeline import IndexReport

    rt = get_runtime()

    # Run in a worker thread so the event loop stays responsive. The pipeline
    # does blocking I/O; leaving it on the loop would freeze the whole app.
    def _run() -> IndexReport:
        return rt.pipeline.run(
            full=req.full,
            respect_thermal=not req.ignore_thermal,
        )

    report = await asyncio.to_thread(_run)
    rt.refresh_graph()

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


@router.get("/health")
async def health() -> dict[str, object]:
    rt = get_runtime()
    return {
        "ok": True,
        "vault": str(rt.settings.vault_path),
        "providers_configured": sum(1 for p in rt.router.providers if p.configured),
    }
