"""Single shared ``Runtime`` per process, lazily built from settings.

FastAPI lets us declare request handlers with type hints and resolve them via
``Depends``. Putting a single Runtime behind a singleton here means the
heavyweight graphs, embedders and reranker weights are reused across requests
without each one paying the cold-start cost -- which is most of the budget on
a fanless Air.

Test fixtures pass their own Runtime through :func:`set_runtime`, so the
production singleton never leaks into tests.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from cortex.config import Settings, load_settings
from cortex.runtime import Runtime, build_runtime

__all__ = ["get_runtime", "reset_runtime", "set_runtime"]

_lock = threading.Lock()
_runtime: Runtime | None = None
_settings: Settings | None = None


def set_runtime(runtime: Runtime | None) -> None:
    """Override the shared runtime. Tests and ``cortex serve --reload`` use this."""
    global _runtime, _settings
    with _lock:
        _runtime = runtime
        _settings = runtime.settings if runtime is not None else None


def reset_runtime() -> None:
    set_runtime(None)


def _build_default_runtime() -> Runtime:
    """Build a long-lived Runtime from the default config path.

    Called once on the first request; subsequent requests reuse it.
    """
    settings = load_settings()
    return build_runtime(settings=settings, offline=False, in_memory=False)


def get_runtime() -> Runtime:
    """Resolve the Runtime for the current request.

    Built lazily so importing this module has no cost beyond a thread lock
    -- the heaviest objects (embedder + reranker + Ollama connection) only
    come into existence on first use.
    """
    global _runtime
    if _runtime is not None:
        return _runtime
    with _lock:
        if _runtime is None:
            _runtime = _build_default_runtime()
        return _runtime


def current_settings() -> Settings:
    return get_runtime().settings


def resolve_vault(vault: str | None) -> Path:
    """Validate a ``--vault`` override path. Used by CLI/server entrypoints."""
    from cortex.cli import err_console  # local to avoid heavy Rich dep at import

    settings = current_settings()
    if vault is None:
        return settings.vault_path
    candidate = Path(vault).expanduser()
    if not candidate.exists():
        err_console.print(f"[red]Vault not found:[/red] {candidate}")
        raise ValueError(candidate)
    return candidate


def app_state_payload() -> dict[str, Any]:
    return {
        "vault_path": str(current_settings().vault_path),
        "vault_name": current_settings().display_vault_name,
    }
