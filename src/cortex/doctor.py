"""Pre-flight diagnostics for the second-brain setup.

``cortex doctor`` runs nine small checks against the *world* -- is Ollama up,
are extras installed, is the LaunchDaemon plist present, can memory notes be
written -- and emits a single ``Ready`` or ``N issues`` summary line on top of a
detailed list. It deliberately does **not** build a full ``Runtime`` for these
checks: a user running ``cortex doctor`` for the first time may not have an
Obsidian vault configured yet, and the doctor should still surface what's
wrong so they have something to act on.
"""

from __future__ import annotations

import json
import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from cortex.config import Settings
from cortex.llm.providers import build_chat_providers
from cortex.llm.router import Router
from cortex.models import Sensitivity

logger = logging.getLogger(__name__)

__all__ = ["Check", "DoctorReport", "Status", "run_doctor"]


class Status(Enum):
    """Severity of an individual check.

    Distinct from ``exit code``: doctor exits non-zero only on ``FAIL``, never
    on ``WARN`` or ``INFO``. The colour is what users see; the enum name is the
    contract for assertions in tests.
    """

    PASS = "green"
    INFO = "blue"
    WARN = "yellow"
    FAIL = "red"

    @property
    def icon(self) -> str:
        return {Status.PASS: "✓", Status.INFO: "i", Status.WARN: "!", Status.FAIL: "✗"}[self]


@dataclass(frozen=True, slots=True)
class Check:
    """One row in the diagnostic table.

    ``hint`` is a one-line remediation; ``details`` are extra breadcrumbs on
    failed checks (the refused providers, the path that wasn't writable, etc.).
    Keep messages short enough to fit a 90-char terminal.
    """

    name: str
    status: Status
    message: str
    hint: str | None = None
    details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DoctorReport:
    """Aggregate of all checks. The single thing callers print."""

    checks: list[Check]

    @property
    def blocking(self) -> int:
        return sum(1 for c in self.checks if c.status is Status.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status is Status.WARN)

    @property
    def infos(self) -> int:
        return sum(1 for c in self.checks if c.status is Status.INFO)

    def summary_line(self) -> str:
        """Single 'Ready / N issues' line the user asked for."""
        if self.blocking == 0 and self.warnings == 0:
            return "Ready"
        total = self.blocking + self.warnings
        return f"{total} issue{'s' if total != 1 else ''}"


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _vault_summary(vault: Path) -> str:
    """Best-effort count of notes and wikilinks, never raises."""
    try:
        notes = list(vault.rglob("*.md"))
    except OSError:
        return "found"

    # Limit the wikilink scan: probing ten thousand files for "[[" on every
    # doctor invocation would be silly. The flag doesn't need to be exact.
    sample = notes[:200]
    with_links = 0
    for path in sample:
        try:
            if "[[" in path.read_text(encoding="utf-8", errors="ignore"):
                with_links += 1
        except OSError:
            continue
    return f"{len(notes)} notes, {with_links}/{len(sample)} sampled have wikilinks"


def check_config(settings: Settings) -> Check:
    # If we got a Settings object at all, it parsed without raising. Trust that.
    _ = settings  # explicit non-use to avoid 'unused' lint; presence is the check
    return Check("Config", Status.PASS, "configuration loaded")


def check_vault(settings: Settings) -> Check:
    vault = settings.vault_path
    if not vault.exists():
        return Check(
            "Vault",
            Status.FAIL,
            f"path does not exist: {vault}",
            hint="Set CORTEX_VAULT, --vault, or vault_path in cortex.toml",
        )
    if not vault.is_dir():
        return Check(
            "Vault",
            Status.FAIL,
            f"path is not a directory: {vault}",
            hint="Point vault_path to the folder containing your markdown notes",
        )
    return Check("Vault", Status.PASS, f"{vault} ({_vault_summary(vault)})")


def check_ollama(settings: Settings) -> Check:
    """Probe Ollama's ``/api/tags`` with a tight timeout.

    Uses stdlib ``urllib`` rather than the cortex ``OllamaEmbedder`` on purpose:
    that client has a 120s default timeout, which is the wrong budget for a
    diagnostic. A 2s ceiling is loud enough to catch a hung daemon and quiet
    enough to stay unnoticeable when nothing is wrong.
    """
    url = settings.ollama_url.rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Check(
            "Ollama",
            Status.FAIL,
            f"unreachable at {url}",
            hint="Start Ollama: `ollama serve`",
            details=[f"{type(exc).__name__}: {exc}"],
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return Check(
            "Ollama",
            Status.FAIL,
            "/api/tags returned a malformed response",
            details=[f"{type(exc).__name__}: {exc}"],
        )

    models = sorted({m.get("name", "") for m in payload.get("models", []) if m.get("name")})
    target = settings.embed_model
    pulled = target in models

    # ``/api/ps`` lists models currently loaded into memory -- a stronger
    # signal than /api/tags, which only reports what has been pulled. Combine
    # both into the message so the user knows whether the model is on disk or
    # resident right now. A loaded model is almost certainly callable; a
    # pulled-but-cold model has to spool on first inference, which on a
    # fanless Air can exceed the cortex 120s embed timeout when called from
    # indexing. Knowledge of either state is the right level of honesty here.
    loaded: set[str] = set()
    try:
        with urllib.request.urlopen(f"{url}/api/ps", timeout=1.0) as resp:
            loaded_payload = json.loads(resp.read())
        loaded = {m.get("name", "") for m in loaded_payload.get("models", []) if m.get("name")}
    except (OSError, ValueError, json.JSONDecodeError):
        # /api/ps is a diagnostic nicety, not a gate: ignore its failures.
        pass

    if pulled and target in loaded:
        return Check(
            "Ollama",
            Status.PASS,
            f"{target} pulled and currently loaded ({len(models)} pulled total)",
        )
    if pulled:
        return Check(
            "Ollama",
            Status.PASS,
            f"{target} pulled ({len(models)} total); not currently loaded -- first inference "
            "will spool",
        )
    return Check(
        "Ollama",
        Status.WARN,
        f"reachable but {target} not pulled",
        hint=f"Run: ollama pull {target}",
        details=[f"available: {', '.join(models[:6])}{'...' if len(models) > 6 else ''}"],
    )


# Pattern for a single assignment inside an ``Environment=`` line. systemd
# accepts whitespace-separated lists (``Environment=K1=v1 K2=v2``), so the
# outer caller splits on whitespace and feeds each token here.
_SYSTEMD_ASSIGNMENT = re.compile(r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*"?\s*$""")


def _candidate_paths() -> tuple[tuple[Path, str], ...]:
    """Resolve the per-platform ollama daemon config paths lazily.

    Building the tuple at module import time would call ``Path.home()``, which
    raises ``RuntimeError`` when ``$HOME`` is unset -- e.g. when Cortex is run
    from inside a sandbox or a system service. Running the resolution on
    each call costs ~10 stat calls total and keeps the import side-effect
    free. Tests monkeypatch ``_OLLAMA_DAEMON_PATHS`` (a snapshot of this
    tuple's signature) to inject fixtures without touching the real
    ``~/Library`` directory.
    """
    home: Path | None = None
    try:
        # Probe ``Path.home()`` AND a stat on its result in one block so a
        # sandbox with no ``$HOME`` (RuntimeError) AND a sandbox where
        # ``$HOME`` points at an unreadable directory (OSError) both fall
        # back to the more conservative system paths. Bundling them keeps
        # mypy happy with a single ``Path | None`` annotation rather than
        # two narrowed branches.
        candidate = Path.home()
        candidate.exists()
        home = candidate
    except (RuntimeError, OSError):
        home = None
    entries: list[tuple[Path, str]] = []
    if home is not None:
        # macOS — Homebrew user-scope daemon (foreground or background).
        entries.append((home / "Library/LaunchAgents/homebrew.mxcl.ollama.plist", "plist"))
        # macOS — official Ollama.app install.
        entries.append((home / "Library/LaunchAgents/com.ollama.ollama.plist", "plist"))
        # Linux — systemd user-scoped unit.
        entries.append((home / ".config/systemd/user/ollama.service", "systemd"))
    # Systemwide installs (no $HOME dependency).
    entries.append((Path("/Library/LaunchDaemons/homebrew.mxcl.ollama.plist"), "plist"))
    entries.append((Path("/Library/LaunchDaemons/com.ollama.ollama.plist"), "plist"))
    entries.append((Path("/etc/systemd/system/ollama.service"), "systemd"))
    entries.append((Path("/lib/systemd/system/ollama.service"), "systemd"))
    return tuple(entries)


OLLAMA_DAEMON_PATHS: tuple[tuple[Path, str], ...] = _candidate_paths()


def _read_plist_env(path: Path) -> dict[str, str]:
    """Return a plist's ``EnvironmentVariables`` dict as plain strings.

    Reads binary or XML plists uniformly. Returns an empty dict if the file
    has no env block. We never raise: a crash here would mask the more useful
    fact that the file is unparseable, which the caller records as FAIL.
    """
    try:
        with path.open("rb") as fh:
            data = plistlib.load(fh)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return {}
    env = data.get("EnvironmentVariables")
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items()}


_SYSTEMD_ENV_PREFIX = re.compile(r"^\s*Environment\s*=\s*(.*)$")
# Inline comments must be preceded by whitespace so values containing ``#``
# (e.g. ``VAR=val#tag``) survive intact.
_SYSTEMD_INLINE_COMMENT = re.compile(r"\s+#.*$")


# Pattern for an entry inside a ``launchctl print`` environment block. The
# launchd output uses ``KEY => VALUE`` (rocket arrow) rather than ``KEY=VALUE``.
_LAUNCHCTL_KV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=>\s*(.*?)\s*$")


def _launchctl_braces_open(line: str) -> bool:
    """True if *line* opens a launchd environment block (``... = {``)."""
    return line.rstrip().endswith("{")


def _is_launchctl_block_close(line: str) -> bool:
    """True if *line* is a closing brace on its own."""
    return line.strip() == "}"


def _parse_launchctl_env_blocks(output: str) -> dict[str, str]:
    """Parse every ``KEY => VALUE`` line inside launchctl environment blocks.

    ``launchctl print`` emits up to three blocks -- inherited environment,
    default environment, environment -- each opened by ``<name> = {`` and
    closed by ``}``. We merge all entries into a single dict with later keys
    winning; this matches what the running process actually sees.
    """
    env: dict[str, str] = {}
    in_block = False
    for line in output.splitlines():
        if _launchctl_braces_open(line):
            in_block = True
            continue
        if _is_launchctl_block_close(line):
            if in_block:
                in_block = False
            continue
        if not in_block:
            continue
        match = _LAUNCHCTL_KV_LINE.match(line)
        if match is None:
            continue
        env[match.group(1)] = match.group(2)
    return env


def _safe_run(args: tuple[str, ...], timeout: float) -> str | None:
    """Run *args* via ``subprocess.run`` and return stdout.

    Returns ``None`` for any failure mode the diagnostic must treat as
    "could not probe": missing binary, OS error, timeout, non-zero exit.
    Never raises.
    """
    try:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


def _probe_launchctl_live() -> dict[str, str] | None:
    """Read the running ollama daemon's environment via ``launchctl print``.

    Tries every conventional label for the homebrew and official Ollama.app
    installs, in user scope first (the homebrew default) and then system
    scope (LaunchDaemon installs). Returns the merged env dict if at least
    one target parses; ``None`` when no ollama service is loaded or the
    tool is unavailable.
    """
    # ``_live_ollama_env`` only routes to this branch when
    # ``sys.platform == 'darwin'``; ``os.getuid`` is always available on
    # macOS. The unguarded call keeps mypy happy -- wrapping in a try/except
    # for OSError is unreachable on the platforms we actually run on.
    uid = os.getuid()
    targets: tuple[str, ...] = (
        f"gui/{uid}/homebrew.mxcl.ollama",
        f"gui/{uid}/com.ollama.ollama",
        "system/homebrew.mxcl.ollama",
        "system/com.ollama.ollama",
    )
    for target in targets:
        out = _safe_run(("launchctl", "print", target), timeout=3.0)
        if out is None:
            continue
        env = _parse_launchctl_env_blocks(out)
        # ``_safe_run`` already gates on non-zero exit code / empty stdout,
        # so we trust launchctl's word: if it printed anything, we read it.
        # An empty dict here means "the daemon is up but no env landed in
        # any of the three blocks" -- treat that as a successful probe,
        # not a failure, because users care about *negative* results too
        # (``launchctl unsetenv OLLAMA_NUM_GPU``).
        return env
    return None


def _parse_systemctl_env_lines(output: str) -> dict[str, str]:
    """Parse every ``Environment=KEY=value ...`` line from systemctl output.

    Each line begins with ``Environment=`` and contains a space-separated
    list of ``KEY=value`` tokens. We reuse the systemd tokeniser regex so
    static-file and live-probe semantics stay in lock-step.
    """
    env: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Environment="):
            continue
        body = stripped[len("Environment=") :]
        for token in body.split():
            token = token.strip('"')
            match = _SYSTEMD_ASSIGNMENT.match(token)
            if match is None:
                continue
            env[match.group(1)] = match.group(2)
    return env


def _probe_systemctl_live() -> dict[str, str] | None:
    """Read the running ollama daemon's environment via ``systemctl show``.

    Tries the conventional unit names. Returns ``None`` if no ollama unit
    is loaded or ``systemctl`` is unavailable.
    """
    for unit in ("ollama.service", "homebrew-mxcl-ollama.service"):
        # ``--property=Environment`` asks systemd for the merged env it would
        # pass to ``ExecStart=``; ``--no-pager`` keeps the output unframed.
        argv = ("systemctl", "show", unit, "--property=Environment", "--no-pager")
        out = _safe_run(argv, timeout=3.0)
        if out is None:
            continue
        # ``_safe_run`` already gates on non-zero exit / empty stdout. An
        # empty dict from the parser means systemd didn't set Environment=
        # for this unit -- still a successful probe, not a failure.
        return _parse_systemctl_env_lines(out)
    return None


def _live_ollama_env() -> dict[str, str] | None:
    """Probe the *running* ollama daemon's actual environment.

    macOS reads via ``launchctl print``; Linux via ``systemctl show``. We
    return the merged env dict for any successful probe, ``None`` if no
    probe picked up an ollama service or the platform has none of these
    tools. Catches drift between the plist/unit value and the env the
    kernel actually launched the daemon with (``brew services restart``
    not run after editing the plist, ``launchctl setenv`` overrides,
    shell-launched ``ollama serve``).
    """
    # Widening the platform value to ``str`` keeps mypy from collapsing
    # this function to "always returns a probe" and flagging the defensive
    # ``return None`` below as unreachable. ``sys.platform`` is typed as a
    # Literal on supported interpreters.
    platform: str = sys.platform
    if platform == "darwin":
        return _probe_launchctl_live()
    if platform.startswith("linux"):
        return _probe_systemctl_live()
    # The remaining ``sys.platform`` values (``win32``, ``freebsd``, ...)
    # have neither launchctl nor systemd; surfacing ``None`` lets the static
    # check still report without making this an FAIL.
    return None


def _read_systemd_env(path: Path) -> dict[str, str]:
    """Parse ``Environment=KEY=value`` lines from a systemd unit.

    systemd accepts a *whitespace-separated* list per line, and a value may
    contain escaped whitespace (``\\\\ ``) or quoted segments. We handle the
    common forms:

    * ``Environment=KEY=value``           → ``{KEY: value}``
    * ``Environment="KEY=value"``         → ``{KEY: value}``
    * ``Environment=K1=v1 K2=v2``         → ``{K1: v1, K2: v2}``
    * trailing ``# comment`` (whitespace-prefixed) stripped
    * line continuations (``\\\\``) joined before parsing

    We deliberately do *not* follow ``EnvironmentFile=`` directives: doing so
    would mean reading additional files and any ambiguity there would be a
    misleading signal. The doctor's job is to surface whether the user
    *explicitly* disabled GPU -- which they generally do inline. The
    EnvironmentFile caveat is surfaced in the user-visible message instead.
    """
    env: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return env

    # Join systemd line continuations first so a single logical Environment=
    # line spans multiple physical lines.
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped.rstrip("\\")
            continue
        if buffer:
            buffer += stripped
            joined.append(buffer)
            buffer = ""
        else:
            joined.append(stripped)
    if buffer:
        joined.append(buffer)

    for line in joined:
        # Comment stripping only past whitespace, so ``KEY=foo#bar`` survives.
        line = _SYSTEMD_INLINE_COMMENT.sub("", line)
        prefix_match = _SYSTEMD_ENV_PREFIX.match(line)
        if prefix_match is None:
            continue
        # Each token may itself be ``KEY=value`` or quoted ``"KEY=value"``.
        # We don't fully implement systemd's tokeniser (multi-quote, embedded
        # spaces) -- a best-effort split on whitespace outside quotes is
        # enough for OLLAMA_NUM_GPU=0, which never contains spaces or quotes.
        tail = prefix_match.group(1)
        for token in tail.split():
            token = token.strip('"')
            match = _SYSTEMD_ASSIGNMENT.match(token)
            if match is None:
                continue
            key, value = match.group(1), match.group(2)
            # Last assignment wins, matching systemd override semantics.
            env[key] = value
    return env


@dataclass(frozen=True, slots=True)
class _OllamaSupervisor:
    """Snapshot of where and how OLLAMA_NUM_GPU is set on this machine.

    Shared by ``check_gpu_path`` (writes the supervisor-side WARN) and
    ``check_inference_path`` (writes the ollama-ps cross-check WARN). The
    boolean ``disabled`` is True iff *any* source -- plist, systemd unit,
    launchctl print, or systemctl show -- sets OLLAMA_NUM_GPU to a falsy
    value. ``UNSET`` here means the source could not be probed, not that
    the value is missing.
    """

    static_value: str | None  # value from plist / unit, or None if no key
    static_path: Path | None  # path we read from
    static_kind: str | None  # "plist" or "systemd" for the file we read
    static_file_present: bool  # was any candidate file found?
    live_value: str | None  # value from running daemon env
    live_probe_succeeded: bool  # False if launchctl/systemctl call failed

    @property
    def disabled(self) -> bool:
        """True iff any source we got evidence from says 0 / falsy.

        A successful probe that finds no override contributes ENABLED;
        a failed probe is treated as no evidence. If static says 0 the
        supervisor is disabled even when the live probe fails, because the
        on-disk config is the user's stated intent.
        """
        return (self.static_value is not None and self.static_value == "0") or (
            self.live_value is not None and self.live_value == "0"
        )

    @property
    def evidence(self) -> bool:
        """True iff we have any successful probe, regardless of findings.

        "Probed the supervisor and didn't find the OLLAMA_NUM_GPU key"
        is itself evidence: the supervisor is policy-enabled by default
        when no override exists. We surface a contradiction against
        ``ollama ps`` only when we have at least one successful probe;
        if every probe failed, the doctor says ``INFO`` and asks the
        user to re-run.
        """
        return self.static_file_present or self.live_probe_succeeded


def _read_ollama_supervisor(_settings: Settings) -> _OllamaSupervisor:
    """Probe plist + unit + launchctl + systemctl for OLLAMA_NUM_GPU.

    First existing file wins -- same semantics as ``check_gpu_path``:
    once we find a real file on disk we trust its full env block,
    including the absence of the OLLAMA_NUM_GPU key, as a statement of
    intent. Scanning further would silently shadow the user's
    user-scope plist with a system-wide one. Never raises.
    """
    static_path: Path | None = None
    static_kind: str | None = None
    static_value: str | None = None
    static_file_present = False
    for path, kind in globals()["OLLAMA_DAEMON_PATHS"]:
        try:
            if not path.exists():
                continue
        except OSError:
            continue
        static_file_present = True
        env = _read_plist_env(path) if kind == "plist" else _read_systemd_env(path)
        static_path = path
        static_kind = kind
        static_value = env.get("OLLAMA_NUM_GPU")
        break

    live_probe_succeeded = True
    live_value: str | None
    _live_env = _live_ollama_env()
    if _live_env is None:
        live_value = None
        live_probe_succeeded = False
    else:
        live_value = _live_env.get("OLLAMA_NUM_GPU")

    return _OllamaSupervisor(
        static_value=static_value,
        static_path=static_path,
        static_kind=static_kind,
        static_file_present=static_file_present,
        live_value=live_value,
        live_probe_succeeded=live_probe_succeeded,
    )


# Sentinel for which compute path a loaded model is on, as exposed by
# ``ollama ps``'s PROCESSOR column. We classify by *qualitative* path,
# not by reading percentage fields -- Ollama has shipped several formats
# ("100% GPU", "32% GPU / 68% CPU", "GPU") and parsing all of them is
# brittle.
class _ModelProcessorKind(Enum):
    GPU = "gpu"  # any GPU contribution
    CPU = "cpu"  # any CPU contribution
    MIXED = "mixed"  # partial offload; both GPU and CPU mentioned
    UNKNOWN = "unknown"


_OLLAMA_PS_PROC_PATTERN = re.compile(r"(\d+)\s*%\s*(GPU|CPU)")


def _parse_ollama_ps_processor(proc_str: str) -> _ModelProcessorKind:
    """Classify an ``ollama ps`` PROCESSOR cell.

    The format Ollama 0.32 emits is ``100% GPU`` for typical cases, but
    partial offload prints ``48% GPU / 52% CPU``. Some earlier releases
    just wrote ``GPU`` or ``CPU``. The presence of *both* terms is
    reported as MIXED so the doctor can flag partial offload
    separately from a clean GPU or CPU run.
    """
    norm = proc_str.strip().lower()
    if not norm:
        return _ModelProcessorKind.UNKNOWN
    has_gpu = "gpu" in norm
    has_cpu = "cpu" in norm
    if has_gpu and has_cpu:
        return _ModelProcessorKind.MIXED
    if has_gpu:
        return _ModelProcessorKind.GPU
    if has_cpu:
        return _ModelProcessorKind.CPU
    return _ModelProcessorKind.UNKNOWN


# Classifies a PROCESSOR cell on a single line. Reads ``100% GPU``,
# ``50% GPU / 50% CPU``, ``48% GPU / 52% CPU``, ``GPU``, ``CPU`` and
# any similar textual form. The regex is conservative: we anchor on
# word boundaries so ``CUSTOM`` is not accidentally matched as ``CPU``.
_OLLAMA_PS_PROCESSOR_FRAGMENT = re.compile(
    r"(?<![\w/])(\d+\s*%\s*GPU(?:\s*/\s*\d+\s*%\s*CPU)?"
    r"|\d+\s*%\s*CPU(?:\s*/\s*\d+\s*%\s*GPU)?"
    r"|\d+\s*%\s*GPU"
    r"|\d+\s*%\s*CPU"
    r"|GPU"
    r"|CPU)(?![\w/])"
)


def _parse_ollama_ps_rows(output: str) -> list[dict[str, str]]:
    """Parse ``ollama ps`` into one dict per data row.

    Returns entries keyed by ``NAME`` and ``PROCESSOR`` -- the only
    columns ``check_inference_path`` reads. ``UNTIL`` is multi-word on
    recent ollama builds, so cropping the row by token index would
    silently misclassify; we crop by *regex match* against the
    PROCESSOR fragment and let the first whitespace token carry the
    name. Header rows are recognised by the literal ``PROCESSOR``
    column name and skipped.
    """
    rows: list[dict[str, str]] = []
    for raw in output.splitlines():
        stripped = raw.strip()
        if not stripped or "PROCESSOR" in stripped.split():
            # Header row or blank.
            continue
        first_space = stripped.find(" ")
        if first_space == -1:
            # Single-token row -- treat it as a NAME with no PROCESSOR.
            rows.append({"NAME": stripped, "PROCESSOR": ""})
            continue
        name = stripped[:first_space].strip()
        proc_match = _OLLAMA_PS_PROCESSOR_FRAGMENT.search(stripped)
        proc = proc_match.group(1).strip() if proc_match else ""
        rows.append({"NAME": name, "PROCESSOR": proc})
    return rows


def _ollama_binary_path() -> str | None:
    """Return the absolute path of ``ollama`` on PATH, or ``None``."""
    return shutil.which("ollama")


def check_inference_path(settings: Settings) -> Check:
    """Cross-check ``ollama ps`` against the supervisor's GPU policy.

    Surfaces two classes of failure the static probe can't catch:

    1. The supervisor says GPU is *enabled* (no override) but every
       loaded model reports zero GPU layers. This is the canonical
       symptom of a Metal/CUDA backend crash, a model too big for VRAM,
       a transient service exception, or a num_layers override that
       only kicks in at run-time.
    2. The supervisor says GPU is *enabled* and at least one model
       still uses some CPU -- typical of partial offload when VRAM is
       tight. We surface this so the user knows inference isn't getting
       the full speed-up they expected.

    If ollama isn't installed, the daemon isn't reachable, or no model
    is loaded, we degrade gracefully to INFO so this check never
    pretends to know something we don't.
    """
    if _ollama_binary_path() is None:
        return Check(
            "Inference path",
            Status.INFO,
            "ollama binary not on PATH",
            hint=(
                "Install Ollama from https://ollama.com to enable "
                "runtime inference path cross-check."
            ),
        )

    out = _safe_run(("ollama", "ps"), timeout=3.0)
    if out is None:
        return Check(
            "Inference path",
            Status.INFO,
            "ollama ps did not respond",
            hint="Is `ollama serve` running?",
        )

    rows = _parse_ollama_ps_rows(out)
    if not rows:
        return Check(
            "Inference path",
            Status.PASS,
            "no models currently loaded",
            details=[
                "ollama ps returned the header row but no data",
                "load a model to engage the runtime cross-check",
            ],
        )

    has_gpu_only = False
    has_cpu_only = False
    has_mixed = False
    has_unknown_proc = False
    per_model_kinds: list[tuple[str, _ModelProcessorKind]] = []
    for row in rows:
        name = row.get("NAME", "<unknown>").strip()
        proc = row.get("PROCESSOR", "").strip()
        kind = _parse_ollama_ps_processor(proc)
        per_model_kinds.append((name, kind))
        if kind is _ModelProcessorKind.GPU:
            has_gpu_only = True
        elif kind is _ModelProcessorKind.CPU:
            has_cpu_only = True
        elif kind is _ModelProcessorKind.MIXED:
            has_mixed = True
        else:
            has_unknown_proc = True

    supervisor = _read_ollama_supervisor(settings)
    details = [
        f"{name}: {row.get('PROCESSOR', '?')}"
        for (name, _kind), row in zip(per_model_kinds, rows, strict=False)
    ]  # Cross-check against supervisor policy.
    if supervisor.disabled:
        # Supervisor says CPU. *Any* GPU contribution while the user has
        # explicitly disabled GPU is the contradiction we want to
        # surface -- even if the same ollama ps output also lists CPU-
        # only or partial-offload models (mixed-model load). On Apple
        # Silicon this is the misreporting symptom; on Linux it's a
        # misread of the disabled key.
        if has_gpu_only:
            return Check(
                "Inference path",
                Status.WARN,
                "OLLAMA_NUM_GPU=0 (supervisor) but loaded models report GPU compute",
                hint=(
                    "The supervisor explicitly disabled GPU, but ollama ps "
                    "shows the loaded model still using GPU. On Apple "
                    "Silicon with unified memory this may be a misleading "
                    "report: verify with `time ollama run <model>` for "
                    "actual inference latency. If the report proves wrong, "
                    "this is the symptom of ollama misreporting on your "
                    "Ollama build."
                ),
                details=[
                    *details,
                    "supervisor: GPU disabled",
                    "loaded model(s): GPU compute reported",
                    "verify with `time ollama run <model>` for actual latency",
                ],
            )
        if has_cpu_only or has_mixed:
            return Check(
                "Inference path",
                Status.PASS,
                "loaded models consistent with OLLAMA_NUM_GPU=0",
                details=details,
            )
        return Check(
            "Inference path",
            Status.PASS,
            "ollama ps reports no recognisable PROCESSOR for loaded models",
            details=details,
        )

    # Supervisor not disabled (no override found). Default expectation is
    # that loaded models should run on GPU.
    if not supervisor.evidence:
        # Couldn't even tell what supervisor says. Don't pretend to know.
        return Check(
            "Inference path",
            Status.INFO,
            "ollama ps available but supervisor state unknown",
            hint=(
                "Could not read either the ollama daemon config or "
                "`launchctl print`. Re-run with verbose logs."
            ),
            details=details,
        )

    if has_cpu_only and not has_gpu_only:
        # The classic "GPU acceleration silently failed" case: the
        # supervisor says enabled, the model loaded, but it's running
        # entirely on CPU. The user typically had no idea.
        return Check(
            "Inference path",
            Status.WARN,
            "loaded models report zero GPU compute despite no NUM_GPU override",
            hint=(
                "GPU acceleration has silently fallen back to CPU. Common "
                "causes: a Metal/CUDA backend crash at startup, a model "
                "exceeding VRAM, an environment variable from a host-side "
                "launchctl deeper than the daemon's own env, or a build of "
                "Ollama that doesn't ship a working backend. Verify the "
                "running log under /opt/homebrew/var/log/ollama.log for "
                "Metal/CUDA compilation errors."
            ),
            details=[
                *details,
                "supervisor: GPU enabled (no OLLAMA_NUM_GPU override)",
                "loaded model(s): CPU compute reported",
                "fixing ollama's backend typically returns ~5x throughput",
            ],
        )

    if has_mixed:
        # Partial offload. The supervisor says GPU but the model spills
        # some layers onto CPU -- the user is paying wall-clock latency
        # they didn't sign up for.
        return Check(
            "Inference path",
            Status.WARN,
            "loaded models use partial CPU compute (offload)",
            hint=(
                "Some layers spilled to CPU -- inference will be slower "
                "than a fully GPU-loaded model. Either shrink the context "
                "size, drop the offload floor, or run a smaller quant."
            ),
            details=[
                *details,
                "supervisor: GPU enabled (no OLLAMA_NUM_GPU override)",
                "loaded model(s): partial GPU/CPU compute",
            ],
        )

    if has_unknown_proc and not has_cpu_only and not has_gpu_only and not has_mixed:
        return Check(
            "Inference path",
            Status.PASS,
            "ollama ps reported no recognisable PROCESSOR for loaded models",
            details=details,
        )

    return Check(
        "Inference path",
        Status.PASS,
        "all loaded models compute on GPU (consistent with no supervisor override)",
        details=details,
    )


def check_gpu_path(_settings: Settings) -> Check:
    """Warn when ``OLLAMA_NUM_GPU`` is set in the ollama daemon config.

    `OLLAMA_NUM_GPU=0` is the standard workaround for the Apple-M5 / Tahoe
    Metal-shader JIT hang: it forces llama.cpp onto the slower compute
    buffers and unblocks inference, at the cost of ~2-5x latency. A user
    who *configured* that override months ago may have forgotten it's still
    active; this check makes the consequence visible every time they run
    ``cortex doctor``.

    We compare *two* sources of truth:

    * **Static** -- the plist on disk or the systemd unit file. This is
      what the user typically edits.
    * **Live** -- ``launchctl print`` on macOS, ``systemctl show
      --property=Environment`` on Linux. This is what the running
      daemon actually loaded. The two can drift: a missed ``brew
      services restart``, a transient ``launchctl setenv``, or a manual
      ``ollama serve`` launched from a shell.

    Severity ladder:

      * static file present, key unset, live daemon also unset
        -> PASS (default GPU acceleration)
      * static file absent, live daemon also unset
        -> INFO (no evidence either way)
      * static value present, live value present and equal
        -> WARN, hint about Metal bug
      * static value present, live value differs
        -> WARN, drift explanation
      * static value present, live value unknown (probe failed)
        -> WARN, hint to ``brew services restart``
      * static value absent, live value present
        -> WARN, hint about transient ``launchctl setenv`` / shell
    """
    last_err: str | None = None
    # Look the snapshot up via ``globals()`` so tests that monkeypatch the
    # module attribute ``OLLAMA_DAEMON_PATHS`` are honoured. We're not
    # caching the path list across calls because that would freeze
    # ``Path.home()`` against the module-import snapshot -- defeating the
    # purpose of building the tuple lazily inside ``_candidate_paths()``.
    # The semantics are *first existing file wins*: the user-installed plist
    # precedes system-wide plists, and once we have a real file on disk we
    # trust its env block alone -- a later candidate does not silently
    # shadow it. This matches the original doctor contract and the test
    # ``test_first_existing_path_wins``.
    static_reading: tuple[Path, str, dict[str, str]] | None = None
    for path, kind in globals()["OLLAMA_DAEMON_PATHS"]:
        try:
            exists = path.exists()
        except OSError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        if not exists:
            continue
        env = _read_plist_env(path) if kind == "plist" else _read_systemd_env(path)
        static_reading = (path, kind, env)
        break
    static_file_present = static_reading is not None
    static_source: tuple[str, Path, str] | None = None
    static_path: Path | None
    static_value: str | None
    if static_reading is not None:
        spath, skind, senv = static_reading
        svalue = senv.get("OLLAMA_NUM_GPU")
        if svalue is not None:
            static_source = (svalue, spath, skind)
        # First-existing-file wins: keep the path for the PASS message
        # even when the env does not set the key, so the user can see
        # which file we inspected.
        static_path = spath
        static_value = svalue
        static_label = f"{spath.name} ({skind})"
    else:
        static_path = None
        static_value = None
        static_label = "(no static config)"

    # Probe the running daemon env. Tolerate every failure mode -- the
    # static source alone is useful enough that we shouldn't fail the whole
    # check on a hiccup in ``launchctl`` or ``systemctl``.
    live_value: str | None
    live_probe_succeeded = True
    _live_env = _live_ollama_env()
    if _live_env is None:
        live_value = None
        live_probe_succeeded = False
    else:
        live_value = _live_env.get("OLLAMA_NUM_GPU")

    if static_source is not None and live_value is not None:
        static_value, static_path, _static_kind = static_source
        if static_value == live_value:
            return Check(
                "GPU path",
                Status.WARN,
                f"OLLAMA_NUM_GPU={static_value} --- inference is CPU-bound "
                f"(static and live daemon agree)",
                hint=(
                    "CPU-only inference is the standard workaround for the Apple "
                    "M5 / macOS Metal shader JIT hang. Remove OLLAMA_NUM_GPU "
                    f"from {static_path} once llama.cpp fixes the upstream bug. "
                    "(EnvironmentFile= contents are not consulted by this check.)"
                ),
                details=[
                    f"static: {static_label}",
                    f"live: {live_value!r}",
                    "static and live match: both CPU-bound",
                ],
            )
        # Drift between static and live.
        return Check(
            "GPU path",
            Status.WARN,
            f"drift between static ({static_value!r}) and live ({live_value!r})",
            hint=(
                f"Static config in {static_path} says OLLAMA_NUM_GPU={static_value} "
                f"but the running daemon has OLLAMA_NUM_GPU={live_value}. Restart "
                "the daemon (`brew services restart ollama`) to bring them into "
                "sync, or fix whichever value is correct. (EnvironmentFile= "
                "contents are not consulted by this check.)"
            ),
            details=[
                f"static: {static_label}",
                f"live: {live_value!r}",
                "static and live disagree --- one of the two needs a refresh",
            ],
        )

    if static_source is not None and live_value is None:
        static_value, static_path, _static_kind = static_source
        if live_probe_succeeded:
            return Check(
                "GPU path",
                Status.WARN,
                f"OLLAMA_NUM_GPU={static_value} set in {static_path.name} but "
                "live daemon env lacks the override",
                hint=(
                    f"`brew services restart ollama` (or the platform equivalent) "
                    f"to apply {static_path} to the running process. "
                    "(EnvironmentFile= contents are not consulted by this check.)"
                ),
                details=[
                    f"static: {static_label}",
                    "live: no override",
                    "static override not yet picked up by the running daemon",
                ],
            )
        # Probe failed badly enough that we couldn't tell; degrade to a
        # softer WARN without pretending we know the live state.
        return Check(
            "GPU path",
            Status.WARN,
            f"OLLAMA_NUM_GPU={static_value} --- inference is CPU-bound",
            hint=(
                "CPU-only inference is the standard workaround for the Apple "
                "M5 / macOS Metal shader JIT hang. Remove OLLAMA_NUM_GPU from "
                f"{static_path} once llama.cpp fixes the upstream bug. "
                "Could not probe the live daemon env to confirm it picked "
                "the override up; the value here reflects the file on disk. "
                "(EnvironmentFile= contents are not consulted by this check.)"
            ),
            details=[
                f"static: {static_label}",
                "live probe: failed or unavailable",
            ],
        )

    if static_source is None and live_value is not None:
        # Transient override: live has it, no static config backs it.
        return Check(
            "GPU path",
            Status.WARN,
            f"live daemon env has OLLAMA_NUM_GPU={live_value} but no static config backs it",
            hint=(
                "This is consistent with a *transient* override: `launchctl "
                "setenv OLLAMA_NUM_GPU=` (macOS) or a shell-launched `ollama "
                "serve` (Linux). Transient overrides do not survive logout, "
                "reboot, or service restart. Persist the setting in a plist "
                "or systemd unit if you want it to stick. "
                "(EnvironmentFile= contents are not consulted by this check.)"
            ),
            details=[
                f"static: {static_label}",
                f"live: {live_value!r}",
                "transient override without a backing config file",
            ],
        )

    # Both unset. Distinguish "we never found a config or live daemon" (INFO)
    # from "config exists but does not set the key, and live also has no
    # override" (PASS).
    if not static_file_present and not live_probe_succeeded:
        if last_err is not None:
            return Check(
                "GPU path",
                Status.FAIL,
                "could not probe any ollama daemon config path",
                details=[last_err],
            )
        return Check(
            "GPU path",
            Status.INFO,
            "no ollama daemon config detected; cannot verify OLLAMA_NUM_GPU override",
            hint=(
                "If you manage ollama manually with `OLLAMA_NUM_GPU` or via "
                "EnvironmentFile=, the diagnostic won't see it."
            ),
        )

    detail_lines = ["EnvironmentFile= is not read by this check; inline Environment= is."]
    if static_source is not None:
        detail_lines.insert(0, f"static: {static_label}")
    if live_probe_succeeded:
        detail_lines.append("live: no override")
    else:
        detail_lines.append("live probe: failed or unavailable")
    return Check(
        "GPU path",
        Status.PASS,
        (
            f"default GPU acceleration ({static_path.name} has no OLLAMA_NUM_GPU; "
            "live daemon env also unset)"
            if static_path is not None
            else (
                "default GPU acceleration (no static config file found; live daemon env also unset)"
            )
        ),
        details=detail_lines,
    )


def check_vector_store(_settings: Settings) -> Check:
    try:
        import lancedb  # noqa: F401
    except ImportError:
        return Check(
            "Vector store",
            Status.WARN,
            "LanceDB not installed; index will be in-memory and lost on restart",
            hint="Install with: pip install 'cortex-brain[index]'",
        )
    return Check("Vector store", Status.PASS, "LanceDB installed")


def check_mcp_wire(_settings: Settings) -> Check:
    """Instantiate a Server without wiring it to a transport.

    The MCP library version that ships with Cortex may not match the one an
    end user installs; the *constructor API* is what we care about, not the
    transport. Catching the crash here turns "MCP silently no-ops" into a loud
    diagnostic.
    """
    try:
        from mcp.server import Server
    except ImportError:
        return Check(
            "MCP",
            Status.INFO,
            "mcp extra not installed",
            hint="Install with: pip install 'cortex-brain[mcp]' if you want MCP integration",
        )

    # Real wire-format regressions are caught by what handlers the Server
    # constructor accepts, not by whether ``Server("name")`` alone builds. We
    # construct a Server with the on_list_tools / on_call_tool kwargs the
    # actual cortex MCP server uses; if the installed mcp version has drifted
    # far enough those kwargs are no longer accepted, doctor reports it before
    # a downstream call_site does. Type the callables as ``Any`` deliberately:
    # mypy's stubs demand precise ServerRequestContext and the concrete
    # parameter types, but those are not what this check is exercising.
    import typing as _typing

    async def _list(_ctx: Any, _params: Any) -> Any:  # pragma: no cover
        return None

    async def _call(_ctx: Any, _params: Any) -> Any:  # pragma: no cover
        return None

    try:
        Server(
            "cortex-doctor",
            on_list_tools=_typing.cast("Any", _list),
            on_call_tool=_typing.cast("Any", _call),
        )
    except (TypeError, AttributeError, ValueError) as exc:
        return Check(
            "MCP",
            Status.FAIL,
            "MCP server constructor rejected the installed mcp version",
            details=[f"{type(exc).__name__}: {exc}"],
        )
    return Check("MCP", Status.PASS, "server constructor accepts the installed mcp version")


def check_rerank_extra(settings: Settings) -> Check:
    if not getattr(settings, "rerank_enabled", False):
        return Check("Rerank extra", Status.INFO, "rerank is disabled in config")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return Check(
            "Rerank extra",
            Status.WARN,
            "rerank_enabled=true but sentence-transformers is missing",
            hint="Install with: pip install 'cortex-brain[rerank]'",
        )
    return Check("Rerank extra", Status.PASS, "sentence-transformers installed")


def check_documents_extra(settings: Settings) -> Check:
    if not getattr(settings, "ingest_documents", False):
        return Check("Documents extra", Status.INFO, "ingest_documents is disabled")
    try:
        import pymupdf4llm  # noqa: F401
    except ImportError:
        return Check(
            "Documents extra",
            Status.WARN,
            "ingest_documents=true but pymupdf4llm is missing",
            hint="Install with: pip install 'cortex-brain[documents]'",
        )
    return Check("Documents extra", Status.PASS, "pymupdf4llm installed")


def check_daemon(_settings: Settings) -> Check:
    plist = Path("/Library/LaunchDaemons/com.jeevesh.cortex.plist")
    if plist.exists():
        return Check("Daemon", Status.PASS, f"installed at {plist}")
    return Check(
        "Daemon",
        Status.WARN,
        "background watcher not installed (indexing only happens when you run `cortex index`)",
        hint="Install with: ./scripts/install-daemon.sh",
    )


def check_privacy(settings: Settings) -> Check:
    """List eligible providers AND refused ones with their reason.

    Showing the refused list is the whole point of ADR-0004: a trained-on-data
    provider sitting next to a no-train one would still be silently downgraded
    by a naive router, and surfacing *that* here is what gives the user a
    way to fix it.
    """
    providers = build_chat_providers(settings.providers)
    router = Router(providers)
    explained = router.explain(Sensitivity.PRIVATE)

    eligible = sorted(p for p, verdict in explained.items() if verdict.startswith("eligible"))
    refused = sorted(f"{p}: {verdict}" for p, verdict in explained.items() if p not in eligible)

    if not eligible:
        return Check(
            "Privacy",
            Status.FAIL,
            "no provider eligible for PRIVATE content",
            hint="Set an API key for a no-training provider, or mark notes public in frontmatter",
            details=refused,
        )

    detail_lines = [f"eligible: {', '.join(eligible)}"]
    detail_lines.extend(f" - {r}" for r in refused)
    return Check(
        "Privacy",
        Status.PASS,
        f"{len(eligible)} provider{'s' if len(eligible) != 1 else ''} eligible for private",
        details=detail_lines,
    )


def check_memory(settings: Settings) -> Check:
    if not settings.memory_enabled:
        return Check("Memory folder", Status.INFO, "memory feature is disabled in config")
    if not settings.vault_path.exists():
        return Check(
            "Memory folder",
            Status.INFO,
            "skipped (vault path does not exist)",
        )
    mem = settings.vault_path / settings.memory_folder
    try:
        mem.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check(
            "Memory folder",
            Status.FAIL,
            f"cannot create {mem}",
            details=[f"OSError: {exc}"],
        )
    # Round-trip a write inside the folder so a read-only mount surfaces as a
    # check failure, not a runtime error three commands later.
    probe = mem / ".cortex-doctor-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(
            "Memory folder",
            Status.FAIL,
            f"cannot write inside {mem}",
            details=[f"OSError: {exc}"],
        )
    return Check("Memory folder", Status.PASS, f"writable at {mem}")


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


_CHECKS = (
    check_config,
    check_vault,
    check_ollama,
    check_gpu_path,
    check_inference_path,
    check_vector_store,
    check_privacy,
    check_mcp_wire,
    check_rerank_extra,
    check_documents_extra,
    check_daemon,
    check_memory,
)


def run_doctor(settings: Settings) -> DoctorReport:
    """Run every check against *settings* and never raise.

    A single network glitch or filesystem oddity must not abort the whole
    diagnostic: failed checks are exactly the *signal* the user came for. The
    swallow-and-record pattern here is the entire reason this function exists
    in addition to the individual check helpers.
    """
    results: list[Check] = []
    for check_fn in _CHECKS:
        try:
            results.append(check_fn(settings))
        except Exception as exc:
            # The doctor MUST NOT crash on a single bad check -- a glitching
            # subprocess is exactly the signal the user came here for.
            logger.debug("doctor check %s crashed", check_fn.__name__, exc_info=True)
            results.append(
                Check(
                    name=check_fn.__name__.removeprefix("check_").replace("_", " ").title(),
                    status=Status.FAIL,
                    message="check raised an unexpected exception",
                    details=[f"{type(exc).__name__}: {exc}"],
                )
            )
    return DoctorReport(checks=results)
