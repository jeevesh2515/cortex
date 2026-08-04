"""Dotenv loading.

Cortex reads credentials from the environment. That works for a shell session,
but it silently fails for the two ways people actually run this: keys written to
a project ``.env`` file, and an MCP server launched by a GUI app that inherits
no shell profile.

Deliberately hand-rolled rather than depending on python-dotenv. The parsing is
twenty lines, and the semantics matter enough to own: **an existing environment
variable always wins over the file**. A key exported in the current shell must
never be shadowed by a stale value in a checked-out ``.env``, because the
resulting failure -- requests authenticating as the wrong account -- is
maddening to debug.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["DOTENV_NAMES", "find_dotenv", "load_dotenv", "parse_dotenv"]

DOTENV_NAMES = (".env", ".env.local")


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse dotenv content into a mapping.

    Handles the conventions people actually use: ``export`` prefixes copied from
    a shell profile, quoted values, inline comments, and blank lines. Malformed
    lines are skipped rather than raised -- a single bad line must not stop the
    process from starting.
    """
    values: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Tolerate `export FOO=bar` pasted from a shell profile.
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue

        value = value.strip()
        if value and value[0] in "\"'":
            # Quoted. Find the *closing* quote and take the interior verbatim,
            # so a value containing '#' survives. Anything after the closing
            # quote is a trailing comment and is discarded -- matching on
            # `value[-1]` instead would fail on `KEY="v"  # note`, leaving the
            # quotes embedded in the value and producing an auth error that
            # looks nothing like a parsing problem.
            quote = value[0]
            closing = value.find(quote, 1)
            value = value[1:closing] if closing != -1 else value[1:]
        else:
            # Unquoted: an inline comment terminates the value.
            hash_index = value.find(" #")
            if hash_index != -1:
                value = value[:hash_index].rstrip()

        values[key] = value

    return values


def find_dotenv(start: Path | None = None, *, depth: int = 4) -> Path | None:
    """Search upward from ``start`` for a dotenv file.

    Walking up matters because the CLI is often invoked from a subdirectory of
    the project, and the file lives at the project root.
    """
    current = (start or Path.cwd()).resolve()
    for _ in range(depth + 1):
        for name in DOTENV_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def load_dotenv(
    path: Path | None = None,
    *,
    start: Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load a dotenv file into ``os.environ``.

    Returns the keys that were actually applied. Values already present in the
    environment are left alone unless ``override`` is set.

    Never logs a value -- only key names -- because this function exists
    specifically to handle secrets.
    """
    target = path or find_dotenv(start)
    if target is None:
        return {}

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", target, exc)
        return {}

    applied: dict[str, str] = {}
    for key, value in parse_dotenv(text).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value

    if applied:
        logger.debug("loaded %d key(s) from %s: %s", len(applied), target, sorted(applied))
    return applied
