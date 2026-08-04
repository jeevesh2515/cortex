"""Obsidian vault parsing.

Obsidian markdown is *not* plain markdown. It carries a link graph the user has
curated by hand, and that graph is the single most valuable retrieval signal in
the vault -- it is a knowledge graph nobody had to pay an LLM to extract.

The parsing rule that matters most: **links and tags inside code must not
count**. A Python snippet containing ``d = {}  # [[not a link]]`` or a shell
line with ``curl host/#anchor`` will otherwise poison the graph with phantom
nodes. We mask fenced blocks, indented blocks and inline code before scanning.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from cortex.models import Note, Sensitivity, WikiLink, content_hash

__all__ = ["extract_links", "extract_tags", "parse_note", "parse_text", "split_frontmatter"]

# --- Frontmatter -----------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

# --- Code masking ----------------------------------------------------------

# Fenced blocks: ``` or ~~~ with matching closing fence. Non-greedy, multiline.
# An unterminated fence swallows to end of document, which matches how Obsidian
# renders it.
_FENCE_RE = re.compile(
    r"^(?P<fence>```+|~~~+)[^\n]*\n.*?(?:^(?P=fence)[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)

# --- Links -----------------------------------------------------------------

# [[target#heading|alias]] with optional leading ! for embeds.
# Target stops at #, | or ]]. Obsidian forbids [ ] | # in note names.
_WIKILINK_RE = re.compile(
    r"(?P<embed>!)?\[\[(?P<target>[^\[\]|#]+?)"
    r"(?:#(?P<heading>[^\[\]|]+?))?"
    r"(?:\|(?P<alias>[^\[\]]*?))?\]\]"
)

# --- Tags ------------------------------------------------------------------

# A tag is # followed by at least one non-numeric char, allowing nesting with /.
# Must be preceded by start-of-line or whitespace so that `foo#bar` and URLs
# with fragments do not match. Obsidian requires tags contain a non-digit.
_TAG_RE = re.compile(r"(?:(?<=\s)|(?<=^))#(?P<tag>[A-Za-z0-9_/\-]*[A-Za-z_/\-][A-Za-z0-9_/\-]*)")

# --- Dataview inline fields ------------------------------------------------

_DATAVIEW_RE = re.compile(r"^\s*(?:-\s*)?(?P<key>[A-Za-z][A-Za-z0-9 _-]*)::\s*(?P<value>.*)$", re.M)


def _mask_code(text: str) -> str:
    """Replace code regions with spaces, preserving offsets and line structure.

    Offsets are preserved so that any future feature wanting character positions
    (e.g. jump-to-source) stays correct against the original text.
    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    masked = _FENCE_RE.sub(blank, text)
    masked = _INLINE_CODE_RE.sub(blank, masked)
    return masked


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the body.

    Returns ``({}, text)`` when absent or malformed. Malformed frontmatter is
    tolerated rather than raised: a single bad note must never abort a vault
    scan of ten thousand files.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw = match.group(1)
    body = text[match.end() :]
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}, body
    if not isinstance(loaded, dict):
        return {}, body
    return loaded, body


def extract_links(text: str, *, mask: bool = True) -> list[WikiLink]:
    """Extract wikilinks in document order, ignoring those inside code."""
    scan = _mask_code(text) if mask else text
    links: list[WikiLink] = []
    for match in _WIKILINK_RE.finditer(scan):
        target = match.group("target").strip()
        if not target:
            continue
        heading = match.group("heading")
        alias = match.group("alias")
        links.append(
            WikiLink(
                target=target,
                heading=heading.strip() if heading else None,
                alias=alias.strip() if alias else None,
                is_embed=match.group("embed") == "!",
            )
        )
    return links


def extract_tags(text: str, *, mask: bool = True) -> set[str]:
    """Extract ``#tags``, ignoring markdown headings, code and URL fragments.

    Nested tags yield their ancestors too: ``#project/cortex/api`` also
    registers ``project`` and ``project/cortex`` so a query for the parent
    matches children.
    """
    scan = _mask_code(text) if mask else text
    # Strip ATX headings ("# Title") -- '#' followed by space is never a tag.
    scan = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+.*$", "", scan, flags=re.MULTILINE)

    tags: set[str] = set()
    for match in _TAG_RE.finditer(scan):
        tag = match.group("tag").rstrip("/")
        if not tag:
            continue
        tags.add(tag)
        parts = tag.split("/")
        for i in range(1, len(parts)):
            tags.add("/".join(parts[:i]))
    return tags


def extract_dataview_fields(text: str, *, mask: bool = True) -> dict[str, str]:
    """Extract Dataview ``key:: value`` inline fields."""
    scan = _mask_code(text) if mask else text
    fields: dict[str, str] = {}
    for match in _DATAVIEW_RE.finditer(scan):
        key = match.group("key").strip().lower()
        value = match.group("value").strip()
        if key and value:
            fields.setdefault(key, value)
    return fields


def _frontmatter_tags(frontmatter: dict[str, Any]) -> set[str]:
    """Read tags from frontmatter, accepting both list and comma-string forms."""
    tags: set[str] = set()
    for key in ("tags", "tag"):
        raw = frontmatter.get(key)
        if raw is None:
            continue
        values: list[str]
        if isinstance(raw, str):
            values = [part.strip() for part in raw.replace(",", " ").split()]
        elif isinstance(raw, (list, tuple)):
            values = [str(item).strip() for item in raw]
        else:
            continue
        for value in values:
            cleaned = value.lstrip("#").strip().rstrip("/")
            if cleaned:
                tags.add(cleaned)
                parts = cleaned.split("/")
                for i in range(1, len(parts)):
                    tags.add("/".join(parts[:i]))
    return tags


def _resolve_sensitivity(frontmatter: dict[str, Any]) -> Sensitivity:
    """Determine whether a note may be sent to a training-tier provider.

    Private by default. Opting a note out requires an explicit, unambiguous
    frontmatter value -- we never infer shareability.
    """
    for key in ("sensitivity", "visibility", "cortex_sensitivity"):
        raw = frontmatter.get(key)
        if isinstance(raw, str) and raw.strip().lower() in {"public", "shareable", "share"}:
            return Sensitivity.PUBLIC
    if frontmatter.get("public") is True:
        return Sensitivity.PUBLIC
    return Sensitivity.PRIVATE


def parse_text(
    text: str,
    *,
    rel_path: str,
    path: Path | None = None,
    mtime: float = 0.0,
) -> Note:
    """Parse raw note text into a :class:`Note`."""
    frontmatter, body = split_frontmatter(text)

    stem = Path(rel_path).stem
    raw_title = frontmatter.get("title")
    title = str(raw_title).strip() if isinstance(raw_title, (str, int, float)) else stem

    tags = extract_tags(body) | _frontmatter_tags(frontmatter)

    return Note(
        path=path or Path(rel_path),
        rel_path=rel_path,
        title=title or stem,
        body=body,
        frontmatter=dict(frontmatter),
        links=extract_links(body),
        tags=tags,
        content_hash=content_hash(text),
        mtime=mtime,
        sensitivity=_resolve_sensitivity(frontmatter),
    )


def parse_note(path: Path, *, vault_root: Path) -> Note:
    """Read and parse a note from disk.

    Undecodable bytes are replaced rather than raising -- vaults accumulate the
    occasional file with a mangled encoding and one of them must not stop a scan.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    try:
        rel = path.resolve().relative_to(vault_root.resolve()).as_posix()
    except ValueError:
        rel = path.name
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return parse_text(text, rel_path=rel, path=path, mtime=mtime)
