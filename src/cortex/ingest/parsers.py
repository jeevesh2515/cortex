"""Non-markdown document parsing.

Obsidian vaults accumulate more than markdown: papers dropped into an
attachments folder, EPUBs, exported web pages. None of that is searchable while
ingestion is markdown-only.

The design constraint is that every extractor must be **optional**. A user who
only keeps markdown should not download a PDF stack, so each format degrades to
"skipped with a reason" rather than an import error. ``available_formats()``
reports what is actually installed, and ``cortex status`` surfaces it -- the
alternative is a vault that silently ignores half its content.

Extractor choice per format, from the 2026 benchmark landscape:

* **PDF** -- PyMuPDF (``pymupdf4llm``) first: 10-50x faster than pure-Python
  alternatives and emits markdown with heading structure intact, which the
  structure-aware chunker then splits properly. `pypdf` is the light fallback.
  Neither does OCR, so a scanned PDF yields little; that is reported rather
  than silently producing an empty note.
* **HTML** -- the stdlib parser with a hand-written extractor. `markitdown` and
  friends are heavier than warranted for what is mostly "drop the chrome".
* **EPUB** -- a zip of XHTML, so it reduces to the HTML path plus unzipping.

Everything returns markdown, so downstream chunking, linking and citation are
identical to a hand-written note.
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DOCUMENT_SUFFIXES",
    "ExtractedDocument",
    "available_formats",
    "extract_document",
    "html_to_markdown",
    "supported_suffixes",
]

DOCUMENT_SUFFIXES = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".epub": "epub",
    ".txt": "text",
}


@dataclass(slots=True)
class ExtractedDocument:
    """Markdown extracted from a non-markdown source."""

    text: str
    title: str = ""
    kind: str = ""
    pages: int = 0
    warning: str = ""
    """Set when extraction succeeded but the result is suspect -- an image-only
    PDF, for instance. Surfaced rather than swallowed so an empty index entry is
    explainable."""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


# --- HTML ------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "form"}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "br",
    "li",
    "tr",
    "blockquote",
    "pre",
    "figure",
}
_HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class _HtmlToMarkdown(HTMLParser):
    """Minimal, structure-preserving HTML to markdown conversion.

    Keeps headings (so the chunker can split on them), list items and links.
    Discards chrome: navigation and boilerplate are the bulk of a saved web page
    and pure noise in an index.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._heading: str | None = None
        self._in_title = False
        self._link: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self._heading = _HEADINGS[tag]
            self.parts.append(f"\n\n{self._heading} ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            self._link = href if href.startswith(("http://", "https://")) else None
            if self._link:
                self.parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in _HEADINGS:
            self._heading = None
            self.parts.append("\n")
        elif tag == "a" and self._link:
            self.parts.append(f"]({self._link})")
            self._link = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        text = re.sub(r"\s+", " ", data)
        if text.strip():
            self.parts.append(text)

    def result(self) -> str:
        joined = "".join(self.parts)
        joined = html.unescape(joined)
        # Collapse the runs of blank lines that block tags inevitably produce.
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        joined = re.sub(r"[ \t]{2,}", " ", joined)
        return joined.strip()


def html_to_markdown(source: str) -> ExtractedDocument:
    """Convert an HTML document to markdown."""
    parser = _HtmlToMarkdown()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:
        logger.debug("html parse issue: %s", exc)
    return ExtractedDocument(text=parser.result(), title=parser.title, kind="html")


# --- PDF -------------------------------------------------------------------


def _pdf_backend() -> str:
    try:
        import pymupdf4llm  # noqa: F401

        return "pymupdf4llm"
    except ImportError:
        pass
    try:
        import fitz  # noqa: F401

        return "pymupdf"
    except ImportError:
        pass
    try:
        import pypdf  # noqa: F401

        return "pypdf"
    except ImportError:
        return ""


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract markdown from a PDF using whichever backend is installed."""
    backend = _pdf_backend()
    if not backend:
        return ExtractedDocument(
            text="",
            kind="pdf",
            warning=(
                "no PDF backend installed; install with: pip install 'cortex-brain[documents]'"
            ),
        )

    try:
        if backend == "pymupdf4llm":
            import pymupdf4llm

            text = pymupdf4llm.to_markdown(str(path), show_progress=False)
            pages = 0
        elif backend == "pymupdf":
            import fitz

            with fitz.open(str(path)) as doc:
                pages = doc.page_count
                text = "\n\n".join(page.get_text("text") for page in doc)
        else:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            pages = len(reader.pages)
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        return ExtractedDocument(text="", kind="pdf", warning=f"{backend} failed: {exc}")

    cleaned = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    warning = ""
    if not cleaned:
        # Almost always a scanned document. None of these backends do OCR, so
        # say so rather than indexing an empty note and looking like a bug.
        warning = "no extractable text (likely a scanned PDF; OCR is not bundled)"
    return ExtractedDocument(text=cleaned, kind="pdf", pages=pages, warning=warning)


# --- EPUB ------------------------------------------------------------------


def extract_epub(path: Path) -> ExtractedDocument:
    """Extract markdown from an EPUB.

    An EPUB is a zip of XHTML, so this unzips and reuses the HTML path. Files
    are taken in archive order, which is close enough to reading order for
    retrieval purposes.
    """
    chunks: list[str] = []
    title = ""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xhtml", ".html", ".htm"))
            ]
            for name in sorted(names):
                try:
                    raw = archive.read(name).decode("utf-8", errors="replace")
                except (KeyError, OSError):
                    continue
                extracted = html_to_markdown(raw)
                if not title and extracted.title:
                    title = extracted.title
                if extracted.text:
                    chunks.append(extracted.text)
    except (zipfile.BadZipFile, OSError) as exc:
        return ExtractedDocument(text="", kind="epub", warning=f"unreadable epub: {exc}")

    return ExtractedDocument(
        text="\n\n".join(chunks).strip(), title=title, kind="epub", pages=len(chunks)
    )


# --- dispatch --------------------------------------------------------------


def extract_document(path: Path) -> ExtractedDocument | None:
    """Extract markdown from a supported non-markdown file.

    Returns ``None`` for unsupported extensions so the caller can skip quietly.
    An extraction that fails returns a document carrying a ``warning`` rather
    than raising: one unreadable attachment must never abort a vault scan.
    """
    kind = DOCUMENT_SUFFIXES.get(path.suffix.lower())
    if kind is None:
        return None

    try:
        if kind == "pdf":
            return extract_pdf(path)
        if kind == "epub":
            return extract_epub(path)
        if kind == "html":
            return html_to_markdown(path.read_text(encoding="utf-8", errors="replace"))
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractedDocument(text=text.strip(), kind="text")
    except OSError as exc:
        return ExtractedDocument(text="", kind=kind, warning=f"unreadable: {exc}")


def available_formats() -> dict[str, bool]:
    """Which formats can actually be extracted right now.

    Surfaced by ``cortex status``: a vault silently ignoring its PDFs because a
    dependency is missing is worse than one that says so.
    """
    return {
        "pdf": bool(_pdf_backend()),
        "html": True,
        "epub": True,
        "text": True,
    }


def supported_suffixes(*, only_available: bool = True) -> set[str]:
    """Extensions worth scanning for."""
    formats = available_formats()
    return {
        suffix
        for suffix, kind in DOCUMENT_SUFFIXES.items()
        if not only_available or formats.get(kind, False)
    }
