"""Temporal query handling.

"What did I work on last week" is one of the most natural questions to ask a
second brain, and pure semantic retrieval answers it badly: the phrase "last
week" carries almost no lexical or semantic signal, so the query degenerates
into matching on whatever else it happens to contain.

Two things fix it, both taken from what the better Obsidian projects do:

1. **Resolve the date range and filter on it**, rather than hoping the embedder
   understands calendars.
2. **Do not apply a top-k cutoff to a temporal query.** "What did I do last
   week" wants coverage, not the 8 best-matching passages. Truncating it gives a
   confidently incomplete answer, which is worse than a slow one.

Deliberately dependency-free. ``dateparser`` handles far more than we need and
pulls in a large dependency tree; the expressions people actually use with a
notes system are a short, closed list.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, timedelta
from pathlib import PurePosixPath
from typing import Any

__all__ = ["DateRange", "extract_note_date", "is_temporal_query", "parse_date_range"]

# Daily-note filename conventions. Obsidian's default is YYYY-MM-DD; the
# compact and dotted forms are common in older vaults.
_DATE_IN_NAME = re.compile(r"(?P<y>\d{4})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})")

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name} | {
    name.lower(): index for index, name in enumerate(calendar.month_abbr) if name
}


@dataclass(frozen=True, slots=True)
class DateRange:
    """An inclusive date range."""

    start: date
    end: date
    label: str = ""

    def contains(self, when: date | None) -> bool:
        if when is None:
            return False
        return self.start <= when <= self.end

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def __str__(self) -> str:
        if self.start == self.end:
            return self.label or self.start.isoformat()
        return self.label or f"{self.start.isoformat()}..{self.end.isoformat()}"


def _week_start(when: date) -> date:
    """Monday of the given date's week (ISO convention)."""
    return when - timedelta(days=when.weekday())


def _month_range(year: int, month: int, label: str = "") -> DateRange:
    last = calendar.monthrange(year, month)[1]
    return DateRange(date(year, month, 1), date(year, month, last), label)


def parse_date_range(query: str, *, today: date | None = None) -> DateRange | None:
    """Extract a date range from natural language, or ``None`` if absent.

    ``today`` is injectable so the behaviour is testable without freezing the
    clock. Returns ``None`` rather than guessing when nothing date-like is
    present -- a false positive here silently truncates an ordinary query to a
    date window, which is a very confusing failure.
    """
    now = today or date.today()
    text = query.lower().strip()

    # --- explicit ISO forms, most specific first -------------------------
    iso_day = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso_day:
        try:
            exact = date(*(int(group) for group in iso_day.groups()))
        except (ValueError, TypeError):
            return None
        return DateRange(exact, exact, exact.isoformat())

    iso_month = re.search(r"\b(\d{4})-(\d{2})\b", text)
    if iso_month:
        iso_year, iso_mon = (int(group) for group in iso_month.groups())
        if 1 <= iso_mon <= 12:
            return _month_range(iso_year, iso_mon, f"{iso_year}-{iso_mon:02d}")

    # --- relative day words ----------------------------------------------
    if re.search(r"\btoday\b", text):
        return DateRange(now, now, "today")
    if re.search(r"\byesterday\b", text):
        prior = now - timedelta(days=1)
        return DateRange(prior, prior, "yesterday")

    # --- "last N days/weeks/months" --------------------------------------
    span = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b", text)
    if span:
        count = int(span.group(1))
        unit = span.group(2)
        if count <= 0:
            return None
        if unit == "day":
            return DateRange(now - timedelta(days=count - 1), now, f"last {count} days")
        if unit == "week":
            return DateRange(now - timedelta(weeks=count), now, f"last {count} weeks")
        if unit == "month":
            return DateRange(now - timedelta(days=30 * count), now, f"last {count} months")
        return DateRange(now - timedelta(days=365 * count), now, f"last {count} years")

    # --- this / last week ------------------------------------------------
    if re.search(r"\bthis week\b", text):
        start = _week_start(now)
        return DateRange(start, start + timedelta(days=6), "this week")
    if re.search(r"\b(?:last|past|previous) week\b", text):
        start = _week_start(now) - timedelta(weeks=1)
        return DateRange(start, start + timedelta(days=6), "last week")

    # --- this / last month -----------------------------------------------
    if re.search(r"\bthis month\b", text):
        return _month_range(now.year, now.month, "this month")
    if re.search(r"\b(?:last|past|previous) month\b", text):
        year = now.year if now.month > 1 else now.year - 1
        month = now.month - 1 if now.month > 1 else 12
        return _month_range(year, month, "last month")

    # --- this / last year ------------------------------------------------
    if re.search(r"\bthis year\b", text):
        return DateRange(date(now.year, 1, 1), date(now.year, 12, 31), "this year")
    if re.search(r"\b(?:last|past|previous) year\b", text):
        prior_year = now.year - 1
        return DateRange(date(prior_year, 1, 1), date(prior_year, 12, 31), "last year")

    # --- named months, optionally with a year ----------------------------
    # Guarded by a leading preposition so that a note *about* "March" does not
    # become a date filter.
    named = re.search(r"\b(?:in|during|from)\s+([a-z]+)(?:\s+(\d{4}))?\b", text)
    if named:
        named_month = _MONTHS.get(named.group(1))
        if named_month:
            named_year = int(named.group(2)) if named.group(2) else now.year
            label = f"{calendar.month_name[named_month]} {named_year}"
            return _month_range(named_year, named_month, label)

    return None


def is_temporal_query(query: str, *, today: date | None = None) -> bool:
    """Whether the query carries a date constraint."""
    return parse_date_range(query, today=today) is not None


def extract_note_date(
    rel_path: str,
    frontmatter: dict[str, Any] | None = None,
    *,
    mtime: float | None = None,
) -> date | None:
    """Best-effort date for a note, in descending order of trustworthiness.

    1. Explicit frontmatter (``date``, ``created``) -- the author said so.
    2. A date in the filename -- the daily-note convention.
    3. Filesystem mtime -- a last resort, and a poor one: syncing a vault
       rewrites mtimes wholesale, so it is only used when nothing better exists.
    """
    frontmatter = frontmatter or {}

    for key in ("date", "created", "day"):
        raw = frontmatter.get(key)
        if raw is None:
            continue
        parsed = _coerce_date(raw)
        if parsed:
            return parsed

    stem = PurePosixPath(rel_path).name
    match = _DATE_IN_NAME.search(stem)
    if match:
        try:
            return date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        except ValueError:
            pass

    if mtime:
        from datetime import datetime

        return datetime.fromtimestamp(mtime, tz=UTC).date()

    return None


def _coerce_date(raw: Any) -> date | None:
    """Coerce a frontmatter value to a date.

    PyYAML already parses unquoted ``2026-08-04`` into a ``date``, so the common
    case needs no work. Quoted values arrive as strings.
    """
    from datetime import datetime

    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, str):
        text = raw.strip()
        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
        if match:
            try:
                return date(*(int(g) for g in match.groups()))
            except ValueError:
                return None
    return None
