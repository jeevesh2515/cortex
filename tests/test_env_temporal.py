"""Tests for dotenv loading, temporal queries and obsidian:// citations.

The dotenv precedence rule gets the most attention: a stale ``.env`` shadowing a
deliberately exported key produces requests authenticating as the wrong account,
which is a miserable thing to debug.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from cortex.env import find_dotenv, load_dotenv, parse_dotenv
from cortex.models import obsidian_uri
from cortex.temporal import (
    DateRange,
    extract_note_date,
    is_temporal_query,
    parse_date_range,
)

# A Wednesday, so week boundaries are unambiguous in the tests.
TODAY = date(2026, 8, 5)


class TestParseDotenv:
    def test_simple_assignment(self) -> None:
        assert parse_dotenv("FOO=bar") == {"FOO": "bar"}

    def test_export_prefix_tolerated(self) -> None:
        # People paste straight out of their shell profile.
        assert parse_dotenv("export FOO=bar") == {"FOO": "bar"}

    def test_double_quotes_stripped(self) -> None:
        assert parse_dotenv('FOO="bar baz"') == {"FOO": "bar baz"}

    def test_single_quotes_stripped(self) -> None:
        assert parse_dotenv("FOO='bar'") == {"FOO": "bar"}

    def test_hash_inside_quotes_preserved(self) -> None:
        # API keys genuinely contain '#'. Treating it as a comment corrupts them.
        assert parse_dotenv("FOO='sk-a#b#c'") == {"FOO": "sk-a#b#c"}

    def test_inline_comment_stripped_when_unquoted(self) -> None:
        assert parse_dotenv("FOO=bar   # a note") == {"FOO": "bar"}

    def test_full_line_comment_ignored(self) -> None:
        assert parse_dotenv("# nothing here\nFOO=bar") == {"FOO": "bar"}

    def test_blank_lines_ignored(self) -> None:
        assert parse_dotenv("\n\nFOO=bar\n\n") == {"FOO": "bar"}

    def test_empty_value_allowed(self) -> None:
        assert parse_dotenv("FOO=") == {"FOO": ""}

    def test_value_containing_equals(self) -> None:
        assert parse_dotenv("FOO=a=b=c") == {"FOO": "a=b=c"}

    def test_malformed_line_skipped(self) -> None:
        assert parse_dotenv("this is not an assignment\nFOO=bar") == {"FOO": "bar"}

    def test_invalid_key_skipped(self) -> None:
        assert parse_dotenv("not-a-key!=x\nFOO=bar") == {"FOO": "bar"}

    def test_realistic_file(self) -> None:
        parsed = parse_dotenv(
            "# Cortex credentials\n"
            "export OPENROUTER_API_KEY=sk-or-v1-abc\n"
            'GROQ_API_KEY="gsk_def"   # groq\n'
            "\n"
            "NVIDIA_API_KEY='nvapi-ghi'\n"
        )
        assert parsed == {
            "OPENROUTER_API_KEY": "sk-or-v1-abc",
            "GROQ_API_KEY": "gsk_def",
            "NVIDIA_API_KEY": "nvapi-ghi",
        }


class TestLoadDotenv:
    def test_applies_to_environ(self, tmp_path: Path, monkeypatch: Any) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_TEST_KEY=from_file\n")
        monkeypatch.delenv("CORTEX_TEST_KEY", raising=False)
        applied = load_dotenv(env_file)
        assert applied == {"CORTEX_TEST_KEY": "from_file"}
        assert os.environ["CORTEX_TEST_KEY"] == "from_file"

    def test_existing_environment_wins(self, tmp_path: Path, monkeypatch: Any) -> None:
        """The rule that prevents a stale .env shadowing a real key."""
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_TEST_KEY=from_file\n")
        monkeypatch.setenv("CORTEX_TEST_KEY", "from_shell")
        applied = load_dotenv(env_file)
        assert applied == {}
        assert os.environ["CORTEX_TEST_KEY"] == "from_shell"

    def test_override_when_asked(self, tmp_path: Path, monkeypatch: Any) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_TEST_KEY=from_file\n")
        monkeypatch.setenv("CORTEX_TEST_KEY", "from_shell")
        load_dotenv(env_file, override=True)
        assert os.environ["CORTEX_TEST_KEY"] == "from_file"

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        assert load_dotenv(tmp_path / "absent") == {}

    def test_finds_file_by_walking_up(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("A=1\n")
        nested = tmp_path / "src" / "deep" / "deeper"
        nested.mkdir(parents=True)
        assert find_dotenv(nested) == tmp_path / ".env"

    def test_returns_none_when_nothing_found(self, tmp_path: Path) -> None:
        assert find_dotenv(tmp_path, depth=0) is None

    def test_does_not_log_values(self, tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("CORTEX_SECRET=super-secret-value\n")
        monkeypatch.delenv("CORTEX_SECRET", raising=False)
        import logging

        with caplog.at_level(logging.DEBUG):
            load_dotenv(env_file)
        assert "super-secret-value" not in caplog.text


class TestParseDateRange:
    def test_no_date_returns_none(self) -> None:
        # A false positive here silently narrows an ordinary query to a window.
        assert parse_date_range("what is reciprocal rank fusion?", today=TODAY) is None

    def test_today(self) -> None:
        window = parse_date_range("what did I do today?", today=TODAY)
        assert window == DateRange(TODAY, TODAY, "today")

    def test_yesterday(self) -> None:
        window = parse_date_range("yesterday's notes", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 8, 4)
        assert window.days == 1

    def test_last_week_is_the_previous_iso_week(self) -> None:
        window = parse_date_range("what did I work on last week?", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 7, 27)  # Monday
        assert window.end == date(2026, 8, 2)  # Sunday
        assert window.days == 7

    def test_this_week(self) -> None:
        window = parse_date_range("this week's log", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 8, 3)
        assert window.days == 7

    def test_last_month_spans_the_whole_month(self) -> None:
        window = parse_date_range("summarise last month", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 7, 1)
        assert window.end == date(2026, 7, 31)

    def test_last_month_crosses_year_boundary(self) -> None:
        window = parse_date_range("last month", today=date(2026, 1, 15))
        assert window is not None
        assert window.start == date(2025, 12, 1)
        assert window.end == date(2025, 12, 31)

    def test_last_n_days(self) -> None:
        window = parse_date_range("notes from the last 3 days", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 8, 3)
        assert window.end == TODAY

    def test_last_n_weeks(self) -> None:
        window = parse_date_range("past 2 weeks", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 7, 22)

    def test_zero_count_rejected(self) -> None:
        assert parse_date_range("last 0 days", today=TODAY) is None

    def test_explicit_iso_day(self) -> None:
        window = parse_date_range("what happened on 2026-03-14?", today=TODAY)
        assert window == DateRange(date(2026, 3, 14), date(2026, 3, 14), "2026-03-14")

    def test_explicit_iso_month(self) -> None:
        window = parse_date_range("notes from 2026-02", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 2, 1)
        assert window.end == date(2026, 2, 28)

    def test_impossible_date_returns_none(self) -> None:
        assert parse_date_range("2026-02-31", today=TODAY) is None

    def test_named_month_with_preposition(self) -> None:
        window = parse_date_range("what did I write in March?", today=TODAY)
        assert window is not None
        assert window.start == date(2026, 3, 1)
        assert window.end == date(2026, 3, 31)

    def test_named_month_with_year(self) -> None:
        window = parse_date_range("notes from January 2025", today=TODAY)
        assert window is not None
        assert window.start == date(2025, 1, 1)

    def test_bare_month_name_is_not_a_filter(self) -> None:
        # A note *about* March must not become a date-filtered query.
        assert parse_date_range("March release planning", today=TODAY) is None

    def test_last_year(self) -> None:
        window = parse_date_range("last year", today=TODAY)
        assert window is not None
        assert window.start == date(2025, 1, 1)
        assert window.end == date(2025, 12, 31)

    def test_is_temporal_query_helper(self) -> None:
        assert is_temporal_query("last week", today=TODAY)
        assert not is_temporal_query("vector databases", today=TODAY)


class TestDateRange:
    def test_contains(self) -> None:
        window = DateRange(date(2026, 8, 1), date(2026, 8, 7))
        assert window.contains(date(2026, 8, 3))
        assert window.contains(date(2026, 8, 1))
        assert window.contains(date(2026, 8, 7))
        assert not window.contains(date(2026, 7, 31))

    def test_contains_none_is_false(self) -> None:
        # An undated note must never satisfy a date filter.
        assert not DateRange(date(2026, 8, 1), date(2026, 8, 7)).contains(None)

    def test_str_uses_label(self) -> None:
        assert str(DateRange(date(2026, 8, 1), date(2026, 8, 7), "last week")) == "last week"

    def test_str_falls_back_to_dates(self) -> None:
        assert "2026-08-01" in str(DateRange(date(2026, 8, 1), date(2026, 8, 7)))


class TestExtractNoteDate:
    def test_iso_filename(self) -> None:
        assert extract_note_date("daily/2026-07-15.md") == date(2026, 7, 15)

    def test_compact_filename(self) -> None:
        assert extract_note_date("journal/20260715.md") == date(2026, 7, 15)

    def test_dotted_filename(self) -> None:
        assert extract_note_date("2026.07.15 meeting.md") == date(2026, 7, 15)

    def test_frontmatter_beats_filename(self) -> None:
        # The author stating a date is more trustworthy than a filename.
        got = extract_note_date("daily/2026-07-15.md", {"date": date(2026, 1, 1)})
        assert got == date(2026, 1, 1)

    def test_frontmatter_string_date(self) -> None:
        assert extract_note_date("a.md", {"created": "2026-05-09"}) == date(2026, 5, 9)

    def test_frontmatter_datetime(self) -> None:
        from datetime import datetime

        got = extract_note_date("a.md", {"date": datetime(2026, 5, 9, 13, 30)})
        assert got == date(2026, 5, 9)

    def test_no_signal_returns_none(self) -> None:
        assert extract_note_date("notes/ideas.md") is None

    def test_mtime_is_last_resort(self) -> None:
        import datetime as dt

        stamp = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.UTC).timestamp()
        assert extract_note_date("notes/ideas.md", mtime=stamp) == date(2026, 6, 1)

    def test_invalid_filename_date_ignored(self) -> None:
        assert extract_note_date("notes/2026-99-99.md") is None

    def test_garbage_frontmatter_ignored(self) -> None:
        assert extract_note_date("notes/x.md", {"date": ["not", "a", "date"]}) is None


class TestObsidianUri:
    def test_basic(self) -> None:
        uri = obsidian_uri("Cooking/Sourdough.md", "MyVault")
        assert uri.startswith("obsidian://open?")
        assert "vault=MyVault" in uri

    def test_path_is_encoded(self) -> None:
        uri = obsidian_uri("notes/a b.md", "V")
        assert "a%20b.md" in uri
        assert "%2F" in uri  # slash encoded, so it is one query value

    def test_vault_name_with_space(self) -> None:
        assert "My%20Vault" in obsidian_uri("a.md", "My Vault")

    def test_heading_anchor(self) -> None:
        uri = obsidian_uri("a.md", "V", heading="Section Two")
        assert uri.endswith("%23Section%20Two")

    def test_no_heading_has_no_anchor(self) -> None:
        assert "%23" not in obsidian_uri("a.md", "V")

    def test_quoted_value_with_trailing_comment(self) -> None:
        # Regression: matching on the last character instead of the closing
        # quote left the quotes embedded in the value.
        assert parse_dotenv('KEY="value"   # trailing note') == {"KEY": "value"}

    def test_unterminated_quote_is_salvaged(self) -> None:
        assert parse_dotenv('KEY="value') == {"KEY": "value"}
