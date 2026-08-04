# ADR 0005: Resolve dates in queries and return coverage, not top-k

**Status:** Accepted · **Date:** 2026-08-04

## Context

"What did I work on last week" is among the most natural questions to ask a
second brain, and pure semantic retrieval answers it badly. The phrase "last
week" carries almost no lexical or semantic signal, so the query collapses into
matching on whatever else it happens to contain. A survey of comparable projects
found the same gap nearly everywhere: only obsidian-copilot and khoj handle
dates properly, and most others rely on the date string appearing in a filename
by luck.

## Decision

Two mechanisms, both narrow on purpose.

**1. Resolve a date window from the query and filter on it.**
`temporal.parse_date_range` handles the closed set of expressions people
actually use with a notes system: today, yesterday, this/last week, month, year,
"last N days", explicit `YYYY-MM-DD` and `YYYY-MM`, and named months behind a
preposition. Every note gets a resolved `note_date` from -- in descending order
of trustworthiness -- explicit frontmatter, a daily-note filename, then mtime.

**2. Do not apply a top-k cutoff to a temporal query.**
"What did I do last week" wants *coverage*. Returning the 8 best-matching
passages produces a confidently incomplete answer, which is worse than a slow
one. When a window is active, retrieval returns everything in it, capped only by
`coverage_limit` to bound the context.

Dependency-free by choice: `dateparser` handles far more than this needs and
brings a large dependency tree for expressions we can enumerate.

## Consequences

**Positive**

- Daily notes become genuinely queryable, which is most of what a journal-style
  vault is for.
- `RetrievalResult.date_range` is surfaced by the CLI and MCP, so the user sees
  how "last week" was interpreted rather than guessing.
- An empty window falls back to unfiltered results *and* reports the window, so
  the caller can say "nothing from last week" instead of silently answering a
  different question.

**Negative and the traps found while building it**

- A false positive is worse than a miss: silently narrowing an ordinary query to
  a date window is very confusing. Hence bare month names do **not** trigger a
  filter -- "March release planning" is a note about March, not a date query.
  Only a preposition ("in March") does.
- **Ordering bug caught by test.** The date filter runs after fusion, so fusion
  must not truncate to `top_k` first, or coverage is silently capped at `top_k`
  no matter how much of the window matched. `fusion_limit` is now raised to
  `coverage_limit` whenever a window is active.
- A dated query filters most candidates away, so the candidate pool has to be
  widened going in (`temporal_overfetch`, default 8x) or the window comes back
  empty despite the vault having content.
- mtime is a poor last resort: syncing a vault rewrites mtimes wholesale. It is
  used only when neither frontmatter nor filename offers anything.

## Rejected

**Indexing dates as a separate filterable column in the vector store.** Cleaner
in principle, and what khoj does with PostgreSQL. Rejected because it would
force schema migrations on both store backends for a filter that is cheap to
apply in Python at these corpus sizes. Worth revisiting past ~500k chunks.
