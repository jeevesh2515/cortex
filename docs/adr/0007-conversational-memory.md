# ADR 0007: Conversational memory as Markdown notes in the vault

**Status:** Accepted · **Date:** 2026-08-04

## Context

Without memory, every session starts cold. A second brain that cannot remember
what you told it last week is a search engine with extra steps.

The obvious implementation is a memory table in the catalog. The survey of
comparable projects showed a better one: basic-memory (3.3k stars) writes agent
memory as ordinary Markdown notes into the vault, so the human and the agent
read and write the same files.

## Decision

Memory notes are plain Markdown written to a `Memory/` folder inside the vault.

**Why not a database.** A separate store is invisible. You could not read what
the system believes about you, correct a wrong inference, or delete something
you regret. Notes in the vault are auditable in Obsidian like anything else,
appear in graph view, and are deleted with a keystroke. They also participate in
retrieval for free -- no second index, no second embedder, no sync problem.

**The part that pays for itself.** A memory note wikilinks the sources it drew
on. Those links join the real link graph, so a memory about sourdough becomes a
hub connecting the notes that answered it. The graph densifies exactly where you
actually think, which then improves graph-expansion retrieval. Verified by test:
after writing a memory, its sources gain a backlink from `Memory/`.

**Off by default per query.** `--remember` is explicit; `memory_auto` exists but
defaults false. Capturing every passing question would fill the vault with
noise, and noise in the vault is worse than no memory at all.

## The failure mode this had to solve

Memory notes are *summaries of primary notes*, so they compete with their own
sources in retrieval. Left uncapped they accumulate and gradually displace the
real content: answers drift towards summarising previous answers, getting more
confident and less grounded with each generation. This is a slow, quiet
degradation that would be very hard to diagnose after the fact.

`memory_max_results` (default 2) caps how much memory can occupy any result set.
Primary sources are never dropped. Tested directly with five memory notes
competing against one primary note.

## Safety

This is the only module that writes to the vault, so:

- Writes are confined to the memory folder. The resolved parent is checked
  against it, so a crafted title cannot escape via `../`. `slugify` already
  strips path separators; the containment check is defence in depth.
- Nothing is ever overwritten -- a collision gets a numeric suffix.
- Writes are atomic: temp file in the same directory, fsync, then `os.replace`.
  An interrupted write leaves a dot-prefixed temp file Obsidian ignores, never a
  half-written note.
- Nothing is ever deleted. `cortex memory` is deliberately read-only; pruning
  belongs in Obsidian, not in a second and worse file manager.
- Memory notes are always `sensitivity: private`, never inheriting public. A
  summary of private notes is private.
