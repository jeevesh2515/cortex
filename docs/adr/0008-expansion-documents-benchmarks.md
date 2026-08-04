# ADR 0008: Local query expansion, optional document ingestion, and measuring instead of guessing

**Status:** Accepted · **Date:** 2026-08-04

## Query expansion, without an LLM

A question is mostly stopwords. "What did I decide about the chunking strategy
for my notes" is thirteen tokens of which three carry signal, and BM25 weights
them all, so the filler actively dilutes scoring.

obsidian-copilot solves this with an LLM call that extracts salient terms and
generates paraphrases. Cortex does it locally instead: it must work offline, an
LLM call before every search is a real latency and thermal cost on a fanless
machine, and the win here comes mostly from *removing* filler rather than
inventing synonyms -- which needs no model.

Extracted: salient terms (a second, tighter lexical query), quoted phrases (an
explicit exact-match instruction, so they lead the variant list), and `#tags`
(a filter the user typed by hand). Each variant becomes its own ranking, so RRF
rewards documents satisfying more than one.

**Two bugs this surfaced, both caught by test:**

1. **Domain stopwords.** "notes" is five characters, so the distinctiveness
   heuristic ranked it a *leading* salient term -- while matching literally every
   document in a notes application. `note`, `notes`, `vault`, `file`, `document`
   are now stopwords.
2. **Shared-dict mutation.** The engine adds weights for variant rankings with
   `setdefault`, and `Runtime.engine()` passes the same `Settings` dict on every
   call, so `fts:1`, `fts:2` … accumulated into global config across queries.
   `fusion_weights` is now copied, not aliased.

## Document ingestion, strictly optional

Vaults accumulate PDFs, EPUBs and saved web pages, none of it searchable while
ingestion is markdown-only. Every extractor is optional and degrades to "skipped
with a reason" rather than an import error, because a markdown-only user should
not download a PDF stack.

Choices follow the 2026 benchmark landscape: `pymupdf4llm` for PDF (10-50x
faster than pure Python, and emits markdown with headings intact so the
structure-aware chunker still works), the stdlib HTML parser with a hand-written
extractor (`markitdown` is heavier than "drop the chrome" warrants), and EPUB as
zip-of-XHTML reusing the HTML path.

Everything becomes markdown, so chunking, linking and citation are identical to
a hand-written note. Imported notes are tagged `cortex/imported` so they are
distinguishable in results.

Neither PDF backend does OCR. A scanned PDF reports that explicitly rather than
silently indexing an empty note and looking like a bug.

**Bug caught by test:** a file that extracted to nothing was still counted as
`indexed` in the report. `index_note` now returns `None` for "nothing
indexable", distinct from `(0, 0)` meaning "a real note whose chunks were
replaced".

## Benchmarks: measure, do not guess

Every retrieval default here was set from published findings on someone else's
corpus. That is a reasonable prior and nothing more. Whether the wikilink graph
helps *your* vault depends on how densely you link; whether reranking earns its
latency depends on your queries.

`cortex bench --cases ground-truth.yaml` reports recall@5/10, MRR, MAP, nDCG@10
and latency percentiles (p95, not mean -- a search that is usually fast and
occasionally takes four seconds feels broken, and an average hides exactly
that). It then **ablates**: each component is disabled in turn and the delta
reported, so a component that costs latency and buys nothing is visible.

Deltas are signed so a component that helps shows positive. A negative delta
means it is actively hurting and should be turned off -- a conclusion the
project should make easy to reach about its own features.

The harness also reports the hash-gate speedup: cold index time over no-op
rescan time. A low figure means something is defeating incremental indexing and
the vault is being needlessly re-embedded.

Ablation toggles components on the live engine rather than rebuilding, so the
index and embedder are identical across runs -- otherwise the comparison
measures cache warmth as much as retrieval. Engine state is restored afterwards;
a benchmark must not leave the system reconfigured.
