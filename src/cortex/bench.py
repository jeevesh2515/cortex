"""Benchmarking and ablation.

Every retrieval knob in Cortex was set from published findings on someone else's
corpus. Those are a reasonable prior and nothing more: whether the wikilink graph
helps *your* vault depends on how densely you link, whether reranking is worth
its latency depends on your queries, and the right chunk size depends on how you
write. Guessing is the normal practice and it is not good enough when measuring
is this cheap.

Two things are measured:

**Throughput.** Indexing rate and query latency percentiles. p95 matters more
than the mean -- a search that is usually fast and occasionally takes four
seconds feels broken, and an average hides exactly that.

**Quality, by ablation.** Given a ground-truth file mapping queries to the notes
that should answer them, each retrieval component is switched off in turn and the
delta reported. A component that costs latency and buys nothing on your vault
should be turned off, and this is the only honest way to find out which.

Ground truth is a small YAML file:

    - query: what did I decide about chunking?
      expect: [Chunking Strategy.md, Retrieval.md]

Twenty queries is enough to be useful. Write them from questions you actually
asked, not questions you invented to make the numbers look good.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AblationResult",
    "BenchCase",
    "IndexBenchmark",
    "QualityReport",
    "average_precision",
    "load_cases",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


@dataclass(frozen=True, slots=True)
class BenchCase:
    """A query and the notes that should answer it."""

    query: str
    expect: frozenset[str]

    @property
    def is_usable(self) -> bool:
        return bool(self.query.strip() and self.expect)


# --- metrics ---------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    """Fraction of expected notes appearing in the top k.

    The headline metric for retrieval: synthesis cannot cite what retrieval never
    surfaced, so a recall failure is unrecoverable downstream.
    """
    if not expected:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return len(expected & set(top)) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: frozenset[str]) -> float:
    """1/rank of the first correct hit. Rewards putting the answer first."""
    for index, note_id in enumerate(_dedupe(retrieved), start=1):
        if note_id in expected:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: Sequence[str], expected: frozenset[str]) -> float:
    """Mean precision at each correct hit. Sensitive to ordering throughout."""
    if not expected:
        return 0.0
    hits = 0
    total = 0.0
    for index, note_id in enumerate(_dedupe(retrieved), start=1):
        if note_id in expected:
            hits += 1
            total += hits / index
    return total / len(expected)


def ndcg_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Discounts logarithmically by position, so it distinguishes "right answer
    third" from "right answer eighth" -- which recall@k cannot.
    """
    import math

    if not expected:
        return 0.0
    top = _dedupe(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, note_id in enumerate(top, start=1)
        if note_id in expected
    )
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def _dedupe(items: Sequence[str]) -> list[str]:
    """First occurrence order. Several chunks from one note count once."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --- reports ---------------------------------------------------------------


@dataclass(slots=True)
class QualityReport:
    """Aggregated retrieval quality over a case set."""

    label: str
    cases: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg_at_10: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    """Queries where nothing expected was retrieved at all. The most useful
    output here: these are the concrete failures worth reading."""

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
        return ordered[index]

    def as_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cases": self.cases,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "total_misses": len(self.misses),
        }


@dataclass(slots=True)
class AblationResult:
    """A baseline plus one report per disabled component."""

    baseline: QualityReport
    variants: list[QualityReport] = field(default_factory=list)

    def deltas(self) -> list[dict[str, Any]]:
        """Change in recall@5 and p50 when each component is removed.

        Signs are chosen so a component that *helps* shows a positive
        ``recall_delta``: disabling it lost you that much.
        """
        rows: list[dict[str, Any]] = []
        for variant in self.variants:
            rows.append(
                {
                    "disabled": variant.label,
                    "recall@5": round(variant.recall_at_5, 4),
                    "recall_delta": round(self.baseline.recall_at_5 - variant.recall_at_5, 4),
                    "ndcg_delta": round(self.baseline.ndcg_at_10 - variant.ndcg_at_10, 4),
                    "p50_saved_ms": round(self.baseline.p50_ms - variant.p50_ms, 1),
                }
            )
        return rows


@dataclass(slots=True)
class IndexBenchmark:
    """Indexing throughput."""

    notes: int = 0
    chunks: int = 0
    elapsed_s: float = 0.0
    reindex_elapsed_s: float = 0.0

    @property
    def notes_per_second(self) -> float:
        return self.notes / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def speedup(self) -> float:
        """How much cheaper a no-op rescan is than a cold index.

        This is the number that proves the content-hash gate is doing its job.
        A low figure means something is defeating it and the vault is being
        needlessly re-embedded.
        """
        if not self.reindex_elapsed_s:
            return 0.0
        return self.elapsed_s / self.reindex_elapsed_s


# --- loading ---------------------------------------------------------------


def load_cases(path: Path) -> list[BenchCase]:
    """Load ground-truth cases from YAML or JSON.

    Malformed entries are skipped with a warning rather than raising: a typo in
    one case should not stop you measuring the other nineteen.
    """
    import json

    text = path.read_text(encoding="utf-8")
    raw: Any
    if path.suffix.lower() in {".json"}:
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    if isinstance(raw, dict):
        # Also accept {query: [expected, ...]}.
        raw = [{"query": key, "expect": value} for key, value in raw.items()]
    if not isinstance(raw, list):
        return []

    cases: list[BenchCase] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        query = str(entry.get("query", "")).strip()
        expected = entry.get("expect") or entry.get("expected") or []
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list):
            continue
        case = BenchCase(
            query=query, expect=frozenset(str(item).strip() for item in expected if item)
        )
        if case.is_usable:
            cases.append(case)
        else:
            logger.warning("skipping unusable bench case: %r", entry)
    return cases


# --- running ---------------------------------------------------------------


def evaluate(
    engine: Any,
    cases: Sequence[BenchCase],
    *,
    label: str = "baseline",
    top_k: int = 10,
    **retrieve_kwargs: Any,
) -> QualityReport:
    """Run every case and aggregate the metrics."""
    report = QualityReport(label=label, cases=len(cases))
    if not cases:
        return report

    recalls5: list[float] = []
    recalls10: list[float] = []
    rrs: list[float] = []
    aps: list[float] = []
    ndcgs: list[float] = []

    for case in cases:
        started = time.perf_counter()
        result = engine.retrieve(case.query, top_k=top_k, **retrieve_kwargs)
        report.latencies_ms.append((time.perf_counter() - started) * 1000)

        retrieved = [scored.chunk.note_id for scored in result.chunks]
        recalls5.append(recall_at_k(retrieved, case.expect, 5))
        recalls10.append(recall_at_k(retrieved, case.expect, 10))
        rrs.append(reciprocal_rank(retrieved, case.expect))
        aps.append(average_precision(retrieved, case.expect))
        ndcgs.append(ndcg_at_k(retrieved, case.expect, 10))

        if not (case.expect & set(retrieved)):
            report.misses.append(case.query)

    report.recall_at_5 = statistics.fmean(recalls5)
    report.recall_at_10 = statistics.fmean(recalls10)
    report.mrr = statistics.fmean(rrs)
    report.map_score = statistics.fmean(aps)
    report.ndcg_at_10 = statistics.fmean(ndcgs)
    return report


def ablate(engine: Any, cases: Sequence[BenchCase], *, top_k: int = 10) -> AblationResult:
    """Measure the contribution of each retrieval component.

    Each variant disables exactly one thing, so the delta is attributable. The
    components are toggled on the engine rather than rebuilt, which keeps the
    index and the embedder identical across runs -- otherwise the comparison
    measures cache warmth as much as retrieval.
    """
    baseline = evaluate(engine, cases, label="baseline", top_k=top_k)
    variants: list[QualityReport] = []

    if getattr(engine, "graph", None) is not None:
        variants.append(evaluate(engine, cases, label="graph", top_k=top_k, use_graph=False))
    if getattr(engine, "reranker", None) is not None:
        variants.append(evaluate(engine, cases, label="reranker", top_k=top_k, rerank=False))
    if getattr(engine, "expansion_enabled", False):
        engine.expansion_enabled = False
        try:
            variants.append(evaluate(engine, cases, label="expansion", top_k=top_k))
        finally:
            engine.expansion_enabled = True

    return AblationResult(baseline=baseline, variants=variants)
