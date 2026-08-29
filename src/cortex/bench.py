"""Benchmarking, evaluation and ablation harness for Cortex.

Provides deterministic retrieval evaluation metrics, latency statistics,
component ablation analysis, run comparison, and structured report exports (JSON, Markdown, CSV).

Measured metrics:
- **Recall@k (k=1, 5, 10)**: Fraction of expected relevant notes surfaced in top k.
- **MRR (Mean Reciprocal Rank)**: 1 / rank of the first relevant document.
- **MAP (Mean Average Precision)**: Position-sensitive mean precision across relevant documents.
- **nDCG@10**: Normalized Discounted Cumulative Gain with logarithmic rank discount.
- **Latency percentiles**: p50, p95, p99, mean, min, max query latency in milliseconds.
- **Component ablation deltas**: Quantitative impact of toggling graph, reranker, and expansion.
- **Environment provenance**: Full record of hardware, versions, models, config, and dataset.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AblationResult",
    "BenchCase",
    "EvalEnvironment",
    "EvalReport",
    "IndexBenchmark",
    "QualityReport",
    "QueryEvalResult",
    "ablate",
    "average_precision",
    "collect_environment",
    "compare_reports",
    "evaluate",
    "hit_at_k",
    "load_cases",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_comparison_markdown",
]


@dataclass(frozen=True, slots=True)
class BenchCase:
    """A benchmark query with expected relevant documents and metadata."""

    query: str
    expect: frozenset[str]
    id: str = ""
    category: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        if self.id:
            return self.id
        import hashlib

        h = hashlib.sha256(self.query.encode("utf-8")).hexdigest()[:8]
        return f"case_{h}"

    @property
    def is_usable(self) -> bool:
        return bool(self.query.strip() and self.expect)


# --- metrics ---------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    """Fraction of expected notes appearing in the top k.

    Denominator: len(expected) -- the total number of ground-truth relevant notes.
    Relevance: Multi-document set overlap (|Expected ∩ Top_k| / |Expected|).

    Note on Recall@1 vs MRR:
    When a query has multiple relevant targets (e.g. |Expected| = 2), Recall@1 has
    a maximum possible mathematical ceiling of 1/2 = 0.50, because only 1 item can
    occupy rank 1. In contrast, MRR (and Hit@1) measures whether ANY relevant target
    was found at rank 1 (1.0 / rank of first hit).
    """
    if not expected or k <= 0:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return len(expected & set(top)) / len(expected)


def hit_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    """Binary hit indicator (Success@k).

    1.0 if AT LEAST ONE expected relevant note is in the deduplicated top k, else 0.0.
    Denominator: 1 (binary satisfaction indicator).
    """
    if not expected or k <= 0:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return 1.0 if bool(expected & set(top)) else 0.0


def reciprocal_rank(retrieved: Sequence[str], expected: frozenset[str]) -> float:
    """1/rank of the first correct hit. Rewards putting the answer first.

    Denominator: rank index (1-based) of the first retrieved note in expected.
    Relevance: First relevant item found.
    """
    for index, note_id in enumerate(_dedupe(retrieved), start=1):
        if note_id in expected:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: Sequence[str], expected: frozenset[str]) -> float:
    """Mean precision at each correct hit. Sensitive to ordering throughout.

    Denominator: len(expected) -- total expected relevant notes.
    Relevance: Precision (hits / rank) evaluated at each true hit position.
    """
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

    Discounts logarithmically by position: DCG@k = sum(1 / log2(i + 1)) for hits.
    Denominator: IDCG@k -- ideal DCG if all expected notes appeared in top ranks.
    """
    if not expected or k <= 0:
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


# --- evaluation structures -------------------------------------------------


@dataclass(slots=True)
class QueryEvalResult:
    """Detailed evaluation result for a single query."""

    case_id: str
    query: str
    expected: list[str]
    retrieved: list[str]
    ranked_chunks: list[dict[str, Any]]
    hit_at_1: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    average_precision: float
    ndcg_at_10: float
    latency_ms: float
    category: str | None = None
    rationale: str | None = None
    is_miss: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "category": self.category,
            "rationale": self.rationale,
            "expected": self.expected,
            "retrieved": self.retrieved,
            "hit@1": round(self.hit_at_1, 4),
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.reciprocal_rank, 4),
            "map": round(self.average_precision, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "latency_ms": round(self.latency_ms, 2),
            "is_miss": self.is_miss,
            "ranked_chunks": self.ranked_chunks,
        }


@dataclass(slots=True)
class QualityReport:
    """Aggregated retrieval quality over a case set."""

    label: str
    cases: int = 0
    hit_at_1: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg_at_10: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    queries: list[QueryEvalResult] = field(default_factory=list)
    category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

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

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))
        return ordered[index]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cases": self.cases,
            "hit@1": round(self.hit_at_1, 4),
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "p99_ms": round(self.p99_ms, 1),
            "mean_ms": round(self.mean_ms, 1),
            "total_misses": len(self.misses),
        }

    def as_dict(self) -> dict[str, Any]:
        row = self.as_row()
        row["misses"] = self.misses
        row["category_metrics"] = {
            cat: {k: round(v, 4) for k, v in m.items()} for cat, m in self.category_metrics.items()
        }
        row["queries"] = [q.as_dict() for q in self.queries]
        return row


@dataclass(slots=True)
class AblationResult:
    """A baseline plus one report per disabled component."""

    baseline: QualityReport
    variants: list[QualityReport] = field(default_factory=list)

    def deltas(self) -> list[dict[str, Any]]:
        """Change in recall@5 and p50 when each component is removed."""
        rows: list[dict[str, Any]] = []
        for variant in self.variants:
            rows.append(
                {
                    "disabled": variant.label,
                    "recall@1": round(variant.recall_at_1, 4),
                    "recall@5": round(variant.recall_at_5, 4),
                    "recall@10": round(variant.recall_at_10, 4),
                    "mrr": round(variant.mrr, 4),
                    "ndcg@10": round(variant.ndcg_at_10, 4),
                    "recall_delta": round(self.baseline.recall_at_5 - variant.recall_at_5, 4),
                    "recall1_delta": round(self.baseline.recall_at_1 - variant.recall_at_1, 4),
                    "recall5_delta": round(self.baseline.recall_at_5 - variant.recall_at_5, 4),
                    "recall10_delta": round(self.baseline.recall_at_10 - variant.recall_at_10, 4),
                    "mrr_delta": round(self.baseline.mrr - variant.mrr, 4),
                    "ndcg_delta": round(self.baseline.ndcg_at_10 - variant.ndcg_at_10, 4),
                    "p50_saved_ms": round(self.baseline.p50_ms - variant.p50_ms, 1),
                }
            )
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "deltas": self.deltas(),
            "variants": [v.as_dict() for v in self.variants],
        }


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
        if not self.reindex_elapsed_s:
            return 0.0
        return self.elapsed_s / self.reindex_elapsed_s


@dataclass(slots=True)
class EvalEnvironment:
    """Provenance and execution environment metadata."""

    timestamp_utc: str
    command: str
    cortex_version: str
    python_version: str
    platform: str
    cpu_count: int
    vault_path: str
    total_notes: int
    total_chunks: int
    store_type: str
    embed_model: str
    embedder_class: str
    embed_dimensions: int
    offline: bool
    rerank_model: str | None
    reranker_class: str | None
    rerank_enabled: bool
    config: dict[str, Any]
    dependency_versions: dict[str, str]
    git_commit: str | None = None
    corpus_fingerprint: str | None = None
    case_file_fingerprint: str | None = None
    dataset_version: str | None = None
    dataset_provenance: str | None = None
    dataset_limitations: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalReport:
    """Comprehensive evaluation report with export capabilities."""

    environment: EvalEnvironment
    quality: QualityReport
    ablation: AblationResult | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "environment": self.environment.as_dict(),
            "dataset_info": self.dataset_info,
            "summary": self.quality.as_row(),
            "category_metrics": self.quality.category_metrics,
            "ablation": self.ablation.as_dict() if self.ablation else None,
            "queries": [q.as_dict() for q in self.quality.queries],
        }

    def to_json(self, path: Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: Path | None = None) -> str:
        q = self.quality
        env = self.environment

        md: list[str] = [
            f"# Cortex Retrieval Evaluation Report — {q.label}",
            f"**Date (UTC):** {env.timestamp_utc}  ·  **Cortex:** {env.cortex_version}  ·  "
            f"**Mode:** {'Offline (deterministic)' if env.offline else 'Dense (live model)'}",
            "",
            "## 1. Summary Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| **Cases Evaluated** | {q.cases} |",
            f"| **Recall@1** | {q.recall_at_1:.4f} |",
            f"| **Recall@5** | {q.recall_at_5:.4f} |",
            f"| **Recall@10** | {q.recall_at_10:.4f} |",
            f"| **MRR** | {q.mrr:.4f} |",
            f"| **MAP** | {q.map_score:.4f} |",
            f"| **nDCG@10** | {q.ndcg_at_10:.4f} |",
            f"| **p50 Latency** | {q.p50_ms:.1f} ms |",
            f"| **p95 Latency** | {q.p95_ms:.1f} ms |",
            f"| **p99 Latency** | {q.p99_ms:.1f} ms |",
            f"| **Total Misses** | {len(q.misses)} |",
            "",
        ]

        if q.category_metrics:
            md.extend(
                [
                    "## 2. Metrics by Category",
                    "",
                    "| Category | Count | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for cat, m in sorted(q.category_metrics.items()):
                cnt = int(m.get("count", 0))
                r1 = m.get("recall@1", 0.0)
                r5 = m.get("recall@5", 0.0)
                r10 = m.get("recall@10", 0.0)
                mrr = m.get("mrr", 0.0)
                ndcg = m.get("ndcg@10", 0.0)
                row_parts = [
                    f"| `{cat}`",
                    f"{cnt}",
                    f"{r1:.3f}",
                    f"{r5:.3f}",
                    f"{r10:.3f}",
                    f"{mrr:.3f}",
                    f"{ndcg:.3f} |",
                ]
                md.append(" | ".join(row_parts))
            md.append("")

        if self.ablation:
            deltas = self.ablation.deltas()
            if deltas:
                md.extend(
                    [
                        "## 3. Component Ablation",
                        "",
                        "| Disabled | Recall@5 | Δ Recall@5 | Δ nDCG@10 | Saved |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for row in deltas:
                    d_rec = row["recall5_delta"]
                    d_ndcg = row["ndcg_delta"]
                    saved = row["p50_saved_ms"]
                    md.append(
                        f"| **{row['disabled']}** | {row['recall@5']:.4f} | "
                        f"{d_rec:+.4f} | {d_ndcg:+.4f} | {saved:+.1f} ms |"
                    )
                md.append("")

        if q.misses:
            md.extend(
                [
                    f"## 4. Query Failures / Misses ({len(q.misses)})",
                    "",
                    "| Query | Expected |",
                    "|---|---|",
                ]
            )
            for q_res in q.queries:
                if q_res.is_miss:
                    md.append(f"| {q_res.query} | `{', '.join(q_res.expected)}` |")
            md.append("")

        md.extend(
            [
                "## 5. Environment & Corpus Provenance",
                "",
                "| Property | Value |",
                "|---|---|",
                f"| **Corpus Vault** | `{env.vault_path}` |",
                f"| **Notes / Chunks** | {env.total_notes} notes / {env.total_chunks} chunks |",
                (
                    f"| **Store / Embedder** | {env.store_type} / "
                    f"{env.embed_model} ({env.embedder_class}) |"
                ),
                f"| **Embedding Dims** | {env.embed_dimensions} |",
                (
                    f"| **Reranker** | {env.rerank_model or 'None'} "
                    f"({env.reranker_class or 'disabled'}) |"
                ),
                (
                    f"| **Python / OS** | Python {env.python_version} "
                    f"({env.platform}, {env.cpu_count} CPUs) |"
                ),
                f"| **Command** | `{env.command}` |",
            ]
        )

        text = "\n".join(md)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: Path | None = None, per_query: bool = True) -> str:
        buf = io.StringIO()
        if per_query:
            fieldnames = [
                "case_id",
                "query",
                "category",
                "recall@1",
                "recall@5",
                "recall@10",
                "mrr",
                "map",
                "ndcg@10",
                "latency_ms",
                "is_miss",
                "expected",
                "top_retrieved",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for q in self.quality.queries:
                writer.writerow(
                    {
                        "case_id": q.case_id,
                        "query": q.query,
                        "category": q.category or "",
                        "recall@1": round(q.recall_at_1, 4),
                        "recall@5": round(q.recall_at_5, 4),
                        "recall@10": round(q.recall_at_10, 4),
                        "mrr": round(q.reciprocal_rank, 4),
                        "map": round(q.average_precision, 4),
                        "ndcg@10": round(q.ndcg_at_10, 4),
                        "latency_ms": round(q.latency_ms, 2),
                        "is_miss": q.is_miss,
                        "expected": "; ".join(q.expected),
                        "top_retrieved": "; ".join(q.retrieved[:5]),
                    }
                )
        else:
            row = self.quality.as_row()
            writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        text = buf.getvalue()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text


# --- run comparison --------------------------------------------------------


def compare_reports(
    baseline: dict[str, Any] | EvalReport, candidate: dict[str, Any] | EvalReport
) -> dict[str, Any]:
    """Compare two evaluation runs and compute deltas."""
    b_dict = baseline.to_dict() if isinstance(baseline, EvalReport) else baseline
    c_dict = candidate.to_dict() if isinstance(candidate, EvalReport) else candidate

    b_sum = b_dict.get("summary", {})
    c_sum = c_dict.get("summary", {})

    metrics = ["recall@1", "recall@5", "recall@10", "mrr", "map", "ndcg@10", "p50_ms", "p95_ms"]
    deltas: dict[str, Any] = {}
    for m in metrics:
        b_val = float(b_sum.get(m, 0.0))
        c_val = float(c_sum.get(m, 0.0))
        deltas[m] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": round(c_val - b_val, 4),
            "pct_change": round(((c_val - b_val) / b_val * 100.0) if b_val > 0 else 0.0, 2),
        }

    b_queries = {q["case_id"]: q for q in b_dict.get("queries", [])}
    c_queries = {q["case_id"]: q for q in c_dict.get("queries", [])}

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for cid, cq in c_queries.items():
        if cid in b_queries:
            bq = b_queries[cid]
            b_ndcg = bq.get("ndcg@10", 0.0)
            c_ndcg = cq.get("ndcg@10", 0.0)
            diff = c_ndcg - b_ndcg
            if diff > 0.001:
                improved.append(cid)
            elif diff < -0.001:
                regressed.append(cid)
            else:
                unchanged.append(cid)

    return {
        "metrics": deltas,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(unchanged),
        "improved_cases": improved,
        "regressed_cases": regressed,
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Cortex Evaluation Run Comparison",
        "",
        "| Metric | Baseline | Candidate | Δ Absolute | % Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, data in comparison.get("metrics", {}).items():
        b_val = data["baseline"]
        c_val = data["candidate"]
        delta = data["delta"]
        pct = data["pct_change"]
        sign = "+" if delta > 0 else ""
        lines.append(f"| `{metric}` | {b_val} | {c_val} | {sign}{delta} | {sign}{pct}% |")

    lines.extend(
        [
            "",
            f"- **Improved cases:** {comparison.get('improved_count', 0)}",
            f"- **Regressed cases:** {comparison.get('regressed_count', 0)}",
            f"- **Unchanged cases:** {comparison.get('unchanged_count', 0)}",
        ]
    )
    return "\n".join(lines)


# --- loading ---------------------------------------------------------------


def load_cases(path: Path) -> list[BenchCase]:
    """Load ground-truth benchmark cases from YAML or JSON.

    Supports both the rich Cortex benchmark schema and legacy formats.
    """
    text = path.read_text(encoding="utf-8")
    raw: Any
    if path.suffix.lower() in {".json"}:
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text)

    cases_list: list[Any] = []
    if isinstance(raw, dict):
        if "cases" in raw and isinstance(raw["cases"], list):
            cases_list = raw["cases"]
        else:
            cases_list = [{"query": k, "expect": v} for k, v in raw.items()]
    elif isinstance(raw, list):
        cases_list = raw

    cases: list[BenchCase] = []
    for entry in cases_list:
        if not isinstance(entry, dict):
            continue
        query = str(entry.get("query", "")).strip()
        expected = entry.get("expect") or entry.get("expected") or []
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list):
            continue
        case_id = str(entry.get("id", "")).strip()
        category = str(entry.get("category", "")).strip() or None
        rationale = str(entry.get("rationale", "")).strip() or None
        expect_set = frozenset(str(item).strip() for item in expected if str(item).strip())

        case = BenchCase(
            query=query,
            expect=expect_set,
            id=case_id,
            category=category,
            rationale=rationale,
            metadata=entry.get("metadata", {}),
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
    """Run every case through the retrieval engine and aggregate metrics."""
    report = QualityReport(label=label, cases=len(cases))
    if not cases:
        return report

    recalls1: list[float] = []
    recalls5: list[float] = []
    recalls10: list[float] = []
    rrs: list[float] = []
    aps: list[float] = []
    ndcgs: list[float] = []

    cat_acc: dict[str, list[dict[str, float]]] = defaultdict(list)

    for case in cases:
        started = time.perf_counter()
        result = engine.retrieve(case.query, top_k=top_k, **retrieve_kwargs)
        latency = (time.perf_counter() - started) * 1000
        report.latencies_ms.append(latency)

        retrieved = [scored.chunk.note_id for scored in result.chunks]
        deduped_retrieved = _dedupe(retrieved)

        h1 = 1.0 if (retrieved and retrieved[0] in case.expect) else 0.0
        r1 = recall_at_k(retrieved, case.expect, 1)
        r5 = recall_at_k(retrieved, case.expect, 5)
        r10 = recall_at_k(retrieved, case.expect, 10)
        rr = reciprocal_rank(retrieved, case.expect)
        ap = average_precision(retrieved, case.expect)
        ndcg = ndcg_at_k(retrieved, case.expect, 10)

        recalls1.append(r1)
        recalls5.append(r5)
        recalls10.append(r10)
        rrs.append(rr)
        aps.append(ap)
        ndcgs.append(ndcg)

        is_miss = not (case.expect & set(retrieved))
        if is_miss:
            report.misses.append(case.query)

        ranked_chunks: list[dict[str, Any]] = [
            {
                "rank": idx,
                "note_id": sc.chunk.note_id,
                "chunk_id": sc.chunk.chunk_id,
                "citation": sc.chunk.citation,
                "score": round(sc.score, 5),
                "components": sorted(sc.components) or [sc.source],
                "is_hit": sc.chunk.note_id in case.expect,
            }
            for idx, sc in enumerate(result.chunks, start=1)
        ]

        query_res = QueryEvalResult(
            case_id=case.case_id,
            query=case.query,
            expected=sorted(case.expect),
            retrieved=deduped_retrieved,
            ranked_chunks=ranked_chunks,
            hit_at_1=h1,
            recall_at_1=r1,
            recall_at_5=r5,
            recall_at_10=r10,
            reciprocal_rank=rr,
            average_precision=ap,
            ndcg_at_10=ndcg,
            latency_ms=latency,
            category=case.category,
            rationale=case.rationale,
            is_miss=is_miss,
        )
        report.queries.append(query_res)

        if case.category:
            cat_acc[case.category].append(
                {
                    "recall@1": r1,
                    "recall@5": r5,
                    "recall@10": r10,
                    "mrr": rr,
                    "map": ap,
                    "ndcg@10": ndcg,
                    "latency_ms": latency,
                }
            )

    report.recall_at_1 = statistics.fmean(recalls1)
    report.recall_at_5 = statistics.fmean(recalls5)
    report.recall_at_10 = statistics.fmean(recalls10)
    report.mrr = statistics.fmean(rrs)
    report.map_score = statistics.fmean(aps)
    report.ndcg_at_10 = statistics.fmean(ndcgs)

    # Aggregate category metrics
    for cat, items in cat_acc.items():
        report.category_metrics[cat] = {
            "count": float(len(items)),
            "recall@1": statistics.fmean([x["recall@1"] for x in items]),
            "recall@5": statistics.fmean([x["recall@5"] for x in items]),
            "recall@10": statistics.fmean([x["recall@10"] for x in items]),
            "mrr": statistics.fmean([x["mrr"] for x in items]),
            "ndcg@10": statistics.fmean([x["ndcg@10"] for x in items]),
            "p50_ms": statistics.median([x["latency_ms"] for x in items]),
        }

    return report


def ablate(engine: Any, cases: Sequence[BenchCase], *, top_k: int = 10) -> AblationResult:
    """Measure the contribution of each retrieval component by disabling in turn."""
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


def _get_pkg_version(name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return "not installed"


def collect_environment(
    runtime: Any,
    command: str = "cortex eval",
    dataset_path: Path | None = None,
) -> EvalEnvironment:
    """Extract full environment provenance and dependency versions."""
    import sys

    from cortex import __version__

    deps = {
        "lancedb": _get_pkg_version("lancedb"),
        "pyarrow": _get_pkg_version("pyarrow"),
        "sentence_transformers": _get_pkg_version("sentence-transformers"),
        "mcp": _get_pkg_version("mcp"),
        "pydantic": _get_pkg_version("pydantic"),
        "typer": _get_pkg_version("typer"),
        "rich": _get_pkg_version("rich"),
        "pymupdf4llm": _get_pkg_version("pymupdf4llm"),
    }

    counts = runtime.catalog.stats()
    s = runtime.settings

    dataset_ver = None
    dataset_prov = None
    dataset_lim = None
    if dataset_path and dataset_path.exists():
        try:
            import yaml

            raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                dataset_ver = str(raw.get("version", "")) or None
                dataset_prov = str(raw.get("provenance", "")) or None
                dataset_lim = str(raw.get("limitations", "")) or None
        except Exception:
            pass

    return EvalEnvironment(
        timestamp_utc=datetime.now(UTC).isoformat(),
        command=command,
        cortex_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        vault_path=str(s.vault_path),
        total_notes=counts["notes"],
        total_chunks=counts["chunks"],
        store_type=type(runtime.store).__name__,
        embed_model=s.embed_model,
        embedder_class=type(runtime.embedder).__name__,
        embed_dimensions=s.embed_dimensions,
        offline=type(runtime.embedder).__name__ == "HashEmbedder",
        rerank_model=s.rerank_model if s.rerank_enabled else None,
        reranker_class=type(runtime.reranker).__name__ if runtime.reranker else None,
        rerank_enabled=bool(s.rerank_enabled and runtime.reranker),
        config={
            "top_k": s.top_k,
            "dense_k": s.dense_k,
            "fts_k": s.fts_k,
            "graph_hops": s.graph_hops,
            "fusion_weights": s.fusion_weights,
            "expansion_enabled": s.expansion_enabled,
            "temporal_enabled": s.temporal_enabled,
        },
        dependency_versions=deps,
        dataset_version=dataset_ver,
        dataset_provenance=dataset_prov,
        dataset_limitations=dataset_lim,
    )
