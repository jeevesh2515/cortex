"""Comprehensive unit and integration tests for the Cortex evaluation harness."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cortex.bench import (
    BenchCase,
    EvalEnvironment,
    EvalReport,
    QualityReport,
    QueryEvalResult,
    ablate,
    average_precision,
    compare_reports,
    evaluate,
    load_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_comparison_markdown,
)
from cortex.catalog import Catalog
from cortex.cli import app
from cortex.index.store import MemoryStore
from cortex.ingest.pipeline import IndexPipeline
from cortex.llm.providers import HashEmbedder
from cortex.retrieve.engine import RetrievalEngine
from cortex.thermal.governor import PowerSource, Reading, StaticProbe

runner = CliRunner()

_FULL_PROBE = StaticProbe(
    Reading(
        power=PowerSource.AC,
        battery_percent=100,
        cpu_speed_limit=100,
        load_average=0.0,
        cpu_count=4,
    )
)


# --- 1. Hand-computed metric tests with exact rankings ----------------------


class TestMetricCalculations:
    def test_recall_at_k_exact_values(self) -> None:
        expected = frozenset({"docA.md", "docB.md"})

        # Both present in top 2
        assert recall_at_k(["docA.md", "docB.md", "docC.md"], expected, k=1) == 0.5
        assert recall_at_k(["docA.md", "docB.md", "docC.md"], expected, k=2) == 1.0
        assert recall_at_k(["docA.md", "docB.md", "docC.md"], expected, k=5) == 1.0

        # One present at rank 1, one at rank 4
        retrieved = ["docA.md", "docX.md", "docY.md", "docB.md"]
        assert recall_at_k(retrieved, expected, k=1) == 0.5
        assert recall_at_k(retrieved, expected, k=3) == 0.5
        assert recall_at_k(retrieved, expected, k=4) == 1.0

        # None present
        assert recall_at_k(["docX.md", "docY.md"], expected, k=5) == 0.0

    def test_recall_edge_cases(self) -> None:
        assert recall_at_k([], frozenset({"docA.md"}), k=5) == 0.0
        assert recall_at_k(["docA.md"], frozenset(), k=5) == 0.0
        assert recall_at_k(["docA.md"], frozenset({"docA.md"}), k=0) == 0.0
        assert recall_at_k(["docA.md"], frozenset({"docA.md"}), k=-1) == 0.0

    def test_recall_deduplication(self) -> None:
        # Multiple chunks from the same note must not count as multiple hits
        retrieved = ["docA.md", "docA.md", "docA.md", "docB.md"]
        expected = frozenset({"docA.md", "docB.md"})
        # After deduplication: ["docA.md", "docB.md"]
        assert recall_at_k(retrieved, expected, k=1) == 0.5
        assert recall_at_k(retrieved, expected, k=2) == 1.0

    def test_reciprocal_rank_exact_values(self) -> None:
        expected = frozenset({"target.md"})

        # Rank 1 -> 1.0
        assert reciprocal_rank(["target.md", "other.md"], expected) == 1.0
        # Rank 2 -> 0.5
        assert reciprocal_rank(["other.md", "target.md"], expected) == 0.5
        # Rank 3 -> 1/3
        assert reciprocal_rank(["x.md", "y.md", "target.md"], expected) == pytest.approx(1.0 / 3.0)
        # Rank 4 -> 0.25
        assert reciprocal_rank(["x.md", "y.md", "z.md", "target.md"], expected) == 0.25
        # Miss -> 0.0
        assert reciprocal_rank(["x.md", "y.md", "z.md"], expected) == 0.0
        # Empty -> 0.0
        assert reciprocal_rank([], expected) == 0.0

    def test_reciprocal_rank_deduplication(self) -> None:
        # Duplicates before hit should collapse
        retrieved = ["x.md", "x.md", "target.md"]
        expected = frozenset({"target.md"})
        # Deduped: ["x.md", "target.md"] -> rank 2
        assert reciprocal_rank(retrieved, expected) == 0.5

    def test_average_precision_exact_values(self) -> None:
        expected = frozenset({"A.md", "B.md"})

        # Perfect ranking: hits at 1 and 2
        # P@1 = 1/1, P@2 = 2/2 -> AP = (1.0 + 1.0) / 2 = 1.0
        assert average_precision(["A.md", "B.md", "C.md"], expected) == 1.0

        # Hits at rank 1 and rank 3:
        # P@1 = 1/1 = 1.0, P@3 = 2/3 -> AP = (1.0 + 2/3) / 2 = 5/6 = 0.8333333333333334
        assert average_precision(["A.md", "X.md", "B.md"], expected) == pytest.approx(5.0 / 6.0)

        # Hits at rank 2 and rank 3:
        # P@2 = 1/2 = 0.5, P@3 = 2/3 -> AP = (1/2 + 2/3) / 2 = (7/6) / 2 = 7/12 = 0.5833333333333334
        assert average_precision(["X.md", "A.md", "B.md"], expected) == pytest.approx(7.0 / 12.0)

        # Only one hit at rank 2:
        # P@2 = 1/2 = 0.5 -> AP = 0.5 / 2 = 0.25
        assert average_precision(["X.md", "A.md", "Y.md"], expected) == 0.25

        # Zero hits -> 0.0
        assert average_precision(["X.md", "Y.md"], expected) == 0.0
        # Empty expected -> 0.0
        assert average_precision(["A.md"], frozenset()) == 0.0

    def test_ndcg_at_k_exact_values(self) -> None:
        expected = frozenset({"A.md", "B.md"})

        # Perfect ranking: A at 1, B at 2
        # DCG = 1/log2(2) + 1/log2(3) = 1 + 0.63092975 = 1.63092975
        # IDCG = 1.63092975 -> nDCG = 1.0
        idcg_2 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(["A.md", "B.md", "C.md"], expected, k=10) == pytest.approx(1.0)

        # Hits at rank 2 and rank 3:
        # DCG = 1/log2(3) + 1/log2(4) = 0.63092975 + 0.5 = 1.13092975
        # nDCG = 1.13092975 / 1.63092975 = 0.6934264
        expected_ndcg = (1.0 / math.log2(3) + 1.0 / math.log2(4)) / idcg_2
        assert ndcg_at_k(["X.md", "A.md", "B.md"], expected, k=10) == pytest.approx(expected_ndcg)

        # Only 1 hit at rank 1:
        # DCG = 1.0 / log2(2) = 1.0 -> nDCG = 1.0 / 1.63092975 = 0.61314719
        assert ndcg_at_k(["A.md", "X.md", "Y.md"], expected, k=10) == pytest.approx(1.0 / idcg_2)

        # Edge cases
        assert ndcg_at_k([], expected, k=10) == 0.0
        assert ndcg_at_k(["A.md"], frozenset(), k=10) == 0.0
        assert ndcg_at_k(["A.md"], expected, k=0) == 0.0


# --- 2. BenchCase and load_cases schema tests -------------------------------


class TestBenchCaseLoading:
    def test_load_rich_cases_yaml(self, tmp_path: Path) -> None:
        case_file = tmp_path / "cases.yaml"
        case_file.write_text(
            """
version: "1.0"
dataset: "test-eval"
provenance: "Unit test cases"
cases:
  - id: "case_01"
    query: "hybrid search vectors"
    expect: ["Retrieval.md", "Embeddings.md"]
    category: "hybrid_core"
    rationale: "Tests multi-doc retrieval."
  - id: "case_02"
    query: "single note search"
    expect: "Solo.md"
""",
            encoding="utf-8",
        )
        cases = load_cases(case_file)
        assert len(cases) == 2
        assert cases[0].id == "case_01"
        assert cases[0].query == "hybrid search vectors"
        assert cases[0].expect == frozenset({"Retrieval.md", "Embeddings.md"})
        assert cases[0].category == "hybrid_core"
        assert cases[0].rationale == "Tests multi-doc retrieval."

        assert cases[1].id == "case_02"
        assert cases[1].expect == frozenset({"Solo.md"})

    def test_load_legacy_list_format(self, tmp_path: Path) -> None:
        case_file = tmp_path / "cases.yaml"
        case_file.write_text(
            "- query: what about chunking?\n  expect: [Chunking.md]\n",
            encoding="utf-8",
        )
        cases = load_cases(case_file)
        assert len(cases) == 1
        assert cases[0].query == "what about chunking?"
        assert cases[0].expect == frozenset({"Chunking.md"})
        assert cases[0].case_id.startswith("case_")

    def test_load_dict_mapping_format(self, tmp_path: Path) -> None:
        case_file = tmp_path / "cases.yaml"
        case_file.write_text("sourdough query: [Sourdough.md]\n", encoding="utf-8")
        cases = load_cases(case_file)
        assert len(cases) == 1
        assert cases[0].query == "sourdough query"
        assert cases[0].expect == frozenset({"Sourdough.md"})

    def test_load_json_format(self, tmp_path: Path) -> None:
        case_file = tmp_path / "cases.json"
        case_file.write_text(
            json.dumps(
                [
                    {
                        "id": "json_1",
                        "query": "query in json",
                        "expect": ["Target.md"],
                        "category": "json_test",
                    }
                ]
            ),
            encoding="utf-8",
        )
        cases = load_cases(case_file)
        assert len(cases) == 1
        assert cases[0].id == "json_1"
        assert cases[0].category == "json_test"

    def test_malformed_entries_skipped_safely(self, tmp_path: Path) -> None:
        case_file = tmp_path / "cases.yaml"
        case_file.write_text(
            """
- query: "valid query"
  expect: ["A.md"]
- query: ""
  expect: ["B.md"]
- query: "missing expect"
- not_a_dict_entry
""",
            encoding="utf-8",
        )
        cases = load_cases(case_file)
        assert len(cases) == 1
        assert cases[0].query == "valid query"


# --- 3. End-to-end evaluation & ablation on synthetic engine -----------------


class TestEvaluationExecution:
    def _setup_engine(self, tmp_path: Path) -> tuple[RetrievalEngine, list[BenchCase]]:
        root = tmp_path / "vault"
        root.mkdir()
        (root / "Retrieval.md").write_text(
            "# Retrieval\n\nHybrid search combines dense vectors with BM25.\nSee [[Embeddings]].\n",
            encoding="utf-8",
        )
        (root / "Embeddings.md").write_text(
            "# Embeddings\n\nDense vector representations capture semantic similarity.\n",
            encoding="utf-8",
        )
        (root / "Sourdough.md").write_text(
            "# Sourdough\n\nFeed the wild yeast starter and bulk ferment.\n",
            encoding="utf-8",
        )

        pipeline = IndexPipeline(
            vault=root,
            store=MemoryStore(),
            catalog=Catalog(),
            embedder=HashEmbedder(dimensions=128),
        )
        pipeline.run()
        engine = RetrievalEngine(
            pipeline.store,
            pipeline.embedder,
            graph=pipeline.build_graph(),
            top_k=5,
        )
        cases = [
            BenchCase(
                id="case_retrieval",
                query="hybrid search dense vectors",
                expect=frozenset({"Retrieval.md"}),
                category="search_tech",
                rationale="Exact hybrid search test.",
            ),
            BenchCase(
                id="case_sourdough",
                query="wild yeast sourdough starter",
                expect=frozenset({"Sourdough.md"}),
                category="baking",
                rationale="Food science query.",
            ),
        ]
        return engine, cases

    def test_evaluate_populates_all_metrics(self, tmp_path: Path) -> None:
        engine, cases = self._setup_engine(tmp_path)
        report = evaluate(engine, cases, top_k=5)

        assert report.cases == 2
        assert 0.0 <= report.recall_at_1 <= 1.0
        assert report.recall_at_5 > 0.0
        assert report.recall_at_10 >= report.recall_at_5
        assert report.mrr > 0.0
        assert report.map_score > 0.0
        assert report.ndcg_at_10 > 0.0
        assert len(report.queries) == 2
        assert report.p50_ms > 0.0
        assert report.p95_ms >= report.p50_ms

        # Check per-query breakdown
        q0 = report.queries[0]
        assert q0.case_id == "case_retrieval"
        assert q0.category == "search_tech"
        assert q0.expected == ["Retrieval.md"]
        assert len(q0.ranked_chunks) > 0
        assert "is_hit" in q0.ranked_chunks[0]

        # Category metrics
        assert "search_tech" in report.category_metrics
        assert "baking" in report.category_metrics

    def test_ablation_produces_deltas(self, tmp_path: Path) -> None:
        engine, cases = self._setup_engine(tmp_path)
        ablation = ablate(engine, cases, top_k=5)

        assert ablation.baseline.cases == 2
        deltas = ablation.deltas()
        assert len(deltas) >= 1
        for row in deltas:
            assert "disabled" in row
            assert "recall@5" in row
            assert "recall5_delta" in row
            assert "ndcg_delta" in row
            assert "p50_saved_ms" in row


# --- 4. Report export formats (JSON, Markdown, CSV) & comparison -------------


class TestReportExportsAndComparison:
    def _dummy_report(self) -> EvalReport:
        env = EvalEnvironment(
            timestamp_utc="2026-08-28T12:00:00Z",
            command="cortex eval",
            cortex_version="0.1.0",
            python_version="3.12.0",
            platform="macOS",
            cpu_count=8,
            vault_path="/test/vault",
            total_notes=5,
            total_chunks=10,
            store_type="MemoryStore",
            embed_model="test-embedder",
            embedder_class="HashEmbedder",
            embed_dimensions=128,
            offline=True,
            rerank_model=None,
            reranker_class=None,
            rerank_enabled=False,
            config={"top_k": 5},
            dependency_versions={"lancedb": "0.36.0"},
        )
        query_res = QueryEvalResult(
            case_id="case_01",
            query="test query",
            expected=["Doc1.md"],
            retrieved=["Doc1.md", "Doc2.md"],
            ranked_chunks=[
                {
                    "rank": 1,
                    "note_id": "Doc1.md",
                    "chunk_id": "Doc1.md#1",
                    "citation": "Doc1.md",
                    "score": 0.05,
                    "components": ["dense"],
                    "is_hit": True,
                }
            ],
            hit_at_1=1.0,
            recall_at_1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            reciprocal_rank=1.0,
            average_precision=1.0,
            ndcg_at_10=1.0,
            latency_ms=15.0,
            category="test_cat",
            rationale="Test rationale.",
            is_miss=False,
        )
        quality = QualityReport(
            label="baseline",
            cases=1,
            recall_at_1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            map_score=1.0,
            ndcg_at_10=1.0,
            latencies_ms=[15.0],
            queries=[query_res],
            category_metrics={
                "test_cat": {
                    "count": 1.0,
                    "recall@1": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr": 1.0,
                    "ndcg@10": 1.0,
                    "latency_ms": 15.0,
                }
            },
        )
        return EvalReport(environment=env, quality=quality)

    def test_json_export_and_roundtrip(self, tmp_path: Path) -> None:
        report = self._dummy_report()
        out_path = tmp_path / "report.json"
        text = report.to_json(out_path)

        assert out_path.exists()
        parsed = json.loads(text)
        assert parsed["schema_version"] == "1.0"
        assert parsed["summary"]["recall@1"] == 1.0
        assert parsed["summary"]["ndcg@10"] == 1.0
        assert len(parsed["queries"]) == 1
        assert parsed["environment"]["store_type"] == "MemoryStore"

    def test_markdown_export(self, tmp_path: Path) -> None:
        report = self._dummy_report()
        out_path = tmp_path / "report.md"
        text = report.to_markdown(out_path)

        assert out_path.exists()
        assert "# Cortex Retrieval Evaluation Report" in text
        assert "| **Recall@1** | 1.0000 |" in text
        assert "| `test_cat` |" in text
        assert "MemoryStore" in text

    def test_csv_export(self, tmp_path: Path) -> None:
        report = self._dummy_report()
        out_path = tmp_path / "report.csv"
        text = report.to_csv(out_path, per_query=True)

        assert out_path.exists()
        assert "case_id,query,category" in text
        assert "case_01,test query,test_cat" in text

    def test_compare_reports(self) -> None:
        r1 = self._dummy_report()
        r2 = self._dummy_report()
        # Simulate improvement in candidate
        r2.quality.recall_at_1 = 1.0
        r1.quality.recall_at_1 = 0.5
        r1.quality.queries[0].ndcg_at_10 = 0.5
        r2.quality.queries[0].ndcg_at_10 = 1.0

        comparison = compare_reports(r1, r2)
        assert "recall@1" in comparison["metrics"]
        assert comparison["metrics"]["recall@1"]["delta"] == 0.5
        assert comparison["improved_count"] == 1
        assert comparison["regressed_count"] == 0

        md = render_comparison_markdown(comparison)
        assert "# Cortex Evaluation Run Comparison" in md
        assert "`recall@1`" in md


# --- 5. CLI end-to-end integration tests -------------------------------------


class TestCLIEvalCommands:
    @pytest.fixture
    def fixture_setup(self, tmp_path: Path) -> tuple[Path, Path]:
        vault_dir = tmp_path / "eval_vault"
        vault_dir.mkdir()
        (vault_dir / "NoteA.md").write_text(
            "# Note A\n\nHybrid retrieval and embeddings.\n", encoding="utf-8"
        )
        (vault_dir / "NoteB.md").write_text(
            "# Note B\n\nSourdough fermentation and starter.\n", encoding="utf-8"
        )

        cases_file = tmp_path / "cases.yaml"
        cases_file.write_text(
            """
cases:
  - id: "case_01"
    query: "hybrid retrieval embeddings"
    expect: ["NoteA.md"]
    category: "tech"
  - id: "case_02"
    query: "sourdough starter fermentation"
    expect: ["NoteB.md"]
    category: "baking"
""",
            encoding="utf-8",
        )
        return vault_dir, cases_file

    def test_cortex_eval_cli_offline(
        self, fixture_setup: tuple[Path, Path], tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        vault_dir, cases_file = fixture_setup
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "cortex_data"))

        json_out = tmp_path / "eval.json"
        md_out = tmp_path / "eval.md"
        csv_out = tmp_path / "eval.csv"

        result = runner.invoke(
            app,
            [
                "eval",
                "--cases",
                str(cases_file),
                "--vault",
                str(vault_dir),
                "--offline",
                "--ignore-thermal",
                "--output-json",
                str(json_out),
                "--output-md",
                str(md_out),
                "--output-csv",
                str(csv_out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Retrieval Quality" in result.stdout
        assert "Recall@1" in result.stdout
        assert "Metrics by Category" in result.stdout
        assert json_out.exists()
        assert md_out.exists()
        assert csv_out.exists()

    def test_cortex_bench_cli_offline(
        self, fixture_setup: tuple[Path, Path], tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        vault_dir, cases_file = fixture_setup
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "cortex_data"))

        result = runner.invoke(
            app,
            [
                "bench",
                "--cases",
                str(cases_file),
                "--vault",
                str(vault_dir),
                "--offline",
                "--ignore-thermal",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Retrieval Quality" in result.stdout
