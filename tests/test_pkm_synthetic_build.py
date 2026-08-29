"""
Tests for Milestone 3C PKM synthetic generator and experiment pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "training"))

from build_pkm_synthetic import SEED_NOTES, generate_pkm_synthetic_dataset  # noqa: E402
from experiment import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from training_schema import (  # noqa: E402
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    validate_no_eval_leakage,
)


class TestPKMSyntheticBuild:
    def test_seed_notes_isolation_from_eval(self) -> None:
        """Verify no seed note title or path overlaps with eval positive IDs."""
        for doc_id, doc_data in SEED_NOTES.items():
            assert doc_id not in EVAL_POSITIVE_IDS
            assert doc_data["title"] not in EVAL_POSITIVE_IDS
            for sec in doc_data["sections"]:
                for query_text, _ in sec["queries"]:
                    assert query_text.strip().lower() not in EVAL_QUERIES

    def test_generate_pkm_synthetic_dataset_counts_and_splits(self) -> None:
        train_exs, dev_exs, review_pack = generate_pkm_synthetic_dataset(200, 50)
        assert len(train_exs) == 200
        assert len(dev_exs) == 50
        assert len(review_pack) == 20

        # Verify review pack structure
        for item in review_pack:
            assert item["review_status"] == "unreviewed"
            assert "reviewer_decision" in item
            assert "query" in item
            assert "positive_id" in item

        # Verify dataset schema
        ds_train = TrainingDataset(dataset_id="test_synth_train", examples=train_exs)
        ds_dev = TrainingDataset(dataset_id="test_synth_dev", examples=dev_exs)
        assert len(ds_train.examples) == 200
        assert len(ds_dev.examples) == 50

        # Verify document disjointness
        train_docs = {ex.positive_id.split("#")[0] for ex in train_exs}
        dev_docs = {ex.positive_id.split("#")[0] for ex in dev_exs}
        assert train_docs & dev_docs == set()

        # Verify zero eval leakage
        violations = validate_no_eval_leakage(train_exs + dev_exs)
        assert violations == []


class TestExperimentMetrics:
    def test_recall_at_k(self) -> None:
        expected = frozenset(["docA", "docB"])
        retrieved = ["docA", "docC", "docD"]
        assert recall_at_k(retrieved, expected, 1) == 0.5
        assert recall_at_k(retrieved, expected, 2) == 0.5
        assert recall_at_k(retrieved, expected, 5) == 0.5

    def test_reciprocal_rank(self) -> None:
        expected = frozenset(["docB"])
        retrieved = ["docA", "docB", "docC"]
        assert reciprocal_rank(retrieved, expected) == 0.5

        retrieved_none = ["docA", "docC"]
        assert reciprocal_rank(retrieved_none, expected) == 0.0

    def test_ndcg_at_k(self) -> None:
        expected = frozenset(["docA"])
        retrieved_rank1 = ["docA", "docB"]
        retrieved_rank2 = ["docB", "docA"]
        assert ndcg_at_k(retrieved_rank1, expected, 10) == 1.0
        assert 0.0 < ndcg_at_k(retrieved_rank2, expected, 10) < 1.0
