"""
Tests for eval/training_schema.py — Milestone 3A validation scaffolding.

These tests verify:
1. Eval asset fingerprints against CHECKSUMS.sha256 (immutability guard)
2. TrainingExample construction — valid and invalid cases
3. All leakage-prevention rules L-1 through L-6
4. TrainingDataset duplicate-ID detection (L-6 batch)
5. validate_no_eval_leakage batch reporter
6. compute_training_id determinism
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Allow importing the schema from eval/ without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from training_schema import (
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    TrainingExample,
    compute_training_id,
    validate_no_eval_leakage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def _make_valid_example(
    *,
    query: str = "how does inverted index BM25 scoring work",
    positive_id: str = "training/doc_001.md",
    positive_text: str = "BM25 scores documents by term frequency and inverse document frequency.",
    hard_negative_ids: list[str] | None = None,
    hard_negative_texts: list[str] | None = None,
    creation_method: str = "mined_bm25",
    notes: str = "BM25 threshold=10.0",
    licence: str = "CC-BY-4.0",
    provenance_url: str | None = None,
    category: str = "lexical_exact",
    split: str = "train",
) -> TrainingExample:
    """Construct a valid TrainingExample with sensible defaults."""
    hard_negative_ids = hard_negative_ids or ["training/doc_002.md"]
    hard_negative_texts = hard_negative_texts or ["Relational databases use B-tree indexes."]
    tid = compute_training_id(query, positive_id, creation_method)
    return TrainingExample(
        training_id=tid,
        split=split,  # type: ignore[arg-type]
        query=query,
        positive_id=positive_id,
        positive_text=positive_text,
        hard_negative_ids=hard_negative_ids,
        hard_negative_texts=hard_negative_texts,
        source="beir/nfcorpus",
        provenance_url=provenance_url,
        licence=licence,
        category=category,
        creation_method=creation_method,  # type: ignore[arg-type]
        created_at="2026-08-29T00:00:00Z",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. Eval asset fingerprint checks (immutability guard)
# ---------------------------------------------------------------------------


class TestEvalAssetFingerprints:
    """Verify that baseline eval assets have not been modified."""

    CHECKSUMS_FILE = REPO_ROOT / "eval" / "CHECKSUMS.sha256"

    def test_checksums_file_exists(self) -> None:
        assert self.CHECKSUMS_FILE.exists(), (
            "eval/CHECKSUMS.sha256 is missing. "
            "Baseline assets must be tracked with their SHA-256 fingerprints."
        )

    def test_all_baseline_assets_match_checksums(self) -> None:
        """Parse CHECKSUMS.sha256 and verify every listed file."""
        lines = self.CHECKSUMS_FILE.read_text().splitlines()
        assert lines, "CHECKSUMS.sha256 must not be empty."
        failures: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected_hash, filepath = line.split("  ", 1)
            full_path = REPO_ROOT / filepath
            if not full_path.exists():
                failures.append(f"MISSING: {filepath}")
                continue
            actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                failures.append(
                    f"HASH MISMATCH: {filepath}\n"
                    f"  expected: {expected_hash}\n"
                    f"  actual:   {actual_hash}"
                )
        assert not failures, "Baseline asset integrity check failed:\n" + "\n".join(failures)

    def test_eval_positive_ids_loaded(self) -> None:
        """Schema must load at least the 11 known eval positive IDs."""
        assert len(EVAL_POSITIVE_IDS) >= 11, (
            f"Expected at least 11 eval positive IDs, got {len(EVAL_POSITIVE_IDS)}"
        )

    def test_eval_queries_loaded(self) -> None:
        assert len(EVAL_QUERIES) >= 25


# ---------------------------------------------------------------------------
# 2. compute_training_id determinism
# ---------------------------------------------------------------------------


class TestComputeTrainingId:
    def test_deterministic(self) -> None:
        a = compute_training_id("q", "doc.md", "human")
        b = compute_training_id("q", "doc.md", "human")
        assert a == b

    def test_different_inputs_produce_different_ids(self) -> None:
        a = compute_training_id("q1", "doc.md", "human")
        b = compute_training_id("q2", "doc.md", "human")
        assert a != b

    def test_method_part_of_hash(self) -> None:
        a = compute_training_id("q", "doc.md", "human")
        b = compute_training_id("q", "doc.md", "mined_bm25")
        assert a != b

    def test_format_is_hex_64_chars(self) -> None:
        tid = compute_training_id("q", "doc.md", "human")
        assert len(tid) == 64
        assert all(c in "0123456789abcdef" for c in tid)


# ---------------------------------------------------------------------------
# 3. Valid training example construction
# ---------------------------------------------------------------------------


class TestValidTrainingExample:
    def test_valid_mined_bm25_example(self) -> None:
        ex = _make_valid_example()
        assert ex.split == "train"
        assert ex.creation_method == "mined_bm25"
        assert ex.training_id == compute_training_id(ex.query, ex.positive_id, ex.creation_method)

    def test_valid_synthetic_llm_example(self) -> None:
        ex = _make_valid_example(
            creation_method="synthetic_llm",
            notes="Generated by gpt-4o; prompt template v1",
        )
        assert ex.creation_method == "synthetic_llm"

    def test_valid_human_example(self) -> None:
        ex = _make_valid_example(
            creation_method="human",
            provenance_url="https://example.com/source-doc",
            notes=None,
        )
        assert ex.provenance_url is not None

    def test_valid_dev_split(self) -> None:
        ex = _make_valid_example(split="dev")
        assert ex.split == "dev"

    def test_multiple_hard_negatives(self) -> None:
        ex = _make_valid_example(
            hard_negative_ids=["training/doc_002.md", "training/doc_003.md"],
            hard_negative_texts=["Text A.", "Text B."],
        )
        assert len(ex.hard_negative_ids) == 2


# ---------------------------------------------------------------------------
# 4. Rule L-1: query isolation
# ---------------------------------------------------------------------------


class TestRuleL1QueryIsolation:
    def test_eval_query_rejected(self) -> None:
        """An exact eval query string must be rejected."""
        # Pick a known eval query from cases.yaml
        if not EVAL_QUERIES:
            pytest.skip("EVAL_QUERIES not loaded (run from repo root)")
        eval_query = next(iter(EVAL_QUERIES))  # lowercase normalised
        # Reconstruct original case to avoid normalisation mismatch
        with pytest.raises(ValueError, match="L-1"):
            _make_valid_example(query=eval_query)

    def test_novel_query_accepted(self) -> None:
        ex = _make_valid_example(query="completely novel query string not in eval set at all")
        assert ex.query.startswith("completely novel")


# ---------------------------------------------------------------------------
# 5. Rule L-2: positive-document isolation
# ---------------------------------------------------------------------------


class TestRuleL2PositiveDocIsolation:
    def test_eval_positive_rejected(self) -> None:
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS not loaded")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        with pytest.raises(ValueError, match="L-2"):
            _make_valid_example(positive_id=eval_doc)

    def test_non_eval_positive_accepted(self) -> None:
        ex = _make_valid_example(positive_id="training/new_doc_not_in_eval.md")
        assert ex.positive_id == "training/new_doc_not_in_eval.md"


# ---------------------------------------------------------------------------
# 6. Rule L-4: hard-negative isolation
# ---------------------------------------------------------------------------


class TestRuleL4HardNegativeIsolation:
    def test_eval_vault_hard_negative_rejected(self) -> None:
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS not loaded")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        with pytest.raises(ValueError, match="L-4"):
            _make_valid_example(
                hard_negative_ids=[eval_doc],
                hard_negative_texts=["Eval vault text."],
            )

    def test_non_eval_hard_negative_accepted(self) -> None:
        ex = _make_valid_example(
            hard_negative_ids=["training/corpus/external_001.md"],
            hard_negative_texts=["External corpus passage."],
        )
        assert "training/corpus/external_001.md" in ex.hard_negative_ids


# ---------------------------------------------------------------------------
# 7. Rule L-6: training_id integrity
# ---------------------------------------------------------------------------


class TestRuleL6TrainingIdIntegrity:
    def test_wrong_training_id_rejected(self) -> None:
        query = "inverted index scoring"
        pid = "training/doc_001.md"
        method = "mined_bm25"
        with pytest.raises(ValueError, match="L-6"):
            TrainingExample(
                training_id="definitely_wrong_hash",
                split="train",
                query=query,
                positive_id=pid,
                positive_text="Text.",
                hard_negative_ids=["training/doc_002.md"],
                hard_negative_texts=["Neg text."],
                source="beir/nfcorpus",
                licence="CC-BY-4.0",
                category="lexical_exact",
                creation_method=method,
                created_at="2026-08-29T00:00:00Z",
                notes="threshold=5.0",
            )

    def test_correct_training_id_accepted(self) -> None:
        ex = _make_valid_example()
        assert ex.training_id == compute_training_id(ex.query, ex.positive_id, ex.creation_method)


# ---------------------------------------------------------------------------
# 8. Additional structural validations
# ---------------------------------------------------------------------------


class TestStructuralValidations:
    def test_eval_split_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_valid_example(split="eval")  # type: ignore[arg-type]

    def test_positive_in_hard_negatives_rejected(self) -> None:
        pid = "training/doc_001.md"
        with pytest.raises(ValueError, match="also be a hard negative"):
            _make_valid_example(
                positive_id=pid,
                hard_negative_ids=[pid],
                hard_negative_texts=["Same doc."],
            )

    def test_mismatched_negative_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            _make_valid_example(
                hard_negative_ids=["training/a.md", "training/b.md"],
                hard_negative_texts=["Only one text."],
            )

    def test_synthetic_llm_without_notes_rejected(self) -> None:
        with pytest.raises(ValueError, match="notes"):
            _make_valid_example(creation_method="synthetic_llm", notes=None)

    def test_human_without_provenance_rejected(self) -> None:
        with pytest.raises(ValueError, match="provenance_url"):
            _make_valid_example(creation_method="human", notes=None, provenance_url=None)

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValueError, match="VALID_CATEGORIES"):
            _make_valid_example(category="not_a_real_category")

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_valid_example(query="")

    def test_empty_hard_negative_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _make_valid_example(
                hard_negative_ids=["training/doc_002.md"],
                hard_negative_texts=["   "],
            )

    def test_too_long_query_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_valid_example(query="x" * 513)


# ---------------------------------------------------------------------------
# 9. TrainingDataset duplicate detection (L-6 batch)
# ---------------------------------------------------------------------------


class TestTrainingDataset:
    def test_single_example_dataset_valid(self) -> None:
        ex = _make_valid_example()
        ds = TrainingDataset(dataset_id="test-v1", examples=[ex])
        assert len(ds.examples) == 1

    def test_duplicate_training_ids_rejected(self) -> None:
        ex = _make_valid_example()
        with pytest.raises(ValueError, match="Duplicate training_id"):
            TrainingDataset(dataset_id="test-v1", examples=[ex, ex])

    def test_two_distinct_examples_accepted(self) -> None:
        ex1 = _make_valid_example(query="how does BM25 scoring work")
        ex2 = _make_valid_example(query="explain HNSW graph construction")
        ds = TrainingDataset(dataset_id="test-v1", examples=[ex1, ex2])
        assert len(ds.examples) == 2


# ---------------------------------------------------------------------------
# 10. validate_no_eval_leakage batch reporter
# ---------------------------------------------------------------------------


class TestLeakageReporter:
    def test_clean_batch_returns_empty_list(self) -> None:
        examples = [_make_valid_example()]
        violations = validate_no_eval_leakage(examples)
        assert violations == []

    def test_leakage_reporter_is_additive(self) -> None:
        """Two clean examples still produce no violations."""
        examples = [
            _make_valid_example(query="BM25 query one"),
            _make_valid_example(query="BM25 query two"),
        ]
        violations = validate_no_eval_leakage(examples)
        assert violations == []
