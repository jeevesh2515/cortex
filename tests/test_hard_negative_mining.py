"""
Tests for Milestone: Deterministic Hard-Negative Mining Pipeline.

Verifies:
1. Okapi BM25 index scoring, tokenization, and deterministic ranking.
2. Training passage corpus extraction and strict eval/dev partition isolation.
3. Zero positive leakage (exact positive & multi-hop supporting facts).
4. Zero eval leakage (L-1, L-2, L-4 rules) and zero dev overlap.
5. List alignment (parallel negative IDs and texts).
6. Human review pack structure, metadata, and unreviewed labelling.
7. Manifest metrics and deterministic repeatability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
TRAINING_DIR = REPO_ROOT / "training"

sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(TRAINING_DIR))

from mine_hard_negatives import (  # noqa: E402
    BM25Index,
    Passage,
    _doc_split,
    _normalise_title,
    _tokenize,
)
from training_schema import (  # noqa: E402
    EVAL_POSITIVE_IDS,
    TrainingExample,
    compute_training_id,
    validate_no_eval_leakage,
)


class TestBM25Index:
    """Unit tests for the standalone deterministic BM25 index."""

    def test_tokenize(self) -> None:
        text = "Hello, World! This is a test 123."
        tokens = _tokenize(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test", "123"]

    def test_bm25_ranking(self) -> None:
        p1 = Passage(
            doc_id="p1",
            title="Quantum Computing",
            text="Quantum computing uses qubits for quantum algorithms.",
            tokens=_tokenize("Quantum computing uses qubits for quantum algorithms."),
        )
        p2 = Passage(
            doc_id="p2",
            title="Baking Sourdough",
            text="Sourdough bread requires flour, water, and wild yeast fermentation.",
            tokens=_tokenize("Sourdough bread requires flour, water, and wild yeast fermentation."),
        )
        p3 = Passage(
            doc_id="p3",
            title="Classical Computing",
            text="Classical computing relies on binary bits 0 and 1 for algorithms.",
            tokens=_tokenize("Classical computing relies on binary bits 0 and 1 for algorithms."),
        )

        index = BM25Index([p1, p2, p3])
        hits = index.search(_tokenize("quantum algorithms"), top_k=3)

        assert len(hits) > 0
        top_idx, top_score = hits[0]
        assert top_idx == 0  # p1 is top hit
        assert top_score > 0.0

        # Deterministic order test
        hits2 = index.search(_tokenize("quantum algorithms"), top_k=3)
        assert hits == hits2


class TestDocumentSplitAndNormalisation:
    """Tests for document normalisation and partition rules."""

    def test_normalise_title(self) -> None:
        assert _normalise_title("First_for_Women") == "First for Women"
        assert _normalise_title("  Arthur's_Magazine  ") == "Arthur's Magazine"

    def test_doc_split_stability(self) -> None:
        split1 = _doc_split("Arthur's Magazine")
        split2 = _doc_split("Arthur's Magazine")
        assert split1 == split2
        assert split1 in {"train", "dev"}


class TestMinedDatasetInvariants:
    """Integration tests verifying the generated mined training dataset."""

    @pytest.fixture(autouse=True)
    def setup_paths(self) -> None:
        self.mined_train_path = TRAINING_DIR / "train_mined.jsonl"
        self.dev_path = TRAINING_DIR / "dev.jsonl"
        self.manifest_path = TRAINING_DIR / "manifest_mined.json"
        self.review_pack_path = TRAINING_DIR / "mined_review_pack.json"

    def test_mined_dataset_structure_and_schema(self) -> None:
        if not self.mined_train_path.exists():
            pytest.skip("train_mined.jsonl not found in working directory")

        lines = [line for line in self.mined_train_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 300, f"Expected 300 training examples, got {len(lines)}"

        examples: list[TrainingExample] = []
        for line in lines:
            data = json.loads(line)
            ex = TrainingExample.model_validate(data)
            examples.append(ex)

            # Check negative list parallelism
            assert len(ex.hard_negative_ids) == len(ex.hard_negative_texts)
            assert len(ex.hard_negative_ids) >= 1
            assert all(len(t) > 0 for t in ex.hard_negative_texts)

            # Check positive leakage
            assert ex.positive_id not in ex.hard_negative_ids
            assert ex.positive_text not in ex.hard_negative_texts

            # Check deduplication
            assert len(ex.hard_negative_ids) == len(set(ex.hard_negative_ids))
            assert len(ex.hard_negative_texts) == len(set(ex.hard_negative_texts))

            # Check training_id integrity
            expected_tid = compute_training_id(ex.query, ex.positive_id, ex.creation_method)
            assert ex.training_id == expected_tid

            # Check notes
            assert ex.notes is not None
            notes_dict = json.loads(ex.notes)
            assert "retriever_breakdown" in notes_dict
            assert "corpus_sha256" in notes_dict

        # Batch leakage check
        violations = validate_no_eval_leakage(examples)
        assert not violations, f"Leakage violations detected:\n{violations}"

    def test_train_dev_eval_disjointness(self) -> None:
        if not self.mined_train_path.exists() or not self.dev_path.exists():
            pytest.skip("Dataset files not available")

        mined_lines = [
            json.loads(line)
            for line in self.mined_train_path.read_text().splitlines()
            if line.strip()
        ]
        dev_lines = [
            json.loads(line) for line in self.dev_path.read_text().splitlines() if line.strip()
        ]

        dev_positive_ids = {e["positive_id"] for e in dev_lines}

        for ex in mined_lines:
            # Check positive ID isolation
            assert ex["positive_id"] not in dev_positive_ids
            assert ex["positive_id"] not in EVAL_POSITIVE_IDS

            # Check mined negative IDs isolation
            for nid in ex["hard_negative_ids"]:
                assert nid not in dev_positive_ids, f"Dev document leakage in negative: {nid}"
                assert nid not in EVAL_POSITIVE_IDS, f"Eval document leakage in negative: {nid}"


class TestReviewPackInvariants:
    """Verifies the human review pack structure and unreviewed status."""

    def test_review_pack_schema(self) -> None:
        review_pack_path = TRAINING_DIR / "mined_review_pack.json"
        if not review_pack_path.exists():
            pytest.skip("mined_review_pack.json not found")

        with review_pack_path.open() as f:
            pack = json.load(f)

        meta = pack["pack_metadata"]
        assert meta["sample_count"] >= 50
        assert meta["review_status"] == "unreviewed"
        assert meta["reviewer_decision"] == "PENDING_HUMAN_REVIEW"
        assert meta["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

        samples = pack["samples"]
        assert len(samples) == meta["sample_count"]

        for sample in samples:
            assert sample["review_status"] == "unreviewed"
            assert sample["reviewer_decision"] == "PENDING_HUMAN_REVIEW"
            assert sample["retriever"] in {"hybrid", "dense", "bm25"}
            assert sample["positive_id"] != sample["negative_id"]
            assert len(sample["query"]) > 0
            assert len(sample["positive_text"]) > 0
            assert len(sample["negative_text"]) > 0


class TestManifestInvariants:
    """Verifies manifest schema and numbers."""

    def test_manifest_schema(self) -> None:
        manifest_path = TRAINING_DIR / "manifest_mined.json"
        if not manifest_path.exists():
            pytest.skip("manifest_mined.json not found")

        with manifest_path.open() as f:
            manifest = json.load(f)

        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["seed"] == 42
        assert manifest["dataset_statistics"]["total_training_examples"] == 300
        assert manifest["dataset_statistics"]["total_mined_negatives"] == 1500
        assert manifest["dataset_statistics"]["total_candidates_retrieved"] > 0
        assert manifest["dataset_statistics"]["total_candidates_rejected"] > 0
        assert manifest["dataset_statistics"]["negatives_per_query_target"] == 5

        # Check that retriever distribution has hybrid, dense, bm25
        ret_dist = manifest["dataset_statistics"]["retriever_distribution"]
        assert ret_dist.get("hybrid", 0) > 0
        assert ret_dist.get("dense", 0) > 0
        assert ret_dist.get("bm25", 0) > 0
