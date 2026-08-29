"""
Tests for Milestone 3B-2 PKM synthetic design and dataset audit invariants.

These tests verify:
1. PKM synthetic example schema compliance under the 3B-2 design specification.
2. Mandatory provenance fields in notes for synthetic_llm examples.
3. Eval containment rules (L-1, L-2, L-4, L-6) for synthetic examples.
4. Quality and isolation invariants of the committed Milestone 3B-1 HotpotQA dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add eval/ and training/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from training_schema import (
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    TrainingExample,
    compute_training_id,
    validate_no_eval_leakage,
)

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _valid_synth_notes() -> str:
    return json.dumps(
        {
            "generator_model": "qwen2.5:7b-instruct",
            "generator_digest": "sha256:1234567890abcdef",
            "prompt_version": "pkm-synth-v1.0",
            "reviewed_by": "human_tester",
            "review_date": "2026-08-29T15:00:00Z",
            "seed_doc_id": "synthetic_vault/pkm_note_001.md",
        }
    )


def _make_synth_example(
    query: str = "How do sourdough starters cultivate wild yeast?",
    pos_id: str = "synthetic_vault/pkm_note_001.md",
    category: str = "dense_semantic",
    notes: str | None = None,
) -> TrainingExample:
    if notes is None:
        notes = _valid_synth_notes()
    tid = compute_training_id(query, pos_id, "synthetic_llm")
    return TrainingExample(
        training_id=tid,
        version="1.0.0",
        split="train",
        query=query,
        positive_id=pos_id,
        positive_text="Sourdough starters capture wild yeast strains such as Candida humilis.",
        hard_negative_ids=[
            "synthetic_vault/pkm_note_002.md",
            "synthetic_vault/pkm_note_003.md",
        ],
        hard_negative_texts=[
            "Commercial baker's yeast uses Saccharomyces cerevisiae exclusively.",
            "Kombucha SCOBY fermentation utilizes acetobacter and osmophilic yeasts.",
        ],
        source="pkm_synthetic_v1",
        provenance_url=None,
        licence="MIT",
        category=category,
        creation_method="synthetic_llm",
        created_at="2026-08-29T15:00:00Z",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. PKM Synthetic Design Schema Tests
# ---------------------------------------------------------------------------


class TestPKMSyntheticDesignSchema:
    def test_valid_synthetic_example(self) -> None:
        ex = _make_synth_example()
        assert ex.creation_method == "synthetic_llm"
        assert ex.licence == "MIT"
        assert ex.source == "pkm_synthetic_v1"
        assert ex.notes is not None
        provenance = json.loads(ex.notes)
        assert provenance["generator_model"] == "qwen2.5:7b-instruct"
        assert provenance["reviewed_by"] == "human_tester"

    def test_synthetic_example_deterministic_id(self) -> None:
        ex = _make_synth_example()
        expected = compute_training_id(ex.query, ex.positive_id, "synthetic_llm")
        assert ex.training_id == expected

    def test_synthetic_example_eval_query_leakage_rejected(self) -> None:
        if not EVAL_QUERIES:
            pytest.skip("EVAL_QUERIES empty")
        eval_q = next(iter(EVAL_QUERIES))
        with pytest.raises(ValueError, match="L-1 violation"):
            _make_synth_example(query=eval_q)

    def test_synthetic_example_eval_positive_leakage_rejected(self) -> None:
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS empty")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        with pytest.raises(ValueError, match="L-2 violation"):
            _make_synth_example(pos_id=eval_doc)

    def test_synthetic_example_eval_vault_hard_negative_rejected(self) -> None:
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS empty")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        tid = compute_training_id("Valid query", "synthetic_vault/safe.md", "synthetic_llm")
        with pytest.raises(ValueError, match="L-4 violation"):
            TrainingExample(
                training_id=tid,
                version="1.0.0",
                split="train",
                query="Valid query",
                positive_id="synthetic_vault/safe.md",
                positive_text="Safe positive text.",
                hard_negative_ids=[eval_doc],
                hard_negative_texts=["Eval negative text."],
                source="pkm_synthetic_v1",
                licence="MIT",
                category="dense_semantic",
                creation_method="synthetic_llm",
                created_at="2026-08-29T15:00:00Z",
                notes=_valid_synth_notes(),
            )


# ---------------------------------------------------------------------------
# 2. HotpotQA 300/50 Dataset Invariant Tests
# ---------------------------------------------------------------------------


class TestHotpotQADatasetInvariants:
    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        self.repo_root = Path(__file__).parent.parent
        self.train_path = self.repo_root / "training" / "train.jsonl"
        self.dev_path = self.repo_root / "training" / "dev.jsonl"

    def test_dataset_files_exist_and_counts(self) -> None:
        if not self.train_path.exists() or not self.dev_path.exists():
            pytest.skip("Dataset files not generated in working directory")

        train_lines = [
            json.loads(line) for line in self.train_path.read_text().splitlines() if line.strip()
        ]
        dev_lines = [
            json.loads(line) for line in self.dev_path.read_text().splitlines() if line.strip()
        ]

        assert len(train_lines) == 300
        assert len(dev_lines) == 50

    def test_dataset_document_disjointness(self) -> None:
        if not self.train_path.exists() or not self.dev_path.exists():
            pytest.skip("Dataset files not generated in working directory")

        train_lines = [
            json.loads(line) for line in self.train_path.read_text().splitlines() if line.strip()
        ]
        dev_lines = [
            json.loads(line) for line in self.dev_path.read_text().splitlines() if line.strip()
        ]

        train_docs = {e["positive_id"] for e in train_lines}
        dev_docs = {e["positive_id"] for e in dev_lines}

        assert len(train_docs) == 300
        assert len(dev_docs) == 50
        assert train_docs & dev_docs == set()

    def test_dataset_leakage_zero(self) -> None:
        if not self.train_path.exists() or not self.dev_path.exists():
            pytest.skip("Dataset files not generated in working directory")

        all_examples = []
        for path in [self.train_path, self.dev_path]:
            for line in path.read_text().splitlines():
                if line.strip():
                    all_examples.append(TrainingExample.model_validate_json(line))

        violations = validate_no_eval_leakage(all_examples)
        assert violations == []

    def test_dataset_pydantic_batch_validation(self) -> None:
        if not self.train_path.exists() or not self.dev_path.exists():
            pytest.skip("Dataset files not generated in working directory")

        train_exs = [
            TrainingExample.model_validate_json(line)
            for line in self.train_path.read_text().splitlines()
            if line.strip()
        ]
        dev_exs = [
            TrainingExample.model_validate_json(line)
            for line in self.dev_path.read_text().splitlines()
            if line.strip()
        ]

        ds_train = TrainingDataset(dataset_id="test_train", examples=train_exs)
        ds_dev = TrainingDataset(dataset_id="test_dev", examples=dev_exs)

        assert len(ds_train.examples) == 300
        assert len(ds_dev.examples) == 50
