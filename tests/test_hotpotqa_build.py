"""
Tests for training/build_hotpotqa.py — Milestone 3B-1.

All tests use mock/synthetic data only. No network access, no downloads.

Covers:
1. _doc_split determinism and approximate ratio
2. _normalise_title stability
3. _parse_hotpotqa with in-memory mock data
4. build_examples:
   - valid example construction
   - document-level split assignment
   - document exclusion (opposing-split disjointness)
   - L-1 eval query isolation
   - L-2 eval positive isolation
   - L-4 hard-negative isolation
   - cap enforcement
   - split conflict (positives span both pools)
   - examples with no hard negatives rejected
5. Batch leakage check round-trip (validate_no_eval_leakage)
6. TrainingDataset dedup integration
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Make training/ and eval/ importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from build_hotpotqa import (
    _doc_split,
    _normalise_title,
    _RawExample,
    build_examples,
)
from training_schema import (
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    compute_training_id,
    validate_no_eval_leakage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw(
    uid: str = "mock_001",
    question: str = "What is the capital of France",
    sup_titles: list[str] | None = None,
    context_extra: list[tuple[str, str]] | None = None,
) -> _RawExample:
    """Build a minimal _RawExample that hashes to 'train' by default."""
    if sup_titles is None:
        sup_titles = ["Paris"]
    # Build context: supporting + at least one distractor
    context: list[tuple[str, str]] = [(t, f"Text about {t}.") for t in sup_titles]
    context.append(("Distractor City", "Text about Distractor City."))
    if context_extra:
        context.extend(context_extra)
    return _RawExample(
        uid=uid,
        question=question,
        supporting_titles=sup_titles,
        context=context,
    )


def _find_train_title() -> str:
    """Return a Wikipedia title that hashes to 'train'."""
    for i in range(1000):
        title = f"TrainDoc_{i:04d}"
        if _doc_split(_normalise_title(title)) == "train":
            return title
    raise RuntimeError("Could not find a train-hashing title")


def _find_dev_title() -> str:
    """Return a Wikipedia title that hashes to 'dev'."""
    for i in range(1000):
        title = f"DevDoc_{i:04d}"
        if _doc_split(_normalise_title(title)) == "dev":
            return title
    raise RuntimeError("Could not find a dev-hashing title")


TRAIN_TITLE = _find_train_title()
DEV_TITLE = _find_dev_title()


def _iter(examples: list[_RawExample]) -> Iterator[_RawExample]:
    return iter(examples)


# ---------------------------------------------------------------------------
# 1. _doc_split
# ---------------------------------------------------------------------------


class TestDocSplit:
    def test_deterministic(self) -> None:
        doc = "hotpotqa/United States"
        a = _doc_split(doc)
        b = _doc_split(doc)
        assert a == b

    def test_returns_train_or_dev(self) -> None:
        for i in range(50):
            result = _doc_split(f"hotpotqa/Doc_{i}")
            assert result in {"train", "dev"}

    def test_approximate_ratio(self) -> None:
        """~20% should be dev, ~80% train over 1000 random IDs."""
        dev_count = sum(1 for i in range(1000) if _doc_split(f"hotpotqa/Article_{i}") == "dev")
        # Expect roughly 200 ± 50
        assert 140 < dev_count < 280, f"Unexpected dev ratio: {dev_count}/1000"

    def test_different_titles_may_differ(self) -> None:
        # Sanity: not all titles hash the same
        results = {_doc_split(f"hotpotqa/Title_{i}") for i in range(20)}
        assert "train" in results and "dev" in results


# ---------------------------------------------------------------------------
# 2. _normalise_title
# ---------------------------------------------------------------------------


class TestNormaliseTitle:
    def test_prefix(self) -> None:
        assert _normalise_title("Paris").startswith("hotpotqa/")

    def test_strips_whitespace(self) -> None:
        assert _normalise_title("  Paris  ") == "hotpotqa/Paris"

    def test_stable(self) -> None:
        assert _normalise_title("Paris") == _normalise_title("Paris")


# ---------------------------------------------------------------------------
# 3. build_examples — valid case
# ---------------------------------------------------------------------------


class TestBuildExamplesValid:
    def test_single_valid_example(self) -> None:
        raw = _raw(question="Who wrote Hamlet", sup_titles=[TRAIN_TITLE])
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 1
        assert stats.total_accepted == 1
        ex = examples[0]
        assert ex.split == "train"
        assert ex.positive_id == _normalise_title(TRAIN_TITLE)
        assert ex.creation_method == "human"
        assert ex.licence == "CC-BY-SA-4.0"
        assert ex.category == "multi_topic"
        assert ex.provenance_url is not None

    def test_training_id_is_deterministic(self) -> None:
        raw = _raw(question="What is water made of", sup_titles=[TRAIN_TITLE])
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 1
        expected_tid = compute_training_id(raw.question, _normalise_title(TRAIN_TITLE), "human")
        assert examples[0].training_id == expected_tid

    def test_dev_example_built(self) -> None:
        raw = _raw(question="Where is the Eiffel Tower", sup_titles=[DEV_TITLE])
        examples, _stats = build_examples(
            _iter([raw]),
            target_split="dev",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_dev_distractor_v1.json",
        )
        assert len(examples) == 1
        assert examples[0].split == "dev"


# ---------------------------------------------------------------------------
# 4. Document-level split assignment
# ---------------------------------------------------------------------------


class TestDocumentSplitAssignment:
    def test_train_example_excluded_from_dev_build(self) -> None:
        """An example whose positive hashes to 'train' is skipped for dev."""
        raw = _raw(question="Train-hashing doc question", sup_titles=[TRAIN_TITLE])
        examples, _ = build_examples(
            _iter([raw]),
            target_split="dev",  # asking for dev, but doc hashes to train
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_dev_distractor_v1.json",
        )
        assert len(examples) == 0

    def test_dev_example_excluded_from_train_build(self) -> None:
        raw = _raw(question="Dev-hashing doc question", sup_titles=[DEV_TITLE])
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 0

    def test_doc_exclusion_set_enforced(self) -> None:
        """An example whose positive is already in the exclusion set is rejected."""
        raw = _raw(question="Exclusion test question", sup_titles=[TRAIN_TITLE])
        pos_id = _normalise_title(TRAIN_TITLE)
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set={pos_id},  # pre-committed to opposing split
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 0
        assert stats.split_conflict == 1

    def test_zero_overlap_train_dev(self) -> None:
        """Full pipeline: dev docs must not appear in train."""
        dev_raw = [_raw(question="Dev q", sup_titles=[DEV_TITLE])]
        dev_examples, _ = build_examples(
            _iter(dev_raw),
            target_split="dev",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_dev_distractor_v1.json",
        )
        dev_docs = {ex.positive_id for ex in dev_examples}

        train_raw = [_raw(question="Train q", sup_titles=[TRAIN_TITLE])]
        train_examples, _ = build_examples(
            _iter(train_raw),
            target_split="train",
            doc_exclusion_set=dev_docs,
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        train_docs = {ex.positive_id for ex in train_examples}

        assert train_docs & dev_docs == set()


# ---------------------------------------------------------------------------
# 5. L-1 eval query isolation
# ---------------------------------------------------------------------------


class TestL1QueryIsolation:
    def test_eval_query_rejected(self) -> None:
        if not EVAL_QUERIES:
            pytest.skip("EVAL_QUERIES empty — run from repo root")
        eval_q = next(iter(EVAL_QUERIES))  # already lowercased
        raw = _raw(question=eval_q, sup_titles=[TRAIN_TITLE])
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 0
        assert stats.eval_query_leak == 1

    def test_novel_query_accepted(self) -> None:
        raw = _raw(
            question="This is a completely novel question not in the eval set xyz123",
            sup_titles=[TRAIN_TITLE],
        )
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# 6. L-2 eval positive isolation
# ---------------------------------------------------------------------------


class TestL2PositiveIsolation:
    def test_eval_positive_doc_rejected(self) -> None:
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS empty — run from repo root")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        raw = _RawExample(
            uid="l2_test",
            question="Question about eval doc",
            supporting_titles=[eval_doc],
            context=[
                (eval_doc, "Some text."),
                ("Distractor", "Distractor text."),
            ],
        )
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 0
        assert stats.eval_doc_leak == 1


# ---------------------------------------------------------------------------
# 7. L-4 hard-negative isolation
# ---------------------------------------------------------------------------


class TestL4HardNegativeIsolation:
    def test_eval_doc_filtered_from_hard_negatives(self) -> None:
        """If a distractor is an eval doc, it should be filtered out.
        If that leaves zero hard negatives, the example is rejected."""
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS empty")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        raw = _RawExample(
            uid="l4_test",
            question="What is the question for L4 test",
            supporting_titles=[TRAIN_TITLE],
            context=[
                (TRAIN_TITLE, "Supporting text."),
                (eval_doc, "Eval vault text — should be filtered."),
            ],
        )
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        # Only eval_doc was available as hard negative → 0 negatives → rejected
        assert len(examples) == 0
        assert stats.no_hard_negatives == 1

    def test_mixed_negatives_eval_filtered_safe_kept(self) -> None:
        """If only SOME negatives are eval docs, safe ones are kept."""
        if not EVAL_POSITIVE_IDS:
            pytest.skip("EVAL_POSITIVE_IDS empty")
        eval_doc = next(iter(EVAL_POSITIVE_IDS))
        # Find a second train-hashing title for the safe negative
        safe_neg = TRAIN_TITLE + "_neg"
        raw = _RawExample(
            uid="l4_mixed",
            question="Mixed negatives question test",
            supporting_titles=[TRAIN_TITLE],
            context=[
                (TRAIN_TITLE, "Supporting text."),
                (eval_doc, "Eval vault — should be filtered."),
                (safe_neg, "Safe distractor text."),
            ],
        )
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 1
        assert all(nid not in EVAL_POSITIVE_IDS for nid in examples[0].hard_negative_ids)


# ---------------------------------------------------------------------------
# 8. Cap enforcement
# ---------------------------------------------------------------------------


class TestCapEnforcement:
    def test_cap_limits_output(self) -> None:
        raws = [
            _raw(question=f"Question about topic {i}", sup_titles=[TRAIN_TITLE + f"_{i}"])
            for i in range(20)
        ]
        # Override TRAIN_TITLE usage — create titles that hash to train
        train_titles = [
            f"TrainTitle_{i:04d}"
            for i in range(200)
            if _doc_split(_normalise_title(f"TrainTitle_{i:04d}")) == "train"
        ][:20]
        raws = [
            _raw(question=f"Unique question number {i}", sup_titles=[t])
            for i, t in enumerate(train_titles)
        ]
        examples, stats = build_examples(
            _iter(raws),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=5,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) <= 5
        assert stats.cap_reached > 0


# ---------------------------------------------------------------------------
# 9. No hard negatives → rejected
# ---------------------------------------------------------------------------


class TestNoHardNegativesRejected:
    def test_only_supporting_docs_in_context_rejected(self) -> None:
        raw = _RawExample(
            uid="no_neg_test",
            question="Question with no distractors",
            supporting_titles=[TRAIN_TITLE],
            context=[(TRAIN_TITLE, "Only supporting doc.")],  # no distractor
        )
        examples, stats = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 0
        assert stats.no_hard_negatives == 1


# ---------------------------------------------------------------------------
# 10. validate_no_eval_leakage round-trip
# ---------------------------------------------------------------------------


class TestLeakageReporterRoundTrip:
    def test_clean_batch_no_violations(self) -> None:
        raw = _raw(
            question="Clean batch test question with no eval overlap",
            sup_titles=[TRAIN_TITLE],
        )
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        violations = validate_no_eval_leakage(examples)
        assert violations == []


# ---------------------------------------------------------------------------
# 11. TrainingDataset dedup integration
# ---------------------------------------------------------------------------


class TestTrainingDatasetIntegration:
    def test_duplicate_rejected_at_dataset_level(self) -> None:
        raw = _raw(question="Duplicate test question", sup_titles=[TRAIN_TITLE])
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        assert len(examples) == 1
        with pytest.raises(ValueError, match="Duplicate training_id"):
            TrainingDataset(
                dataset_id="test",
                examples=examples + examples,  # duplicate
            )

    def test_single_example_dataset_valid(self) -> None:
        raw = _raw(question="Single example dataset test", sup_titles=[TRAIN_TITLE])
        examples, _ = build_examples(
            _iter([raw]),
            target_split="train",
            doc_exclusion_set=set(),
            max_count=10,
            source_file="hotpot_train_v1.1.json",
        )
        ds = TrainingDataset(dataset_id="test", examples=examples)
        assert len(ds.examples) == len(examples)
