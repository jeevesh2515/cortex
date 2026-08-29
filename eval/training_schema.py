"""
Versioned training-data schema and leakage validation for Cortex Milestone 3A.

This module defines:
- TrainingExample: the canonical Pydantic model for a single training pair
- validate_no_eval_leakage: checks a batch of examples against the frozen
  eval/cases.yaml positive set and query strings
- compute_training_id: deterministic SHA-256 ID for deduplication

Rules enforced here:
  L-1  Query not identical to any eval query
  L-2  positive_id not in EVAL_POSITIVE_IDS
  L-3  (documentation; relevance-label reuse cannot be checked structurally)
  L-4  hard_negative_ids not from eval vault (prefix check)
  L-5  (documentation; vault path isolation)
  L-6  training_id == sha256(query + positive_id + creation_method)
  L-7  (caller responsibility; cosine check requires embedding model)

NOTE: This schema is Milestone 3A scaffolding only.
No training data exists yet. Corpus construction requires explicit approval
of Approvals A-D in eval/AUDIT.md before Milestone 3B begins.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Frozen eval positive set — derived from eval/cases.yaml at schema load time.
# These IDs must NEVER appear as positive_id in any training or dev example.
# ---------------------------------------------------------------------------

_EVAL_CASES_PATH = "eval/cases.yaml"

# Eagerly load eval positive IDs so violations are caught at import time.
# In CI the cwd is always the repo root; callers in other contexts must
# os.chdir() first or patch _EVAL_POSITIVE_IDS.
try:
    with open(_EVAL_CASES_PATH) as _f:
        _eval_data = yaml.safe_load(_f)
    EVAL_POSITIVE_IDS: frozenset[str] = frozenset(
        doc_id for case in _eval_data.get("cases", []) for doc_id in case.get("expect", [])
    )
    EVAL_QUERIES: frozenset[str] = frozenset(
        case["query"].strip().lower() for case in _eval_data.get("cases", []) if "query" in case
    )
except FileNotFoundError:
    # Allow import outside repo root (e.g. standalone tests with mocked data)
    EVAL_POSITIVE_IDS = frozenset()
    EVAL_QUERIES = frozenset()

# Prefix used by all eval vault documents.
_EVAL_VAULT_PREFIXES: tuple[str, ...] = (
    "eval/vault/",
    "Retrieval.md",
    "Embeddings.md",
    "Sourdough.md",
    "Flour Chemistry.md",
    "Fermentation Chemistry.md",
    "Pectin Chains.md",
    "Thermal Management.md",
    "Vector Databases.md",
    "Distractor Search.md",
    "daily/2026-08-01.md",
    "daily/2026-08-15.md",
)

# Allowed creation methods.
CreationMethod = Literal["human", "synthetic_llm", "mined_bm25", "mined_dense", "mined_hybrid"]

# Allowed splits. "eval" is intentionally absent.
Split = Literal["train", "dev"]

# Allowed categories (must match eval category vocabulary).
VALID_CATEGORIES = frozenset(
    {
        "dense_semantic",
        "lexical_exact",
        "hybrid_core",
        "wikilink_graph",
        "temporal_query",
        "distractor_resistance",
        "multi_topic",
        "general",  # permitted for training-only examples
    }
)


def compute_training_id(query: str, positive_id: str, creation_method: str) -> str:
    """Deterministic SHA-256 ID for a training example (Rule L-6)."""
    payload = f"{query}\x00{positive_id}\x00{creation_method}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TrainingExample(BaseModel):
    """
    A single contrastive training example for embedding fine-tuning.

    Every field is required except `provenance_url` and `notes`.
    `training_id` is computed and validated automatically.
    """

    training_id: str = Field(
        description=(
            "SHA-256(query + positive_id + creation_method). "
            "Computed by compute_training_id(); validated on construction."
        )
    )
    version: str = Field(default="1.0.0", description="Schema version (semver).")
    split: Split = Field(description="'train' or 'dev' only; never 'eval'.")
    query: str = Field(min_length=1, max_length=512, description="Natural-language query.")
    positive_id: str = Field(
        min_length=1,
        description="Stable document/chunk ID of the relevant passage.",
    )
    positive_text: str = Field(min_length=1, description="Full text of the positive passage.")
    hard_negative_ids: list[str] = Field(
        min_length=1,
        description="At least one hard-negative document/chunk ID.",
    )
    hard_negative_texts: list[str] = Field(
        min_length=1,
        description="Parallel text for each hard negative ID.",
    )
    source: str = Field(min_length=1, description="Source corpus name, e.g. 'beir/nfcorpus'.")
    provenance_url: str | None = Field(
        default=None,
        description="Canonical URL for the source document (required for 'human' examples).",
    )
    licence: str = Field(min_length=1, description="SPDX identifier or 'proprietary'.")
    category: str = Field(description="Query type; must be in VALID_CATEGORIES.")
    creation_method: CreationMethod = Field(description="How this example was created.")
    created_at: str = Field(min_length=1, description="ISO-8601 UTC timestamp.")
    notes: str | None = Field(
        default=None,
        description=(
            "Required for synthetic_llm (declare model) and "
            "mined_* (declare threshold); optional otherwise."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_not_in_eval(cls, v: str) -> str:
        """L-1: query must not be identical to any eval query."""
        if v.strip().lower() in EVAL_QUERIES:
            raise ValueError(
                f"Query '{v[:80]}...' is identical to an eval query (L-1 violation). "
                "Eval queries must never enter the training set."
            )
        return v

    @field_validator("positive_id")
    @classmethod
    def positive_not_eval_asset(cls, v: str) -> str:
        """L-2: positive_id must not be an eval positive."""
        if v in EVAL_POSITIVE_IDS:
            raise ValueError(
                f"positive_id '{v}' is an eval positive document (L-2 violation). "
                "Eval positive documents must never appear as training positives."
            )
        return v

    @field_validator("category")
    @classmethod
    def category_is_valid(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"category '{v}' is not in VALID_CATEGORIES {sorted(VALID_CATEGORIES)}."
            )
        return v

    @field_validator("hard_negative_ids")
    @classmethod
    def hard_negatives_not_eval_vault(cls, v: list[str]) -> list[str]:
        """L-4: hard negatives must not come from eval vault."""
        for nid in v:
            if nid in EVAL_POSITIVE_IDS or any(nid.startswith(pfx) for pfx in _EVAL_VAULT_PREFIXES):
                raise ValueError(
                    f"hard_negative_id '{nid}' is from the eval vault (L-4 violation). "
                    "Hard negatives must be sourced from the approved training corpus only."
                )
        return v

    @field_validator("hard_negative_texts")
    @classmethod
    def hard_negative_texts_non_empty(cls, v: list[str]) -> list[str]:
        for text in v:
            if not text.strip():
                raise ValueError("All hard_negative_texts must be non-empty.")
        return v

    @model_validator(mode="after")
    def validate_id_and_consistency(self) -> "TrainingExample":
        """L-6: training_id must match deterministic hash."""
        expected = compute_training_id(self.query, self.positive_id, self.creation_method)
        if self.training_id != expected:
            raise ValueError(
                f"training_id mismatch (L-6 violation). "
                f"Expected {expected!r}, got {self.training_id!r}. "
                "Use compute_training_id(query, positive_id, creation_method)."
            )

        # hard_negative_ids and hard_negative_texts must be parallel lists.
        if len(self.hard_negative_ids) != len(self.hard_negative_texts):
            raise ValueError("hard_negative_ids and hard_negative_texts must have equal length.")

        # positive_id must not appear in hard_negative_ids.
        if self.positive_id in self.hard_negative_ids:
            raise ValueError(
                f"positive_id '{self.positive_id}' appears in hard_negative_ids. "
                "A positive document cannot also be a hard negative."
            )

        # synthetic_llm and mined_* examples must declare details in notes.
        if self.creation_method in {"synthetic_llm", "mined_bm25", "mined_dense", "mined_hybrid"}:
            if not self.notes:
                raise ValueError(
                    f"creation_method '{self.creation_method}' requires a non-empty 'notes' "
                    "field declaring the generating model or relevance threshold."
                )

        # human examples should have a provenance_url.
        if self.creation_method == "human" and not self.provenance_url:
            raise ValueError("creation_method 'human' requires provenance_url pointing to source.")

        return self


class TrainingDataset(BaseModel):
    """A named, versioned collection of TrainingExamples."""

    dataset_id: str = Field(
        min_length=1, description="Stable dataset name, e.g. 'cortex-train-v1'."
    )
    schema_version: str = Field(default="1.0.0")
    description: str = Field(default="")
    examples: list[TrainingExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_duplicate_ids(self) -> "TrainingDataset":
        """L-6: all training_ids must be unique within the dataset."""
        seen: set[str] = set()
        for ex in self.examples:
            if ex.training_id in seen:
                raise ValueError(
                    f"Duplicate training_id '{ex.training_id}' (L-6 violation). "
                    "Each training example must have a unique deterministic ID."
                )
            seen.add(ex.training_id)
        return self


def validate_no_eval_leakage(
    examples: list[TrainingExample],
) -> list[str]:
    """
    Check a list of already-validated TrainingExamples for eval leakage.

    Returns a list of human-readable violation strings. An empty list means
    no leakage was detected.

    Note: L-1 and L-2 are already enforced by TrainingExample field
    validators. This function provides an additional batch-level summary
    report, and checks L-4 at the batch level with the full EVAL_POSITIVE_IDS
    set in scope.
    """
    violations: list[str] = []
    for ex in examples:
        # L-1 batch check (defensive)
        if ex.query.strip().lower() in EVAL_QUERIES:
            violations.append(f"[L-1] {ex.training_id}: query matches eval query")
        # L-2 batch check (defensive)
        if ex.positive_id in EVAL_POSITIVE_IDS:
            violations.append(
                f"[L-2] {ex.training_id}: positive_id '{ex.positive_id}' is eval positive"
            )
        # L-4 batch check
        for nid in ex.hard_negative_ids:
            if nid in EVAL_POSITIVE_IDS:
                violations.append(f"[L-4] {ex.training_id}: hard_negative '{nid}' is eval positive")
    return violations
