# Milestone: Deterministic Hard-Negative Mining Report

**Date**: 2026-08-29 15:53:59 UTC  
**Pipeline**: `deterministic_hard_negative_mining`  
**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Revision: `c97b51e065bf022ab3b544eb40778c1db1755106`)  
**Lexical Model**: Deterministic Okapi BM25 ($k_1=1.5, b=0.75$)  
**Seed**: `42`  

---

## 1. Executive Summary

The deterministic hard-negative mining pipeline processed all **300 approved training queries** from HotpotQA. A total of **12769 candidate passages** were retrieved across BM25 and dense retrieval pools. Strict leakage and disjointness filters rejected **569 invalid/colliding candidates**, producing a total of **1500 validated hard negatives** across the dataset (5 per query).

---

## 2. Hard-Negative Distribution by Retriever Source

| Retriever Type | Selected Negatives Count | Percentage | Characteristics |
|---|---|---|---|
| **Hybrid Overlap** | 565 | 37.7% | Ranked high in both BM25 and Dense pools; challenging for both lexical and semantic matchers. |
| **Dense (Semantic)** | 608 | 40.5% | High cosine similarity without exact keyword overlap; teaches fine-grained topical boundaries. |
| **BM25 (Lexical)** | 327 | 21.8% | High keyword overlap without semantic entailment; prevents false-positive keyword matching. |
| **Total** | **1500** | **100.0%** | **Balanced multi-modal negative supervision** |

---

## 3. Candidate Rejection Breakdown

| Rejection Reason | Count | Invariant Enforced |
|---|---|---|
| `EXACT_POSITIVE` | 296 | Prevent training contradiction (cannot be negative of itself) |
| `MULTIHOP_SUPPORTING_FACT` | 270 | Prevent second supporting fact of multi-hop query from being penalized |
| `DUPLICATE_TEXT` | 3 | Exact whitespace-duplicate passage text elimination |

---

## 4. Human Review Pack

- **File Location**: [`training/mined_review_pack.json`](file:///Users/jeeveshsingale/Developer/cortex/training/mined_review_pack.json)
- **Sample Size**: 60 sampled query/positive/negative triplets
- **Review Status**: `unreviewed`
- **Reviewer Decision**: `PENDING_HUMAN_REVIEW`
- **Provenance**: Every sample includes source retriever, rank, score, and model revision.

---

## 5. Quality Gate & Isolation Audit

| Check | Target | Actual | Verdict |
|---|---|---|---|
| **Eval Query Leakage (L-1)** | 0 | 0 | **PASSED** |
| **Eval Positive Leakage (L-2)** | 0 | 0 | **PASSED** |
| **Eval Negative Leakage (L-4)** | 0 | 0 | **PASSED** |
| **Dev Positive Overlap** | 0 | 0 | **PASSED** |
| **Positive/Negative Overlap** | 0 | 0 | **PASSED** |
| **Negative List Parallelism** | IDs == Texts | Matched | **PASSED** |
| **Deterministic Repeatability** | Identical SHA-256 | Verified | **PASSED** |

---

## 6. Proposed Embedder Retraining Ablation Plan

With high-quality mined hard negatives now available, the proposed next experiment will compare:
1. **Baseline Model**: `sentence-transformers/all-MiniLM-L6-v2` (zero-shot)
2. **Experiment 3C (Random In-Batch Negatives)**: MNRL on 500 pairs without explicit mined negatives
3. **Experiment 4A (Hard Negatives Only)**: MultipleNegativesRankingLoss with 1 positive + 5 mined hard negatives per query
4. **Experiment 4B (Combined PKM + Mined HotpotQA)**: Fine-tuning on 300 mined HotpotQA pairs + 200 PKM synthetic pairs

Evaluation metrics will be computed across both held-out evaluation benchmarks:
- **HotpotQA Dev Split (50 examples)**: Recall@1/5/10, nDCG@10, MRR
- **Cortex Eval v1 (25 queries, 11 vault notes)**: Recall@1/5/10, nDCG@10, MRR, zero regression tolerance
