# Milestone 3C: Embedding Fine-Tuning Experiment Report

**Date**: 2026-08-29T14:22:59.599426+00:00  
**Status**: Completed  
**Recommendation Verdict**: **INVESTIGATE**  

---

## 1. Experiment Overview & Model Metadata

| Property | Value |
|---|---|
| **Base Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Model Licence** | `Apache-2.0` |
| **Model Revision** | `c97b51e065bf022a...` |
| **Parameter Count** | 22,713,216 (22.7M) |
| **Training Set Size** | 500 pairs (300 HotpotQA + 200 PKM Synthetic) |
| **Device** | `mps` (arm) |
| **Training Duration** | 105.32s |
| **Hyperparameters** | Epochs: 3, Batch Size: 16, LR: 2e-05 |
| **Loss Function** | `MultipleNegativesRankingLoss` (InfoNCE) |

---

## 2. Evaluation Results Summary

### 2.1 HotpotQA Dev Set (50 held-out multi-hop pairs)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| **Recall@1** | 0.8000 | 0.8200 | **+0.0200** |
| **Recall@5** | 0.9400 | 0.9600 | **+0.0200** |
| **Recall@10** | 0.9600 | 1.0000 | **+0.0400** |
| **MRR** | 0.8575 | 0.8773 | **+0.0199** |
| **nDCG@10** | 0.8808 | 0.9066 | **+0.0258** |
| **Avg Query Latency** | 20.94 ms | 26.21 ms | +5.27 ms |

### 2.2 Frozen Cortex Eval v1 (25 queries, 11 vault notes)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| **Recall@1** | 0.7000 | 0.7000 | **+0.0000** |
| **Recall@5** | 0.9600 | 0.9600 | **+0.0000** |
| **Recall@10** | 1.0000 | 1.0000 | **+0.0000** |
| **MRR** | 0.8700 | 0.8767 | **+0.0067** |
| **nDCG@10** | 0.8960 | 0.9023 | **+0.0063** |
| **Avg Query Latency** | 9.02 ms | 10.08 ms | +1.06 ms |

---

## 3. Regression Analysis

### 3.1 Cortex Eval v1 Per-Query Regressions (0 detected)

> **Zero regressions detected on the frozen Cortex Eval v1 benchmark.** All 25 queries matched or exceeded baseline performance.

---

## 4. Production & Architectural Impact

- **Default Configuration**: Cortex's production configuration (`qwen3-embedding:0.6b` via Ollama / hybrid retrieval) remains **100% unchanged**.
- **Model Checkpoints**: Checkpoints are stored in `training/checkpoints/` (gitignored).
- **Isolation Verification**: Training used strictly `combined_train.jsonl`. No validation or evaluation assets were exposed to model parameter updates.

---

## 5. Recommendation & Next Steps

**Verdict**: **INVESTIGATE**

**Rationale**:
Fine-tuning demonstrated solid metric retention/gains on retrieval with minimal regression. Recommend further investigation with scaling and domain weighting before replacing production defaults.

**Proposed Next Milestones**:
1. Keep the experimental pipeline versioned in `training/experiment.py`.
2. Expand the synthetic PKM seed corpus with additional technical domains for higher representation diversity.
3. Investigate cross-encoder reranking vs bi-encoder tuning in subsequent milestones.
