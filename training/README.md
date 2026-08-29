# Training & Fine-Tuning Pipeline (Milestones 3B–3C)

This directory contains the data acquisition, synthetic pair generation, and embedding fine-tuning pipelines for Cortex.

> **Production Invariant**: This directory contains experimental research pipelines and pilot benchmarks. Cortex's default production runtime configuration (`qwen3-embedding:0.6b` via Ollama with hybrid dense + BM25 retrieval) remains **100% unchanged**.

---

## 1. Quick Start & Reproduction

### Step 1: Build HotpotQA Dataset (Milestone 3B-1)
Downloads canonical HotpotQA (Yang et al., EMNLP 2018) raw data, verifies checksums, filters 0-eval-leakage pairs, and constructs document-disjoint splits:
```bash
# Build HotpotQA train (≤300) and dev (≤50) splits
python training/build_hotpotqa.py

# Skip download if raw files already exist in training/raw/
python training/build_hotpotqa.py --skip-download
```

### Step 2: Generate PKM Synthetic Dataset (Milestone 3C)
Constructs 200 train and 50 dev contrastive retrieval pairs from 20 newly authored project-owned seed markdown documents across 12 technical domains:
```bash
# Generate PKM synthetic pairs and review sample pack
python training/build_pkm_synthetic.py
```

### Step 3: Run Fine-Tuning Experiment (Milestone 3C)
Combines HotpotQA and PKM synthetic data (500 pairs total), trains `sentence-transformers/all-MiniLM-L6-v2`, and benchmarks baseline vs fine-tuned models on HotpotQA dev (50 pairs) and the frozen Cortex Eval v1 benchmark (25 queries):
```bash
# Run full training and evaluation harness
python training/experiment.py
```

---

## 2. Model & Experiment Configuration

| Property | Value |
|---|---|
| **Base Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Model Licence** | Apache-2.0 |
| **Model Git Revision** | `c97b51e065bf022ab3b544eb40778c1db1755106` |
| **Parameter Count** | 22,713,216 (22.7M) |
| **Training Set Size** | 500 pairs (300 HotpotQA + 200 PKM Synthetic) |
| **Hardware / Device** | Apple Silicon GPU (`mps`) or CPU |
| **Hyperparameters** | 3 Epochs, Batch Size 16, LR 2e-5, Warmup 10%, Weight Decay 0.01 |
| **Loss Function** | `MultipleNegativesRankingLoss` (InfoNCE) |

---

## 3. Key Experiment Results (Baseline vs Fine-Tuned)

### 3.1 HotpotQA Dev Set (50 held-out multi-hop pairs, 200 candidate pool)
| Metric | Baseline (`all-MiniLM-L6-v2`) | Fine-Tuned (500 pairs) | Delta |
|---|---|---|---|
| **Recall@1** | 0.8000 | 0.8200 | **+0.0200** (+2.5%) |
| **Recall@5** | 0.9400 | 0.9600 | **+0.0200** (+2.1%) |
| **Recall@10** | 0.9600 | 1.0000 | **+0.0400** (+4.2%) |
| **MRR** | 0.8575 | 0.8773 | **+0.0199** (+2.3%) |
| **nDCG@10** | 0.8808 | 0.9066 | **+0.0258** (+2.9%) |
| **Avg Query Latency** | 20.94 ms | 26.21 ms | +5.27 ms |

### 3.2 Frozen Cortex Eval v1 (25 queries, 11 vault notes)
| Metric | Baseline (`all-MiniLM-L6-v2`) | Fine-Tuned (500 pairs) | Delta |
|---|---|---|---|
| **Recall@1** | 0.7000 | 0.7000 | **+0.0000** |
| **Recall@5** | 0.9600 | 0.9600 | **+0.0000** |
| **Recall@10** | 1.0000 | 1.0000 | **+0.0000** |
| **MRR** | 0.8700 | 0.8767 | **+0.0067** (+0.8%) |
| **nDCG@10** | 0.8960 | 0.9023 | **+0.0063** (+0.7%) |
| **Avg Query Latency** | 9.02 ms | 10.08 ms | +1.06 ms |
| **Per-Query Regressions** | — | — | **0 regressions** |

Full analysis is available in [`eval/EXPERIMENT_REPORT_3C.md`](../eval/EXPERIMENT_REPORT_3C.md).

---

## 4. Data Provenance & Review Status

1. **HotpotQA**:
   - CC BY-SA 4.0 (Yang et al., EMNLP 2018).
   - See [`training/SOURCES.md`](SOURCES.md) and [`eval/LICENSING.md`](../eval/LICENSING.md) for full attribution, license terms, and weight distribution restrictions.
2. **PKM Synthetic Pairs**:
   - MIT licensed, 100% generated from project-authored seed notes.
   - **Review Status Notice**: All synthetic examples are tagged with `review_status: "unreviewed"` and `reviewer_decision: "PENDING_HUMAN_REVIEW"`.
   - A representative 20-sample review pack is maintained in [`training/synthetic_review_pack.json`](synthetic_review_pack.json). Synthetic examples must not be represented as human-reviewed without explicit human audit.
3. **Rejected Corpora**:
   - NQ/DPR (CC-BY-NC 4.0), NF-Corpus (custom restrictive), SciFact (ODC-By / publisher copyright), and MS MARCO (non-commercial) are permanently excluded from this repository.

---

## 5. Limitations & Scope

- **Pilot Scale**: The dataset size (500 training pairs) and dev evaluation size (50 pairs) are intended as a small-scale proof-of-concept for domain adaptation.
- **Evaluation Power**: The 25-query frozen evaluation suite serves as a regression smoke-test, but larger held-out benchmark suites are required to detect sub-percentage quality deltas with high statistical confidence.
- **Model Scope**: Limited to bi-encoder fine-tuning of `all-MiniLM-L6-v2`. Cross-encoder rerankers, LoRA, and hard-negative mining across massive corpora remain future exploratory topics.

---

## 6. Committed Files & Structure

| File | Committed | Description |
|---|---|---|
| `build_hotpotqa.py` | ✅ | HotpotQA download, leakage filter, and split builder |
| `build_pkm_synthetic.py` | ✅ | PKM seed documents and synthetic pair generator |
| `experiment.py` | ✅ | Fine-tuning and evaluation experiment harness |
| `manifest.json` | ✅ | HotpotQA dataset metadata and hashes |
| `manifest_synthetic.json` | ✅ | PKM synthetic dataset metadata and hashes |
| `synthetic_review_pack.json` | ✅ | 20-sample unreviewed audit pack |
| `experiment_results_3c.json` | ✅ | Machine-readable experiment results |
| `SOURCES.md` | ✅ | Licensing and attribution records |
| `README.md` | ✅ | This documentation |
| `raw/` | ❌ gitignored | Official HotpotQA raw downloads |
| `*.jsonl` | ❌ gitignored | Generated intermediate training sets |
| `checkpoints/` | ❌ gitignored | PyTorch / HuggingFace model checkpoints |
