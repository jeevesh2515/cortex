# Milestone 3B-2: Decision Record & Training Data Design

**Date**: 2026-08-29  
**Status**: Completed — Research and Design Only  
**Branch**: `ml-training-upgrade`  
**Commit Verified**: `f6f8b98`  

---

## Executive Summary

This decision record formalises the outcomes of Milestone 3B-2:
1. **NQ/DPR Passage Corpus Audit**: Formally **REJECTED**. While Wikipedia text is CC BY-SA, the canonical DPR passage segmentation and retrieval training pairs produced by Facebook AI Research (`facebookresearch/DPR`) are licensed under **CC-BY-NC 4.0** (Non-Commercial). This violates the project's commercial permissibility requirements.
2. **PKM Synthetic-Pair Design**: Established end-to-end specifications for generating 200 train / 50 dev contrastive pairs from newly authored, project-owned PKM markdown notes, with strict provenance, human review rubric, and 5-layer leakage prevention.
3. **HotpotQA 300/50 Quality Audit**: Complete audit of the Milestone 3B-1 dataset. 100% schema compliance, 0 eval leakage violations, 0 duplicate/near-duplicate query clusters, and exactly 0 document overlap between train (300 docs) and dev (50 docs).
4. **Scaling Decision**: Retain the 300 train / 50 dev HotpotQA dataset as the fixed external baseline split. Expand to 500 train / 100 dev only by adding the 200/50 PKM synthetic dataset in Milestone 3C upon explicit approval.

---

## 1. NQ / DPR Passage Corpus Audit

### 1.1 Source Identification & Architecture

| Component | Canonical Source | Authors / Owner | License |
|---|---|---|---|
| **Query & QA Annotations** | Natural Questions (NQ) | Kwiatkowski et al. (2019), Google LLC | CC BY-SA 3.0 / Apache 2.0 |
| **Passage Corpus** (`psgs_w100.tsv.gz`) | Facebook AI Research DPR | Karpukhin et al. (2020), Meta/Facebook | **CC-BY-NC 4.0** |
| **Retrieval Split Pairs** (`nq-train.json`, `nq-dev.json`) | Facebook AI Research DPR | Karpukhin et al. (2020), Meta/Facebook | **CC-BY-NC 4.0** |

- **Canonical Passage Download URL**: `https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`
- **Snapshot & Size**: English Wikipedia dump from December 20, 2018; segmented into 21,015,324 non-overlapping 100-word blocks.
- **Hash Availability**: No author-published canonical SHA-256 hash exists in the repository; hashes must be computed dynamically.

### 1.2 Distinction Between NQ QA and DPR Retrieval Corpus

A critical architectural distinction exists between Google's Natural Questions and the DPR retrieval corpus:
1. **Google Natural Questions**: A reading comprehension dataset consisting of Google Search queries paired with full HTML Wikipedia pages and token span annotations (long/short answers). It does **not** provide segmented passage retrieval chunks or contrastive negative rankings.
2. **DPR Retrieval Corpus**: Created by Facebook AI Research by segmenting Wikipedia into 100-word passages and mining BM25 / dense positive and negative passages for each NQ question.

### 1.3 License & Commercial / Redistribution Status

- The `facebookresearch/DPR` repository and all distributed data assets (including `psgs_w100.tsv.gz` and `nq-*.json`) are released under **CC-BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
- The NonCommercial restriction explicitly forbids commercial use and exploitation.
- While raw Wikipedia text is CC BY-SA 3.0/4.0, using the curated passage segmentation, passage IDs, and query-passage pairings created by Facebook AI Research binds the downstream user to the CC-BY-NC 4.0 license of the DPR distribution.

### 1.4 Notice & Attribution Obligations

Under CC-BY-NC 4.0:
- Must give appropriate credit to Karpukhin et al. (2020) and Facebook AI Research.
- Must provide a link to the license and indicate if changes were made.
- Derived works cannot be distributed for commercial purposes.

### 1.5 Audit Verdict

> **VERDICT: REJECTED**  
> DPR Wikipedia passages and NQ-DPR retrieval pairs are **REJECTED** due to the **CC-BY-NC 4.0** non-commercial restriction. This project maintains a commercially permissible codebase and rejects datasets with non-commercial limitations (consistent with the previous rejection of MS MARCO). Rights cannot be inferred from raw Wikipedia text when using Facebook's curated passage corpus and relevance pairs.

---

## 2. PKM Synthetic-Pair Design

### 2.1 Permitted Seed Content

To eliminate copyright risk and eval leakage, synthetic data generation will use exclusively project-authored markdown seeds:
- **PERMITTED**: Newly authored synthetic PKM notes written explicitly for Cortex training, covering realistic note-taking domains:
  - Technical architecture notes & design patterns
  - Project planning and retrospectives
  - Daily journals and personal logs
  - Domain-specific study notes (science, engineering, culinary chemistry)
- **STRICTLY PROHIBITED**:
  - `eval/vault/*` notes (eval containment)
  - `eval/cases.yaml` queries (eval containment)
  - HotpotQA or NQ/DPR Wikipedia texts
  - Private / personal user vaults
  - Third-party copyrighted documents

### 2.2 Generation Specification & Metadata Schema

- **Generator Model**: Local deterministic execution (e.g. `qwen2.5:7b-instruct` / `ollama` with locked digest and `temperature: 0.0`).
- **Prompt Template**:
  ```text
  You are an expert Personal Knowledge Management (PKM) annotator.
  Given the following synthetic note, generate:
  1. A natural search query that a user would write when looking for this note.
  2. The exact relevant text excerpt from the note (positive).
  3. Two plausible but incorrect text excerpts (hard negatives) on related topics that do NOT answer the query.

  Format as strict JSON:
  {
    "query": "...",
    "positive_text": "...",
    "hard_negative_texts": ["...", "..."],
    "category": "dense_semantic | lexical_exact | hybrid_core | wikilink_graph | temporal_query"
  }
  ```
- **Provenance Metadata**:
  Every synthetic example must populate `creation_method: "synthetic_llm"` and `notes` with valid JSON:
  ```json
  {
    "generator_model": "qwen2.5:7b-instruct",
    "generator_digest": "sha256:...",
    "prompt_version": "pkm-synth-v1.0",
    "reviewed_by": "human_reviewer_id",
    "review_date": "2026-08-29T15:00:00Z",
    "seed_doc_id": "synthetic_vault/note_012.md"
  }
  ```

### 2.3 Human-Review Rubric & Acceptance Criteria

Every candidate synthetic example must pass 4-point human review before inclusion:
1. **Target Relevance (Pass/Fail)**: The query must be unambiguous and directly answered by `positive_text`.
2. **Negative Discrimination (Pass/Fail)**: `hard_negative_texts` must share lexical/topical similarity but must NOT satisfy the query (no false negatives).
3. **Query Naturalness (Pass/Fail)**: Query must reflect genuine PKM search behavior, avoiding stilted LLM phrasing or prompt leaks.
4. **Formatting & Bounds (Pass/Fail)**: Query length 5–256 chars, positive length 100–1000 chars, exact parallel list lengths for negatives.

### 2.4 Leakage & Isolation Rules

1. **L-1 (Eval Query Isolation)**: Exact and normalized lowercased substring check against all 25 queries in `eval/cases.yaml`.
2. **L-2 (Eval Positive Isolation)**: Positive ID check against all 11 IDs in `EVAL_POSITIVE_IDS`.
3. **L-4 (Eval Hard Negative Isolation)**: Negative ID check against `_EVAL_VAULT_PREFIXES`.
4. **L-6 (Deterministic Hash)**: `training_id = SHA-256(query + positive_id + "synthetic_llm")`.
5. **Cross-Corpus Isolation**: Cosine/substring deduplication against HotpotQA `training/train.jsonl` and `training/dev.jsonl`.
6. **Document-Disjoint Splits**: Synthetic seed documents are assigned to `train` (80%) or `dev` (20%) at the document level prior to generation.

*Scope Note*: No synthetic examples have been generated in Milestone 3B-2.

---

## 3. HotpotQA 300/50 Dataset Quality Audit

A comprehensive quality audit was conducted on `training/train.jsonl` (300 examples) and `training/dev.jsonl` (50 examples).

### 3.1 Distribution & Metric Summary

| Metric | Train Split | Dev Split | Total |
|---|---|---|---|
| **Example Count** | 300 | 50 | 350 |
| **Category** | `multi_topic` (100%) | `multi_topic` (100%) | 350 (100%) |
| **Licence** | `CC-BY-SA-4.0` | `CC-BY-SA-4.0` | 350 (100%) |
| **Hard Negatives per Example** | Exactly 3 (100%) | Exactly 3 (100%) | Exactly 3 (100%) |
| **Unique Positive Documents** | 300 | 50 | 350 |
| **Document Overlap (Train ∩ Dev)** | **0** | **0** | **0** |
| **Missing Schema Fields** | 0 | 0 | 0 |
| **Duplicate `training_id`s** | 0 | 0 | 0 |
| **Exact Duplicate Queries** | 0 | 0 | 0 |
| **60-Char Prefix Clusters** | 0 | 0 | 0 |
| **Eval Leakage Violations (L-1/L-2/L-4)**| **0** | **0** | **0** |

### 3.2 Text Length Profile

- **Positive Passage Length**:
  - Minimum: 68 characters
  - Maximum: 1,669 characters
  - Mean: 420 characters
  - Passages < 50 characters: 0

### 3.3 Sample Review (Accepted Examples)

1. **Example `cbc689...` (Train)**:
   - *Query*: "Which magazine was started first Arthur's Magazine or First for Women?"
   - *Positive Document*: `hotpotqa/Arthur's Magazine` (Text: *"Arthur's Magazine (1844–1846) was an American literary periodical..."*)
   - *Hard Negative*: `hotpotqa/Radio City (Indian radio station)` (Text: *"Radio City is India's first private FM radio station..."*)
   - *Assessment*: High-quality multi-hop retrieval pair; hard negative challenges lexical matching on "first" while remaining non-relevant.
2. **Example `95eb8d...` (Train)**:
   - *Query*: "The Oberoi family is part of a hotel company that has a head office in what city"
   - *Positive Document*: `hotpotqa/Oberoi family` (Text: *"The Oberoi family is an Indian family that is famous for its involvement in hotels..."*)
   - *Hard Negative*: `hotpotqa/The Ritz-Carlton Jakarta` (Text: *"The Ritz-Carlton Jakarta is a hotel and skyscraper..."*)
   - *Assessment*: Clear semantic match with strong topical distractor (hotel domain).

### 3.4 Ingestion Rejections

During dataset construction:
- **Eval Positive Taint**: Distractor articles matching eval positive titles were automatically filtered from hard negative lists.
- **Split Consistency**: Multi-hop examples where supporting facts hashed to conflicting splits were rejected to maintain strict document-level isolation.

---

## 4. Scaling Decision

### 4.1 Comparative Analysis

| Option | Size | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **Option A: Retain 300 / 50 HotpotQA** | 300 train / 50 dev | Verified 100% clean, zero leakage, zero document overlap, fully audited. | Single category (`multi_topic`). | **RECOMMENDED FOR BASELINE** |
| **Option B: Expand HotpotQA to 500 / 100** | 500 train / 100 dev | Reaches round number using existing script. | Increases volume in single category without adding domain/query diversity. | **REJECTED (Arbitrary scaling)** |
| **Option C: Hybrid 300 HotpotQA + 200 PKM Synth** | 500 train / 100 dev | Adds PKM domain diversity, lexical/temporal query types, 100% project-owned text. | Requires synthetic generation and human review execution. | **RECOMMENDED FOR MILESTONE 3C** |

### 4.2 Scaling Policy & Prerequisites

Expansion beyond 300 train / 50 dev must NOT be done merely to hit round numbers. Expansion will proceed to 500 train / 100 dev in Milestone 3C only when:
1. The synthetic PKM generation pipeline and seed notes are approved.
2. 200 train + 50 dev PKM synthetic pairs pass 100% human review under the rubric in §2.3.
3. Training loss and contrastive evaluation on the 300/50 baseline establish a reference benchmark.

---

## 5. Required Approvals for Next Steps

To proceed with future milestones, explicit approvals are required:

- **[APPROVAL 3B-2.1: NQ/DPR Rejection Confirmation]**: Confirm formal rejection of NQ/DPR passage retrieval datasets due to CC-BY-NC 4.0.
- **[APPROVAL 3B-2.2: PKM Synthetic Generation Plan]**: Approve generation of 200 train / 50 dev PKM synthetic examples under the specification and review rubric in §2.
- **[APPROVAL 3B-2.3: Baseline Pilot Lock]**: Lock the 300 train / 50 dev HotpotQA dataset as the official external baseline component.
