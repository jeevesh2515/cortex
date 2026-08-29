# Milestone 3A — Evaluation-Validity Audit & Training-Data Design

> **Status**: Approved baseline locked at commit `c64e93c`.
> **Branch**: `ml-training-upgrade`
> **Baseline assets are immutable.** Do not edit `eval/baseline_dense.json`,
> `eval/baseline_dense.md`, `eval/cases.yaml`, or `eval/vault/` without
> creating a new versioned baseline with a fresh commit hash.

---

## 1. Baseline Asset Fingerprints

The following SHA-256 hashes were recorded at commit `c64e93c` and must be
verified before any future re-evaluation run is accepted as the current
baseline.

| Asset | SHA-256 |
|---|---|
| `eval/baseline_dense.json` | `64c190af30f28bd2ffa0eac1bf170c90d5ce858e14de1db67235a4207daf7ed0` |
| `eval/baseline_dense.md` | `0d37a6c2b854fd362cf75dd12986110fa20c52c0bd8b39aebedfab9fcc00dd8f` |
| `eval/cases.yaml` | `001e75638c7a667ba6ef53910a1ad9cbbaba367e26aaa2eb74b17ea509db9acd` |
| `eval/vault/Distractor Search.md` | `3ee280a8df5399631c8a853b593459adfb231b78b4f6092cd5dbd95ecf19f6b3` |
| `eval/vault/Embeddings.md` | `affe4e83104a5b0991659fb4bb590a801e73dd1ed782419586d6b1a1066058e4` |
| `eval/vault/Fermentation Chemistry.md` | `b166b341cbb16c7674e13dd8bc2c5a3dadc0c11d57317d0529fd19a7b640ae2f` |
| `eval/vault/Flour Chemistry.md` | `b7c1f0e08aa067ead6fde09cab78e2a9c1a39c03d96c637ce105edb2a9435d7e` |
| `eval/vault/Pectin Chains.md` | `c6604d41e5977896e0d25f9932ae7c08815c60516b8bef306df9a1034c19e201` |
| `eval/vault/Retrieval.md` | `3fecdd76622aa05ae0042f2c83f2921a2503c106267a81cd06ac112fea9ce6d7` |
| `eval/vault/Sourdough.md` | `c48fea324d95983d620e0792cb1af90026d8c3a46e0fac1dca4e6b9c1d9b3a26` |
| `eval/vault/Thermal Management.md` | `3c32e08aa292057f3132e7c28978ddbc7028b2628d443b85d6ea65c3f1bd1cc2` |
| `eval/vault/Vector Databases.md` | `731cfd2719333f9ff6c2ffb562275d4080745672e6b54b96f3228f333f27312e` |
| `eval/vault/daily/2026-08-01.md` | `6f900cac33010024602ce9a422e306e7f8c1f34d6a051938919c0bc8a341cb01` |
| `eval/vault/daily/2026-08-15.md` | `8f775deec16aeea8eabeeded24b979e350708423e03d319d8635c9b8f5717268` |

Verification command:

```bash
sha256sum -c eval/CHECKSUMS.sha256
```

---

## 2. Statistical Usefulness Audit

### 2.1 Query Count by Category

| Category | Queries | Fraction |
|---|---|---|
| `dense_semantic` | 10 | 40% |
| `multi_topic` | 4 | 16% |
| `lexical_exact` | 4 | 16% |
| `wikilink_graph` | 2 | 8% |
| `temporal_query` | 2 | 8% |
| `distractor_resistance` | 2 | 8% |
| `hybrid_core` | 1 | 4% |
| **Total** | **25** | 100% |

**Observation**: `dense_semantic` is over-represented at 40%. Categories
`hybrid_core`, `wikilink_graph`, `temporal_query`, and `distractor_resistance`
each have 1-2 queries — statistically insufficient for category-level
conclusions. A single case flip changes the category metric by 50-100 pp.

### 2.2 Query Count by Expected Document

| Document | Queries Where Relevant | Case IDs |
|---|---|---|
| `Retrieval.md` | 4 | `retrieval_01`, `retrieval_05`, `paraphrase_01`, `paraphrase_07` |
| `Sourdough.md` | 4 | `baking_01`, `baking_04`, `paraphrase_03`, `paraphrase_06` |
| `Flour Chemistry.md` | 4 | `baking_02`, `baking_03`, `baking_04`, `paraphrase_03` |
| `Pectin Chains.md` | 3 | `biochem_02`, `biochem_03`, `paraphrase_05` |
| `Fermentation Chemistry.md` | 3 | `biochem_01`, `biochem_03`, `paraphrase_06` |
| `daily/2026-08-15.md` | 3 | `temporal_02`, `temporal_03`, `paraphrase_08` |
| `Embeddings.md` | 2 | `retrieval_02`, `paraphrase_04` |
| `Thermal Management.md` | 2 | `retrieval_04`, `paraphrase_02` |
| `Vector Databases.md` | 2 | `retrieval_03`, `distractor_02` |
| `daily/2026-08-01.md` | 2 | `temporal_01`, `paraphrase_07` |
| `Distractor Search.md` | 1 | `distractor_01` |

`Distractor Search.md` has one query — its signal is a coin flip. Three notes
(`Retrieval`, `Sourdough`, `Flour Chemistry`) account for 12 of 25
query-document pairs, creating metric skew.

### 2.3 Candidate Pool

- Total indexed notes: 11. One chunk per note; chunk candidates = note candidates.
- All 11 notes appear in at least one eval case.
- With `top_k=10` and 11 candidates, Recall@10 is trivially 1.0 for any
  working retrieval system (91% of corpus). Recall@5 covers 45% of corpus.

### 2.4 Near-Duplicate Query Pairs

None of these are exact duplicates, but they share enough signal that a model
memorising one will trivially pass the other:

| Case A | Case B | Shared Target |
|---|---|---|
| `retrieval_04_thermal_fanless` ("cortex prevent macbook air from getting hot") | `paraphrase_02_cpu_heat_throttling` ("pmset speed limit drops when mac heats up") | Both -> `Thermal Management.md` |
| `retrieval_02_embeddings_concept` ("semantic vector similarity in continuous space") | `paraphrase_04_qwen_embedding_speed` ("small dense embedding models low latency MTEB") | Both -> `Embeddings.md` |
| `baking_01_starter_feeding` ("how often to feed sourdough starter and bulk ferment") | `paraphrase_03_high_protein_flour` ("best flour protein percentage for sourdough") | Both include `Sourdough.md` |
| `biochem_01_fermentation_kinetics` ("polysaccharide breakdown metabolic pathways") | `paraphrase_06_wild_yeast_microbiology` ("lactic acid bacteria and wild yeast symbiosis") | Both include `Fermentation Chemistry.md` |
| `temporal_02_aug_15_benchmarks` ("what metrics did I implement on 2026-08-15") | `temporal_03_query_expansion_work` ("when did I work on local query expansion") | Both -> `daily/2026-08-15.md` |
| `paraphrase_07_bidirectional_links` ("avoid heavy graphrag bidirectional obsidian links") | `retrieval_05_graph_expansion` ("wikilink traversal second order context") | Both include `Retrieval.md` |

Effective independent query count: approximately 18-20 of 25.

### 2.5 Near-Ceiling Metric Risk

| Metric | Baseline | Risk |
|---|---|---|
| Recall@1 | 0.900 | Only 10% headroom; 3 case regressions = -0.12 |
| Recall@5 | 1.000 | **Saturated** — cannot detect regressions |
| Recall@10 | 1.000 | **Saturated by corpus size** — uninformative at N=11 |
| MRR | 1.000 | **Saturated** — all first relevant hits are rank-1 |
| MAP | 0.9933 | Near-saturated |
| nDCG@10 | 0.9968 | Near-saturated |

At N=25, the 95% Wilson confidence interval on Recall@1=0.90 is ±0.12.
Two models at 0.90 and 0.92 Recall@1 are statistically indistinguishable.

### 2.6 Minimum Scale Before Quality Deltas Can Be Trusted

| Confidence Target | Required Queries | Required Documents | Rationale |
|---|---|---|---|
| Detect ±5 pp at 80% power | >= 200 | >= 100 | TREC-style minimum |
| Detect ±2 pp at 80% power | >= 1,000 | >= 500 | MTEB-scale |
| Per-category (>= 30/cat, 7 cats) | >= 210 | >= 50 | Poisson stability |
| Embedding A/B (0.90 vs 0.92 MRR) | >= 500 | >= 200 | z-test at alpha=0.05 |

**Practical minimum**: 200 queries x 50 unique documents, no single
document covering more than 10% of queries, no category with fewer than
20 cases. Current fixture = fast regression smoke-test only.

---

## 3. Strict Split Policy

### 3.1 Three-Way Split Definition

```
eval/      — FROZEN; immutable regression and quality gate
training/  — to be created in Milestone 3B (requires licensing approval)
dev/       — held-out tuning set; created alongside training/
```

### 3.2 Leakage-Prevention Rules (Binding; enforced in training_schema.py)

**L-1 Query isolation.**
No training or dev example may use a query string that is identical to any
eval query. (Near-duplicate cosine check > 0.92 is recommended at ingest
time but requires the embedding model; the hard identical-string check is
enforced unconditionally.)

**L-2 Positive-document isolation.**
The note IDs in `eval/cases.yaml[*].expect` are the eval positive set.
These IDs may not appear as `positive_id` in any training or dev example.

**L-3 No relevance-label reuse.**
Query-document relevance labels in `eval/cases.yaml` must not be used to
construct training positives, hard negatives, or contrastive pairs.

**L-4 Hard-negative isolation.**
Hard negatives must not be sourced from `eval/vault/`. Use only documents
from the licensed training corpus approved in Milestone 3B.

**L-5 Vault-document isolation.**
Documents in `eval/vault/` are eval-only. They must not be copied,
paraphrased, or chunked into any training corpus.

**L-6 Deterministic ID deduplication.**
Every training example carries a `training_id` = SHA-256(query +
positive_id + creation_method). Duplicate IDs are rejected at validation.

**L-7 Near-duplicate cluster control.**
Before a batch enters the training set, cosine-similarity check at
threshold 0.95 against existing training queries (configurable). Examples
inflating near-duplicate clusters are flagged and excluded.

### 3.3 Split Allocation (Target for Milestone 3B)

| Split | Fraction | Purpose |
|---|---|---|
| `train` | 70% | Embedding fine-tuning, contrastive loss |
| `dev` | 15% | Hyperparameter selection, early stopping |
| `eval` | 15% | eval/cases.yaml + future licensed additions |

Splits are assigned at **source level**, not at random, to prevent
domain-leakage. All examples from one source document go into one split.

---

## 4. Versioned Training-Data Schema

Full Pydantic schema: `eval/training_schema.py`. Fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `training_id` | `str` | YES | SHA-256(query + positive_id + creation_method); unique |
| `version` | `str` | YES | Schema semver e.g. "1.0.0" |
| `split` | `"train"/"dev"` | YES | Never "eval" |
| `query` | `str` | YES | 1-512 chars |
| `positive_id` | `str` | YES | Stable doc/chunk ID |
| `positive_text` | `str` | YES | Full passage text |
| `hard_negative_ids` | `list[str]` | YES | >= 1; must differ from positive |
| `hard_negative_texts` | `list[str]` | YES | Parallel text for each negative |
| `source` | `str` | YES | Corpus name e.g. "beir/nfcorpus" |
| `provenance_url` | `str or None` | no | Canonical source URL |
| `licence` | `str` | YES | SPDX identifier or "proprietary" |
| `category` | `str` | YES | Matches eval category vocabulary |
| `creation_method` | enum | YES | "human"/"synthetic_llm"/"mined_bm25"/"mined_dense" |
| `created_at` | `str` | YES | ISO-8601 UTC |
| `notes` | `str or None` | no | Free-text annotation |

### 4.1 Synthetic vs. Human-Authored Policy

| Method | Allowed | Constraint |
|---|---|---|
| `human` | YES (preferred) | Must have `provenance_url` |
| `synthetic_llm` | YES | Declare model in `notes`; max 50% of any category |
| `mined_bm25` | YES | Declare relevance threshold in `notes` |
| `mined_dense` | YES | Declare cosine threshold in `notes` |

---

## 5. Source Plan for a Larger Licensed Corpus

**Proposal only. No data will be downloaded until Approvals A-D are received.**

### 5.1 Candidate Sources

| Source | Size | Licence | Domain Fit |
|---|---|---|---|
| BEIR / NF-Corpus | 3,633 docs, 2,590 queries | CC BY-SA 4.0 | Dense semantic; scientific |
| BEIR / SciFact | 5,183 docs, 1,109 queries | CC BY 4.0 | Claim retrieval; good distractors |
| StackExchange CS/AI | ~200K Q-A pairs | CC BY-SA 4.0 | Technical how-to; similar to dev notes |
| Wikipedia DPR subset | <=50K passages | CC BY-SA 3.0 | General knowledge; large negatives pool |
| PKM synthetic pairs | 1,000-2,000 (proposed) | MIT (project) | Personal vault structure, wikilinks |
| MS MARCO | 8.8M passages | Non-commercial | Web search; use for negatives only |

### 5.2 Recommended Phase 1 (Milestone 3B, pending approval)

BEIR NF-Corpus + BEIR SciFact + 500 PKM synthetic pairs:
- 500-1,000 training examples
- 100 dev examples from held-out BEIR splits
- 50 new eval queries on a NEW separate vault (never eval/vault/)

### 5.3 Licensing Obligations

All sources must be attributed in `training/SOURCES.md` before any
fine-tuning run. CC BY-SA requires share-alike on derivative datasets.

---

## 6. Approvals Required Before Milestone 3B

**[APPROVAL A] Corpus licensing.**
Approve use of BEIR NF-Corpus (CC BY-SA 4.0) and BEIR SciFact (CC BY 4.0)
as training data sources. Approve attribution in `training/SOURCES.md`.

**[APPROVAL B] Synthetic data cap.**
Approve the 50% synthetic-per-category cap and the requirement that
generating model and prompt template be recorded in each example's `notes`.

**[APPROVAL C] Scale targets.**
Approve an initial training set of 500-1,000 examples with 100 dev examples,
before any full-scale expansion.

**[APPROVAL D] Eval-vault isolation.**
Confirm that `eval/vault/` and `eval/cases.yaml` are permanently frozen as
eval-only assets and that a new, separate vault will be built for any
Milestone 3B eval expansion.

---

## 7. Files Changed in Milestone 3A

| File | Action |
|---|---|
| `eval/AUDIT.md` | New |
| `eval/CHECKSUMS.sha256` | New — machine-verifiable hashes |
| `eval/training_schema.py` | New — Pydantic schema + leakage validators |
| `tests/test_training_schema.py` | New — validation test scaffolding |
| `eval/README.md` | Updated — fingerprint table + link to audit |

No retrieval logic, models, defaults, baselines, eval cases, or vault files
were modified.
