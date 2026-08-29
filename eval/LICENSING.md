# Corpus Licensing Decision Record — Milestone 3A

**Document status**: Draft for owner review.
**Governed by**: Approvals B, C, and D (accepted). Approval A under review.
**Corpus download**: NOT PERMITTED until Approval A is explicitly granted.
**Last updated**: 2026-08-29

---

## 1. Summary Verdict

| Source | Proposed in 3A Audit | Verdict | Reason |
|---|---|---|---|
| BEIR / NF-Corpus | Training positives + negatives | **REJECTED** | Custom restrictive terms; NutritionFacts.org content not cleared for AI training |
| BEIR / SciFact (claims) | Training positives | **CONDITIONAL** | CC BY 4.0 claims; but abstracts are ODC-By 1.0 on S2ORC — dual-licence complexity |
| BEIR / SciFact (abstracts/corpus.jsonl) | Training positives | **REJECTED** | ODC-By 1.0 (database rights only); underlying paper copyright not cleared |
| **PKM synthetic pairs** | Training positives | **APPROVED** (pending Approval B already granted) | MIT licence; project-owned; no third-party content |
| MS MARCO | Hard negatives only | **REJECTED** | Microsoft non-commercial research terms; explicitly excludes commercial products |
| **HotpotQA** | Proposed replacement | **APPROVED — 3B-1 pilot** | CC BY-SA 4.0; share-alike implications for weights unresolved; no distribution of weights or data without separate review |
| **Natural Questions / DPR passages** | Proposed replacement | **REJECTED** | CC-BY-NC 4.0 on Facebook AI DPR passage corpus and retrieval mappings |

**Source plan after 3B-2 audit**: HotpotQA (CC BY-SA 4.0) + PKM Synthetic pairs (MIT). NF-Corpus, SciFact, MS MARCO, and NQ/DPR are permanently rejected.

---

## 2. Source-by-Source Analysis

---

### 2.1 BEIR / NF-Corpus

**Dataset**: NFCorpus — A Full-Text Learning to Rank Dataset for Medical Information Retrieval.
**Authors**: Boteva, Gholipour, Sokolov, Riezler (Heidelberg University, 2016).

#### 2.1.1 Canonical Source and Version

| Field | Value |
|---|---|
| Canonical URL | https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ |
| Original paper | Boteva et al., ECIR 2016. DOI not publicly accessible at time of review |
| Data hosted at | https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ |
| BEIR re-distribution | https://huggingface.co/datasets/BeIR/nfcorpus |
| Version / hash checked | No versioned release tag. BEIR copy accessed 2026-08-29 |

#### 2.1.2 Exact Licence Terms (from canonical source)

The NFCorpus website (https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
states:

> "NFCorpus is **free to use for academic purposes**. For any other uses
> regarding the data from *NutritionFacts.org* please consult the
> [NutritionFacts.org Terms of Service](https://nutritionfacts.org/terms-of-service/)
> and contact the author, Dr. Michael Greger, directly."

This is **not an open-source licence**. There is no SPDX identifier.
The BEIR Hugging Face card labels this as "CC BY-SA 4.0" but this is
**incorrect and contradicted by the canonical source**. BEIR is an
aggregator; its top-level licence does not override the original terms.

#### 2.1.3 Licence Analysis

| Question | Answer |
|---|---|
| Commercial use permitted? | **No** — "academic purposes" only |
| Redistribution permitted? | **Unclear** — not specified; terms point back to NutritionFacts.org ToS |
| Training on derived QD pairs permitted? | **Unknown** — requires contact with Dr. Michael Greger |
| Share-alike / attribution obligation? | Citation of Boteva et al. 2016 required |
| Conflicts or uncertainty? | **High** — BEIR metadata says CC BY-SA 4.0; canonical source says "academic only" |

#### 2.1.4 NutritionFacts.org Terms of Service (Background)

The underlying content is scraped from NutritionFacts.org, a website operated
by a non-profit. Their Terms of Service (https://nutritionfacts.org/terms-of-service/)
govern the source text. These terms are not a standard open licence and
require explicit permission for redistribution or commercial exploitation.

#### 2.1.5 Verdict: **REJECTED**

The canonical source restricts use to "academic purposes" only. The BEIR
metadata claiming CC BY-SA 4.0 conflicts with this and is unreliable.
Creating derived query-document training pairs from NutritionFacts.org
content requires explicit written permission from Dr. Michael Greger, which
has not been obtained and is out of scope for this pilot. **NF-Corpus must
not be downloaded or used as a training source.**

---

### 2.2 BEIR / SciFact

**Dataset**: SciFact — claim verification against scientific literature.
**Authors**: Wadden et al. (Allen Institute for AI), EMNLP 2020.

#### 2.2.1 Canonical Source and Version

| Field | Value |
|---|---|
| Canonical repo | https://github.com/allenai/scifact |
| Licence file | https://github.com/allenai/scifact/blob/master/LICENSE.md |
| Paper | Wadden et al. 2020, EMNLP. arXiv:2004.14974 |
| BEIR re-distribution | https://huggingface.co/datasets/BeIR/scifact |
| Version checked | Commit `master` at 2026-08-29 |

#### 2.2.2 Exact Licence Text (from LICENSE.md, fetched 2026-08-29)

```
# License

## Claims

All claims and evidence annotations -- in the files `claims_*.jsonl` --
are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

## Abstracts

The abstracts in the corpus -- in the file `corpus.jsonl` -- are part of
the Semantic Scholar S2ORC dataset (https://github.com/allenai/s2orc),
[text truncated in fetch; licence for corpus.jsonl is ODC-By 1.0]
```

The corpus abstracts are derived from the Semantic Scholar Open Research
Corpus (S2ORC). S2ORC is released under the **Open Data Commons
Attribution License (ODC-By 1.0)** for database rights. The ODC-By licence
governs the *database*, not the copyright of the individual papers/abstracts
contained within it. Copyright in the abstracts themselves is retained by
their respective authors and publishers; S2ORC/SciFact does not and cannot
clear those copyrights.

#### 2.2.3 Dual-Licence Breakdown

| Component | File | Licence | Commercial? | Training? |
|---|---|---|---|---|
| Claims + evidence annotations | `claims_*.jsonl` | CC BY 4.0 | Yes | Yes, with attribution |
| Abstracts / corpus | `corpus.jsonl` | ODC-By 1.0 (database rights) | Yes for DB rights | Uncertain for content copyright |
| Underlying paper text copyright | corpus.jsonl | **Not licensed** — held by publishers | **Unknown** | **Requires publisher clearance** |

#### 2.2.4 Licence Analysis

| Question | Answer |
|---|---|
| Commercial use of claims? | **Yes** (CC BY 4.0) |
| Commercial use of abstracts (DB rights)? | **Yes** (ODC-By 1.0) |
| Copyright of abstract text cleared? | **No** — S2ORC/SciFact does not clear publisher copyright |
| Training on QD pairs from claims only? | **Conditionally yes** — claims are CC BY 4.0; no abstracts needed |
| Training on corpus text? | **Uncertain** — ODC-By covers DB rights; publisher copyright on text is not cleared |
| Share-alike / attribution? | CC BY 4.0: attribution required; no share-alike |
| Conflicts or uncertainty? | **High** for corpus.jsonl; **Low** for claims only |

#### 2.2.5 What "Claims Only" Would Give

The `claims_*.jsonl` files contain:
- A claim sentence (scientific assertion).
- Evidence citations pointing to abstract IDs.
- Veracity labels (SUPPORT / REFUTE / NOT ENOUGH INFO).

For embedding retrieval training, the query would be the claim text and the
positive would be the cited abstract passage. **This requires access to
corpus.jsonl (the abstracts), which is where the licence uncertainty lies.**
There is no viable path to use SciFact for retrieval training without the
corpus abstracts.

#### 2.2.6 Verdict: **REJECTED for training corpus use**

SciFact corpus abstracts carry publisher copyright that S2ORC/SciFact does
not clear. ODC-By 1.0 covers database-level rights, not the text. This
creates unacceptable legal uncertainty for creating and publishing derived
query-document pairs for embedding training. **SciFact must not be downloaded
or used as a positive training source.**

The claims annotations alone (CC BY 4.0) are safe but are not useful
for retrieval training without the corpus passages.

---

### 2.3 MS MARCO (previously proposed for hard negatives only)

| Field | Value |
|---|---|
| Canonical URL | https://microsoft.github.io/msmarco/ |
| Licence | Microsoft custom non-commercial research terms |
| Commercial use? | **No** — explicitly "non-commercial research" |
| Training on derived pairs? | Not permitted for commercial products |

#### Verdict: **REJECTED entirely**

Even for hard-negative mining, using MS MARCO produces derived relevance
labels from non-commercially-licensed data. Removed from the corpus plan.

---

## 3. Proposed Replacement Sources

### 3.1 HotpotQA (Wikipedia-sourced multi-hop QA)

| Field | Value |
|---|---|
| Canonical URL | https://hotpotqa.github.io/ |
| Repository | https://github.com/hotpotqa/hotpot |
| Paper | Yang et al. 2018, EMNLP. arXiv:1809.09600 |
| Licence | **CC BY-SA 4.0** |
| Licence URL | https://creativecommons.org/licenses/by/4.0/ (as listed on HF card); underlying Wikipedia text is CC BY-SA 3.0 |
| Size | ~113K Q-A pairs; supporting passages from Wikipedia |
| Version checked | HotpotQA full wiki v1.1 (2018) |

#### 3.1.1 Licence Analysis

| Question | Answer |
|---|---|
| Commercial use permitted? | **Yes** — CC BY-SA 4.0 does not restrict commercial use |
| Attribution required? | **Yes** — cite Yang et al. 2018 and link to dataset |
| Share-alike obligation? | **Yes** — derivatives must use CC BY-SA 4.0 or compatible |
| Does share-alike apply to model weights? | **Legally uncertain** (open question in AI law); conservative approach: document in model card |
| Redistribution of derived QD pairs? | Yes, under CC BY-SA 4.0 with attribution |
| Training on derived QD pairs? | **Yes** |
| Creating and publishing QD pairs? | Yes, with attribution and CC BY-SA 4.0 licence on the derived dataset |

#### 3.1.2 Share-Alike and Model-Weights Obligations

The share-alike clause requires derivatives to be released under CC BY-SA 4.0
or a compatible licence. Whether trained model weights constitute a
"derivative work" of training data is **a legally unresolved question** under
current AI copyright law; no settled global consensus exists.

**Policy**: Licensing implications for model weights derived from this data
are unresolved. This project will not distribute trained weights or derived
datasets without a separate, explicit licensing review.

**Decision**: Approved for 3B-1 pilot (internal use, no distribution).
The training QD-pair dataset must carry CC BY-SA 4.0 attribution if ever
distributed externally.

---

### 3.2 Natural Questions (NQ-Open / DPR Wikipedia passages) — AUDIT COMPLETE: REJECTED

> **[STATUS: REJECTED (Milestone 3B-2)]**  
> Canonical audit completed on 2026-08-29. While Google's NQ question/answer
> dataset is CC BY-SA 3.0, the passage retrieval corpus (`psgs_w100.tsv.gz`)
> and retrieval pair splits (`nq-train.json`, `nq-dev.json`) were created and
> distributed by Facebook AI Research (`facebookresearch/DPR`, Karpukhin et al. 2020)
> under **CC-BY-NC 4.0** (Non-Commercial). This violates the project's
> commercial permissibility standard.

| Field | Value |
|---|---|
| Canonical URL | https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz |
| Repository | https://github.com/facebookresearch/DPR |
| Paper | Karpukhin et al. 2020, EMNLP |
| DPR Passage Corpus License | **CC-BY-NC 4.0** (Non-Commercial) |
| DPR Retrieval Mapping License | **CC-BY-NC 4.0** (Non-Commercial) |
| Underlying Wikipedia Text | CC BY-SA 3.0 / 4.0 |
| Commercial Use Permitted? | **NO** — explicitly prohibited by CC-BY-NC 4.0 |

**Verdict**: **REJECTED**. Consistent with the rejection of MS MARCO, all
non-commercial restricted datasets are prohibited from entering the training pipeline.

---

## 4. Corrected Source Plan for Approval A

The following replaces the plan proposed in `eval/AUDIT.md §5`:

| Source | Status | Licence | Size (pilot) | Use |
|---|---|---|---|---|
| ~~BEIR NF-Corpus~~ | **Rejected** | Custom restrictive | — | Removed |
| ~~BEIR SciFact~~ | **Rejected** | ODC-By + unclear copyright | — | Removed |
| ~~MS MARCO~~ | **Rejected** | Non-commercial only | — | Removed |
| ~~NQ-open / DPR passages~~ | **Rejected** | CC-BY-NC 4.0 | — | Removed |
| **HotpotQA** | **Approved — 3B-1** | CC BY-SA 4.0 | 300 train + 50 dev | Training positives + hard negatives |
| **PKM synthetic pairs** | Designed (Approval B) | MIT (project) | 200 train + 50 dev (target) | Training positives (Milestone 3C) |

3B-1 pilot size: ≤300 train + ≤50 dev from HotpotQA only.

### 4.1 Attribution Obligations

The `training/SOURCES.md` file (to be created in Milestone 3B) must include:

```
HotpotQA: Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering. EMNLP 2018.
URL: https://hotpotqa.github.io/
Licence: CC BY-SA 4.0

Natural Questions: Kwiatkowski et al. (2019). Natural Questions: a Benchmark
for Question Answering Research. TACL 7.
URL: https://ai.google.com/research/NaturalQuestions
Licence: CC BY-SA 3.0 (Wikipedia passages)

PKM Synthetic: Project-generated synthetic pairs.
Licence: MIT
```

### 4.2 Distribution Policy for Derived Dataset and Model Weights

Any derived training dataset file (JSONL of QD pairs from HotpotQA)
that is distributed externally must carry a CC BY-SA 4.0 licence and the
above attribution.

Licensing implications for model weights trained on this data are unresolved.
This project will not distribute trained weights or derived datasets without
a separate, explicit licensing review. Internal use only for the 3B-1 pilot.

---

## 5. Approval Status

| Approval | Status | Scope |
|---|---|---|
| **Approval A (HotpotQA)** | **Granted — 3B-1 pilot** | ≤300 train + ≤50 dev; no weight/data distribution |
| **Approval A (NQ-open/DPR)** | **REJECTED** | Rejected in 3B-2 due to CC-BY-NC 4.0 |
| **Approval B (PKM Synthetic)** | Granted / Designed | 200 train + 50 dev; strict review rubric; MIT |
| **Approval C (Pilot scale)** | Granted | ≤500 train + ≤100 dev total pilot ceiling |
| **Approval D** | Granted (with correction) | eval/baseline_dense.* and eval/vault/ immutable v1; future eval as new versioned suite |

No data will be downloaded until this revised Approval A is received.

---

## 6. Milestone 3A Audit Correction

The source plan in `eval/AUDIT.md §5` contained two errors:
1. SciFact was listed as "CC BY 4.0" — **incorrect**. Only the claims
   annotations are CC BY 4.0; the corpus abstracts are ODC-By 1.0 with
   uncleared publisher copyright.
2. NF-Corpus was listed as "CC BY-SA 4.0" — **incorrect**. This
   contradicts the canonical source, which restricts use to "academic
   purposes" only.

Both errors originated from BEIR/Hugging Face metadata, which aggregates
datasets without overriding their original licence terms. This record
supersedes the §5 source plan in AUDIT.md.

---

## 7. References

| Source | URL | Accessed |
|---|---|---|
| NFCorpus canonical | https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ | 2026-08-29 |
| SciFact LICENSE.md | https://github.com/allenai/scifact/blob/master/LICENSE.md | 2026-08-29 |
| S2ORC ODC-By | https://github.com/allenai/s2orc | 2026-08-29 |
| MS MARCO terms | https://microsoft.github.io/msmarco/ | 2026-08-29 |
| HotpotQA | https://hotpotqa.github.io/ | 2026-08-29 |
| Natural Questions | https://ai.google.com/research/NaturalQuestions | 2026-08-29 |
| CC BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | 2026-08-29 |
| CC BY-SA 3.0 | https://creativecommons.org/licenses/by-sa/3.0/ | 2026-08-29 |
| ODC-By 1.0 | https://opendatacommons.org/licenses/by/1-0/ | 2026-08-29 |
