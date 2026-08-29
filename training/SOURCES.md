# Training Data Sources — Cortex Milestone 3B-1

**Status**: HotpotQA approved for 3B-1 pilot.
**Governed by**: `eval/LICENSING.md` §4 and §5.
**Distribution policy**: Trained weights and derived datasets will not be
distributed without a separate, explicit licensing review
(`eval/LICENSING.md §4.2`).

---

## HotpotQA (Pilot — ≤300 train, ≤50 dev)

| Field | Value |
|---|---|
| Dataset name | HotpotQA |
| Authors | Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning |
| Paper | Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP 2018*. arXiv:1809.09600 |
| Canonical URL | https://hotpotqa.github.io/ |
| Repository | https://github.com/hotpotqa/hotpot |
| Licence | **CC BY-SA 4.0** |
| Licence URL | https://creativecommons.org/licenses/by-sa/4.0/ |
| Underlying source | Wikipedia text (CC BY-SA 3.0); supporting-fact labels by crowd workers |
| Files used | `hotpot_train_v1.1.json`, `hotpot_dev_distractor_v1.json` |
| Canonical download host | http://curtis.ml.cmu.edu/datasets/hotpot/ |
| File hashes | See `training/manifest.json` (populated by `build_hotpotqa.py`) |

### Attribution (required by CC BY-SA 4.0)

```
Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R.,
& Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering. In Proceedings of EMNLP 2018.
https://hotpotqa.github.io/
Licence: CC BY-SA 4.0
```

### Obligations

- Attribution must be included in any downstream documentation, model
  card, or published report that uses these training examples.
- Any externally distributed derived dataset (JSONL of QD pairs) must
  carry CC BY-SA 4.0 and the above attribution.
- Licensing implications for model weights are unresolved. This project
  will not distribute trained weights or derived datasets without a
  separate licensing review.

---

## Designed Sources (Milestone 3B-2 Design / Milestone 3C Target)

| Source | Status | Scope |
|---|---|---|
| **PKM synthetic pairs** | **Designed** (Approval B granted) | 200 train + 50 dev; project-authored markdown seeds; MIT license |

---

## Rejected Sources (permanent)

| Source | Reason |
|---|---|
| BEIR / NF-Corpus | "Academic purposes only" — custom restrictive terms; NutritionFacts.org permission required |
| BEIR / SciFact (corpus.jsonl) | ODC-By 1.0 covers DB rights only; publisher copyright on abstracts not cleared |
| MS MARCO | Microsoft non-commercial research terms; explicitly excludes commercial products |
| Natural Questions / DPR passages (`psgs_w100.tsv.gz`, `nq-*.json`) | **CC-BY-NC 4.0** (Facebook AI Research) — Non-Commercial restriction incompatible with project terms |
