# Training Data — Milestone 3B-1

This directory contains the HotpotQA retrieval training pipeline for Cortex.

## Quick start

```bash
# Download HotpotQA and build validated JSONL (takes 5-10 min on first run)
python training/build_hotpotqa.py

# Skip download if raw files are already present
python training/build_hotpotqa.py --skip-download

# Dry run: validate without writing JSONL
python training/build_hotpotqa.py --dry-run
```

## Files

| File | Committed | Description |
|---|---|---|
| `build_hotpotqa.py` | ✅ | Download + validation + JSONL builder |
| `SOURCES.md` | ✅ | Licence and attribution record |
| `manifest.json` | ✅ | Machine-readable counts, hashes, validation results |
| `README.md` | ✅ | This file |
| `raw/` | ❌ gitignored | Official HotpotQA JSON downloads |
| `train.jsonl` | ❌ gitignored | ≤300 validated TrainingExample objects |
| `dev.jsonl` | ❌ gitignored | ≤50 validated TrainingExample objects |

## Isolation guarantees

| Rule | Scope | Mechanism |
|---|---|---|
| L-1 | Query isolation | No query matches any `eval/cases.yaml` query (exact, lowercased) |
| L-2 | Positive-doc isolation | No `positive_id` in `EVAL_POSITIVE_IDS` |
| L-4 | Hard-negative isolation | No `hard_negative_id` from eval vault |
| L-6 | Deterministic ID | `training_id = SHA-256(query + positive_id + creation_method)` |
| DOC | Document-level split | Train and dev share zero positive document IDs |

## Split policy

Document IDs (normalised Wikipedia article titles) are assigned to splits
by `SHA-256(doc_id)[0]`: values < 52 → dev (~20%), ≥ 52 → train (~80%).
An example is accepted into split X only if all its positive documents hash
to X and none appear in the opposing split's committed document set.

## Licence

HotpotQA source data: CC BY-SA 4.0.
See `SOURCES.md` for full attribution and distribution obligations.
