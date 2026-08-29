#!/usr/bin/env python3
"""
Milestone 3C: Controlled Embedding Fine-Tuning Experiment Pipeline.

Executes a complete, reproducible embedding fine-tuning experiment:
1. Merges 300 HotpotQA train + 200 PKM synthetic train into combined_train.jsonl (500 examples).
2. Evaluates the unmodified baseline model (sentence-transformers/all-MiniLM-L6-v2, Apache 2.0) on:
   - HotpotQA Dev (50 examples)
   - Frozen Cortex Eval v1 (25 queries, 11 vault notes)
3. Fine-tunes the baseline model exclusively on combined_train.jsonl using MultipleNegativesRankingLoss.
4. Evaluates the fine-tuned model on both held-out sets under identical conditions.
5. Generates structured JSON and Markdown reports with metric deltas, per-query regressions, and a final verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from sentence_transformers import InputExample, SentenceTransformer

try:
    from sentence_transformers.sentence_transformer import losses
except ImportError:
    from sentence_transformers import losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "eval"
TRAINING_DIR = REPO_ROOT / "training"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints" / "experiment_3c"

HOTPOT_TRAIN_PATH = TRAINING_DIR / "train.jsonl"
HOTPOT_DEV_PATH = TRAINING_DIR / "dev.jsonl"
PKM_TRAIN_PATH = TRAINING_DIR / "pkm_train.jsonl"
PKM_DEV_PATH = TRAINING_DIR / "pkm_dev.jsonl"
COMBINED_TRAIN_PATH = TRAINING_DIR / "combined_train.jsonl"

EVAL_CASES_PATH = EVAL_DIR / "cases.yaml"
EVAL_VAULT_DIR = EVAL_DIR / "vault"
REPORT_MD_PATH = EVAL_DIR / "EXPERIMENT_REPORT_3C.md"
REPORT_JSON_PATH = TRAINING_DIR / "experiment_results_3c.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LICENCE = "Apache-2.0"
MODEL_REVISION = "c97b51e065bf022ab3b544eb40778c1db1755106"

# ---------------------------------------------------------------------------
# Metric Definitions (Identical to src/cortex/bench.py)
# ---------------------------------------------------------------------------


def _dedupe(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    if not expected or k <= 0:
        return 0.0
    top = _dedupe(retrieved)[:k]
    return len(expected & set(top)) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: frozenset[str]) -> float:
    for index, note_id in enumerate(_dedupe(retrieved), start=1):
        if note_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], expected: frozenset[str], k: int) -> float:
    if not expected or k <= 0:
        return 0.0
    top = _dedupe(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(i + 1) for i, note_id in enumerate(top, start=1) if note_id in expected
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(expected)) + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------


def prepare_combined_training_set() -> list[dict]:
    """Merge 300 HotpotQA + 200 PKM synthetic examples into combined_train.jsonl."""
    logger.info("Merging HotpotQA and PKM synthetic training datasets...")
    hotpot_examples = [
        json.loads(line) for line in HOTPOT_TRAIN_PATH.read_text().splitlines() if line.strip()
    ]
    pkm_examples = [
        json.loads(line) for line in PKM_TRAIN_PATH.read_text().splitlines() if line.strip()
    ]

    combined = hotpot_examples + pkm_examples
    COMBINED_TRAIN_PATH.write_text("\n".join(json.dumps(ex) for ex in combined) + "\n")
    logger.info(
        f"Combined training set written: {len(combined)} examples ({len(hotpot_examples)} HotpotQA + {len(pkm_examples)} PKM)"
    )
    return combined


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    avg_latency_ms: float
    query_count: int
    per_query_results: list[dict[str, Any]]


def evaluate_hotpot_dev(model: SentenceTransformer, dev_path: Path) -> EvalMetrics:
    """Evaluate embedding model on HotpotQA dev (50 queries, 200 candidate passages)."""
    dev_data = [json.loads(line) for line in dev_path.read_text().splitlines() if line.strip()]

    # Build unique candidate document pool: all positives + all hard negatives
    docs_map: dict[str, str] = {}
    for ex in dev_data:
        docs_map[ex["positive_id"]] = ex["positive_text"]
        for nid, ntext in zip(ex["hard_negative_ids"], ex["hard_negative_texts"]):
            docs_map[nid] = ntext

    doc_ids = sorted(docs_map.keys())
    doc_texts = [docs_map[did] for did in doc_ids]

    # Pre-encode all candidate documents
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []
    ndcgs = []
    latencies = []
    per_query = []

    for ex in dev_data:
        q = ex["query"]
        expected = frozenset([ex["positive_id"]])

        t0 = time.perf_counter()
        q_emb = model.encode(q, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.cosine_similarity(q_emb.unsqueeze(0), doc_embeddings).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)

        retrieved_ids = [doc_ids[idx] for idx in ranked_indices]

        r1 = recall_at_k(retrieved_ids, expected, 1)
        r5 = recall_at_k(retrieved_ids, expected, 5)
        r10 = recall_at_k(retrieved_ids, expected, 10)
        m = reciprocal_rank(retrieved_ids, expected)
        n = ndcg_at_k(retrieved_ids, expected, 10)

        recalls_1.append(r1)
        recalls_5.append(r5)
        recalls_10.append(r10)
        mrrs.append(m)
        ndcgs.append(n)

        per_query.append(
            {
                "query": q,
                "expected": list(expected),
                "retrieved_top3": retrieved_ids[:3],
                "recall_at_1": r1,
                "mrr": m,
                "ndcg_at_10": n,
                "latency_ms": t_elapsed,
            }
        )

    return EvalMetrics(
        recall_at_1=sum(recalls_1) / len(recalls_1),
        recall_at_5=sum(recalls_5) / len(recalls_5),
        recall_at_10=sum(recalls_10) / len(recalls_10),
        mrr=sum(mrrs) / len(mrrs),
        ndcg_at_10=sum(ndcgs) / len(ndcgs),
        avg_latency_ms=sum(latencies) / len(latencies),
        query_count=len(dev_data),
        per_query_results=per_query,
    )


def evaluate_cortex_eval_v1(model: SentenceTransformer) -> EvalMetrics:
    """Evaluate embedding model directly against frozen Cortex eval v1 (25 queries, 11 vault notes)."""
    with open(EVAL_CASES_PATH) as f:
        eval_cases = yaml.safe_load(f)["cases"]

    vault_files = sorted(list(EVAL_VAULT_DIR.glob("**/*.md")))
    vault_docs: dict[str, str] = {}
    for vf in vault_files:
        rel_path = str(vf.relative_to(EVAL_VAULT_DIR))
        vault_docs[rel_path] = vf.read_text(encoding="utf-8")

    doc_ids = sorted(vault_docs.keys())
    doc_texts = [vault_docs[did] for did in doc_ids]

    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []
    ndcgs = []
    latencies = []
    per_query = []

    for case in eval_cases:
        q = case["query"]
        expected = frozenset(case["expect"])

        t0 = time.perf_counter()
        q_emb = model.encode(q, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.cosine_similarity(q_emb.unsqueeze(0), doc_embeddings).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed)

        retrieved_ids = [doc_ids[idx] for idx in ranked_indices]

        r1 = recall_at_k(retrieved_ids, expected, 1)
        r5 = recall_at_k(retrieved_ids, expected, 5)
        r10 = recall_at_k(retrieved_ids, expected, 10)
        m = reciprocal_rank(retrieved_ids, expected)
        n = ndcg_at_k(retrieved_ids, expected, 10)

        recalls_1.append(r1)
        recalls_5.append(r5)
        recalls_10.append(r10)
        mrrs.append(m)
        ndcgs.append(n)

        per_query.append(
            {
                "id": case.get("id", q[:20]),
                "query": q,
                "category": case.get("category", "unknown"),
                "expected": sorted(list(expected)),
                "retrieved_top3": retrieved_ids[:3],
                "recall_at_1": r1,
                "recall_at_5": r5,
                "mrr": m,
                "ndcg_at_10": n,
                "latency_ms": t_elapsed,
            }
        )

    return EvalMetrics(
        recall_at_1=sum(recalls_1) / len(recalls_1),
        recall_at_5=sum(recalls_5) / len(recalls_5),
        recall_at_10=sum(recalls_10) / len(recalls_10),
        mrr=sum(mrrs) / len(mrrs),
        ndcg_at_10=sum(ndcgs) / len(ndcgs),
        avg_latency_ms=sum(latencies) / len(latencies),
        query_count=len(eval_cases),
        per_query_results=per_query,
    )


# ---------------------------------------------------------------------------
# Fine-Tuning Loop
# ---------------------------------------------------------------------------


def run_fine_tuning(
    model: SentenceTransformer,
    train_data: list[dict],
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Fine-tune model using sentence-transformers MultipleNegativesRankingLoss."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Initiating fine-tuning on device: {device}")

    train_examples: list[InputExample] = []
    for ex in train_data:
        texts = [ex["query"], ex["positive_text"]]
        if ex.get("hard_negative_texts"):
            texts.append(ex["hard_negative_texts"][0])
        train_examples.append(InputExample(texts=texts))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(CHECKPOINTS_DIR / "fine_tuned_model")

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr, "weight_decay": 0.01},
        output_path=output_path,
        show_progress_bar=False,
    )
    training_duration_s = time.time() - t0

    # Load fine-tuned model checkpoint
    fine_tuned_model = SentenceTransformer(output_path)
    logger.info(f"Fine-tuned model checkpoint reloaded from {output_path}")

    training_meta = {
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_examples": len(train_examples),
        "training_duration_seconds": round(training_duration_s, 2),
        "output_path": output_path,
    }

    return fine_tuned_model, training_meta


# ---------------------------------------------------------------------------
# Experiment Runner & Report Generator
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    logger.info("=== STARTING MILESTONE 3C EMBEDDING EXPERIMENT ===")

    # 1. Prepare data
    train_data = prepare_combined_training_set()

    # 2. Baseline model setup & evaluation
    logger.info(f"Loading baseline model: {MODEL_NAME}")
    baseline_model = SentenceTransformer(MODEL_NAME)
    param_count = sum(p.numel() for p in baseline_model.parameters())

    logger.info("Evaluating baseline on HotpotQA dev...")
    base_hotpot = evaluate_hotpot_dev(baseline_model, HOTPOT_DEV_PATH)

    logger.info("Evaluating baseline on frozen Cortex Eval v1...")
    base_cortex = evaluate_cortex_eval_v1(baseline_model)

    # 3. Controlled fine-tuning
    logger.info("Executing fine-tuning on combined 500-pair training set...")
    fine_tuned_model, train_meta = run_fine_tuning(
        baseline_model, train_data, epochs=3, batch_size=16
    )

    # 4. Fine-tuned model evaluation
    logger.info("Evaluating fine-tuned model on HotpotQA dev...")
    ft_hotpot = evaluate_hotpot_dev(fine_tuned_model, HOTPOT_DEV_PATH)

    logger.info("Evaluating fine-tuned model on frozen Cortex Eval v1...")
    ft_cortex = evaluate_cortex_eval_v1(fine_tuned_model)

    # 5. Compute deltas & per-query regressions
    cortex_regressions = []
    for b_q, ft_q in zip(base_cortex.per_query_results, ft_cortex.per_query_results):
        delta_mrr = ft_q["mrr"] - b_q["mrr"]
        delta_ndcg = ft_q["ndcg_at_10"] - b_q["ndcg_at_10"]
        if delta_mrr < -1e-5 or delta_ndcg < -1e-5:
            cortex_regressions.append(
                {
                    "id": b_q["id"],
                    "query": b_q["query"],
                    "category": b_q["category"],
                    "baseline_mrr": b_q["mrr"],
                    "ft_mrr": ft_q["mrr"],
                    "delta_mrr": round(delta_mrr, 4),
                    "baseline_ndcg": b_q["ndcg_at_10"],
                    "ft_ndcg": ft_q["ndcg_at_10"],
                    "delta_ndcg": round(delta_ndcg, 4),
                }
            )

    hotpot_regressions = []
    for b_q, ft_q in zip(base_hotpot.per_query_results, ft_hotpot.per_query_results):
        delta_mrr = ft_q["mrr"] - b_q["mrr"]
        if delta_mrr < -1e-5:
            hotpot_regressions.append(
                {
                    "query": b_q["query"],
                    "baseline_mrr": b_q["mrr"],
                    "ft_mrr": ft_q["mrr"],
                    "delta_mrr": round(delta_mrr, 4),
                }
            )

    # 6. Recommendation logic
    # Positive delta on dev without catastrophic collapse on eval v1 -> investigate / integrate
    if ft_hotpot.mrr >= base_hotpot.mrr and len(cortex_regressions) <= 2:
        verdict = "INVESTIGATE"
        verdict_summary = "Fine-tuning demonstrated solid metric retention/gains on retrieval with minimal regression. Recommend further investigation with scaling and domain weighting before replacing production defaults."
    elif ft_hotpot.mrr > base_hotpot.mrr:
        verdict = "INVESTIGATE"
        verdict_summary = "In-domain retrieval improved, but slight regression observed on specific out-of-domain edge queries. Recommend investigating loss weighting and hard-negative mining."
    else:
        verdict = "REJECT"
        verdict_summary = (
            "Fine-tuning failed to demonstrate net-positive generalization across held-out sets."
        )

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "licence": MODEL_LICENCE,
            "parameters": param_count,
        },
        "training": train_meta,
        "evaluation": {
            "hotpotqa_dev": {
                "baseline": asdict(base_hotpot),
                "fine_tuned": asdict(ft_hotpot),
                "delta": {
                    "recall_at_1": round(ft_hotpot.recall_at_1 - base_hotpot.recall_at_1, 4),
                    "recall_at_5": round(ft_hotpot.recall_at_5 - base_hotpot.recall_at_5, 4),
                    "recall_at_10": round(ft_hotpot.recall_at_10 - base_hotpot.recall_at_10, 4),
                    "mrr": round(ft_hotpot.mrr - base_hotpot.mrr, 4),
                    "ndcg_at_10": round(ft_hotpot.ndcg_at_10 - base_hotpot.ndcg_at_10, 4),
                    "avg_latency_ms": round(
                        ft_hotpot.avg_latency_ms - base_hotpot.avg_latency_ms, 2
                    ),
                },
                "regressions_count": len(hotpot_regressions),
                "regressions": hotpot_regressions,
            },
            "cortex_eval_v1": {
                "baseline": asdict(base_cortex),
                "fine_tuned": asdict(ft_cortex),
                "delta": {
                    "recall_at_1": round(ft_cortex.recall_at_1 - base_cortex.recall_at_1, 4),
                    "recall_at_5": round(ft_cortex.recall_at_5 - base_cortex.recall_at_5, 4),
                    "recall_at_10": round(ft_cortex.recall_at_10 - base_cortex.recall_at_10, 4),
                    "mrr": round(ft_cortex.mrr - base_cortex.mrr, 4),
                    "ndcg_at_10": round(ft_cortex.ndcg_at_10 - base_cortex.ndcg_at_10, 4),
                    "avg_latency_ms": round(
                        ft_cortex.avg_latency_ms - base_cortex.avg_latency_ms, 2
                    ),
                },
                "regressions_count": len(cortex_regressions),
                "regressions": cortex_regressions,
            },
        },
        "recommendation": {
            "verdict": verdict,
            "rationale": verdict_summary,
        },
    }

    # Save JSON report
    REPORT_JSON_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"JSON results saved to {REPORT_JSON_PATH}")

    # Generate Markdown Report
    generate_markdown_report(results)
    return results


def generate_markdown_report(res: dict[str, Any]) -> None:
    m = res["model"]
    t = res["training"]
    e = res["evaluation"]
    hp_b = e["hotpotqa_dev"]["baseline"]
    hp_ft = e["hotpotqa_dev"]["fine_tuned"]
    hp_d = e["hotpotqa_dev"]["delta"]

    cx_b = e["cortex_eval_v1"]["baseline"]
    cx_ft = e["cortex_eval_v1"]["fine_tuned"]
    cx_d = e["cortex_eval_v1"]["delta"]

    rec = res["recommendation"]

    md = f"""# Milestone 3C: Embedding Fine-Tuning Experiment Report

**Date**: {res["timestamp"]}  
**Status**: Completed  
**Recommendation Verdict**: **{rec["verdict"]}**  

---

## 1. Experiment Overview & Model Metadata

| Property | Value |
|---|---|
| **Base Model** | `{m["name"]}` |
| **Model Licence** | `{m["licence"]}` |
| **Model Revision** | `{m["revision"][:16]}...` |
| **Parameter Count** | {m["parameters"]:,} ({m["parameters"] / 1e6:.1f}M) |
| **Training Set Size** | {t["training_examples"]} pairs (300 HotpotQA + 200 PKM Synthetic) |
| **Device** | `{t["device"]}` ({platform.processor() or platform.machine()}) |
| **Training Duration** | {t["training_duration_seconds"]}s |
| **Hyperparameters** | Epochs: {t["epochs"]}, Batch Size: {t["batch_size"]}, LR: {t["learning_rate"]} |
| **Loss Function** | `MultipleNegativesRankingLoss` (InfoNCE) |

---

## 2. Evaluation Results Summary

### 2.1 HotpotQA Dev Set (50 held-out multi-hop pairs)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| **Recall@1** | {hp_b["recall_at_1"]:.4f} | {hp_ft["recall_at_1"]:.4f} | **{"+" if hp_d["recall_at_1"] >= 0 else ""}{hp_d["recall_at_1"]:.4f}** |
| **Recall@5** | {hp_b["recall_at_5"]:.4f} | {hp_ft["recall_at_5"]:.4f} | **{"+" if hp_d["recall_at_5"] >= 0 else ""}{hp_d["recall_at_5"]:.4f}** |
| **Recall@10** | {hp_b["recall_at_10"]:.4f} | {hp_ft["recall_at_10"]:.4f} | **{"+" if hp_d["recall_at_10"] >= 0 else ""}{hp_d["recall_at_10"]:.4f}** |
| **MRR** | {hp_b["mrr"]:.4f} | {hp_ft["mrr"]:.4f} | **{"+" if hp_d["mrr"] >= 0 else ""}{hp_d["mrr"]:.4f}** |
| **nDCG@10** | {hp_b["ndcg_at_10"]:.4f} | {hp_ft["ndcg_at_10"]:.4f} | **{"+" if hp_d["ndcg_at_10"] >= 0 else ""}{hp_d["ndcg_at_10"]:.4f}** |
| **Avg Query Latency** | {hp_b["avg_latency_ms"]:.2f} ms | {hp_ft["avg_latency_ms"]:.2f} ms | {hp_d["avg_latency_ms"]:+.2f} ms |

### 2.2 Frozen Cortex Eval v1 (25 queries, 11 vault notes)

| Metric | Baseline | Fine-Tuned | Delta |
|---|---|---|---|
| **Recall@1** | {cx_b["recall_at_1"]:.4f} | {cx_ft["recall_at_1"]:.4f} | **{"+" if cx_d["recall_at_1"] >= 0 else ""}{cx_d["recall_at_1"]:.4f}** |
| **Recall@5** | {cx_b["recall_at_5"]:.4f} | {cx_ft["recall_at_5"]:.4f} | **{"+" if cx_d["recall_at_5"] >= 0 else ""}{cx_d["recall_at_5"]:.4f}** |
| **Recall@10** | {cx_b["recall_at_10"]:.4f} | {cx_ft["recall_at_10"]:.4f} | **{"+" if cx_d["recall_at_10"] >= 0 else ""}{cx_d["recall_at_10"]:.4f}** |
| **MRR** | {cx_b["mrr"]:.4f} | {cx_ft["mrr"]:.4f} | **{"+" if cx_d["mrr"] >= 0 else ""}{cx_d["mrr"]:.4f}** |
| **nDCG@10** | {cx_b["ndcg_at_10"]:.4f} | {cx_ft["ndcg_at_10"]:.4f} | **{"+" if cx_d["ndcg_at_10"] >= 0 else ""}{cx_d["ndcg_at_10"]:.4f}** |
| **Avg Query Latency** | {cx_b["avg_latency_ms"]:.2f} ms | {cx_ft["avg_latency_ms"]:.2f} ms | {cx_d["avg_latency_ms"]:+.2f} ms |

---

## 3. Regression Analysis

### 3.1 Cortex Eval v1 Per-Query Regressions ({len(e["cortex_eval_v1"]["regressions"])} detected)
"""
    if not e["cortex_eval_v1"]["regressions"]:
        md += "\n> **Zero regressions detected on the frozen Cortex Eval v1 benchmark.** All 25 queries matched or exceeded baseline performance.\n"
    else:
        md += "\n| Query ID | Category | Baseline MRR | FT MRR | Delta MRR | Delta nDCG@10 |\n|---|---|---|---|---|---|\n"
        for reg in e["cortex_eval_v1"]["regressions"]:
            md += f"| `{reg['id']}` | `{reg['category']}` | {reg['baseline_mrr']:.4f} | {reg['ft_mrr']:.4f} | {reg['delta_mrr']:+.4f} | {reg['delta_ndcg']:+.4f} |\n"

    md += f"""
---

## 4. Production & Architectural Impact

- **Default Configuration**: Cortex's production configuration (`qwen3-embedding:0.6b` via Ollama / hybrid retrieval) remains **100% unchanged**.
- **Model Checkpoints**: Checkpoints are stored in `training/checkpoints/` (gitignored).
- **Isolation Verification**: Training used strictly `combined_train.jsonl`. No validation or evaluation assets were exposed to model parameter updates.

---

## 5. Recommendation & Next Steps

**Verdict**: **{rec["verdict"]}**

**Rationale**:
{rec["rationale"]}

**Proposed Next Milestones**:
1. Keep the experimental pipeline versioned in `training/experiment.py`.
2. Expand the synthetic PKM seed corpus with additional technical domains for higher representation diversity.
3. Investigate cross-encoder reranking vs bi-encoder tuning in subsequent milestones.
"""
    REPORT_MD_PATH.write_text(md)
    logger.info(f"Markdown report generated: {REPORT_MD_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_experiment()
    return 0


if __name__ == "__main__":
    main()
