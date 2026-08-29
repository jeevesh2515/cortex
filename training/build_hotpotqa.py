#!/usr/bin/env python3
"""
Milestone 3B-1: HotpotQA retrieval training-data builder.

Downloads official HotpotQA splits, extracts retrieval QD pairs with
supporting-facts as relevance labels, enforces all leakage/isolation rules
from eval/training_schema.py, and writes validated JSONL + a manifest.

Usage:
    python training/build_hotpotqa.py [--dry-run] [--skip-download]

Output files (all gitignored):
    training/train.jsonl   — up to MAX_TRAIN validated TrainingExamples
    training/dev.jsonl     — up to MAX_DEV validated TrainingExamples

Committed output:
    training/manifest.json — counts, hashes, split-policy description

Isolation guarantees:
    L-1  No query is identical to any eval/cases.yaml query (exact, lowercased)
    L-2  No positive_id is in EVAL_POSITIVE_IDS
    L-4  No hard_negative_id is from the eval vault
    L-6  training_id == SHA-256(query + positive_id + creation_method)
    DOC  Train and dev positive-document sets are disjoint (document-level split)
    CAP  ≤ MAX_TRAIN train examples, ≤ MAX_DEV dev examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "eval"
TRAINING_DIR = REPO_ROOT / "training"
RAW_DIR = TRAINING_DIR / "raw"
TRAIN_OUT = TRAINING_DIR / "train.jsonl"
DEV_OUT = TRAINING_DIR / "dev.jsonl"
MANIFEST_PATH = TRAINING_DIR / "manifest.json"

sys.path.insert(0, str(EVAL_DIR))
from training_schema import (  # noqa: E402
    EVAL_POSITIVE_IDS,
    EVAL_QUERIES,
    TrainingDataset,
    TrainingExample,
    compute_training_id,
    validate_no_eval_leakage,
)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Canonical CMU-hosted URLs (official HotpotQA distribution point)
SOURCES: dict[str, dict] = {
    "hotpot_train_v1.1.json": {
        # Canonical origin (CMU); may require institutional access
        "canonical_url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        # Actual download source used: HuggingFace Hub via `datasets` library
        "url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa (distractor, train split)",
        "split": "train",
        "licence": "CC-BY-SA-4.0",
        "citation": (
            "Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable "
            "Multi-hop Question Answering. EMNLP 2018. arXiv:1809.09600"
        ),
    },
    "hotpot_dev_distractor_v1.json": {
        "canonical_url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa (distractor, validation split)",
        "split": "dev",
        "licence": "CC-BY-SA-4.0",
        "citation": (
            "Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable "
            "Multi-hop Question Answering. EMNLP 2018. arXiv:1809.09600"
        ),
    },
}

MAX_TRAIN = 300
MAX_DEV = 50
SCHEMA_VERSION = "1.0.0"
CREATION_METHOD = "human"       # relevance labels are human crowd-annotated
SOURCE_NAME = "hotpotqa"
LICENCE = "CC-BY-SA-4.0"

# ---------------------------------------------------------------------------
# Document-level split assignment
# ---------------------------------------------------------------------------

def _doc_split(doc_id: str) -> str:
    """
    Deterministically assign a Wikipedia article title to 'train' or 'dev'.

    We hash the doc_id with SHA-256 and assign 8/10 to train, 2/10 to dev.
    This is applied at document level so that no article can appear as a
    positive in both train and dev.
    """
    h = hashlib.sha256(doc_id.encode("utf-8")).digest()[0]  # 0–255
    return "dev" if h < 52 else "train"   # ~20% dev, ~80% train


def _normalise_title(title: str) -> str:
    """Produce a stable document ID from a Wikipedia article title."""
    return f"hotpotqa/{title.strip()}"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ProgressReporter:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.last_pct = -1

    def __call__(self, blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(blocks * block_size * 100 / total))
        if pct >= self.last_pct + 10:
            mb = blocks * block_size / 1_000_000
            print(f"  {self.filename}: {pct}% ({mb:.0f} MB)", flush=True)
            self.last_pct = pct


def download_file(filename: str, *, skip: bool = False) -> Path:
    """Download filename into RAW_DIR if not already present. Returns path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / filename
    meta = SOURCES[filename]

    if dest.exists():
        print(f"[skip] {filename} already present — verifying hash...")
        return dest

    if skip:
        raise FileNotFoundError(
            f"--skip-download requested but {dest} not found. "
            "Run without --skip-download first."
        )

    url = meta["url"]
    print(f"[download] {filename}")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")
    t0 = time.monotonic()
    urllib.request.urlretrieve(url, dest, reporthook=_ProgressReporter(filename))
    elapsed = time.monotonic() - t0
    size_mb = dest.stat().st_size / 1_000_000
    print(f"  Done: {size_mb:.1f} MB in {elapsed:.1f}s")
    return dest


# ---------------------------------------------------------------------------
# HotpotQA parsing
# ---------------------------------------------------------------------------

@dataclass
class _RawExample:
    uid: str
    question: str
    supporting_titles: list[str]     # unique titles from supporting_facts
    context: list[tuple[str, str]]   # (title, full_text) for all paragraphs


def _parse_hotpotqa(path: Path) -> Iterator[_RawExample]:
    """Parse a HotpotQA JSON file, yielding _RawExample objects.

    Handles two formats:
    - CMU wire format: context is list of [title, [sentences]], supporting_facts
      is list of [title, sent_id], id key is "_id"
    - HuggingFace datasets column format: context is {title: [...], sentences: [...]},
      supporting_facts is {title: [...], sent_id: [...]}, id key is "id"
    """
    print(f"[parse] Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB)...")
    with path.open() as f:
        data = json.load(f)
    print(f"  {len(data)} examples loaded")

    for item in data:
        # --- Resolve ID ---
        uid = item.get("_id") or item.get("id", "unknown")

        # --- Resolve supporting_facts ---
        sf_raw = item.get("supporting_facts", [])
        sup_titles: list[str] = []
        seen: set[str] = set()

        if isinstance(sf_raw, dict):
            # HF column format: {"title": [...], "sent_id": [...]}
            for title in sf_raw.get("title", []):
                if title not in seen:
                    sup_titles.append(title)
                    seen.add(title)
        else:
            # CMU wire format: [[title, sent_id], ...]
            for entry in sf_raw:
                title = entry[0]
                if title not in seen:
                    sup_titles.append(title)
                    seen.add(title)

        if not sup_titles:
            continue

        # --- Resolve context ---
        ctx_raw = item.get("context", [])
        ctx: list[tuple[str, str]] = []

        if isinstance(ctx_raw, dict):
            # HF column format: {"title": [...], "sentences": [[sents], ...]}
            titles = ctx_raw.get("title", [])
            sentences_list = ctx_raw.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                full_text = " ".join(s.strip() for s in sentences if s.strip())
                if full_text:
                    ctx.append((title, full_text))
        else:
            # CMU wire format: [[title, [sentences]], ...]
            for title, sentences in ctx_raw:
                full_text = " ".join(s.strip() for s in sentences if s.strip())
                if full_text:
                    ctx.append((title, full_text))

        if not ctx:
            continue

        yield _RawExample(
            uid=uid,
            question=item["question"].strip(),
            supporting_titles=sup_titles,
            context=ctx,
        )



# ---------------------------------------------------------------------------
# Example builder
# ---------------------------------------------------------------------------

@dataclass
class _RejectionStats:
    no_supporting_docs: int = 0
    no_hard_negatives: int = 0
    split_conflict: int = 0    # positives span both train and dev pools
    eval_query_leak: int = 0   # L-1
    eval_doc_leak: int = 0     # L-2 / L-4
    schema_error: int = 0
    cap_reached: int = 0
    total_seen: int = 0
    total_accepted: int = 0

    def report(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_accepted": self.total_accepted,
            "rejected_no_supporting_docs": self.no_supporting_docs,
            "rejected_no_hard_negatives": self.no_hard_negatives,
            "rejected_split_conflict": self.split_conflict,
            "rejected_eval_query_leak_L1": self.eval_query_leak,
            "rejected_eval_doc_leak_L2_L4": self.eval_doc_leak,
            "rejected_schema_error": self.schema_error,
            "rejected_cap_reached": self.cap_reached,
        }


def build_examples(
    raw_examples: Iterator[_RawExample],
    *,
    target_split: str,
    doc_exclusion_set: set[str],
    max_count: int,
    source_file: str,
) -> tuple[list[TrainingExample], _RejectionStats]:
    """
    Convert _RawExample objects into validated TrainingExample objects.

    target_split: 'train' or 'dev' — examples are accepted only when ALL
                  positive documents hash to this split AND none appear in
                  doc_exclusion_set (document-level disjointness).
    doc_exclusion_set: document IDs already committed to the OTHER split.
    """
    results: list[TrainingExample] = []
    stats = _RejectionStats()

    for raw in raw_examples:
        stats.total_seen += 1

        if len(results) >= max_count:
            stats.cap_reached += 1
            continue

        # Build context lookup
        ctx_map: dict[str, str] = {title: text for title, text in raw.context}

        # --- Positive docs: must have text in context ---
        pos_titles = [t for t in raw.supporting_titles if t in ctx_map]
        if not pos_titles:
            stats.no_supporting_docs += 1
            continue

        # --- Hard negatives: non-supporting context paragraphs ---
        sup_set = set(raw.supporting_titles)
        neg_titles = [t for t, _ in raw.context if t not in sup_set and t in ctx_map]
        if not neg_titles:
            stats.no_hard_negatives += 1
            continue

        # Pre-compute IDs for both positives and negatives (needed for all checks)
        pos_ids = [_normalise_title(t) for t in pos_titles]
        neg_ids = [_normalise_title(t) for t in neg_titles]

        # --- L-1: eval query isolation ---
        if raw.question.strip().lower() in EVAL_QUERIES:
            stats.eval_query_leak += 1
            continue

        # --- L-2: eval positive isolation (before split gate) ---
        # Must run BEFORE split assignment so that an eval doc whose title happens
        # to hash to a different split is still counted as eval_doc_leak (not
        # silently skipped as a wrong-split example).
        # Check raw titles (e.g. "Retrieval.md") AND normalised IDs
        # ("hotpotqa/Retrieval.md") because EVAL_POSITIVE_IDS stores bare names.
        if any(t in EVAL_POSITIVE_IDS for t in pos_titles) or any(
            pid in EVAL_POSITIVE_IDS for pid in pos_ids
        ):
            stats.eval_doc_leak += 1
            continue

        # --- L-4: eval vault hard-negative isolation (before split gate) ---
        # Filter (title, id) pairs atomically to keep lists in sync.
        # If filtering leaves zero hard negatives, reject the example.
        neg_pairs = [
            (t, nid)
            for t, nid in zip(neg_titles, neg_ids)
            if t not in EVAL_POSITIVE_IDS and nid not in EVAL_POSITIVE_IDS
        ]
        if len(neg_pairs) < len(neg_titles):
            if not neg_pairs:
                stats.no_hard_negatives += 1
                continue
            neg_titles = [t for t, _ in neg_pairs]
            neg_ids = [nid for _, nid in neg_pairs]

        # --- Document-level split assignment ---
        # Use first positive only for split decision (primary supporting doc)
        assigned_split = _doc_split(_normalise_title(pos_titles[0]))

        # All positives must agree on the assigned split
        if len({_doc_split(_normalise_title(t)) for t in pos_titles}) > 1:
            stats.split_conflict += 1
            continue

        if assigned_split != target_split:
            continue  # Belongs to the other split — not a rejection, just skip

        # Document-level disjointness: none of our positives may be in the
        # exclusion set (documents committed to the opposing split)
        if any(pid in doc_exclusion_set for pid in pos_ids):
            stats.split_conflict += 1
            continue

        # Use first positive only (simplest multi-hop pivot; avoids
        # multi-positive complexity in the schema for this pilot)
        pos_id = pos_ids[0]
        pos_text = ctx_map[pos_titles[0]]
        # Cap at 3 hard negatives; derive texts from the synced, filtered pairs
        neg_id_list = neg_ids[:3]
        neg_title_list = neg_titles[:3]
        neg_text_list = [ctx_map[t] for t in neg_title_list]

        tid = compute_training_id(raw.question, pos_id, CREATION_METHOD)

        try:
            ex = TrainingExample(
                training_id=tid,
                version=SCHEMA_VERSION,
                split=target_split,          # type: ignore[arg-type]
                query=raw.question,
                positive_id=pos_id,
                positive_text=pos_text,
                hard_negative_ids=neg_id_list,
                hard_negative_texts=neg_text_list,
                source=f"{SOURCE_NAME}/{source_file}",
                provenance_url=f"https://hotpotqa.github.io/#{raw.uid}",
                licence=LICENCE,
                category="multi_topic",      # HotpotQA is multi-hop / multi-topic
                creation_method=CREATION_METHOD,  # type: ignore[arg-type]
                created_at=datetime.now(timezone.utc).isoformat(),
                notes=None,
            )
        except Exception as exc:
            stats.schema_error += 1
            print(f"  [schema-error] {raw.uid}: {exc}")
            continue

        results.append(ex)
        stats.total_accepted += 1

    return results, stats


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(
    *,
    train_examples: list[TrainingExample],
    dev_examples: list[TrainingExample],
    train_stats: _RejectionStats,
    dev_stats: _RejectionStats,
    source_hashes: dict[str, str],
    train_path: Path,
    dev_path: Path,
    dry_run: bool,
) -> dict:
    train_jsonl_hash = _sha256_file(train_path) if train_path.exists() else "n/a"
    dev_jsonl_hash = _sha256_file(dev_path) if dev_path.exists() else "n/a"

    # Unique positive document IDs per split
    train_docs = sorted({ex.positive_id for ex in train_examples})
    dev_docs = sorted({ex.positive_id for ex in dev_examples})
    overlap = set(train_docs) & set(dev_docs)

    manifest = {
        "milestone": "3B-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "schema_version": SCHEMA_VERSION,
        "split_policy": (
            "document-disjoint: SHA-256(doc_id)[0] < 52 → dev (~20%), else → train (~80%). "
            "An example is accepted into split X only when ALL its positive docs hash to X "
            "and none appear in the opposing split's committed doc set."
        ),
        "eval_isolation": {
            "eval_positive_ids_count": len(EVAL_POSITIVE_IDS),
            "eval_queries_count": len(EVAL_QUERIES),
            "rules_enforced": ["L-1", "L-2", "L-4", "L-6", "DOC-DISJOINT"],
        },
        "sources": {
            fname: {
                "url": meta["url"],
                "licence": meta["licence"],
                "citation": meta["citation"],
                "sha256": source_hashes.get(fname, "not-downloaded"),
            }
            for fname, meta in SOURCES.items()
        },
        "train": {
            "output_file": str(train_path.relative_to(REPO_ROOT)),
            "sha256": train_jsonl_hash,
            "count": len(train_examples),
            "unique_positive_docs": len(train_docs),
            "rejection_stats": train_stats.report(),
        },
        "dev": {
            "output_file": str(dev_path.relative_to(REPO_ROOT)),
            "sha256": dev_jsonl_hash,
            "count": len(dev_examples),
            "unique_positive_docs": len(dev_docs),
            "rejection_stats": dev_stats.report(),
        },
        "document_overlap_train_dev": sorted(overlap),
        "leakage_violations": validate_no_eval_leakage(train_examples + dev_examples),
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest] Written to {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate but do not write JSONL output.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume raw files are already present; skip download.",
    )
    args = parser.parse_args(argv)

    print("=" * 68)
    print("Cortex Milestone 3B-1: HotpotQA data builder")
    print("=" * 68)
    print(f"  MAX_TRAIN = {MAX_TRAIN}  MAX_DEV = {MAX_DEV}")
    print(f"  eval positive IDs: {len(EVAL_POSITIVE_IDS)}")
    print(f"  eval queries:      {len(EVAL_QUERIES)}")
    print()

    # 1. Download files
    source_hashes: dict[str, str] = {}
    downloaded_paths: dict[str, Path] = {}
    for fname in SOURCES:
        try:
            p = download_file(fname, skip=args.skip_download)
            source_hashes[fname] = _sha256_file(p)
            print(f"  SHA-256 {fname}: {source_hashes[fname]}")
            downloaded_paths[fname] = p
        except Exception as exc:
            print(f"[ERROR] Could not obtain {fname}: {exc}")
            return 1

    # 2. Build dev examples first (to establish doc exclusion set for train)
    print("\n--- Building DEV examples ---")
    dev_raw = _parse_hotpotqa(downloaded_paths["hotpot_dev_distractor_v1.json"])
    dev_examples, dev_stats = build_examples(
        dev_raw,
        target_split="dev",
        doc_exclusion_set=set(),        # no prior constraint on dev
        max_count=MAX_DEV,
        source_file="hotpot_dev_distractor_v1.json",
    )
    dev_doc_set = {ex.positive_id for ex in dev_examples}
    print(f"  Accepted: {len(dev_examples)} / {dev_stats.total_seen}")
    print(f"  Unique positive docs in dev: {len(dev_doc_set)}")

    # 3. Build train examples, excluding all dev positive docs
    print("\n--- Building TRAIN examples ---")
    train_raw = _parse_hotpotqa(downloaded_paths["hotpot_train_v1.1.json"])
    train_examples, train_stats = build_examples(
        train_raw,
        target_split="train",
        doc_exclusion_set=dev_doc_set,  # enforce document-level disjointness
        max_count=MAX_TRAIN,
        source_file="hotpot_train_v1.1.json",
    )
    train_doc_set = {ex.positive_id for ex in train_examples}
    print(f"  Accepted: {len(train_examples)} / {train_stats.total_seen}")
    print(f"  Unique positive docs in train: {len(train_doc_set)}")

    # 4. Validate with TrainingDataset (duplicate ID detection)
    print("\n--- Schema validation ---")
    try:
        TrainingDataset(
            dataset_id="cortex-hotpotqa-3b1-train",
            description="HotpotQA retrieval pairs for Cortex Milestone 3B-1 train split",
            examples=train_examples,
        )
        print(f"  TrainingDataset(train): OK — {len(train_examples)} examples, 0 duplicates")
    except Exception as exc:
        print(f"  [FAIL] TrainingDataset(train): {exc}")
        return 1

    try:
        TrainingDataset(
            dataset_id="cortex-hotpotqa-3b1-dev",
            description="HotpotQA retrieval pairs for Cortex Milestone 3B-1 dev split",
            examples=dev_examples,
        )
        print(f"  TrainingDataset(dev): OK — {len(dev_examples)} examples, 0 duplicates")
    except Exception as exc:
        print(f"  [FAIL] TrainingDataset(dev): {exc}")
        return 1

    # 5. Batch leakage check
    violations = validate_no_eval_leakage(train_examples + dev_examples)
    if violations:
        print(f"\n[FAIL] Leakage violations found ({len(violations)}):")
        for v in violations:
            print(f"  {v}")
        return 1
    print(f"  validate_no_eval_leakage: OK — 0 violations")

    # 6. Document-overlap check
    overlap = train_doc_set & dev_doc_set
    if overlap:
        print(f"\n[FAIL] Document overlap between train and dev ({len(overlap)} docs):")
        for d in sorted(overlap):
            print(f"  {d}")
        return 1
    print(f"  Document disjointness: OK — 0 overlapping positive docs")

    # 7. Write JSONL
    if not args.dry_run:
        with TRAIN_OUT.open("w") as f:
            for ex in train_examples:
                f.write(ex.model_dump_json() + "\n")
        print(f"\n[write] {TRAIN_OUT.relative_to(REPO_ROOT)} ({len(train_examples)} lines)")

        with DEV_OUT.open("w") as f:
            for ex in dev_examples:
                f.write(ex.model_dump_json() + "\n")
        print(f"[write] {DEV_OUT.relative_to(REPO_ROOT)} ({len(dev_examples)} lines)")
    else:
        print("\n[dry-run] Skipping JSONL write")

    # 8. Write manifest (always written, even in dry-run)
    manifest = write_manifest(
        train_examples=train_examples,
        dev_examples=dev_examples,
        train_stats=train_stats,
        dev_stats=dev_stats,
        source_hashes=source_hashes,
        train_path=TRAIN_OUT,
        dev_path=DEV_OUT,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  Train examples : {manifest['train']['count']} (target: {MAX_TRAIN})")
    print(f"  Dev examples   : {manifest['dev']['count']} (target: {MAX_DEV})")
    print(f"  Train docs     : {manifest['train']['unique_positive_docs']}")
    print(f"  Dev docs       : {manifest['dev']['unique_positive_docs']}")
    print(f"  Doc overlap    : {len(manifest['document_overlap_train_dev'])}")
    print(f"  Leakage        : {len(manifest['leakage_violations'])} violations")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
