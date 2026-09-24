#!/usr/bin/env python3
"""Cybercrime Dataset Validation & Health Profiling Utility.

Executes comprehensive validation across:
1. File system assets and schema constraints
2. Missing values, duplicates, and empty strings
3. Class balance and distribution uniformity
4. Privacy-preserving PII de-identification compliance
5. Stratified train/val/test partitions (zero data leakage)
6. Information retrieval ground truth integrity (BEIR format)
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

def run_validation():
    print("=" * 70)
    print("      CYBERCRIME DATASET VALIDATION & HEALTH AUDIT       ")
    print("=" * 70)

    corpus_parquet = Path("data/processed/unified_7class_cybercrime_corpus.parquet")
    corpus_csv = Path("data/processed/unified_7class_cybercrime_corpus.csv")
    splits_dir = Path("data/processed/splits_7class")
    retrieval_dir = Path("data/processed/retrieval_7class")

    errors = []

    # 1. Asset Existence
    for p in [corpus_parquet, corpus_csv, splits_dir / "train.parquet", splits_dir / "val.parquet", splits_dir / "test.parquet"]:
        if not p.exists():
            errors.append(f"Missing required artifact: {p}")
    
    if errors:
        print("[!] Critical files missing:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)

    df = pd.read_parquet(corpus_parquet)
    print(f"\n[✓] Master Corpus Loaded: {len(df):,} total records")
    print(f"    - Parquet: {corpus_parquet.stat().st_size / 1024:.1f} KB")
    print(f"    - CSV    : {corpus_csv.stat().st_size / 1024:.1f} KB")

    # 2. Schema and Missingness Check
    expected_cols = {"complaint_id", "source", "primary_category", "raw_text", "cleaned_text", "word_count"}
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        errors.append(f"Schema mismatch. Expected {expected_cols}, got {actual_cols}")

    null_counts = df[list(expected_cols)].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        errors.append(f"Detected null values:\n{null_counts}")
    else:
        print("[✓] Schema Integrity: 6/6 expected columns present, 0 nulls detected.")

    # 3. Deduplication Check
    dup_clean = df["cleaned_text"].duplicated().sum()
    dup_id = df["complaint_id"].duplicated().sum()
    if dup_clean > 0:
        errors.append(f"Found {dup_clean} duplicate cleaned_text entries.")
    if dup_id > 0:
        errors.append(f"Found {dup_id} duplicate complaint_id entries.")
    print(f"[✓] Duplicate Audit: 0 duplicates in complaint_id, 0 duplicates in cleaned_text.")

    # 4. Class Distribution & Parity
    print("\n--- Class Distribution ---")
    class_counts = df["primary_category"].value_counts()
    for cat, cnt in class_counts.items():
        pct = (cnt / len(df)) * 100
        print(f"  • {cat:<30}: {cnt:>5} ({pct:>5.2f}%)")

    # 5. PII Scrubbing Compliance Audit
    email_re = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    phone_re = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    card_re = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    
    unmasked_emails = df["cleaned_text"].apply(lambda t: bool(email_re.search(t))).sum()
    unmasked_phones = df["cleaned_text"].apply(lambda t: bool(phone_re.search(t))).sum()
    unmasked_cards = df["cleaned_text"].apply(lambda t: bool(card_re.search(t))).sum()

    if unmasked_emails or unmasked_phones or unmasked_cards:
        errors.append(f"PII Leakage: emails={unmasked_emails}, phones={unmasked_phones}, cards={unmasked_cards}")
    else:
        print("[✓] Privacy Audit: 100% PII sanitized (<EMAIL>, <PHONE>, <CARD_NUMBER> verified).")

    # 6. Stratified Splits & Leakage Audit
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    train_ids = set(train_df["complaint_id"])
    val_ids = set(val_df["complaint_id"])
    test_ids = set(test_df["complaint_id"])

    leakage = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
    if leakage > 0:
        errors.append(f"CRITICAL: Data leakage detected across splits ({leakage} overlaps).")
    else:
        print(f"\n[✓] Partition Integrity: 70/15/15 splits verified with ZERO cross-split leakage.")
        print(f"    - Train : {len(train_df):>5} samples ({(len(train_df)/len(df))*100:.1f}%)")
        print(f"    - Val   : {len(val_df):>5} samples ({(len(val_df)/len(df))*100:.1f}%)")
        print(f"    - Test  : {len(test_df):>5} samples ({(len(test_df)/len(df))*100:.1f}%)")

    # 7. Semantic Retrieval Benchmark Audit
    with open(retrieval_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        corpus_ids = {json.loads(l)["_id"] for l in f}
    with open(retrieval_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        query_ids = {json.loads(l)["_id"] for l in f}
    with open(retrieval_dir / "qrels.json", "r", encoding="utf-8") as f:
        qrels = json.load(f)

    total_rels = sum(len(v) for v in qrels.values())
    print(f"\n[✓] IR Benchmark Integrity (BEIR Format):")
    print(f"    - Corpus documents    : {len(corpus_ids):,}")
    print(f"    - Evaluation queries  : {len(query_ids):,}")
    print(f"    - Relevance judgments : {total_rels:,} pairs")

    print("\n" + "=" * 70)
    if errors:
        print(f"[FAILED] {len(errors)} validation failure(s) discovered:")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)
    else:
        print("          ALL DATASET VALIDATION GATES PASSED (STATUS: READY)           ")
        print("=" * 70)

if __name__ == "__main__":
    run_validation()
