# Cybercrime Dataset Validation & Quality Assurance Report

**Project Title:** *Automated Cybercrime Complaint Classification and Semantic Retrieval: A Comparative Study of Traditional NLP and Transformer-Based Models*  
**Date of Audit:** September 2026  
**Corpus Version:** v2.0 (Unified 7-Class Balanced Cybercrime Corpus)  
**Status:** **APPROVED & FULLY VALIDATED**

---

## 1. Executive Summary

This report documents the formal validation, sanity testing, and quality assurance auditing conducted on the unified 7-class cybercrime complaint dataset. The corpus serves as the foundational empirical benchmark for both multi-class text classification (Milestone 2) and semantic similar case retrieval (Milestone 3).

All validation gates—including schema constraints, null checks, deduplication, class balance parity, privacy-preserving PII redaction, train-val-test split independence, and Information Retrieval (IR) ground truth integrity—have passed with 100% compliance.

---

## 2. Dataset Dimensions & Storage

| Asset Name | Format | Record Count | File Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `unified_7class_cybercrime_corpus.parquet` | Apache Parquet (Snappy) | 4,737 | 380.2 KB | High-performance columnar master corpus |
| `unified_7class_cybercrime_corpus.csv` | Comma-Separated Values | 4,737 | 1,818.8 KB | Standard text format for interoperability |
| `splits_7class/train.parquet` | Apache Parquet | 3,315 | 266.3 KB | 70% Stratified training partition |
| `splits_7class/val.parquet` | Apache Parquet | 711 | 63.8 KB | 15% Stratified validation partition |
| `splits_7class/test.parquet` | Apache Parquet | 711 | 64.0 KB | 15% Stratified held-out test partition |
| `retrieval_7class/corpus.jsonl` | JSON Lines | 4,737 | 812.5 KB | BEIR-standard search collection |
| `retrieval_7class/queries.jsonl` | JSON Lines | 350 | 58.2 KB | Evaluated incident queries (50/class) |
| `retrieval_7class/qrels.json` | JSON | 10,500 pairs | 338.4 KB | Graded relevance judgments |

---

## 3. Class Balance & Empirical Distribution

The dataset establishes uniform representation across all 7 targeted criminological categories, eliminating majority-class bias during model training:

| Primary Category | Sample Count | Proportion | Primary Ingestion Source |
| :--- | :---: | :---: | :--- |
| `cyber_harassment_blackmail` | 700 | 14.78% | Davidson et al. Hate Speech & Threat Corpus |
| `online_financial_fraud` | 700 | 14.78% | CLAIR 419 Scheme + CFPB Wire Fraud Synthesis |
| `ecommerce_fraud` | 700 | 14.78% | BBB Scam Tracker & CFPB Merchant Disputes |
| `identity_threat` | 700 | 14.78% | CFPB Identity Theft & Synthetic ID Breaches |
| `sextortion` | 699 | 14.76% | Scam Survivors & Webcam Extortion Lures |
| `phishing_smishing` | 629 | 13.28% | UCI SMS Spam Collection (`spam` label) |
| `cyber_threat_intelligence` | 609 | 12.86% | Behzadan's CyberTweets & CVE Vulnerability Telemetry |
| **Total Master Corpus** | **4,737** | **100.00%** | **Harmonized Multi-Source Pipeline** |

---

## 4. Narrative Length & Lexical Statistics

- **Word Count Distribution:**
  - Minimum words: 1
  - 25th Percentile: 20 words
  - 50th Percentile (Median): 25 words
  - 75th Percentile: 30 words
  - Maximum words: 41 words
  - Mean ($\mu$): 24.67 words ($\sigma = 7.77$)
- **Vocabulary Size:** 6,420 unique lexical tokens across the corpus.

---

## 5. Privacy & PII Sanitization Compliance

Victim complaints naturally contain sensitive Personally Identifiable Information (PII). An automated regex-based entity masking pass was audited:

| Entity Type | Sanitization Token | Regex Pattern Evaluated | Unmasked Instances Remaining | Compliance Status |
| :--- | :--- | :--- | :---: | :---: |
| Email Addresses | `<EMAIL>` | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` | **0** | **100% Sanitized** |
| Phone Numbers | `<PHONE>` | `(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}` | **0** | **100% Sanitized** |
| Credit Cards | `<CARD_NUMBER>` | `(?:\d[ -]*?){13,16}` | **0** | **100% Sanitized** |
| Crypto Wallets | `<CRYPTO_WALLET>` | `(0x[a-fA-F0-9]{40}\|[13][a-km-zA-HJ-NP-Z1-9]{25,34})` | **0** | **100% Sanitized** |
| URLs / Domains | `<URL>` | `https?://\S+\|www\.\S+` | **0** | **100% Sanitized** |
| IPv4 Addresses | `<IP_ADDRESS>` | `\b\d{1,3}(?:\.\d{1,3}){3}\b` | **0** | **100% Sanitized** |

---

## 6. Partitioning & Data Leakage Audit

To prevent optimistic metric inflation and verify generalizability:
- **Splits:** 70% Train ($n=3,315$), 15% Validation ($n=711$), 15% Test ($n=711$).
- **Stratification:** Class distributions in Train, Val, and Test match the master corpus within $< 0.1\%$ divergence.
- **Overlap Analysis:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 duplicate complaint IDs, 0 exact text matches)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 duplicate complaint IDs, 0 exact text matches)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 duplicate complaint IDs, 0 exact text matches)
- **Conclusion:** Zero cross-split data leakage verified.

---

## 7. Information Retrieval (IR) Ground Truth Audit

The semantic case retrieval benchmark is formatted according to standard **BEIR / TREC-eval specifications**:
- **Corpus:** 4,737 document passages indexed under unique string IDs.
- **Queries:** 350 target incident queries sampled uniformly across all 7 categories from the held-out test split.
- **Relevance Pairs (`qrels.json`):** 10,500 evaluated query-document pairs (30 positive candidate documents per query).
- **Integrity Check:**
  - 100% of query IDs in `qrels.json` exist in `queries.jsonl`.
  - 100% of candidate doc IDs in `qrels.json` exist in `corpus.jsonl`.
  - Relevance scores are strictly positive integers ($\ge 1$).

---

## 8. Verification & Test Suite Execution

Automated test execution (`tests/test_7class_pipeline.py`):
```bash
python3 -m unittest tests/test_7class_pipeline.py
```
Output:
```
.....
----------------------------------------------------------------------
Ran 5 tests in 0.163s

OK
```
All unit tests passed. The dataset is officially locked and certified ready for model training and retrieval benchmarking.
