"""Comprehensive Information Retrieval Evaluation Suite.

Evaluates and compares:
1. Sparse BM25Okapi
2. Sparse TF-IDF Cosine
3. Dense Bi-Encoder (all-MiniLM-L6-v2 / all-mpnet-base-v2)
4. Hybrid Reciprocal Rank Fusion (BM25 + Dense Bi-Encoder)

Outputs unified BEIR benchmark metrics (NDCG, MRR, MAP, Recall, Precision).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever, TfidfCosineRetriever
from src.retrieval.dense_retriever import DenseBiEncoderRetriever, HybridRRFRetriever
from src.retrieval.evaluate_retrieval import evaluate_retriever

RETRIEVAL_DIR = PROJECT_ROOT / "data/processed/retrieval_7class"
REPORTS_DIR = PROJECT_ROOT / "reports"

def run_comprehensive_retrieval_benchmark(dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print("=" * 85)
    print("   CYBERCRIME SIMILAR CASE RETRIEVAL: COMPREHENSIVE MULTI-MODEL BENCHMARK   ")
    print("=" * 85)

    corpus_path = RETRIEVAL_DIR / "corpus.jsonl"
    queries_path = RETRIEVAL_DIR / "queries.jsonl"
    qrels_path = RETRIEVAL_DIR / "qrels.json"

    print("\n[*] Loading corpus passages from corpus.jsonl...")
    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"[✓] Loaded {len(corpus):,} corpus documents.")

    print("[*] Loading evaluation queries from queries.jsonl...")
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"[✓] Loaded {len(queries):,} evaluation queries.")

    print("[*] Loading ground truth relevance judgments from qrels.json...")
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    print(f"[✓] Loaded {len(qrels):,} qrel query mappings.")

    # 1. Sparse BM25Okapi
    print("\n>>> 1. Evaluating Sparse BM25Okapi...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    bm25_res = evaluate_retriever("BM25Okapi (Sparse)", bm25, queries, qrels, top_k=20)

    # 2. Sparse TF-IDF Cosine
    print(">>> 2. Evaluating Sparse TF-IDF Cosine...")
    tfidf = TfidfCosineRetriever(max_features=10000)
    tfidf.index(corpus)
    tfidf_res = evaluate_retriever("TF-IDF Cosine (Sparse)", tfidf, queries, qrels, top_k=20)

    # 3. Dense Bi-Encoder
    print(f"\n>>> 3. Evaluating Dense Bi-Encoder ({dense_model_name})...")
    dense = DenseBiEncoderRetriever(model_name=dense_model_name)
    dense.index(corpus, batch_size=64)
    dense_res = evaluate_retriever(f"Dense Bi-Encoder ({dense_model_name.split('/')[-1]})", dense, queries, qrels, top_k=20)

    # 4. Hybrid Reciprocal Rank Fusion
    print("\n>>> 4. Evaluating Hybrid Reciprocal Rank Fusion (BM25 + Dense)...")
    hybrid = HybridRRFRetriever(sparse_retriever=bm25, dense_retriever=dense, rrf_k=60)
    hybrid_res = evaluate_retriever("Hybrid RRF (BM25 + Dense)", hybrid, queries, qrels, top_k=20)

    all_results = [dense_res, hybrid_res, tfidf_res, bm25_res]
    res_df = pd.DataFrame(all_results).sort_values(by="NDCG@10", ascending=False).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "retrieval_comprehensive_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 85)
    print("                     OVERALL IR BENCHMARK COMPARISON TABLE                      ")
    print("=" * 85)
    print(res_df.to_string(index=False))
    print("=" * 85)
    print(f"[✓] Benchmark results saved to {out_csv}\n")

if __name__ == "__main__":
    model = "sentence-transformers/all-MiniLM-L6-v2"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    run_comprehensive_retrieval_benchmark(dense_model_name=model)
