"""Information Retrieval Evaluation Suite for Cybercrime Similar Case Retrieval.

Evaluates BM25 and TF-IDF Cosine models on the BEIR/TREC benchmark against qrels.json.
Metrics:
- Mean Reciprocal Rank (MRR@10)
- Normalized Discounted Cumulative Gain (NDCG@5, NDCG@10)
- Mean Average Precision (MAP@10)
- Recall@1, Recall@5, Recall@10, Recall@20
- Precision@1, Precision@5, Precision@10
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever, TfidfCosineRetriever

RETRIEVAL_DIR = Path("data/processed/retrieval_7class")
REPORTS_DIR = Path("reports")

def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def compute_ndcg(retrieved_ids: List[str], ground_truth: Dict[str, int], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k."""
    relevances = [ground_truth.get(did, 0) for did in retrieved_ids[:k]]
    actual_dcg = compute_dcg(relevances, k)

    ideal_relevances = sorted(ground_truth.values(), reverse=True)
    ideal_dcg = compute_dcg(ideal_relevances, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_mrr(retrieved_ids: List[str], ground_truth: Dict[str, int], k: int = 10) -> float:
    """Computes Reciprocal Rank at rank k (rank of first relevant document)."""
    for i, did in enumerate(retrieved_ids[:k]):
        if ground_truth.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def compute_recall(retrieved_ids: List[str], ground_truth: Dict[str, int], k: int) -> float:
    """Computes Recall at rank k (relevant retrieved / total relevant)."""
    if not ground_truth:
        return 0.0
    retrieved_rel = sum(1 for did in retrieved_ids[:k] if ground_truth.get(did, 0) > 0)
    return retrieved_rel / len(ground_truth)

def compute_precision(retrieved_ids: List[str], ground_truth: Dict[str, int], k: int) -> float:
    """Computes Precision at rank k."""
    if k == 0:
        return 0.0
    retrieved_rel = sum(1 for did in retrieved_ids[:k] if ground_truth.get(did, 0) > 0)
    return retrieved_rel / k

def compute_ap(retrieved_ids: List[str], ground_truth: Dict[str, int], k: int = 10) -> float:
    """Computes Average Precision at rank k."""
    if not ground_truth:
        return 0.0
    running_hits = 0
    running_sum = 0.0
    for i, did in enumerate(retrieved_ids[:k]):
        if ground_truth.get(did, 0) > 0:
            running_hits += 1
            running_sum += running_hits / (i + 1)
    num_rel = min(len(ground_truth), k)
    return running_sum / num_rel if num_rel > 0 else 0.0

def evaluate_retriever(
    name: str,
    retriever: Any,
    queries: List[Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 20
) -> Dict[str, Any]:
    """Runs batch queries and evaluates all IR metrics."""
    ndcg_5_list, ndcg_10_list = [], []
    mrr_10_list = []
    map_10_list = []
    r1_list, r5_list, r10_list, r20_list = [], [], [], []
    p1_list, p5_list, p10_list = [], [], []

    start_time = time.time()
    for q in queries:
        qid = q["_id"]
        qtext = q["text"]
        gt = qrels.get(qid, {})

        results = retriever.query(qtext, top_k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        ndcg_5_list.append(compute_ndcg(retrieved_ids, gt, k=5))
        ndcg_10_list.append(compute_ndcg(retrieved_ids, gt, k=10))
        mrr_10_list.append(compute_mrr(retrieved_ids, gt, k=10))
        map_10_list.append(compute_ap(retrieved_ids, gt, k=10))

        r1_list.append(compute_recall(retrieved_ids, gt, k=1))
        r5_list.append(compute_recall(retrieved_ids, gt, k=5))
        r10_list.append(compute_recall(retrieved_ids, gt, k=10))
        r20_list.append(compute_recall(retrieved_ids, gt, k=20))

        p1_list.append(compute_precision(retrieved_ids, gt, k=1))
        p5_list.append(compute_precision(retrieved_ids, gt, k=5))
        p10_list.append(compute_precision(retrieved_ids, gt, k=10))

    elapsed = time.time() - start_time
    avg_latency_ms = (elapsed / len(queries)) * 1000.0

    return {
        "Retriever": name,
        "NDCG@10": round(float(np.mean(ndcg_10_list)), 4),
        "NDCG@5": round(float(np.mean(ndcg_5_list)), 4),
        "MRR@10": round(float(np.mean(mrr_10_list)), 4),
        "MAP@10": round(float(np.mean(map_10_list)), 4),
        "Recall@1": round(float(np.mean(r1_list)), 4),
        "Recall@5": round(float(np.mean(r5_list)), 4),
        "Recall@10": round(float(np.mean(r10_list)), 4),
        "Recall@20": round(float(np.mean(r20_list)), 4),
        "Precision@10": round(float(np.mean(p10_list)), 4),
        "Latency/Query (ms)": round(avg_latency_ms, 2)
    }

def main():
    print("=" * 80)
    print("      CYBERCRIME SIMILAR CASE RETRIEVAL: IR BENCHMARK SUITE       ")
    print("=" * 80)

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

    # 1. Evaluate BM25Okapi
    print("\n>>> Indexing and Evaluating BM25Okapi...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    bm25_metrics = evaluate_retriever("BM25Okapi", bm25, queries, qrels, top_k=20)

    # 2. Evaluate TF-IDF Cosine
    print(">>> Indexing and Evaluating TF-IDF Cosine...")
    tfidf_retriever = TfidfCosineRetriever(max_features=10000)
    tfidf_retriever.index(corpus)
    tfidf_metrics = evaluate_retriever("TF-IDF Cosine", tfidf_retriever, queries, qrels, top_k=20)

    results = [bm25_metrics, tfidf_metrics]
    res_df = pd.DataFrame(results).sort_values(by="NDCG@10", ascending=False).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "retrieval_baseline_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("                RETRIEVAL BENCHMARK COMPARISON TABLE                 ")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)
    print(f"[✓] Retrieval results saved to {out_csv}\n")

if __name__ == "__main__":
    main()
