"""Semantic Retrieval Ground-Truth & Evaluation Suite Builder.

Prepares evaluation corpus, queries, and qrels (relevance judgments) in standard
BEIR / TREC format to benchmark Sparse (BM25), Dense (SBERT), and Hybrid (RRF) retrieval.
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

PROCESSED_DIR = Path("data/processed")
RETRIEVAL_DIR = PROCESSED_DIR / "retrieval"

def build_retrieval_benchmark(num_queries: int = 300) -> None:
    corpus_path = PROCESSED_DIR / "unified_cybercrime_corpus.parquet"
    test_path = PROCESSED_DIR / "splits" / "test.parquet"

    if not corpus_path.exists() or not test_path.exists():
        raise FileNotFoundError("Master corpus or test split missing. Run split_builder.py first.")

    corpus_df = pd.read_parquet(corpus_path)
    test_df = pd.read_parquet(test_path)

    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build document search pool (corpus.jsonl)
    corpus_file = RETRIEVAL_DIR / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for _, row in corpus_df.iterrows():
            doc = {
                "_id": str(row["complaint_id"]),
                "title": str(row["primary_category"]),
                "text": str(row["cleaned_text"]),
                "metadata": {
                    "source": str(row["source"]),
                    "original_issue": str(row["original_issue"])
                }
            }
            f.write(json.dumps(doc) + "\n")
    print(f"[✓] Document pool saved: {len(corpus_df)} docs -> {corpus_file}")

    # 2. Sample evaluation queries from test split (queries.jsonl)
    queries_sample = test_df.sample(n=min(num_queries, len(test_df)), random_state=42)
    queries_file = RETRIEVAL_DIR / "queries.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        for _, row in queries_sample.iterrows():
            query = {
                "_id": str(row["complaint_id"]),
                "text": str(row["cleaned_text"])
            }
            f.write(json.dumps(query) + "\n")
    print(f"[✓] Evaluation queries saved: {len(queries_sample)} queries -> {queries_file}")

    # 3. Construct relevance judgments (qrels.json)
    # Graded relevance:
    # Score 2: Same primary category AND same original issue / sub-topic (High semantic affinity)
    # Score 1: Same primary category (Topical relevance)
    print("[*] Generating ground truth relevance judgments (qrels)...")
    qrels: Dict[str, Dict[str, int]] = {}
    
    # Pre-index corpus by category and issue for fast matching
    cat_index = corpus_df.groupby("primary_category")["complaint_id"].apply(set).to_dict()
    issue_index = corpus_df.groupby(["primary_category", "original_issue"])["complaint_id"].apply(set).to_dict()

    for _, q_row in queries_sample.iterrows():
        q_id = str(q_row["complaint_id"])
        cat = q_row["primary_category"]
        issue = q_row["original_issue"]

        qrels[q_id] = {}
        
        # High relevance matches (same issue)
        same_issue_docs = issue_index.get((cat, issue), set())
        for doc_id in same_issue_docs:
            if doc_id != q_id:
                qrels[q_id][str(doc_id)] = 2

        # Broader category matches (limit to top 20 to maintain concise evaluation)
        same_cat_docs = cat_index.get(cat, set()) - same_issue_docs
        for doc_id in list(same_cat_docs)[:20]:
            if doc_id != q_id:
                qrels[q_id][str(doc_id)] = 1

    qrels_file = RETRIEVAL_DIR / "qrels.json"
    with open(qrels_file, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)

    total_rels = sum(len(docs) for docs in qrels.values())
    avg_rels = total_rels / len(qrels) if qrels else 0
    print(f"[✓] Qrels saved -> {qrels_file}")
    print(f"    Total query-document judgments: {total_rels} (Avg {avg_rels:.1f} relevant cases/query)")

if __name__ == "__main__":
    build_retrieval_benchmark()
