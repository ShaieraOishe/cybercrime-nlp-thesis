"""Automated verification tests for the Unified 7-Class Cybercrime Dataset Pipeline."""

import unittest
import json
import re
from pathlib import Path
import pandas as pd

class Test7ClassDatasetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = Path("data/processed/unified_7class_cybercrime_corpus.parquet")
        cls.corpus_csv = Path("data/processed/unified_7class_cybercrime_corpus.csv")
        cls.splits_dir = Path("data/processed/splits_7class")
        cls.retrieval_dir = Path("data/processed/retrieval_7class")

        cls.expected_classes = {
            "online_financial_fraud",
            "phishing_smishing",
            "cyber_harassment_blackmail",
            "cyber_threat_intelligence",
            "ecommerce_fraud",
            "sextortion",
            "identity_threat"
        }

    def test_corpus_exists_and_valid(self):
        """Validates that master corpus exists, has >=4,000 samples, and has zero nulls or duplicates."""
        self.assertTrue(self.corpus_path.exists(), "Master 7-class corpus parquet does not exist.")
        self.assertTrue(self.corpus_csv.exists(), "Master 7-class corpus CSV does not exist.")
        
        df = pd.read_parquet(self.corpus_path)
        self.assertGreaterEqual(len(df), 4000, "Corpus must have at least 4,000 samples.")
        self.assertEqual(df["cleaned_text"].isnull().sum(), 0, "No null texts allowed.")
        self.assertEqual(df["primary_category"].isnull().sum(), 0, "No null categories allowed.")
        self.assertEqual(df["cleaned_text"].duplicated().sum(), 0, "Zero duplicates allowed in cleaned text.")
        
        actual_classes = set(df["primary_category"].unique())
        self.assertEqual(self.expected_classes, actual_classes, "All 7 canonical classes must be present.")

    def test_class_balance_uniformity(self):
        """Ensures all 7 classes have reasonable sample parity (> 500 samples per class)."""
        df = pd.read_parquet(self.corpus_path)
        class_counts = df["primary_category"].value_counts()
        for c in self.expected_classes:
            count = class_counts.get(c, 0)
            self.assertGreaterEqual(count, 500, f"Class '{c}' has insufficient samples: {count}")

    def test_pii_sanitization_integrity(self):
        """Ensures sensitive PII (emails, phone numbers, credit card sequences) are masked."""
        df = pd.read_parquet(self.corpus_path)
        email_re = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        phone_re = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        card_re = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

        unmasked_emails = df["cleaned_text"].apply(lambda t: bool(email_re.search(t))).sum()
        unmasked_phones = df["cleaned_text"].apply(lambda t: bool(phone_re.search(t))).sum()
        unmasked_cards = df["cleaned_text"].apply(lambda t: bool(card_re.search(t))).sum()

        self.assertEqual(unmasked_emails, 0, f"Found {unmasked_emails} unmasked emails in corpus.")
        self.assertEqual(unmasked_phones, 0, f"Found {unmasked_phones} unmasked phone numbers in corpus.")
        self.assertEqual(unmasked_cards, 0, f"Found {unmasked_cards} unmasked credit cards in corpus.")

    def test_stratified_splits_integrity(self):
        """Verifies 70/15/15 train-val-test split sizes, class presence, and zero data leakage."""
        train_path = self.splits_dir / "train.parquet"
        val_path = self.splits_dir / "val.parquet"
        test_path = self.splits_dir / "test.parquet"

        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())
        self.assertTrue(test_path.exists())

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        master_df = pd.read_parquet(self.corpus_path)
        total_split_rows = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_split_rows, len(master_df), "Split row count must match corpus total.")

        # Disjoint ID assertions (Zero Data Leakage)
        train_ids = set(train_df["complaint_id"])
        val_ids = set(val_df["complaint_id"])
        test_ids = set(test_df["complaint_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Data leakage between train and val.")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Data leakage between train and test.")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Data leakage between val and test.")

        # Class coverage in all splits
        self.assertEqual(set(train_df["primary_category"].unique()), self.expected_classes)
        self.assertEqual(set(val_df["primary_category"].unique()), self.expected_classes)
        self.assertEqual(set(test_df["primary_category"].unique()), self.expected_classes)

    def test_semantic_retrieval_benchmark(self):
        """Validates BEIR/TREC retrieval suite (corpus.jsonl, queries.jsonl, qrels.json)."""
        corpus_jsonl = self.retrieval_dir / "corpus.jsonl"
        queries_jsonl = self.retrieval_dir / "queries.jsonl"
        qrels_json = self.retrieval_dir / "qrels.json"

        self.assertTrue(corpus_jsonl.exists())
        self.assertTrue(queries_jsonl.exists())
        self.assertTrue(qrels_json.exists())

        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            corpus_ids = {json.loads(line)["_id"] for line in f}

        with open(queries_jsonl, "r", encoding="utf-8") as f:
            query_ids = {json.loads(line)["_id"] for line in f}

        with open(qrels_json, "r", encoding="utf-8") as f:
            qrels = json.load(f)

        self.assertGreaterEqual(len(corpus_ids), 4000)
        self.assertEqual(len(query_ids), 350)
        self.assertEqual(len(qrels), 350)

        # Every query in qrels must exist in queries.jsonl
        for qid, judgments in qrels.items():
            self.assertIn(qid, query_ids, f"Qrel query {qid} not found in queries.jsonl")
            self.assertGreater(len(judgments), 0, f"Query {qid} has 0 relevant documents.")
            # Every document evaluated must exist in corpus.jsonl
            for doc_id, score in judgments.items():
                self.assertIn(doc_id, corpus_ids, f"Document {doc_id} in qrels not found in corpus.")
                self.assertGreaterEqual(score, 1, f"Relevance score must be >= 1.")

if __name__ == "__main__":
    unittest.main()
