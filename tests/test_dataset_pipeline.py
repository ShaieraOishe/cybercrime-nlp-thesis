"""Automated verification tests for the Cybercrime Dataset Pipeline."""

import unittest
import json
from pathlib import Path
import pandas as pd

class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.corpus_path = Path("data/processed/unified_cybercrime_corpus.parquet")
        self.splits_dir = Path("data/processed/splits")
        self.retrieval_dir = Path("data/processed/retrieval")

    def test_corpus_exists_and_valid(self):
        self.assertTrue(self.corpus_path.exists(), "Master corpus parquet does not exist.")
        df = pd.read_parquet(self.corpus_path)
        self.assertGreaterEqual(len(df), 1000, "Corpus must have at least 1,000 samples.")
        self.assertEqual(df["cleaned_text"].isnull().sum(), 0, "No null texts allowed.")
        self.assertEqual(df["primary_category"].isnull().sum(), 0, "No null categories allowed.")
        
        # Check all 6 classes exist
        expected_classes = {
            "identity_theft_impersonation",
            "financial_cyber_fraud",
            "phishing_social_engineering",
            "ransomware_extortion",
            "unauthorized_access_hacking",
            "denial_of_service_disruption",
        }
        actual_classes = set(df["primary_category"].unique())
        self.assertEqual(expected_classes, actual_classes, "All 6 classes must be present.")

    def test_splits_integrity(self):
        train_path = self.splits_dir / "train.parquet"
        val_path = self.splits_dir / "val.parquet"
        test_path = self.splits_dir / "test.parquet"

        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())
        self.assertTrue(test_path.exists())

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        total_split_rows = len(train_df) + len(val_df) + len(test_df)
        master_df = pd.read_parquet(self.corpus_path)
        self.assertEqual(total_split_rows, len(master_df), "Split row count must equal corpus total.")

        # Ensure no data leakage (disjoint IDs)
        train_ids = set(train_df["complaint_id"])
        val_ids = set(val_df["complaint_id"])
        test_ids = set(test_df["complaint_id"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Data leakage between train and val.")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Data leakage between train and test.")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Data leakage between val and test.")

    def test_retrieval_ground_truth(self):
        corpus_jsonl = self.retrieval_dir / "corpus.jsonl"
        queries_jsonl = self.retrieval_dir / "queries.jsonl"
        qrels_json = self.retrieval_dir / "qrels.json"

        self.assertTrue(corpus_jsonl.exists())
        self.assertTrue(queries_jsonl.exists())
        self.assertTrue(qrels_json.exists())

        with open(qrels_json, "r") as f:
            qrels = json.load(f)

        self.assertGreater(len(qrels), 0, "Qrels must contain queries.")
        for qid, judgments in qrels.items():
            self.assertGreater(len(judgments), 0, f"Query {qid} has no relevant documents.")

if __name__ == "__main__":
    unittest.main()
