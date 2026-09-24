"""High-Performance BM25Okapi and Sparse Vector Space Information Retrieval Engines.

Implements:
1. Pure NumPy BM25Okapi with Robertson-Spärck Jones IDF and document length normalization.
2. TF-IDF Vector Space Model with Cosine Similarity.
3. BEIR/TREC-compatible query ranking interfaces for cybercrime similar case retrieval.
"""

import math
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BM25Retriever:
    """Inverted index and BM25Okapi scoring engine."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_ids: List[str] = []
        self.doc_lengths: np.ndarray = np.array([])
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_idx: freq}
        self.idf: Dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple, fast alphanumeric regex tokenizer in lowercase."""
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, corpus: List[Dict[str, str]]) -> "BM25Retriever":
        """Builds inverted index and calculates IDF weights.

        Args:
            corpus: List of dicts with '_id' and 'text'.
        """
        self.corpus_ids = [doc["_id"] for doc in corpus]
        self.doc_count = len(corpus)
        lengths = []
        self.inverted_index = {}

        for doc_idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc["text"])
            lengths.append(len(tokens))
            term_freqs: Dict[str, int] = {}
            for t in tokens:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = freq

        self.doc_lengths = np.array(lengths, dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lengths)) if self.doc_count > 0 else 1.0

        # Calculate Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            # Robertson-Spärck Jones formula
            self.idf[term] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Scores all indexed documents against the query and returns top_k (doc_id, score)."""
        tokens = self._tokenize(query_text)
        if not tokens or self.doc_count == 0:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float32)
        len_norm = 1.0 - self.b + self.b * (self.doc_lengths / self.avg_doc_len)

        for term in tokens:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings.items():
                denom = tf + self.k1 * len_norm[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += term_idf * (numerator / denom)

        # Retrieve top_k indices
        if top_k >= self.doc_count:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [(self.corpus_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results[:top_k]

class TfidfCosineRetriever:
    """TF-IDF Vector Space retrieval baseline with Cosine Similarity."""

    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.corpus_ids: List[str] = []
        self.tfidf_matrix = None

    def index(self, corpus: List[Dict[str, str]]) -> "TfidfCosineRetriever":
        self.corpus_ids = [doc["_id"] for doc in corpus]
        texts = [doc["text"] for doc in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        if top_k >= len(self.corpus_ids):
            top_indices = np.argsort(sims)[::-1]
        else:
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = [(self.corpus_ids[idx], float(sims[idx])) for idx in top_indices if sims[idx] > 0]
        return results[:top_k]
