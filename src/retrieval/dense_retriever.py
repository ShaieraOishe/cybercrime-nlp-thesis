"""Dense Bi-Encoder & Hybrid Reciprocal Rank Fusion (RRF) Retrieval Engines.

Implements:
1. DenseBiEncoderRetriever: Dense semantic retrieval using Sentence-Transformers (e.g. all-mpnet-base-v2).
2. HybridRRFRetriever: Reciprocal Rank Fusion combining Sparse (BM25) and Dense representations.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

class DenseBiEncoderRetriever:
    """Dense bi-encoder retrieval using pre-trained sentence embedding models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.corpus_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.cache_path = Path("data/processed/retrieval_7class/corpus_dense_embeddings.npy")
        self.cache_ids_path = Path("data/processed/retrieval_7class/corpus_dense_ids.json")

    def _load_model(self):
        if self.model is None:
            import os
            from sentence_transformers import SentenceTransformer
            import torch
            if self.device is None:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name, device=self.device)

    def index(self, corpus: List[Dict[str, str]], batch_size: int = 64, use_cache: bool = True) -> "DenseBiEncoderRetriever":
        self.corpus_ids = [doc["_id"] for doc in corpus]

        # Check for cached precomputed embeddings
        if use_cache and self.cache_path.exists() and self.cache_ids_path.exists():
            with open(self.cache_ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == self.corpus_ids:
                self.corpus_embeddings = np.load(self.cache_path)
                return self

        self._load_model()
        texts = [doc["text"] for doc in corpus]
        # Generate normalized dense embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        self.corpus_embeddings = embeddings
        
        # Save cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, embeddings)
        with open(self.cache_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_ids, f)

        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        self._load_model()
        if self.corpus_embeddings is None or len(self.corpus_ids) == 0:
            return []

        q_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # Inner product on normalized vectors is cosine similarity
        sims = np.dot(self.corpus_embeddings, q_emb)

        if top_k >= len(self.corpus_ids):
            top_indices = np.argsort(sims)[::-1]
        else:
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(self.corpus_ids[i], float(sims[i])) for i in top_indices[:top_k]]

class HybridRRFRetriever:
    """Hybrid Reciprocal Rank Fusion (RRF) combining Sparse BM25 and Dense Bi-Encoder."""

    def __init__(self, sparse_retriever: Any, dense_retriever: Any, rrf_k: int = 60):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def query(self, query_text: str, top_k: int = 10, candidate_k: int = 50) -> List[Tuple[str, float]]:
        sparse_results = self.sparse_retriever.query(query_text, top_k=candidate_k)
        dense_results = self.dense_retriever.query(query_text, top_k=candidate_k)

        rrf_scores: Dict[str, float] = {}

        # Sparse contribution
        for rank, (doc_id, _) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Dense contribution
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by fused RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:top_k]
