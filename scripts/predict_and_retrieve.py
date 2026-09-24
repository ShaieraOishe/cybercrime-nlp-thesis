#!/usr/bin/env python3
"""Unified Cybercrime Complaint Triage & Semantic Case Retrieval Tool.

Given any victim complaint narrative:
1. Sanitizes PII (masks emails, phones, credit cards, crypto wallets).
2. Classifies the incident into one of the 7 cybercrime categories using either
   a traditional ML model (Logistic Regression / Naive Bayes) or a fine-tuned Transformer (DistilBERT).
3. Retrieves the top-K most semantically similar historical cases using BM25, Dense Bi-Encoder, or Hybrid RRF.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.text_cleaner import CybercrimeTextCleaner
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseBiEncoderRetriever, HybridRRFRetriever

def load_triage_engine(
    classifier_type: str = "transformer",
    retriever_type: str = "hybrid",
    checkpoint_dir: Path = Path("models/checkpoints"),
    transformer_dir: Path = Path("models/transformers/distilbert-base-uncased/best_model"),
    retrieval_dir: Path = Path("data/processed/retrieval_7class")
):
    cleaner = CybercrimeTextCleaner(mask_pii=True)

    # 1. Load Classifier
    classifier_fn = None
    if classifier_type == "transformer" and transformer_dir.exists():
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(str(transformer_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(transformer_dir)).to(device)
        model.eval()

        def predict_transformer(text: str) -> Tuple[str, float]:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                conf = probs[pred_idx].item() * 100
                category = model.config.id2label[pred_idx]
            return category, conf

        classifier_fn = predict_transformer
    else:
        # Fallback to ML baseline
        vectorizer = joblib.load(checkpoint_dir / "tfidf_vectorizer.joblib")
        label_encoder = joblib.load(checkpoint_dir / "label_encoder.joblib")
        if (checkpoint_dir / "logisticregression_model.joblib").exists():
            model = joblib.load(checkpoint_dir / "logisticregression_model.joblib")
        else:
            model = joblib.load(checkpoint_dir / "multinomialnb_model.joblib")

        def predict_ml(text: str) -> Tuple[str, float]:
            X = vectorizer.transform([text])
            pred_idx = model.predict(X)[0]
            cat = label_encoder.inverse_transform([pred_idx])[0]
            conf = 100.0
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                conf = probs[pred_idx] * 100.0
            return cat, conf

        classifier_fn = predict_ml

    # 2. Load Corpus & Index Retriever
    corpus = []
    with open(retrieval_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    corpus_lookup = {doc["_id"]: doc for doc in corpus}

    bm25 = BM25Retriever(k1=1.5, b=0.75).index(corpus)

    if retriever_type in ["dense", "hybrid"]:
        dense = DenseBiEncoderRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2").index(corpus)
        if retriever_type == "hybrid":
            retriever = HybridRRFRetriever(sparse_retriever=bm25, dense_retriever=dense, rrf_k=60)
        else:
            retriever = dense
    else:
        retriever = bm25

    return cleaner, classifier_fn, retriever, corpus_lookup

def triage_complaint(
    narrative: str,
    classifier_type: str = "transformer",
    retriever_type: str = "hybrid",
    top_k: int = 3
):
    cleaner, classifier_fn, retriever, corpus_lookup = load_triage_engine(
        classifier_type=classifier_type,
        retriever_type=retriever_type
    )

    print("\n" + "=" * 80)
    print("                 CYBERCRIME COMPLAINT TRIAGE & RETRIEVAL REPORT                 ")
    print("=" * 80)
    print(f"\n[Raw Input Narrative]:\n\"{narrative}\"")

    # Step 1: PII Sanitization
    cleaned = cleaner.clean(narrative)
    print(f"\n[Sanitized / PII Redacted]:\n\"{cleaned}\"")

    # Step 2: Prediction
    pred_category, conf = classifier_fn(cleaned)
    engine_name = "Fine-Tuned DistilBERT (Transformer)" if classifier_type == "transformer" else "TF-IDF + Logistic Regression"
    print("\n" + "-" * 80)
    print(f"[*] PREDICTED CATEGORY:  {pred_category.upper()} (Confidence: {conf:.1f}%)")
    print(f"[*] Classification Model: {engine_name}")
    print("-" * 80)

    # Step 3: Retrieval
    print(f"\n[*] Top-{top_k} Semantically Similar Cases ({retriever_type.upper()} Index):")
    results = retriever.query(cleaned, top_k=top_k)

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = corpus_lookup.get(doc_id, {})
        category = doc.get("title", "unknown")
        source = doc.get("metadata", {}).get("source", "N/A")
        text = doc.get("text", "")
        print(f"\n  #{rank} [Score: {score:.4f}] - ID: {doc_id} | Category: {category} | Source: {source}")
        print(f"      Passage: \"{text}\"")

    print("\n" + "=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Triage cybercrime complaint narrative.")
    parser.add_argument("narrative", nargs="?", default=(
        "I was contacted on WhatsApp by someone claiming to be JPMorgan Chase alerting me to "
        "an unauthorized wire of $15,000. They told me to go to http://secure-chase-auth-92.com to verify my password."
    ))
    parser.add_argument("--classifier", choices=["transformer", "ml"], default="transformer")
    parser.add_argument("--retriever", choices=["hybrid", "dense", "bm25"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    triage_complaint(
        narrative=args.narrative,
        classifier_type=args.classifier,
        retriever_type=args.retriever,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
