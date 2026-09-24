"""End-to-End Preprocessing and Feature Extraction Pipeline.

Provides standardized text normalization, TF-IDF vectorization,
and deterministic label encoding across train, validation, and test splits.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.text_cleaner import CybercrimeTextCleaner

DEFAULT_SPLITS_DIR = Path("data/processed/splits_7class")
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")

class PreprocessingPipeline:
    """Orchestrates text cleaning, label encoding, and TF-IDF feature extraction."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2
    ):
        self.cleaner = CybercrimeTextCleaner(mask_pii=True)
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            stop_words="english"
        )
        self.is_fitted = False

    def fit(self, train_texts: pd.Series, train_labels: pd.Series) -> "PreprocessingPipeline":
        """Fits label encoder and TF-IDF vectorizer strictly on training data."""
        # Clean texts if not already cleaned
        cleaned_texts = [self.cleaner.clean(t) for t in train_texts]
        self.label_encoder.fit(train_labels)
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        return self

    def transform(self, texts: pd.Series, labels: Optional[pd.Series] = None):
        """Transforms raw or cleaned texts into TF-IDF sparse matrix and encoded labels."""
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before transforming.")
        
        cleaned_texts = [self.cleaner.clean(t) for t in texts]
        X = self.vectorizer.transform(cleaned_texts)
        
        if labels is not None:
            y = self.label_encoder.transform(labels)
            return X, y
        return X

    def fit_transform(self, train_texts: pd.Series, train_labels: pd.Series):
        """Fits on training data and returns transformed features and encoded labels."""
        self.fit(train_texts, train_labels)
        return self.transform(train_texts, train_labels)

    def save(self, output_dir: Path = DEFAULT_CHECKPOINT_DIR) -> None:
        """Serializes fitted vectorizer and label encoder to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.label_encoder, output_dir / "label_encoder.joblib")
        print(f"[✓] Preprocessing pipeline saved to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path = DEFAULT_CHECKPOINT_DIR) -> "PreprocessingPipeline":
        """Loads fitted vectorizer and label encoder from disk."""
        pipe = cls()
        pipe.vectorizer = joblib.load(input_dir / "tfidf_vectorizer.joblib")
        pipe.label_encoder = joblib.load(input_dir / "label_encoder.joblib")
        pipe.is_fitted = True
        return pipe

def load_preprocessed_splits(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    max_features: int = 10000,
    save_pipeline: bool = True
) -> Tuple[Any, Any, Any, Any, Any, Any, PreprocessingPipeline]:
    """Convenience helper to load and vectorize train, val, and test splits.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, pipeline)
    """
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    pipeline = PreprocessingPipeline(max_features=max_features)
    X_train, y_train = pipeline.fit_transform(train_df["cleaned_text"], train_df["primary_category"])
    X_val, y_val = pipeline.transform(val_df["cleaned_text"], val_df["primary_category"])
    X_test, y_test = pipeline.transform(test_df["cleaned_text"], test_df["primary_category"])

    if save_pipeline:
        pipeline.save()

    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline
