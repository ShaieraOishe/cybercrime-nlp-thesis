"""Traditional Machine Learning Baselines for Cybercrime Complaint Classification.

Trains and evaluates:
1. Multinomial Naive Bayes (MultinomialNB)
2. Logistic Regression (LogisticRegression with balanced class weights)
3. Linear Support Vector Classifier (LinearSVC)
4. Random Forest Classifier (RandomForestClassifier)

Saves checkpoints, confusion matrices, and formal classification benchmark tables.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path and configure writable matplotlib dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import time
import argparse
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.preprocessing.pipeline import load_preprocessed_splits

CHECKPOINT_DIR = Path("models/checkpoints")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

def get_models(seed: int = 42) -> Dict[str, Any]:
    """Returns baseline model registry."""
    return {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "LinearSVC": LinearSVC(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=None, n_jobs=-1, random_state=seed
        )
    }

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], model_name: str, save_path: Path):
    """Plots and saves normalized confusion matrix heatmap."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title(f"Normalized Confusion Matrix: {model_name}", fontsize=14, pad=15)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_benchmarks(seed: int = 42, max_features: int = 10000):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      CYBERCRIME COMPLAINT CLASSIFICATION: ML BASELINE SUITE       ")
    print("=" * 75)

    print("\n[*] Vectorizing splits using TF-IDF (1-2 grams, max_features=10,000)...")
    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = load_preprocessed_splits(
        max_features=max_features, save_pipeline=True
    )
    classes = list(pipeline.label_encoder.classes_)
    print(f"[✓] Feature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"[✓] Target classes ({len(classes)}): {classes}")

    models = get_models(seed=seed)
    results = []

    for name, model in models.items():
        print(f"\n>>> Training {name}...")
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Validation set evaluation
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)

        # Test set evaluation
        start_test = time.time()
        test_preds = model.predict(X_test)
        infer_time = time.time() - start_test
        test_acc = accuracy_score(y_test, test_preds)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
        _, _, weighted_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)

        print(f"    - Train time : {train_time:.3f}s | Test inference: {infer_time:.4f}s")
        print(f"    - Val Acc    : {val_acc:.4f}  | Val Macro-F1  : {val_f1:.4f}")
        print(f"    - Test Acc   : {test_acc:.4f}  | Test Macro-F1 : {test_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds)
        fig_path = FIGURES_DIR / f"cm_{name.lower()}.png"
        plot_confusion_matrix(cm, classes, name, fig_path)

        # Save model checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(model, ckpt_path)

        results.append({
            "Model": name,
            "Val Accuracy": round(val_acc, 4),
            "Val Macro-F1": round(val_f1, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test Macro-Prec": round(test_p, 4),
            "Test Macro-Rec": round(test_r, 4),
            "Test Macro-F1": round(test_f1, 4),
            "Test Weighted-F1": round(weighted_f1, 4),
            "Train Time (s)": round(train_time, 3),
            "Inference Time (s)": round(infer_time, 4)
        })

    # Summary table
    res_df = pd.DataFrame(results).sort_values(by="Test Macro-F1", ascending=False).reset_index(drop=True)
    csv_path = REPORTS_DIR / "classification_baseline_results.csv"
    res_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print("                    FINAL BENCHMARK COMPARISON TABLE                     ")
    print("=" * 75)
    print(res_df.to_string(index=False))
    print("=" * 75)
    print(f"[✓] Benchmark results saved to {csv_path}")
    print(f"[✓] Confusion matrices saved to {FIGURES_DIR}")
    print(f"[✓] Model checkpoints saved to {CHECKPOINT_DIR}\n")

    # Detailed report for best model
    best_model_name = res_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_preds = best_model.predict(X_test)
    print(f"Detailed Classification Report ({best_model_name} on Held-Out Test Split):")
    print(classification_report(y_test, best_preds, target_names=classes, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run traditional ML classification baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10000)
    args = parser.parse_args()
    run_benchmarks(seed=args.seed, max_features=args.max_features)
