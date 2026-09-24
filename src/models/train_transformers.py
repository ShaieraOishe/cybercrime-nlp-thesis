"""Transformer Fine-Tuning Pipeline for Cybercrime Complaint Classification.

Supports:
- distilbert-base-uncased (Fast lightweight baseline)
- roberta-base (Contextual bidirectional benchmark)
- jackaduma/SecBERT (Cybersecurity domain-adapted transformer)

Leverages Apple Silicon MPS / CUDA hardware acceleration and outputs
confusion matrices and formal classification reports.
"""

import os
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import time
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

SPLITS_DIR = PROJECT_ROOT / "data/processed/splits_7class"
CHECKPOINTS_DIR = PROJECT_ROOT / "models/transformers"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

class CybercrimeDataset(Dataset):
    """PyTorch Dataset for text sequence classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False  # Collator handles dynamic padding
        )
        encoding["labels"] = label
        return encoding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }

def train_transformer(
    model_name: str = "distilbert/distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42
):
    model_alias = model_name.split("/")[-1].lower()
    output_dir = CHECKPOINTS_DIR / model_alias
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"      TRANSFORMER FINE-TUNING: {model_name}       ")
    print("=" * 75)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[*] Hardware Acceleration Target: {device.upper()}")

    # 2. Load splits
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["primary_category"])
    val_labels = label_encoder.transform(val_df["primary_category"])
    test_labels = label_encoder.transform(test_df["primary_category"])
    classes = list(label_encoder.classes_)
    num_labels = len(classes)

    print(f"[*] Loaded partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"[*] Target Classes ({num_labels}): {classes}")

    # 3. Load Tokenizer & Model
    print(f"\n[*] Downloading/Loading Tokenizer and Pre-Trained Weights for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)}
    )

    train_dataset = CybercrimeDataset(train_df["cleaned_text"].tolist(), train_labels.tolist(), tokenizer, max_length)
    val_dataset = CybercrimeDataset(val_df["cleaned_text"].tolist(), val_labels.tolist(), tokenizer, max_length)
    test_dataset = CybercrimeDataset(test_df["cleaned_text"].tolist(), test_labels.tolist(), tokenizer, max_length)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=1,
        seed=seed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 5. Train
    print("\n>>> Starting Fine-Tuning...")
    start_train = time.time()
    trainer.train()
    train_time = time.time() - start_train
    print(f"[✓] Training completed in {train_time:.1f}s")

    # 6. Evaluate on Held-out Test Set
    print("\n[*] Evaluating on Held-Out Test Split (n=711)...")
    start_test = time.time()
    test_results = trainer.predict(test_dataset)
    infer_time = time.time() - start_test
    test_preds = np.argmax(test_results.predictions, axis=-1)

    test_metrics = compute_metrics((test_results.predictions, test_labels))
    print(f"    - Test Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"    - Test Macro-F1  : {test_metrics['macro_f1']:.4f}")
    print(f"    - Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")

    # 7. Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples", xticklabels=classes, yticklabels=classes)
    plt.title(f"Normalized Confusion Matrix: {model_alias.upper()}", fontsize=14, pad=15)
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"cm_{model_alias}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] Saved confusion matrix to {fig_path}")

    # 8. Save final model & tokenizer
    final_model_dir = output_dir / "best_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"[✓] Best checkpoint & tokenizer saved to {final_model_dir}")

    # 9. Append to benchmark table
    res_entry = {
        "Model": model_alias.upper(),
        "Architecture": model_name,
        "Test Accuracy": round(test_metrics["accuracy"], 4),
        "Test Macro-Prec": round(test_metrics["macro_precision"], 4),
        "Test Macro-Rec": round(test_metrics["macro_recall"], 4),
        "Test Macro-F1": round(test_metrics["macro_f1"], 4),
        "Test Weighted-F1": round(test_metrics["weighted_f1"], 4),
        "Train Time (s)": round(train_time, 2),
        "Inference Latency (s)": round(infer_time, 4)
    }

    csv_path = REPORTS_DIR / "transformer_benchmark_results.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_df = existing_df[existing_df["Model"] != model_alias.upper()]
        updated_df = pd.concat([existing_df, pd.DataFrame([res_entry])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([res_entry])
    updated_df.to_csv(csv_path, index=False)
    print(f"[✓] Appended results to {csv_path}")

    print("\nDetailed Test Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=classes, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune transformer for cybercrime complaint classification.")
    parser.add_argument("--model-name", type=str, default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_transformer(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed
    )
