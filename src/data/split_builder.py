"""Dataset Splitter.

Produces strictly stratified Train (70%), Validation (15%), and Test (15%) splits
maintaining category distributions across all classes.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = PROCESSED_DIR / "splits"

def create_stratified_splits(seed: int = 42) -> None:
    corpus_file = PROCESSED_DIR / "unified_cybercrime_corpus.parquet"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Master corpus not found at {corpus_file}")

    df = pd.read_parquet(corpus_file)
    print(f"[*] Loaded {len(df)} total records from {corpus_file}")

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["primary_category"]
    )

    # Second split: Split temp 50/50 -> 15% Val, 15% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["primary_category"]
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    print("\n[✓] Created stratified splits:")
    for name, split_df in splits.items():
        parquet_path = SPLITS_DIR / f"{name}.parquet"
        csv_path = SPLITS_DIR / f"{name}.csv"
        split_df.to_parquet(parquet_path, index=False)
        split_df.to_csv(csv_path, index=False)
        pct = (len(split_df) / len(df)) * 100
        print(f"    - {name.capitalize():<5}: {len(split_df):>5} samples ({pct:>4.1f}%) -> {parquet_path}")

    # Verify distribution parity
    print("\n--- Distribution Parity Check (Percent per category in Train vs Test) ---")
    train_dist = train_df["primary_category"].value_counts(normalize=True) * 100
    test_dist = test_df["primary_category"].value_counts(normalize=True) * 100
    comparison = pd.DataFrame({"Train %": train_dist, "Test %": test_dist})
    print(comparison.round(2))

if __name__ == "__main__":
    create_stratified_splits()
