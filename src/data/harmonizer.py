"""Master Corpus Harmonizer.

Combines processed records from CFPB, EuRepoC, CyberGuard, and custom agency sources
into a unified, deduplicated master research corpus.
"""

from pathlib import Path
import pandas as pd
from typing import List

PROCESSED_DIR = Path("data/processed")

def harmonize_all_sources() -> pd.DataFrame:
    """Discovers and merges all processed source parquets into a master dataset."""
    source_files = [
        PROCESSED_DIR / "cfpb_cybercrime.parquet",
        PROCESSED_DIR / "eurepoc_incidents.parquet",
        PROCESSED_DIR / "cyberguard_incidents.parquet",
        PROCESSED_DIR / "custom_agency_complaints.parquet"
    ]

    dfs: List[pd.DataFrame] = []
    for sf in source_files:
        if sf.exists():
            df = pd.read_parquet(sf)
            print(f"[✓] Loaded {len(df)} records from {sf.name} (Source: {df['source'].iloc[0]})")
            dfs.append(df)
        else:
            print(f"[-] Warning: {sf.name} not found, skipping.")

    if not dfs:
        raise RuntimeError("No processed source datasets found in data/processed/!")

    master_df = pd.concat(dfs, ignore_index=True)
    initial_count = len(master_df)

    # 1. Null handling: strictly drop any records with missing critical fields
    master_df = master_df.dropna(subset=["cleaned_text", "primary_category"])
    
    # 2. Text quality filter: minimum word length
    master_df = master_df[master_df["cleaned_text"].str.split().str.len() >= 5]

    # 3. Deduplication: exact duplicate cleaned texts
    master_df = master_df.drop_duplicates(subset=["cleaned_text"])
    final_count = len(master_df)
    print(f"[✓] Deduplication: reduced from {initial_count} to {final_count} unique records.")

    # Sort & reset index
    master_df = master_df.reset_index(drop=True)

    master_parquet = PROCESSED_DIR / "unified_cybercrime_corpus.parquet"
    master_csv = PROCESSED_DIR / "unified_cybercrime_corpus.csv"
    master_df.to_parquet(master_parquet, index=False)
    master_df.to_csv(master_csv, index=False)
    print(f"\n[✓] Saved unified master corpus ({len(master_df)} records) to:")
    print(f"    - {master_parquet}")
    print(f"    - {master_csv}")

    print("\n================ Master Corpus Summary ================")
    print("Distribution by Source:")
    print(master_df["source"].value_counts())
    print("\nDistribution by Primary Cybercrime Category:")
    print(master_df["primary_category"].value_counts())
    print("=======================================================\n")

    return master_df

if __name__ == "__main__":
    harmonize_all_sources()
