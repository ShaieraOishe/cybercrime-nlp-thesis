"""Plug-and-play Generic Custom / Institutional Dataset Adapter.

Allows seamless ingestion of any external CSV/JSON cybercrime complaints
(e.g., from local police stations, CERTs, or research partners).
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd

from src.preprocessing.text_cleaner import CybercrimeTextCleaner
from src.data.taxonomy import map_text_to_taxonomy, UNIFIED_CLASSES

CUSTOM_RAW_DIR = Path("data/raw/custom")
PROCESSED_DIR = Path("data/processed")

def generate_sample_agency_data(num_samples: int = 500) -> Path:
    """Generates a realistic sample police/agency complaints CSV file for testing."""
    import random
    CUSTOM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    sample_file = CUSTOM_RAW_DIR / "sample_agency_complaints.csv"
    
    cities = ["Metropolis", "Gotham", "Riverdale", "Star City", "Central City"]
    crimes = [
        ("Online Crypto Investment Scam", "Victim lost $35,000 after being directed by an acquaintance to transfer Ethereum into a fake decentralized wallet app."),
        ("Corporate Email Compromise", "Financial controller received a spoofed email from the CEO demanding an urgent wire payment of $54,000 to an offshore vendor."),
        ("Identity Theft / Bank Loan", "Complainant discovered that their national ID was used to secure an online personal loan of $12,000 without their knowledge."),
        ("Ransomware Hostage", "Local dental clinic reported an attack that encrypted all patient history files, demanding 2 Bitcoin in payment."),
        ("Social Media Account Hijack", "Victim reported that their Instagram and email accounts were hijacked via SIM swapping and used to solicit money from friends."),
        ("DDoS Attack on School Portal", "District online learning management system was subjected to an amplification DDoS attack during final examinations.")
    ]

    records = []
    for i in range(num_samples):
        crime_title, narrative = random.choice(crimes)
        city = random.choice(cities)
        records.append({
            "police_report_number": f"PR-{city[:3].upper()}-2024-{1000 + i}",
            "jurisdiction": f"{city} Cyber Crime Unit",
            "incident_summary": f"[{city} Police Bureau] {narrative} Contact officer email: fraud.unit@{city.lower()}police.gov.",
            "reported_offense": crime_title,
            "loss_amount_usd": random.randint(1000, 100000)
        })

    df = pd.DataFrame(records)
    df.to_csv(sample_file, index=False)
    print(f"[✓] Created sample agency dataset at {sample_file} ({len(df)} records)")
    return sample_file

def ingest_custom_dataset(
    input_file: Path,
    text_col: str,
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
    output_name: str = "custom_agency_complaints"
) -> pd.DataFrame:
    """Ingests, cleans, and standardizes an arbitrary custom cybercrime dataset."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {input_path}")

    print(f"[*] Ingesting custom dataset from {input_path}...")
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix in [".json", ".jsonl"]:
        df = pd.read_json(input_path, lines=(input_path.suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    print(f"[*] Raw rows loaded: {len(df)}")
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in file. Available columns: {list(df.columns)}")

    # Clean missing values
    df = df.dropna(subset=[text_col]).copy()
    
    cleaner = CybercrimeTextCleaner(mask_pii=True)
    df["cleaned_text"] = df[text_col].astype(str).apply(cleaner.clean)

    # Assign IDs
    if id_col and id_col in df.columns:
        df["complaint_id"] = df[id_col].astype(str)
    else:
        df["complaint_id"] = [f"custom_{i}" for i in range(len(df))]

    # Map labels to unified taxonomy
    if label_col and label_col in df.columns:
        # Use existing label text to guide mapping
        df["primary_category"] = (df[label_col].astype(str) + " " + df["cleaned_text"]).apply(map_text_to_taxonomy)
        df["original_issue"] = df[label_col].astype(str)
    else:
        df["primary_category"] = df["cleaned_text"].apply(map_text_to_taxonomy)
        df["original_issue"] = "Custom Agency Report"

    df["source"] = "custom_agency"
    df["raw_text"] = df[text_col].astype(str)
    df["product"] = "Institutional Complaint"
    df["date_received"] = "2024-06-01"

    standard_cols = ["complaint_id", "source", "date_received", "product", "raw_text", "cleaned_text", "primary_category", "original_issue"]
    df_standard = df[standard_cols]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PROCESSED_DIR / f"{output_name}.parquet"
    out_csv = PROCESSED_DIR / f"{output_name}.csv"
    df_standard.to_parquet(out_parquet, index=False)
    df_standard.to_csv(out_csv, index=False)

    print(f"[✓] Successfully processed and harmonized {len(df_standard)} records into {out_parquet}")
    print("\n--- Category Breakdown ---")
    print(df_standard["primary_category"].value_counts())
    return df_standard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Cybercrime Ingestion Adapter")
    parser.add_argument("--input", type=str, help="Path to custom CSV or JSON file")
    parser.add_argument("--text-col", type=str, default="incident_summary", help="Name of narrative text column")
    parser.add_argument("--label-col", type=str, default="reported_offense", help="Name of category/offense column")
    parser.add_argument("--id-col", type=str, default="police_report_number", help="Name of ID column")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample agency CSV for testing")
    args = parser.parse_args()

    if args.generate_sample or not args.input:
        sample_path = generate_sample_agency_data(num_samples=500)
        ingest_custom_dataset(
            input_file=sample_path,
            text_col="incident_summary",
            label_col="reported_offense",
            id_col="police_report_number",
            output_name="custom_agency_complaints"
        )
    else:
        ingest_custom_dataset(
            input_file=Path(args.input),
            text_col=args.text_col,
            label_col=args.label_col,
            id_col=args.id_col
        )
