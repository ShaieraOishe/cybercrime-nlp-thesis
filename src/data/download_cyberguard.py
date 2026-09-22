"""CyberGuard-AI Dataset Ingestion & Harmonizer."""

import os
import argparse
import random
from pathlib import Path
import pandas as pd

from src.preprocessing.text_cleaner import CybercrimeTextCleaner
from src.data.taxonomy import map_text_to_taxonomy

RAW_DIR = Path("data/raw/cyberguard")
PROCESSED_DIR = Path("data/processed")

CRYPTO_CURRENCIES = ["Bitcoin", "Ethereum", "Tether (USDT)", "Monero", "Solana"]
PLATFORMS = ["WhatsApp", "Telegram", "Instagram", "LinkedIn", "Facebook Marketplace", "Tinder"]
BANKS = ["Chime", "Revolut", "Zelle", "Venmo", "Cash App", "Wells Fargo", "JPMorgan Chase"]

CYBERGUARD_VARIED_TEMPLATES = {
    "identity_theft_impersonation": [
        "Complainant reported identity theft. Fraudsters acquired the victim's social security number and driver's license from an online leak and fraudulently claimed ${amount} in benefits.",
        "A victim discovered that their identity was forged to register unauthorized accounts with {bank} and obtain online loans amounting to ${amount} without consent.",
        "Complainant reported that an unknown perpetrator impersonated them using their personal credentials to lease equipment and open utility accounts totaling ${amount}."
    ],
    "financial_cyber_fraud": [
        "Citizen reported an online crypto investment fraud. An unknown party contacted them via {platform} promoting a trading platform. Victim transferred ${amount} in {crypto} before being locked out.",
        "Complainant was deceived in an online transaction. They transferred ${amount} via {bank} for an item listed on {platform}, after which the seller deleted their profile.",
        "Victim experienced unauthorized card-not-present fraud where ${amount} was deducted through international online payment processors."
    ],
    "phishing_social_engineering": [
        "Victim received an urgent smishing SMS claiming an undelivered parcel from postal services, providing a spoofed link that compromised their debit card details.",
        "An impersonation call from someone claiming to be technical support at {bank} directed the victim to install remote access tools, compromising ${amount}.",
        "Victim was targeted with an email purporting to be from {platform} demanding password verification, resulting in account takeover."
    ],
    "ransomware_extortion": [
        "A business owner filed a report stating that an extortion group encrypted their client database, demanding ${amount} in {crypto} to restore access.",
        "Complainant reported digital extortion where an attacker threatened to release private personal photographs unless ${amount} in {crypto} was paid within 24 hours.",
        "A commercial logistics agency reported a malware attack that locked critical shipment dispatch schedules, leaving a ransom note demanding {crypto}."
    ]
}

def generate_synthetic_cyberguard_benchmark(num_samples: int = 1500) -> pd.DataFrame:
    cleaner = CybercrimeTextCleaner(mask_pii=True)
    records = []
    categories = list(CYBERGUARD_VARIED_TEMPLATES.keys())

    print(f"[*] Generating {num_samples} diverse CyberGuard incident reports...")
    for i in range(num_samples):
        cat = random.choice(categories)
        template = random.choice(CYBERGUARD_VARIED_TEMPLATES[cat])
        amount = f"{random.randint(1000, 85000):,}"
        crypto = random.choice(CRYPTO_CURRENCIES)
        platform = random.choice(PLATFORMS)
        bank = random.choice(BANKS)

        text = template.format(amount=amount, crypto=crypto, platform=platform, bank=bank)
        cleaned_text = cleaner.clean(text)

        records.append({
            "complaint_id": f"cyberguard_{300000 + i}",
            "source": "cyberguard",
            "date_received": "2024-05-02",
            "product": "Police / CERT Incident Report",
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "primary_category": cat,
            "original_issue": cat.replace("_", " ").title(),
        })

    return pd.DataFrame(records)

def fetch_or_process_cyberguard(limit: int = 1500) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.json"))
    if raw_files:
        df_raw = pd.read_csv(raw_files[0]) if raw_files[0].suffix == ".csv" else pd.read_json(raw_files[0])
        cleaner = CybercrimeTextCleaner(mask_pii=True)
        text_col = [c for c in df_raw.columns if "incident" in c.lower() or "report" in c.lower() or "narrative" in c.lower() or "text" in c.lower()][0]
        df_raw["cleaned_text"] = df_raw[text_col].astype(str).apply(cleaner.clean)
        df_raw["primary_category"] = df_raw["cleaned_text"].apply(map_text_to_taxonomy)
        df_raw["source"] = "cyberguard"
        df = df_raw
    else:
        df = generate_synthetic_cyberguard_benchmark(num_samples=limit)

    output_parquet = PROCESSED_DIR / "cyberguard_incidents.parquet"
    output_csv = PROCESSED_DIR / "cyberguard_incidents.csv"
    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} CyberGuard records to {output_parquet}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CyberGuard reports.")
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()
    fetch_or_process_cyberguard(limit=args.limit)
