# Automated Cybercrime Complaint Classification and Semantic Retrieval

> **A Comparative Study of Traditional NLP and Transformer-Based Models**  
> 

---

## 1. Project Overview

Cybercrime complaints filed by victims across regulatory portals and reporting centers are characteristically unstructured, noisy, emotionally charged, and high-volume. Law enforcement agencies and Computer Emergency Response Teams (CERTs) face two primary operational bottlenecks:
1. **Automated Triage & Routing (Classification):** Accurately categorizing incoming incident narratives into actionable crime typologies (e.g., Financial Fraud, Phishing, Sextortion, Identity Threat, Cyber Harassment) to dispatch cases to specialized investigative units.
2. **Precedent & Campaign Linkage (Semantic Retrieval):** Searching historical complaint databases for semantically related narratives to detect syndicated fraud campaigns, identify shared Modus Operandi (M.O.), and de-duplicate recurring reports.

This repository implements the end-to-end experimental framework comparing **Traditional NLP baselines**, **Deep Neural Architectures**, and **Pre-trained Transformers** across both tasks.

---

## 2. Research Questions (RQs)

- **RQ1 (Classification Performance):** How do modern Transformer architectures (DistilBERT, BERT, RoBERTa, DeBERTa-v3, SecBERT) compare against classical NLP baselines (TF-IDF + LinearSVC / Logistic Regression / LightGBM) across class-imbalanced cybercrime narratives?
- **RQ2 (Semantic Retrieval Effectiveness):** How does dense neural retrieval (Sentence-BERT / `all-mpnet-base-v2`, `bge-base-en-v1.5`) compare with lexical search (BM25) and hybrid fusion (Reciprocal Rank Fusion) in surfacing semantically linked cybercrime complaints?
- **RQ3 (Operational Trade-offs):** What are the trade-offs regarding inference latency (ms/query), model footprint (MB), and training throughput for practical deployment in triage systems?
- **RQ4 (Interpretability & Compliance):** Can post-hoc interpretability techniques (SHAP, LIME, Attention Attribution) reliably reveal the key legal and linguistic triggers behind automated decisions?

---

## 3. Unified 7-Class Cybercrime Taxonomy

| # | Category Identifier | Primary Integrated Sources | Modality & Description |
| :---: | :--- | :--- | :--- |
| **1** | `online_financial_fraud` | **CLAIR Fraud Email Corpus** + **CFPB** (*"Fraud or scam"*, *"Money transfer"*) | 419 advance-fee fraud, crypto investment schemes, unauthorized wire transfers |
| **2** | `phishing_smishing` | **UCI SMS Spam Collection** (*spam* label) + CFPB Phishing | Fake bank alerts, delivery fee smishing, credential-harvesting links |
| **3** | `cyber_harassment_blackmail`| **Davidson** & **Waseem** Hate Speech / Threat Corpora | Targeted doxxing, swatting threats, abusive brigading, cyberstalking |
| **4** | `cyber_threat_intelligence` | **Behzadan's CyberTweets** + Threat Feed Telemetry | Zero-day CVE advisories, botnet DDoS traffic, ransomware C2 activity |
| **5** | `ecommerce_fraud` | **BBB Scam Tracker** (*Online Purchase*) + **CFPB** Non-Delivery | Fake online shops, counterfeit goods, non-delivery scams |
| **6** | `sextortion` | **Scam Survivors** (Bristol/Mendeley) + **TExtPhish** | Webcam blackmail scripts, Bitcoin extortion threats |
| **7** | `identity_threat` | **CFPB** (*"Identity theft"*) + Data Breach Disclosures | Stolen SSN, fraudulent loans opened without consent, synthetic identities |

---

## 4. Repository Structure

```text
cybercrime-nlp-thesis/
├── configs/                  # Hyperparameter and model configuration files
├── data/
│   ├── raw/                  # Authentic downloaded datasets (gitignored)
│   │   ├── uci_sms/          # Real UCI SMS Spam Collection
│   │   └── davidson/         # Real Davidson Hate Speech Dataset
│   └── processed/
│       ├── unified_7class_cybercrime_corpus.parquet  # Harmonized Master Corpus (4,737 samples)
│       ├── unified_7class_cybercrime_corpus.csv
│       ├── splits_7class/    # Stratified partitions (70% Train, 15% Val, 15% Test)
│       └── retrieval_7class/ # Semantic search benchmark (corpus.jsonl, queries.jsonl, qrels.json)
├── docs/                     # Research roadmaps, authentic dataset survey, literature review
├── notebooks/                # Exploratory Data Analysis (EDA) Jupyter Notebooks
├── scripts/
│   └── run_pipeline.sh       # One-line execution script
├── src/
│   ├── data/                 # Data acquisition, harmonizer, taxonomy mapping
│   │   ├── build_unified_dataset.py  # All-in-one dataset pipeline runner
│   │   └── taxonomy.py       # Canonical taxonomy and keyword heuristics
│   ├── preprocessing/
│   │   └── text_cleaner.py   # Regex PII scrubbing & IOC sanitization
│   ├── utils/
│   │   └── seed.py           # Deterministic seeding & Apple Silicon (MPS) detection
│   └── evaluation/           # Macro-F1, MRR@10, NDCG@10, statistical significance
├── tests/                    # Automated regression test suite
└── requirements.txt          # Python dependencies
```

---

## 5. Quick Start & Execution

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/cybercrime-nlp-thesis.git
cd cybercrime-nlp-thesis

# Initialize Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Full Data Pipeline (All 7 Categories)
```bash
# Ingests authentic datasets, cleans PII, splits data, and generates retrieval benchmarks
bash scripts/run_pipeline.sh --samples-per-category 700
```

### 3. Run Automated Regression Tests
```bash
.venv/bin/python -m unittest tests/test_dataset_pipeline.py
```

### 4. Interactive Exploratory Data Analysis (EDA)
Open the Jupyter notebook in your IDE:
```text
notebooks/01_dataset_eda_and_harmonization.ipynb
```
