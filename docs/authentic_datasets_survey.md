# Comprehensive Catalog of Authentic, Publicly Accessible Cybercrime Datasets

**Thesis Topic:** *Automated Cybercrime Complaint Classification and Semantic Retrieval: A Comparative Study of Traditional NLP and Transformer-Based Models*

---

## 🏆 Summary Matrix of Authentic Datasets

| Dataset | Primary Authority / Source | Volume (Narratives) | Typology & Attack Vectors Covered | License / Terms | PII Sanitization Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CFPB Consumer Complaint Database** | US Federal Regulator (CFPB) | **>1,600,000** total (>350k cyber/fraud) | Identity Theft, Account Takeover, Wire Fraud, Crypto Scams, Phishing Impersonation | Public Domain (U.S. Govt Work / CC0) | ✅ Pre-masked by federal legal teams (`XXXX`) |
| **2. BBB Scam Tracker Academic Corpus** | Better Business Bureau / Zenodo | **~300,000+** victim reports | 25+ Scam Types (Tech support, Romance scam, Fake check, Crypto scam, Extortion) | Open Academic Research License | ✅ Contact info scrubbed at source |
| **3. EuRepoC (European Repo of Cyber Incidents)** | Univ. of Heidelberg / SWP / Zenodo | **~3,000+** verified reports | Ransomware, DDoS, Wiper Malware, System Intrusion, Supply Chain Breach | Creative Commons Attribution 4.0 (CC BY 4.0) | ✅ Enterprise/agency reports; no consumer PII |
| **4. DIFrauD Benchmark** | Hugging Face (ACL/EMNLP '23-'24) | **95,854** standardized texts | 7 Domains: Phishing emails, SMS Smishing, Job Scams, Impersonation, Fake reviews | Apache 2.0 / MIT Open Source | ✅ Pre-cleaned & standardized for BERT |
| **5. VCDB (VERIS Community Database)** | Verizon Risk Team / GitHub | **>9,500+** breach narratives | VERIS 4A Framework: Hacking, Malware (Ransomware), Social (Phishing), Misuse | CC BY 4.0 International | ✅ Organization-level; anonymized personnel |
| **6. Privacy Rights Clearinghouse (PRC)** | Privacy Rights Clearinghouse / Data.gov | **~11,000+** breach narratives | HACK (Exploit/Malware), DISC (Leaked DB), CARD (CNP Fraud), INSD (Insider) | CC BY-NC 4.0 (Academic use permitted) | ✅ Fully sanitized breach accounts |
| **7. M-SET (Social Engineering Threat Dataset)** | Zenodo (2024 Release) | **624** curated cases | Phishing, Baiting, Pretexting, Scareware, Malware Dropper, Benign | CC BY 4.0 (DOI: 10.5281/zenodo.15235123) | ✅ Anonymized entities & lures |
| **8. Cyberbullying Multi-Class Corpus** | IEEE TCSS Benchmark / Kaggle | **~47,000** samples | Cyberstalking, Harassment, Ethnic Hate, Gender Harassment | CC BY-SA 4.0 | ✅ Masked user handles |

---

## 1. Top Recommended Victim Complaint Repositories

### 1.1 CFPB Consumer Complaint Database (Cybercrime & Fraud Slice)
* **Direct Download URL:**
  ```bash
  curl -o cfpb_complaints.csv "https://data.consumerfinance.gov/api/views/s6ew-56td/rows.csv?accessType=DOWNLOAD"
  ```
* **Python Streaming API Access:**
  ```python
  import pandas as pd
  url = "https://data.consumerfinance.gov/resource/s6ew-56td.json?$where=consumer_complaint_narrative IS NOT NULL&$limit=50000"
  df = pd.read_json(url)
  ```
* **Key Fields:** `consumer_complaint_narrative` (50–1,200 words), `product`, `sub_product`, `issue`, `sub_issue`.
* **Why it's essential:** Authentic victim emotional descriptions, natural class imbalance, pre-masked PII.

### 1.2 Better Business Bureau (BBB) Scam Tracker Academic Dataset
* **Official URL:** https://www.bbb.org/scamtracker
* **Zenodo Research Mirror:** DOI [10.5281/zenodo.4764835](https://doi.org/10.5281/zenodo.4764835)
* **Key Fields:** `Description` (victim narrative), `Scam Type` (25+ categories), `Dollar Loss` (numerical financial loss), `Contact Method` (Email, SMS, Phone), `Payment Method` (Crypto, Wire, Gift Card).
* **Why it's essential:** Includes actual monetary loss amounts, enabling triage severity regression alongside classification.

---

## 2. Technical Cyber Incident & Threat Intelligence Repositories

### 2.1 EuRepoC (European Repository of Cyber Incidents)
* **Zenodo Direct Download:**
  ```bash
  wget "https://zenodo.org/records/10058362/files/eurepoc_data.csv"
  ```
* **Key Fields:** `incident_description`, `attack_vector` (*Ransomware, Data Breach, DDoS, Wiper, Phishing*), `threat_actor_attributed`, `target_sector`.
* **Why it's essential:** Independent corroboration of real-world high-impact attacks by academic institutions.

### 2.2 VCDB (VERIS Community Database)
* **GitHub Repository:** https://github.com/vz-risk/vcdb
* **Direct Clone:**
  ```bash
  git clone --depth 1 https://github.com/vz-risk/vcdb.git
  ```
* **Key Fields:** Detailed incident summaries with standardized VERIS 4A taxonomy tags (Hacking, Malware, Social, Misuse).

---

## 3. Peer-Reviewed NLP Benchmark Corpora

### 3.1 DIFrauD Benchmark (Hugging Face)
* **Hugging Face Hub:** https://huggingface.co/datasets/difraud/difraud
* **Python Access:**
  ```python
  from datasets import load_dataset
  dataset = load_dataset("difraud/difraud")
  ```
* **Key Fields:** 95,854 standardized samples across 7 deception domains (phishing emails, job scams, smishing SMS).
