# Curated Dataset Guide for Cybercrime Complaint Classification & Retrieval

This document provides a comprehensive survey of viable, publicly accessible datasets across major research platforms (Government Data Portals, Hugging Face, Kaggle, Zenodo, and GitHub).

---

## 🏆 Top Recommendation: CFPB Consumer Complaint Database (Cyber & Fraud Slice)

* **Source:** [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/)
* **Mirrors:** Available on Kaggle (`selener/consumer-complaint-database`) and Hugging Face (`cfpb-consumer-complaints`).
* **License:** Public Domain (U.S. Government Work).
* **Total Volume:** Over 4 million complaints (>1.5 million with full unstructured victim narratives).
* **Cybercrime-Specific Subset:** Over 250,000+ narratives.
* **Relevant Categories (Issue & Sub-Issue):**
  1. *Identity Theft & Impersonation* (Account opened without consent, stolen SSN/credentials)
  2. *Unauthorized Electronic Fund Transfers & Wire Fraud* (Online banking breach, unauthorized crypto transfer)
  3. *Online Scams & Phishing* (Advance fee scam, impersonation of financial entity, fake investment)
  4. *Card Not Present (CNP) / Card Fraud* (Stolen card details used for online purchases)
* **Strengths:**
  - Real human victim narratives (50–500+ words), capturing genuine victim emotions and technical confusion.
  - Pre-sanitized and anonymized by federal regulators (`xxxx` masks), removing legal liability.
  - Highly recognized and cited in peer-reviewed NLP and forensic literature.
  - Perfect for **Semantic Retrieval**: Many complaints describe the exact same modus operandi, fraudulent website, or syndicate scheme.

---

## 🛡️ Top Incident & Threat Repositories

### 1. EuRepoC (European Repository of Cyber Incidents)
* **Source:** [eurepoc.eu](https://eurepoc.eu/) / Zenodo (DOI: [10.5281/zenodo.10058362](https://doi.org/10.5281/zenodo.10058362))
* **Volume:** ~2,500+ verified global cyber incidents.
* **Fields:** Unstructured narrative summary, attack vector (Ransomware, Data Theft/Breach, DDoS, Wiper, Phishing), threat actor attribution, target industry sector.
* **Best For:** High-level cyber incident classification and threat intelligence retrieval.

### 2. VCDB (VERIS Community Database)
* **Source:** [veriscommunity.net](http://veriscommunity.net/) / GitHub (`vz-risk/vcdb`)
* **Volume:** Over 8,000+ real-world cyber security incident narratives categorized under the VERIS framework.
* **Labels:** Threat Actions (Hacking, Social, Malware, Misuse, Physical, Error), Variety (Phishing, Ransomware, Brute force, SQLi).

---

## 🤖 NLP & Hugging Face Benchmarks

### 1. DIFrauD (Domain-Independent Fraud Detection Benchmark)
* **Source:** Hugging Face [`difraud/difraud`](https://huggingface.co/datasets/difraud/difraud)
* **Volume:** 95,854 text samples across 7 domains.
* **Domains:** Phishing emails, Job scams, SMS smishing, Fake news, Deceptive product reviews.
* **Format:** JSONL with standardized 80/10/10 train-val-test splits, pre-cleaned for Transformer tokenizers.

### 2. CyberGuard AI Incident Classification Dataset
* **Source:** GitHub [`dhruvldrp9/CyberGuard-AI-Hackathon`](https://github.com/dhruvldrp9/CyberGuard-AI-Hackathon)
* **Volume:** 31,000+ cyber incident reports.
* **Labels:** Hierarchical threat classes (general cyber threat vs. granular attack vector).

### 3. Zenodo Multiclass Social Engineering Threat Dataset
* **Source:** Zenodo (DOI: [10.5281/zenodo.15235123](https://doi.org/10.5281/zenodo.15235123))
* **Volume:** 624 curated messages.
* **Labels:** Phishing, Malware, Scareware, Baiting, Pretexting, Benign.
* **Best For:** Zero-shot / Few-shot evaluation or evaluation test bed.

---

## 👥 Social Cybercrime & Harassment Datasets

### 1. Cyberbullying Multi-Class Dataset
* **Source:** Kaggle (`saurabhshahane/cyberbullying-dataset`)
* **Volume:** ~47,000 text samples.
* **Labels:** Cyberstalking, gender-based harassment, religious hate, ethnic cyberbullying, non-cyberbullying.

---

## 💡 Recommended Thesis Corpus Formulation Strategy

For a publication-grade thesis, we recommend constructing **`CyberComplaint-Bench`**:
1. **Primary Corpus (80%):** A curated 30,000–50,000 complaint subset of the **CFPB Database**, filtered for cybercrime typologies (Identity Theft, Online Scam/Phishing, Unauthorized Wire/Crypto Transfer, Account Takeover).
2. **Supplemental Real Incidents (20%):** Samples from **EuRepoC** or **CyberGuard** to incorporate technical attack vectors (Ransomware, Hacking, Malware).
3. **Retrieval Benchmark:** A gold-standard test set of 200–500 complaint queries mapped to known similar cases sharing the same Modus Operandi (M.O.).
