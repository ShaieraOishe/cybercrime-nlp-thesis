# Comprehensive Thesis Research Roadmap & Methodology

**Thesis Title:** *Automated Cybercrime Complaint Classification and Semantic Retrieval: A Comparative Study of Traditional NLP and Transformer-Based Models*

---

## 🎯 Phase 1: Problem Formulation & Dataset Strategy

### 1.1 Objective Definition
- **Dual-Task Architecture:**
  1. **Supervised Classification:** Multi-class (or multi-label) classification predicting cybercrime typology and triage priority from victim narrative descriptions.
  2. **Information Retrieval (Semantic Search):** Nearest-neighbor retrieval to identify related complaints (same campaign, shared modus operandi, or duplicate cases).
- **Core Research Novelty & Angle:**
  - Most existing works focus solely on binary detection (phishing vs. legitimate, cyberbullying vs. normal). This research comprehensively evaluates multi-class real-world categorization coupled with semantic case retrieval.
  - Rigorous benchmarking of domain-adapted models (SecBERT, CyBERT) vs. general models (DeBERTa-v3, RoBERTa) vs. high-speed traditional baselines (Linear SVM, LightGBM) under imbalanced, long-tail class distributions.

### 1.2 Dataset Strategy & Sources
Cybercrime complaint texts require handling privacy (PII) and availability challenges:
1. **Public Benchmarks:**
   - **Consumer Financial Protection Bureau (CFPB) Complaints:** Millions of consumer complaints with narrative texts, categorized by issue (Identity Theft, Credit Card Fraud, Money Transfer Scam, Cyber Extortion). Fully open & public.
   - **Cyberbullying & Hate Speech corpora:** Twitter/Reddit/Kaggle cyberbullying datasets.
   - **Phishing & Scam corpora:** Nazario phishing corpus, SpamAssassin, SMS scam datasets, Kaggle cybercrime datasets.
2. **Synthetic / Augmented Datasets:**
   - LLM-generated incident reports conditioned on MITRE ATT&CK / FBI IC3 annual report case summaries to augment under-represented classes (e.g. Ransomware, BEC / Business Email Compromise).
3. **Data Governance & Ethics:**
   - Anonymization protocol: Regex + NER to replace personally identifiable information (PII) with generic tokens (`<EMAIL>`, `<PHONE>`, `<CARD_NUMBER>`, `<CRYPTO_WALLET>`).

---

## 🧹 Phase 2: Exploratory Data Analysis (EDA) & Preprocessing

- **Distribution Analysis:** Class frequencies, label co-occurrence matrix, class imbalance ratio.
- **Narrative Length Profiling:** Token count distributions (determining `max_seq_length` for transformer tokenizers vs. classical n-grams).
- **Lexical Analysis:** Most informative n-grams per class via TF-IDF log-odds ratio and Chi-square feature selection.
- **Train / Validation / Test Splitting:**
  - Stratified 70/15/15 split (or 80/10/10) to preserve minority class ratios.
  - Ensuring no data leakage across splits prior to vectorizer fitting or threshold calibration.

---

## 🤖 Phase 3: Automated Classification Benchmark

### 3.1 Model Progression
1. **Baselines (Zero/Naive baseline):**
   - Stratified Dummy Classifier / Majority Class baseline.
2. **Traditional NLP + Machine Learning:**
   - Feature extractors: Unigram + Bigram TF-IDF (sublinear TF scaling, max 10k-20k features).
   - Classifiers: Multinomial Naive Bayes, L2-regularized Logistic Regression, Linear Support Vector Classifier (LinearSVC), and Gradient Boosting (LightGBM/XGBoost).
3. **Classical Deep Learning (Neural Baselines):**
   - Pre-trained GloVe/FastText word embeddings + Bidirectional LSTM (BiLSTM) with Self-Attention or 1D-CNN.
4. **General Pre-trained Transformers:**
   - DistilBERT-base-uncased (lightweight, low-latency baseline).
   - BERT-base-uncased & RoBERTa-base (standard robust encoders).
   - DeBERTa-v3-base (disentangled attention, state-of-the-art encoder for NLU).
5. **Domain-Adapted Cybersecurity Transformers:**
   - `JackBAI/SecBERT` or `CyBERT` (pre-trained on cybersecurity corpora, CVEs, and technical threat feeds).

### 3.2 Training & Optimization Protocol
- Class weight adjustments (e.g., inverse class frequency or Focal Loss) to address heavy class imbalance.
- Mixed precision training (FP16) on GPU / MPS.
- Learning rate warmup with linear/cosine decay (`AdamW`, $lr \in [2\times 10^{-5}, 5\times 10^{-5}]$).

---

## 🔍 Phase 4: Semantic Retrieval & Precedent Matching

### 4.1 Retrieval Paradigms
1. **Lexical / Sparse Search:**
   - BM25 (Okapi BM25) implemented via `rank-bm25` over tokenized complaint narratives.
2. **Dense Semantic Retrieval (Bi-Encoders):**
   - Sentence-Transformers: `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`.
   - Vector Indexing: FAISS (IndexFlatIP with normalized cosine embeddings or HNSW for scalable retrieval).
3. **Hybrid Retrieval (Sparse + Dense Fusion):**
   - Reciprocal Rank Fusion (RRF):
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
     where $k \approx 60$.
4. **Two-Stage Retrieval (Cross-Encoder Re-ranking):**
   - Retrieve Top-50 via Dense/Hybrid, then re-rank using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

---

## 📊 Phase 5: Empirical Comparison, Interpretability & Statistical Rigor

### 5.1 Metrics Suite
- **Classification:** Macro-Averaged Precision, Recall, and F1-Score; Weighted F1; Per-class F1; Confusion Matrix; ROC-AUC.
- **Retrieval:**
  - Mean Reciprocal Rank (MRR@10)
  - Mean Average Precision (MAP@k)
  - Normalized Discounted Cumulative Gain (NDCG@10)
  - Recall@k ($k \in \{1, 5, 10, 20\}$)
- **Operational Viability:**
  - Inference Latency: Batch size 1 latency (ms/sample) on CPU vs. GPU/MPS.
  - Model Footprint: Checkpoint size (MB) and peak memory usage during inference.

### 5.2 Statistical Significance Testing
- McNemar’s Test: To assess whether differences in classification accuracy/F1 between Transformer and Traditional baselines are statistically significant ($p < 0.05$).
- 95% Bootstrap Confidence Intervals: 1,000 bootstrap iterations over test predictions to establish confidence bands.

### 5.3 Model Interpretability
- **Traditional Models:** Top informative features (coefficients and feature importances).
- **Transformer Models:** Local Interpretable Model-agnostic Explanations (LIME) or Integrated Gradients to highlight narrative phrases that triggered the cybercrime classification.

---

## 📝 Phase 6: Thesis Structure & Milestone Timeline

- **Chapter 1:** Introduction, Problem Statement, Motivation & Research Questions.
- **Chapter 2:** Literature Review (Cybercrime landscape, NLP in Law Enforcement, Transformer evolution, Dense Information Retrieval).
- **Chapter 3:** Methodology (Data preparation, PII sanitization, Model Architectures, Retrieval Framework).
- **Chapter 4:** Experimental Setup (Datasets, Hyperparameters, Evaluation Metrics, Hardware & Software environment).
- **Chapter 5:** Results & Comparative Analysis (Classification benchmarks, Retrieval benchmarks, Efficiency trade-offs).
- **Chapter 6:** Discussion, Error Analysis & Interpretability (Case studies, failure modes, limitations).
- **Chapter 7:** Conclusion, Ethical Implications & Future Work.
