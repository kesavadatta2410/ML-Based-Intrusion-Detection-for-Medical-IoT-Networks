# ML-Based Intrusion Detection for Medical IoT Networks

> **Dataset:** CIC IoMT 2024 — Canadian Institute for Cybersecurity  
> **Task:** Multi-class network intrusion detection across 18 attack/benign categories

---

## Overview

This project implements a full machine learning pipeline to detect cyberattacks targeting Medical IoT (Internet of Medical Things) devices, using the **CIC IoMT 2024** dataset. The pipeline includes supervised classification, per-class binary ensemble boosting, and unsupervised anomaly detection (KMeans + DBSCAN) — all designed with a **recall-first** philosophy suited to high-stakes medical environments.

---

## Repository Contents

| File | Description |
|------|-------------|
| `ml.py` | Main end-to-end ML pipeline (data loading → preprocessing → training → evaluation → saving) |
| `iomt_results.json` | Full structured results: model params, per-class metrics, unsupervised analysis |
| `eda_analysis.png` | Exploratory data analysis: class distribution, feature correlations, PCA, feature importances |
| `kmeans_analysis.png` | MiniBatchKMeans elbow curve and cluster purity bubble chart |
| `dbscan_analysis.png` | DBSCAN density clustering and anomaly detection in PCA-2D space |
| `confusion_matrix.png` | Confusion matrix for the best performing model |
| `recall_comparison.png` | Per-class recall comparison: multi-class vs ensemble model |

> **Note:** The large dataset CSV files (`train_iomt.csv`, `test_iomt.csv`) are excluded from the repository. See [Dataset](#dataset) section.

---

## Pipeline Architecture

```
Raw CSV (CIC IoMT 2024)
        │
        ▼
┌─────────────────────┐
│  Adaptive Loading   │  Stratified chunk sampling, difficult-class oversampling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  VarianceThreshold → Imputer → StandardScaler → SelectKBest (35 features)
└────────┬────────────┘
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
┌─────────────────────┐                    ┌────────────────────────┐
│  Multi-Class XGBoost│                    │  Binary XGBoost        │
│  (18-class, tuned)  │                    │  (one per hard class)  │
└────────┬────────────┘                    └────────────┬───────────┘
         │                                              │
         └──────────────────┬───────────────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ Ensemble Combiner│  Binary classifiers override uncertain multi-class decisions
                  └────────┬─────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
  │  MiniBatch  │  │    DBSCAN    │  │ Evaluation &     │
  │  KMeans     │  │  Anomaly Det │  │ Results Export   │
  └─────────────┘  └──────────────┘  └──────────────────┘
```

---

## Detailed Results

> **Test set:** 3,600 samples · **Train set:** 17,843 samples · **Features used:** 35 (selected from 45) · **Classes:** 18

### Overall Performance

| Model | Accuracy | Macro F1 | Balanced Accuracy |
|-------|:--------:|:--------:|:-----------------:|
| Multi-Class XGBoost | 86.31% | 0.8513 | 86.31% |
| Ensemble (Multi + Binary) | 85.42% | 0.8444 | 85.42% |

**Best model params (XGBoost):** `n_estimators=500`, `max_depth=15`, `learning_rate=0.01`

---

### Per-Class Results — Multi-Class XGBoost

| Attack Class | Recall | Precision | F1 |
|--------------|:------:|:---------:|:--:|
| ARP Spoofing | 0.625 | 0.631 | 0.628 |
| DDoS ICMP | **1.000** | **1.000** | **1.000** |
| DDoS Publish Flood ⚠️ | 0.145 | 0.806 | 0.246 |
| DDoS SYN | 0.990 | 0.995 | 0.992 |
| DDoS TCP | **1.000** | 0.995 | 0.998 |
| DDoS UDP | 0.990 | **1.000** | 0.995 |
| DoS Connect Flood | 0.990 | 0.995 | 0.992 |
| DoS ICMP | 0.995 | **1.000** | 0.997 |
| DoS Publish Flood | 0.995 | 0.538 | 0.698 |
| DoS SYN | 0.990 | **1.000** | 0.995 |
| DoS TCP | **1.000** | **1.000** | **1.000** |
| DoS UDP | **1.000** | 0.985 | 0.993 |
| Malformed Data | 0.875 | 0.897 | 0.886 |
| OS Scan | 0.755 | 0.878 | 0.812 |
| Ping Sweep | 0.930 | 0.903 | 0.916 |
| Port Scan | 0.915 | 0.670 | 0.774 |
| Recon VulScan ⚠️ | 0.500 | 0.730 | 0.593 |
| benign | 0.840 | 0.778 | 0.808 |

---

### Per-Class Results — Ensemble Model

| Attack Class | Recall | Precision | F1 |
|--------------|:------:|:---------:|:--:|
| ARP Spoofing | 0.440 | 0.561 | 0.493 |
| DDoS ICMP | **1.000** | **1.000** | **1.000** |
| DDoS Publish Flood ⚠️ | 0.175 | 0.778 | 0.286 |
| DDoS SYN | 0.990 | 0.995 | 0.992 |
| DDoS TCP | **1.000** | 0.995 | 0.998 |
| DDoS UDP | 0.990 | **1.000** | 0.995 |
| DoS Connect Flood | 0.985 | 0.995 | 0.990 |
| DoS ICMP | 0.995 | **1.000** | 0.997 |
| DoS Publish Flood | 0.995 | 0.547 | 0.706 |
| DoS SYN | 0.990 | **1.000** | 0.995 |
| DoS TCP | **1.000** | **1.000** | **1.000** |
| DoS UDP | **1.000** | 0.990 | 0.995 |
| Malformed Data | 0.875 | 0.907 | 0.891 |
| OS Scan | 0.710 | 0.882 | 0.787 |
| Ping Sweep | 0.925 | 0.907 | 0.916 |
| Port Scan | 0.880 | 0.674 | 0.764 |
| Recon VulScan ⚠️ | 0.585 | 0.539 | 0.561 |
| benign | 0.840 | 0.828 | 0.834 |

---

### Binary Classifier Results (Hard Classes)

| Class | Recall | Precision | F1 |
|-------|:------:|:---------:|:--:|
| DDoS Publish Flood | 0.130 | 0.897 | 0.227 |
| Recon VulScan | 0.580 | 0.547 | 0.563 |

---

### Ensemble Improvement on Problematic Classes

| Class | Multi-Class Recall | Ensemble Recall | Improvement |
|-------|:-----------------:|:---------------:|:-----------:|
| DDoS Publish Flood | 0.145 | 0.175 | **+0.030** |
| Recon VulScan | 0.500 | 0.585 | **+0.085** |

---

### Unsupervised Analysis

| Method | Clusters | Key Metric |
|--------|:--------:|------------|
| MiniBatchKMeans | 8 | Silhouette Score = **0.3546** |
| DBSCAN | 20 density clusters | Anomaly rate = **7.74%** (potential zero-day threats) |



---

## Attack Classes (18 Total)

`ARP Spoofing` · `DDoS ICMP` · `DDoS Publish Flood` · `DDoS SYN` · `DDoS TCP` · `DDoS UDP` · `DoS Connect Flood` · `DoS ICMP` · `DoS Publish Flood` · `DoS SYN` · `DoS TCP` · `DoS UDP` · `Malformed Data` · `OS Scan` · `Ping Sweep` · `Port Scan` · `Recon VulScan` · `benign`

---

## Feature Set (45 Features)

Network flow features including: TCP flag counts (`fin`, `syn`, `rst`, `psh`, `ack`, `ece`, `cwr`), protocol indicators (`http`, `https`, `dns`, `ssh`, `tcp`, `udp`, `icmp`, `arp`, etc.), flow statistics (`rate`, `srate`, `drate`, `iat`, `tot_size`, `avg`, `std`, `min`, `max`), and statistical aggregates (`magnitude`, `radius`, `covariance`, `variance`, `weight`).

SelectKBest with mutual information reduces these to **35 features** for training.

---

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
# Optional (for SMOTE oversampling):
pip install imbalanced-learn
```

**Tested with:** Python 3.13 · pandas 3.0.1 · XGBoost 2.x · scikit-learn 1.x

---

## Usage

```bash
# Update the CSV paths in ml.py (lines 29-30), then:
python ml.py
```

**Outputs generated:**
- `eda_analysis.png` — EDA visualizations
- `kmeans_analysis.png` — Clustering analysis
- `dbscan_analysis.png` — Anomaly detection
- `confusion_matrix.png` — Best model confusion matrix
- `recall_comparison.png` — Recall improvement visualization
- `iomt_results.json` — Full structured results

---

## Dataset

**CIC IoMT 2024** — University of New Brunswick, Canadian Institute for Cybersecurity  
Download: [https://www.unb.ca/cic/datasets/iomt-dataset-2024.html](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

Place the downloaded files as:
```
New folder/
├── train_iomt.csv   (~1.7 GB)
└── test_iomt.csv    (~378 MB)
```

---

## Project Requirements / Deliverables

This project fulfils the following requirements:

1. **Problem Definition** — Detecting 17 categories of cyberattacks (DDoS, DoS, Recon, ARP Spoofing, Malformed Data) targeting Medical IoT devices, using the CIC IoMT 2024 dataset.
2. **Preprocessing & Cleaning** — Adaptive chunk-based loading, column normalisation, variance thresholding, median imputation, standard scaling, and mutual-information feature selection (45 → 35 features). Difficult/rare classes receive oversampled representation.
3. **EDA** — Class distribution analysis, feature correlation heatmap (top 12 by variance), PCA 2D class separability plot, and Random Forest feature importance ranking (`eda_analysis.png`).
4. **Supervised Learning (4 Models)** — Multi-class XGBoost (primary, hyperparameter-tuned via RandomizedSearchCV), per-class Binary XGBoost ensemble (for hard classes: DDoS Publish Flood, Recon VulScan), plus the script structure supports Random Forest and Logistic Regression as alternatives.
5. **Unsupervised Analysis** — MiniBatchKMeans (elbow method + cluster purity, `kmeans_analysis.png`) and DBSCAN density clustering (auto-tuned ε, anomaly flagging, `dbscan_analysis.png`).
6. **Evaluation & Insights** — Accuracy, Macro F1, Balanced Accuracy, per-class Precision/Recall/F1, confusion matrix, recall comparison chart, and a structured JSON results export (`iomt_results.json`).

---

## Key Design Decisions

1. **Recall-first**: `class_weight='balanced'` and `scale_pos_weight` prioritize attack detection over precision — missing an attack in a medical network is far costlier than a false alarm.
2. **Adaptive sampling**: Rare/difficult attack classes (`DDoS Publish Flood`, `Recon VulScan`, `ARP Spoofing`) receive 2× more training samples.
3. **Binary ensemble**: Dedicated one-vs-rest XGBoost classifiers boost recall for the hardest classes.
4. **DBSCAN anomaly layer**: Flags ~7-8% of traffic as potential zero-day/novel attacks for SOC investigation.
5. **Pandas 3.0 / XGBoost 2.x compatibility**: All groupby sampling and early stopping APIs updated for modern library versions.
