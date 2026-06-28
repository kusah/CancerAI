# CancerAI Data Dictionary

## Dataset Overview

This project uses a TCGA-derived RNA-Seq gene expression dataset for multi-class cancer classification.

---

## Feature Dataset (`data.csv`)

| Column | Description |
|---------|-------------|
| Unnamed: 0 | Sample Identifier |
| gene_0 – gene_20530 | RNA-Seq gene expression values |

Total Samples: **801**

Total Gene Features: **20,531**

---

## Label Dataset (`label.csv`)

| Column | Description |
|---------|-------------|
| Unnamed: 0 | Sample Identifier |
| Class | Cancer Type |

---

## Cancer Classes

| Label | Cancer |
|--------|---------|
| BRCA | Breast Cancer |
| COAD | Colon Adenocarcinoma |
| KIRC | Kidney Renal Clear Cell Carcinoma |
| LUAD | Lung Adenocarcinoma |
| PRAD | Prostate Adenocarcinoma |

---

## Data Quality

- Missing Values: 0
- Duplicate Samples: 0
- Gene Features: Numeric (`float64`)
- Sample ID: String
- Target Variable: Categorical

---

## Planned Preprocessing

- Rename `Unnamed: 0` to `Sample_ID`
- Merge `data.csv` and `label.csv`
- Encode cancer labels
- Save processed dataset