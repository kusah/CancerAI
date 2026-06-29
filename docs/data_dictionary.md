# Data Dictionary

## Dataset

Gene Expression Cancer RNA-Seq Dataset

---

## Feature Dataset (data.csv)

| Column | Description |
|---------|-------------|
| Unnamed: 0 | Sample Identifier |
| gene_0 - gene_20530 | RNA-Seq Gene Expression Values |

Total Samples: 801

Total Features: 20,531

---

## Label Dataset (labels.csv)

| Column | Description |
|---------|-------------|
| Unnamed: 0 | Sample Identifier |
| Class | Cancer Type |

---

## Processed Dataset

| Column | Description |
|---------|-------------|
| Sample_ID | Unique Sample Identifier |
| gene_0 - gene_20530 | Gene Expression Values |
| Cancer_Type | Encoded Target Label |

---

## Cancer Label Mapping

| Label | Encoded Value |
|--------|--------------:|
| BRCA | 0 |
| COAD | 1 |
| KIRC | 2 |
| LUAD | 3 |
| PRAD | 4 |

---

## Data Quality

Missing Values: 0

Duplicate Samples: 0

Gene Features: float64

Target Variable: Encoded Integer