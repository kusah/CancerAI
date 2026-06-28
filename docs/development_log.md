# CancerAI Development Log

## Project
CancerAI – Explainable Multi-Cancer Classification using RNA-Seq Gene Expression

---

# Sprint 1 – Project Foundation

### Completed

- Created project structure
- Configured Python virtual environment
- Set up GitHub repository
- Implemented configuration module (`config.py`)
- Implemented reusable data loader (`data_loader.py`)
- Created dataset validator (`validator.py`)
- Created dataset statistics module (`statistics.py`)
- Selected TCGA-derived RNA-Seq dataset
- Downloaded dataset (`data.csv`, `label.csv`)
- Loaded dataset successfully
- Performed initial exploratory data analysis
- Generated cancer type distribution visualization

### Key Findings

- Samples: 801
- Gene Features: 20,531
- Cancer Classes: 5
- Missing Values: 0
- Duplicate Samples: 0

### Next Sprint

- Preprocess dataset
- Merge feature and label datasets
- Rename columns
- Encode target labels
- Save processed dataset