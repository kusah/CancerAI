# CancerAI Development Log

## Sprint 1

### Completed

- Project structure created
- Virtual environment configured
- GitHub repository initialized
- Configuration module created
- Data loader implemented
- Dataset downloaded
- Initial EDA completed
- Dataset statistics generated
- Cancer distribution visualization created

---

## Sprint 2

### Completed

- Renamed dataset columns
- Merged feature and label datasets
- Encoded cancer labels using LabelEncoder
- Saved processed dataset
- Saved label encoder for deployment

---

## Current Progress

- Dataset Ready
- EDA Completed
- Preprocessing Completed

---

## Sprint 3 – Baseline Machine Learning

### Completed

- Loaded processed dataset
- Performed train-test split using stratified sampling
- Standardized gene expression features
- Trained Logistic Regression model
- Evaluated model performance

### Results

- Accuracy: 98.14%
- Macro F1-score: 0.97
- Weighted F1-score: 0.98

### Observation

The Logistic Regression model achieved excellent performance with only a few misclassifications between COAD and LUAD.

# Sprint 4 – Model Benchmarking Framework

## Date
1/7/2026

## Completed

- Refactored `train.py` into reusable functions.
- Created modular ML training pipeline.
- Added `load_dataset()`.
- Added `prepare_data()`.
- Added `split_data()`.
- Added `scale_data()`.
- Added `train_and_evaluate()`.
- Added `generate_reports()`.
- Added `save_model()`.
- Added Decision Tree classifier.
- Generated evaluation reports automatically.
- Saved Decision Tree model.
- Generated Decision Tree confusion matrix.

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 98.14% |
| Decision Tree | 95.65% |

## Outcome

The reusable training pipeline now supports multiple machine learning models with minimal code changes.