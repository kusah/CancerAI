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

# Sprint 5 – Random Forest Benchmarking

## Date
(Enter today's date)

## Completed

- Added Random Forest classifier to the reusable training pipeline.
- Implemented automatic model benchmarking.
- Added training time tracking.
- Generated Random Forest evaluation reports.
- Generated Random Forest confusion matrix (CSV & PNG).
- Saved Random Forest model.
- Created and updated `model_comparison.csv`.

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 98.14% |
| Decision Tree | 95.65% |
| Random Forest | 98.76% |

## Observations

- Random Forest achieved the highest accuracy so far.
- Decision Tree performance improved when using an ensemble approach.
- The benchmarking framework now updates automatically for every trained model.


## Sprint 6 – Hyperparameter Tuning

### Completed
- Created reusable tuning pipeline (`tune.py`)
- Integrated GridSearchCV with 5-fold cross-validation
- Tuned Random Forest hyperparameters
- Saved tuned model and tuning results

### Observation
- Default Random Forest hyperparameters achieved the best cross-validation performance (99.69%), indicating that the baseline configuration was already optimal among the tested combinations.

## Sprint 7 – XGBoost Hyperparameter Tuning

### Completed
- Extended reusable tuning pipeline for XGBoost.
- Performed GridSearchCV with 5-fold cross-validation.
- Saved tuned model and tuning reports.
- Compared tuned XGBoost with tuned Random Forest.

### Best Parameters
- learning_rate = 0.1
- max_depth = 3
- n_estimators = 100
- subsample = 0.8

### Best CV Accuracy
99.06%