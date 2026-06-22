# Financial Fraud Detection

> End-to-end binary classification pipeline for financial fraud detection, benchmarking six model families — Logistic Regression, Elastic Net, Decision Tree, Random Forest, SVM, and a Stacking Ensemble — on imbalanced financial transaction data. The project covers data engineering, model training with parallel computation, ROC-based comparison, and variable importance analysis.

---

## Overview

Financial fraud detection is one of the highest-stakes machine learning applications: false negatives (missed fraud) generate direct losses, while false positives (false alarms) erode customer trust. This project builds a full detection pipeline on a synthetic financial transactions dataset, with explicit attention to **class imbalance**, **feature engineering from timestamps**, **missing value imputation**, and **multi-model benchmarking**.

The final ensemble model combines the probability outputs of all five base models using a meta-learner (Elastic Net), capturing complementary signals that no single model exploits alone.

---

## Dataset

The full dataset (769MB) exceeds GitHub's file size limit and is not included in this repository. A sample of 10,000 rows (`financial_data_sample.csv`) is provided for reference and to allow running the script locally on a reduced scale. The full dataset is available upon request.

Synthetic financial transaction dataset containing features including:

- Transaction amount, type, and payment channel
- Sender/receiver account identifiers
- Timestamp (converted to `hour_of_day` feature)
- Time since last transaction (minutes, with NA imputation)
- Geographic anomaly score
- Behavioral risk signals
- Binary target: `is_fraud` (highly imbalanced — fraud is a rare event)

---

## Methodology

### Feature Engineering & Preprocessing

| Step | Detail |
|------|--------|
| Variable selection | Dropped `transaction_id`, `ip_address`, `device_hash` (identifiers with no predictive signal); dropped `fraud_type` (leaks the target) and `location` (proxied by `geo_anomaly_score`) |
| Timestamp decomposition | Extracted `hour_of_day` from ISO datetime; dropped raw timestamp |
| Missing value imputation | `time_since_last_transaction`: imputed first by user-level median (per `sender_account`), then by global median for remaining NAs; accounts dropped post-imputation |
| Discretization | `time_since_last_transaction` binned into 6 temporal categories (<1 min, 1–5 min, ..., 60+ min) |
| Class balancing | Undersampled non-fraud class to 1:1 ratio in training set; test set left unbalanced to reflect real-world conditions |

### Models Trained

All models were trained with 5-fold cross-validation optimizing AUC-ROC.

| Model | Implementation | Notes |
|-------|---------------|-------|
| Logistic Regression | `glm` (binomial) | Baseline interpretable model |
| Elastic Net | `glmnet` | Grid search over α ∈ [0,1] and λ ∈ 10⁻³–10⁰; automatic feature selection |
| Decision Tree | `rpart` | Tuned complexity parameter; visualized final tree |
| Random Forest | `rf` (randomForest) | Grid over `mtry` ∈ {3, 5, 7}; parallelized over 4 cores via `doParallel` |
| SVM (Linear) | `svmLinear` | Trained on random subsample (10K rows) for computational tractability; grid over C ∈ {0.01, 0.1, 1, 10} |
| Stacking Ensemble | `glmnet` meta-learner | Level-1 predictions from all 5 base models used as features; meta-model is Elastic Net |

### Evaluation
- ROC curves plotted simultaneously for all 6 models (`viridis` color scale)
- AUC computed for each model; best model identified programmatically
- Resampled performance summary via `caret::resamples` with boxplots
- Top-5 variable importance from Elastic Net (most interpretable regularized model)

---

## Key Visualizations

- **Multi-model ROC comparison** — 6 curves on a single plot with color-coded legend
- **Variable importance** — Top 5 predictors from Elastic Net (bar chart)
- **Fraud by time interval** — Proportion of fraudulent transactions per `time_since_last_transaction` category
- **Fraud heatmap** — Fraud rate by hour-of-day × payment channel (identifies high-risk windows)
- **Decision tree diagram** — Interpretable visual of the rpart final model

---

## Key Findings

- Fraud transactions show elevated rates in **off-hours windows** (00:00–04:00) and in specific **payment channels**, consistent with real-world fraud patterns.
- **Time since last transaction** is among the most discriminating features: fraud is disproportionately concentrated in transactions occurring within seconds or minutes of the previous one (velocity patterns).
- The **Stacking Ensemble** achieved the highest AUC, with the Random Forest as the best single-model baseline.
- The Elastic Net automatically zeroed out several low-signal features, leaving a compact and interpretable coefficient set.

---

## Tech Stack

| Package | Role |
|---------|------|
| `caret` | Model training, CV, evaluation framework |
| `glmnet` | Elastic Net (base + meta-learner) |
| `rpart` / `rpart.plot` | Decision Tree |
| `randomForest` | Random Forest |
| `kernlab` | SVM |
| `pROC` | ROC / AUC |
| `doParallel` | Parallel computation for RF |
| `ggplot2`, `viridis`, `scales` | Visualization |
| `tidyverse` | Data manipulation |

---

## Project Structure

```
financial-fraud-detection/
├── analysis.R                      # Full pipeline script
└── financial_data_sample.csv       # Sample dataset (10,000 rows — see Data section)
```

---

## How to Run

```r
install.packages(c("tidyverse", "caret", "glmnet", "rpart", "rpart.plot",
                   "randomForest", "e1071", "pROC", "kernlab", "viridis",
                   "doParallel", "ggplot2", "scales"))

source("analysis.R")
```

> Note: Random Forest training uses `doParallel` with 4 cores. Adjust `makeCluster(n)` based on your machine.

---

## Skills Demonstrated

`Fraud Detection` · `Binary Classification` · `Ensemble Learning` · `Stacking` · `Elastic Net` · `Random Forest` · `SVM` · `Feature Engineering` · `Class Imbalance` · `Parallel Computing` · `ROC / AUC` · `R` · `caret`
