# Credit Card Fraud Detection (Tabular ML Project)

## 1. Project overview

This project applies supervised machine learning to detect potentially fraudulent credit card transactions.  
Given historical transaction data with features such as amount and anonymized customer attributes, the goal is to predict whether a new transaction is **fraud** or **not fraud** so that high‑risk operations can be flagged for further review.

This is a **binary classification** problem (fraud = 1, non‑fraud = 0).

## 2. Dataset

- Source: Credit Card Fraud Detection dataset on Kaggle  
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- Description:
  - Transactions made by European cardholders over two days.
  - 284,807 transactions with 492 frauds.
  - Strongly imbalanced dataset: frauds are a very small fraction of all transactions.
  - Features are mostly anonymized numeric values (V1, V2, ..., V28), plus `Time`, `Amount`, and target column `Class`.

## 3. Objectives

In this project I aim to:

1. Understand the structure and characteristics of the dataset through **exploratory data analysis (EDA)**.
2. Build a first **baseline classification model** using classical ML algorithms (e.g., Logistic Regression).
3. Evaluate the model with metrics suitable for imbalanced classification:
   - Precision, recall, F1‑score
   - Confusion matrix
   - ROC–AUC
4. Prepare the project for future improvements:
   - Better handling of class imbalance.
   - More advanced models (e.g., tree‑based methods).
   - API and deployment in later weeks.

## 4. Project structure

Planned structure:

```text
ml-tabular-project/
├── data/              # Raw data (not committed if large)
├── notebooks/
│   ├── 01_baseline.ipynb   # First baseline model
│   └── 02_eda.ipynb        # Exploratory data analysis
├── src/               # Python modules (to be filled in later)
├── README.md
└── requirements.txt   # Project dependencies (to be added)

## Current Progress (Day 4 - Week 1)

### Dataset Statistics

- **Total transactions**: 284,807 [web:14][web:15]
- **Fraud cases (Class 1)**: 492 (0.17%) [web:14][web:15]
- **Non-fraud cases (Class 0)**: 284,315 (99.83%) [web:14][web:15]
- **Key challenge**: The dataset is extremely imbalanced; fraud is very rare.

### Baseline Model Results

**Model**: Logistic Regression (default settings, no explicit class balancing)

**Performance on Class 1 (Fraud):**

- Precision: **0.84** (≈ 0.8378)
- Recall: **0.63** (≈ 0.6327)
- F1-Score: **0.72** (≈ 0.7209)
- Overall Accuracy: **99.92%** (≈ 0.9992)

**Key insight**:  
Accuracy is very high mainly because 99.83% of transactions are non-fraud, so accuracy is **misleading** for this imbalanced problem. The main issue is **recall**: the model still **misses about 37% of fraud cases**, which is risky in a real fraud detection system. [web:17][web:21]

### EDA Key Findings

1. **Class imbalance**  
   - Only 0.17% of transactions are fraud (492 out of 284,807). [web:14][web:15]
   - Most models will default to predicting “non-fraud”, so special techniques are needed (class weighting, resampling, etc.). [web:17][web:21]

2. **Amount & Time**  
   - Very weak linear correlation with fraud (near zero), so they cannot separate fraud vs non-fraud on their own. [web:21]
   - Fraud transactions appear over a similar amount range as legitimate ones (no simple “high amount = fraud” rule).

3. **V features (PCA-transformed)**  
   - Several V features show clear separation between fraud and non-fraud in histograms.
   - **V14, V12, V17** show strong differences in distribution between Class 0 and Class 1.
   - **V10, V4, V16** show some separation but are weaker.
   - These features are likely to be very important for modeling. [web:21]

4. **Correlation insights**  
   - V14 has a moderate (negative) correlation with Class, confirming it as a strong predictor. [web:21]
   - V12 and V17 also show noticeable correlation with Class.
   - Amount has very weak correlation with Class, matching the earlier EDA finding. [web:21]

### Next Steps (Days 5–7)

- Implement class imbalance handling:
  - Try class weighting in Logistic Regression.
  - Experiment with SMOTE (oversampling) and undersampling the majority class. [web:17][web:22]
- Train more powerful models:
  - Random Forest, XGBoost, and compare them with the baseline logistic regression. [web:21]
- Use EDA insights:
  - Focus on strong features such as V14, V12, V17 (and possibly V10, V16) in feature engineering.
- **Target for Week 2**:
  - Recall > 75% for fraud, while keeping precision > 80%.
