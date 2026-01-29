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
