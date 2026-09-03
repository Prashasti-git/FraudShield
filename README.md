# FraudShield — ML-Based Banking Fraud Detection

A comparative study of classification algorithms for detecting fraudulent
banking transactions on the FraudShield_Banking_Data dataset
(50,000 transactions, 25 features).

## Project structure

FraudShield/
├── data/
│   ├── raw/            # Original, untouched CSV goes here
│   └── processed/       # Cleaned / split data saved between notebooks
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_cleaning.ipynb
│   ├── 04_features.ipynb
│   ├── 05_split.ipynb
│   ├── 06_unsupervised.ipynb
│   ├── 07_baseline_models.ipynb
│   ├── 08_imbalance.ipynb
│   ├── 09_model_comparison.ipynb
│   └── 10_feature_importance.ipynb
├── models/              # Saved trained models (.pkl / .joblib)
├── reports/
│   └── figures/         # Exported plots for the writeup / slides
├── app/                 # Deployment code (Flask/FastAPI/Streamlit)
├── requirements.txt
└── README.md

## Setup

    python -m venv venv
    source venv/bin/activate      # (venv\Scripts\activate on Windows)
    pip install -r requirements.txt
    jupyter lab

## Workflow

Notebooks are numbered in the order they should be run. Each phase's
output (cleaned data, splits, trained models) is saved to disk so later
notebooks don't need to repeat earlier steps.
