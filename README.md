# FraudShield--Credit Card Fraud Detection

A supervised machine learning project that detects fraudulent credit card transactions using a comparative study of classification algorithms, with a focus on handling severe class imbalance the right way.

## Overview

Banking and card fraud costs billions annually, and the core challenge isn't just building a classifier it's building one that works when fraud makes up less than 2% of all transactions. This project walks through the full lifecycle of a fraud detection system: exploratory analysis, feature engineering, imbalance handling, model comparison, and (eventually) deployment.

Rather than chasing accuracy  which is a misleading metric on imbalanced data — this project centers evaluation on **Precision, Recall, F1-Score, and ROC-AUC**, and explicitly investigates the trade-off between catching fraud and disrupting legitimate customers.

## Dataset

**[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)** (uditjain13, Kaggle)

- 20,000 transactions, 26 columns
- Target: `is_fraud` (binary) — 1.70% fraud rate (339 fraud / 19,661 normal)
- No missing values, no duplicate rows
- Interpretable, named features across several categories:
  - **Monetary**: `amount_usd`, `account_balance_usd`
  - **Merchant/Card**: `merchant_category`, `card_type`, `card_age_months`, `is_new_merchant`, `merchant_risk_score`
  - **Channel/Auth**: `channel`, `device_type`, `auth_method`, `used_vpn`
  - **Behavioral**: `hours_since_last_txn`, `txn_count_last_24h`, `velocity_score`, `cvv_retry_count`, `prior_disputes`
  - **Risk flags**: `is_foreign_transaction`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`
  - **Time**: `time_of_day_hour`, `day_of_week`
  - **Demographic/Geo**: `customer_age`, `distance_from_home_km`

## Project Structure

```
FraudShield/
├── data/
│   ├── raw/                    # Original, untouched dataset
│   └── processed/              # Train/test splits, scaled data
├── notebooks/
│   ├── 01_data_overview.ipynb      # Initial inspection
│   ├── 02_eda.ipynb                # Univariate/bivariate/multivariate analysis
│   ├── 03_cleaning.ipynb           # Outlier detection & policy decisions
│   ├── 04_features.ipynb           # Encoding, scaling
│   ├── 05_split.ipynb              # Stratified train/test split
│   ├── 07_baseline_models.ipynb    # Individual classifiers, default params
│   ├── 08_imbalance.ipynb          # SMOTE vs. class-weighting comparison
│   ├── 09_model_comparison.ipynb   # Full comparison, tuning, cross-validation
│   └── 10_feature_importance.ipynb # Interpretability, tied back to EDA
├── models/                     # Saved trained models & scaler
├── reports/
│   └── figures/                 # Exported plots
├── app/                         # Deployment code (Flask/Streamlit)
├── requirements.txt
└── README.md
```

## Methodology

1. **Exploratory Data Analysis**-- identified which features actually correlate with fraud (risk flags like `is_ai_generated_scam_attempt` and `ip_country_mismatch` showed 4-6x higher fraud rates; `cvv_retry_count`, `velocity_score`, and `merchant_risk_score` were the strongest numeric predictors).
2. **Outlier Analysis** — rather than blindly removing statistical outliers, fraud rate was compared *within* outliers vs. the overall dataset for every numeric column. Outliers in fraud-correlated features showed 3-6x higher fraud rates, confirming they were genuine signal, not noise — so no outlier removal was performed.
3. **Feature Engineering** — categorical variables one-hot encoded, boolean flags normalized to 0/1, numeric features standardized (scaler fit only on training data to avoid leakage).
4. **Class Imbalance Handling** — SMOTE and `class_weight='balanced'` are compared directly rather than assuming SMOTE is automatically the right choice.
5. **Model Comparison** — Logistic Regression, Decision Tree, KNN, SVM, Random Forest, and XGBoost are evaluated on identical train/test splits using Precision, Recall, F1, and ROC-AUC, with hyperparameter tuning and stratified cross-validation.
6. **Threshold Tuning** — the default 0.5 classification threshold is examined explicitly, since optimal fraud-detection thresholds often differ from 0.5 depending on the cost trade-off between missed fraud and false alarms.
7. **Feature Importance** — the best-performing model's feature importances are compared back against the original EDA findings to confirm consistency.

## Tech Stack

- **Language**: Python 3.13
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Imbalance handling**: imbalanced-learn (SMOTE)
- **Environment**: Jupyter Lab

## Setup

```bash
git clone <your-repo-url>
cd FraudShield
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Place the dataset CSV in `data/raw/` before running the notebooks. Notebooks are numbered in the order they should be executed each phase's output is saved to `data/processed/` or `models/` so later notebooks don't repeat earlier work.

## Results

*(To be filled in once model comparison is complete final metrics table, best performing model, and key takeaways will go here.)*

## Future Work

- REST API deployment (Flask/FastAPI) for real time inference
- SHAP/LIME integration for per prediction explainability
- Cost-sensitive learning incorporating the real-world asymmetry between false positives and false negatives
- Precision-recall curve analysis (more informative than ROC-AUC at this fraud rate)
- Probability calibration checks

## Author

Prashasti

---

*This project treats fraud detection as a supervised classification problem, deliberately excluding unsupervised/clustering approaches to keep the scope focused clustering-based anomaly detection is planned as a separate project.*
