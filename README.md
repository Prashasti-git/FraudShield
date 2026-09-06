# FraudShield — Credit Card Fraud Detection

A supervised machine learning project that detects fraudulent credit card transactions using a comparative study of classification algorithms, with a focus on handling severe class imbalance the right way — from raw data to a deployed, working API.

## Live Demo

Try it yourself: **[fraudshield-api-qd6q.onrender.com](https://fraudshield-api-qd6q.onrender.com)**

*(Free-tier hosting — the first request after a period of inactivity may take 30-60 seconds to wake up.)*

## Overview

Banking and card fraud costs billions annually, and the core challenge isn't just building a classifier — it's building one that works when fraud makes up less than 2% of all transactions. This project walks through the full lifecycle of a fraud detection system: exploratory analysis, feature engineering, imbalance handling, model comparison, threshold tuning, and deployment.

Rather than chasing accuracy — which is a misleading metric on imbalanced data — this project centers evaluation on **Precision, Recall, F1-Score, and ROC-AUC**, and explicitly investigates the trade-off between catching fraud and disrupting legitimate customers.

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
│   ├── 06_baseline_models.ipynb    # Individual classifiers, default params
│   ├── 07_imbalance.ipynb          # SMOTE vs. class-weighting comparison
│   ├── 08_model_comparison.ipynb   # Full comparison, tuning, cross-validation
│   └── 09_feature_importance.ipynb # Interpretability, tied back to EDA
├── models/                     # Saved champion model, scaler, threshold, feature columns
├── reports/
│   └── figures/                 # Exported plots
├── app/
│   ├── app.py                   # Flask REST API
│   └── templates/
│       └── index.html           # Dark-themed interactive interface
├── requirements.txt
└── README.md
```

## Methodology

1. **Exploratory Data Analysis** — identified which features actually correlate with fraud. Risk flags like `is_ai_generated_scam_attempt` and `ip_country_mismatch` showed 4-6x higher fraud rates than baseline; `cvv_retry_count`, `velocity_score`, and `merchant_risk_score` were the strongest numeric predictors.
2. **Outlier Analysis** — rather than blindly removing statistical outliers, fraud rate was compared *within* outliers vs. the overall dataset for every numeric column. Outliers in fraud-correlated features showed 3-6x higher fraud rates, confirming they were genuine signal, not noise — so no outlier removal was performed.
3. **Feature Engineering** — categorical variables one-hot encoded, boolean flags normalized to 0/1, numeric features standardized (scaler fit only on training data to avoid leakage), yielding 49 final features.
4. **Class Imbalance Handling** — SMOTE and `class_weight='balanced'` were compared directly across every model rather than assuming one approach fits all. Finding: the best strategy is model-dependent — Logistic Regression's untouched baseline outperformed both resampling techniques, while SVM and KNN required imbalance handling just to produce usable predictions at all.
5. **Model Comparison** — Logistic Regression, Decision Tree, KNN, SVM, Random Forest, and XGBoost were evaluated on identical train/test splits using Precision, Recall, F1, and ROC-AUC, with hyperparameter tuning (GridSearchCV) and 5-fold stratified cross-validation.
6. **Threshold Tuning** — the default 0.5 classification threshold was tested explicitly against the F1-optimal threshold for each model. This turned out to be the single most impactful lever in the whole project — more impactful than imbalance handling or hyperparameter tuning combined.
7. **Feature Importance** — the champion model's coefficients were compared back against the original EDA findings to confirm consistency, and to surface predictors (like `auth_method`) that EDA's initial pass hadn't fully explored.

## Results

**Champion model: Logistic Regression** (`C=10`, `penalty='l1'`, `solver='liblinear'`, decision threshold = 0.267)

| Metric | Score |
|---|---|
| F1-Score | 0.516 |
| ROC-AUC | 0.9492 |
| Precision | 0.57 |
| Recall | 0.47 |

Out of 68 real fraud cases in the test set, the model correctly identifies 32 (47% recall) while maintaining 57% precision on its fraud flags — a defensible trade-off given the dataset's rarity of fraud (1.7%) and the modest strength of individual predictors.

**Full model comparison (best threshold-tuned F1 per model):**

| Model | Approach | F1 |
|---|---|---|
| **Logistic Regression** | baseline + tuned threshold | **0.520** |
| XGBoost | SMOTE + tuned threshold | 0.354 |
| SVM | class_weight + tuned threshold | 0.321 |
| XGBoost | scale_pos_weight + tuned threshold | 0.300 |
| Random Forest | SMOTE + tuned threshold | 0.288 |
| Random Forest | class_weight + tuned threshold | 0.256 |
| Decision Tree | baseline | 0.214 |
| KNN | SMOTE + tuned threshold | 0.147 |

**Key findings:**
- **Threshold tuning mattered more than anything else.** Logistic Regression's F1 jumped from 0.330 to 0.520 purely by moving the decision threshold from 0.5 to 0.236 — no resampling or reweighting involved.
- **Imbalance-handling strategy is model-dependent, not universal.** SVM and KNN were completely broken at baseline (F1 = 0, predicting only the majority class) and needed SMOTE/class_weight to function. Decision Tree performed *worse* with imbalance handling than without. Logistic Regression needed none at all.
- **Hyperparameter tuning added almost nothing** beyond what threshold tuning alone achieved (0.516 vs. 0.520) — confirming the model wasn't struggling with its internal parameters, but with where the decision boundary was drawn.
- **Feature importance strongly validated the EDA**, and also surfaced `auth_method` (No Authentication / PIN / Biometric) as a major predictor that the initial exploratory pass hadn't fully investigated — a useful reminder that model interpretation can catch gaps in earlier analysis.

## Deployment

The champion model is served via a Flask REST API with a custom dark-themed web interface:

- **`POST /predict`** — accepts a JSON transaction and returns fraud probability + verdict
- **`GET /health`** — health check
- **`GET /`** — interactive web interface with a live gauge visualization

Deployed on [Render](https://render.com) at **[fraudshield-api-qd6q.onrender.com](https://fraudshield-api-qd6q.onrender.com)**.

### Example request

```bash
curl -X POST https://fraudshield-api-qd6q.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount_usd": 450.00,
    "account_balance_usd": 1200.00,
    "merchant_category": "Crypto Exchange",
    "card_type": "Credit",
    "card_age_months": 24,
    "is_new_merchant": 0,
    "merchant_risk_score": 0.8,
    "channel": "Online",
    "device_type": "Mobile",
    "auth_method": "No Authentication",
    "used_vpn": 1,
    "hours_since_last_txn": 0.5,
    "txn_count_last_24h": 8,
    "velocity_score": 0.9,
    "cvv_retry_count": 3,
    "prior_disputes": 1,
    "is_foreign_transaction": 1,
    "ip_country_mismatch": 1,
    "billing_shipping_mismatch": 1,
    "is_ai_generated_scam_attempt": 0,
    "time_of_day_hour": 3,
    "day_of_week": 6,
    "customer_age": 29,
    "distance_from_home_km": 350.0
  }'
```

Response:
```json
{
  "fraud_probability": 0.994,
  "is_fraud": true,
  "threshold_used": 0.2672
}
```

## Tech Stack

- **Language**: Python 3.13
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Imbalance handling**: imbalanced-learn (SMOTE)
- **Backend**: Flask, Flask-CORS, Gunicorn
- **Environment**: Jupyter Lab
- **Deployment**: Render

## Local Setup

```bash
git clone https://github.com/Prashasti-git/FraudShield.git
cd FraudShield
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Place the dataset CSV in `data/raw/` before running the notebooks. Notebooks are numbered in the order they should be executed — each phase's output is saved to `data/processed/` or `models/` so later notebooks don't repeat earlier work.

To run the API and interface locally:
```bash
cd app
python3 app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Future Work

- SHAP/LIME integration for per-prediction explainability
- Cost-sensitive learning incorporating the real-world asymmetry between false positives and false negatives
- Precision-recall curve analysis alongside ROC-AUC (more informative at this fraud rate)
- Probability calibration checks
- Persistent/authenticated API access if used beyond a demo context

## Author

Prashasti

---

*This project treats fraud detection as a supervised classification problem, deliberately excluding unsupervised/clustering approaches to keep the scope focused — clustering-based anomaly detection is planned as a separate project.*
