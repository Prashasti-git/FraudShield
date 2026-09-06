from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
CORS(app)
# Load everything once, at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'champion_model.pkl'))
threshold = joblib.load(os.path.join(MODEL_DIR, 'champion_threshold.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

BOOL_COLS = ['is_foreign_transaction', 'ip_country_mismatch', 'billing_shipping_mismatch',
             'is_ai_generated_scam_attempt', 'used_vpn', 'is_new_merchant']
CAT_COLS = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']


def preprocess(transaction: dict) -> pd.DataFrame:
    df = pd.DataFrame([transaction])

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df_encoded = pd.get_dummies(df, columns=CAT_COLS)

    # Ensure every training column exists, in the exact same order;
    # missing dummy columns (categories not present in this single request) become 0
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns=feature_columns)
    return df_scaled


@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.get_json()
        if transaction is None:
            return jsonify({'error': 'No JSON body provided'}), 400

        X = preprocess(transaction)
        fraud_probability = model.predict_proba(X)[0, 1]
        is_fraud = bool(fraud_probability >= threshold)

        return jsonify({
            'is_fraud': is_fraud,
            'fraud_probability': round(float(fraud_probability), 4),
            'threshold_used': round(float(threshold), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)