# webapp/app.py
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ===== MODEL LOAD (cached) =====
@st.cache_resource
def load_model():
    # Build safe path relative to this file
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model_xgb_rsv_bal.pkl"))
    # Fallback to alternative name (case-insensitive) if needed:
    if not os.path.exists(model_path):
        # try common alternative names
        alt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        for fname in os.listdir(alt) if os.path.exists(alt) else []:
            if fname.lower().startswith("model") and fname.lower().endswith((".pkl", ".joblib", ".sav")):
                model_path = os.path.join(alt, fname)
                break

    
    try:
        with open(model_path, "rb") as f:
            mdl = pickle.load(f)
        return mdl
    except Exception as e:
        st.error(f"Failed to load model. Check path and file. Error: {e}")
        return None

model = load_model()

st.title("💡 Customer Churn Prediction")
st.write("Fill in customer details below to predict churn.")

# ===== UI FIELDS =====
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

def tri_feature(label):
    # if no internet: choose No internet service automatically
    if internet_service == "No":
        return "No internet service"
    return st.selectbox(label, ["No", "Yes", "No internet service"])

online_security = tri_feature("Online Security")
online_backup = tri_feature("Online Backup")
device_protection = tri_feature("Device Protection")
tech_support = tri_feature("Tech Support")
streaming_tv = tri_feature("Streaming TV")
streaming_movies = tri_feature("Streaming Movies")

paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=50.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=1e7, value=500.0, step=0.1)

# ===== EXPECTED TRAINING SCHEMA (exact column names) =====
EXPECTED_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# ===== HELPER: build a single-row feature vector matching EXPECTED_COLS =====
def build_features():
    gender_map = {"Male": 0, "Female": 1}
    yn_map = {"No": 0, "Yes": 1}

    row = {c: 0 for c in EXPECTED_COLS}
    # basic features
    row.update({
        'gender': int(gender_map[gender]),
        'SeniorCitizen': int(yn_map[senior]),
        'Partner': int(yn_map[partner]),
        'Dependents': int(yn_map[dependents]),
        'tenure': int(tenure),
        'PhoneService': int(yn_map[phone_service]),
        'PaperlessBilling': int(yn_map[paperless_billing]),
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges),
    })

    # MultipleLines
    if multiple_lines == "No phone service":
        row['MultipleLines_No phone service'] = 1
    elif multiple_lines == "Yes":
        row['MultipleLines_Yes'] = 1

    # InternetService
    if internet_service == "Fiber optic":
        row['InternetService_Fiber optic'] = 1
    elif internet_service == "No":
        row['InternetService_No'] = 1

    # internet-dependent tri state features
    def set_tri(prefix, value):
        if value == "No internet service":
            col = f"{prefix}_No internet service"
            if col in row:
                row[col] = 1
        elif value == "Yes":
            col = f"{prefix}_Yes"
            if col in row:
                row[col] = 1

    set_tri("OnlineSecurity", online_security)
    set_tri("OnlineBackup", online_backup)
    set_tri("DeviceProtection", device_protection)
    set_tri("TechSupport", tech_support)
    set_tri("StreamingTV", streaming_tv)
    set_tri("StreamingMovies", streaming_movies)

    # Contract
    if contract == "One year":
        row['Contract_One year'] = 1
    elif contract == "Two year":
        row['Contract_Two year'] = 1

    # PaymentMethod
    if payment_method == "Credit card (automatic)":
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        row['PaymentMethod_Electronic check'] = 1
    elif payment_method == "Mailed check":
        row['PaymentMethod_Mailed check'] = 1

    # build DataFrame ensuring column order
    df = pd.DataFrame([row], columns=EXPECTED_COLS)

    # enforce dtypes
    int_cols = [c for c in EXPECTED_COLS if c not in ['MonthlyCharges', 'TotalCharges']]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    df[['MonthlyCharges', 'TotalCharges']] = df[['MonthlyCharges', 'TotalCharges']].astype(float)

    return df

input_df = build_features()
st.markdown("*Data sent to model:*")
st.dataframe(input_df)

# ===== PREDICTION =====
threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("🔮 Predict!"):
    if model is None:
        st.error("Model not loaded. Check logs and model path.")
    else:
        try:
            # If this model is a pipeline that includes preprocessing, passing raw input_df is correct.
            probs = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[:, 1]
            pred = model.predict(input_df)
            pred_class = int(np.array(pred).ravel()[0])
            churn_prob = float(probs[0]) if probs is not None else None

            if churn_prob is not None:
                proba_text = f" (churn probability: {churn_prob:.1%})"
            else:
                proba_text = ""

            if pred_class == 1:
                if churn_prob is not None and churn_prob < threshold:
                    st.warning(f"Model predicted CHURN but probability {churn_prob:.2f} < threshold {threshold}. {proba_text}")
                st.error("❌ This customer is likely to CHURN." + proba_text)
            else:
                if churn_prob is not None and churn_prob >= threshold:
                    st.warning(f"Model predicted NO-CHURN but probability {churn_prob:.2f} >= threshold {threshold}. {proba_text}")
                st.success("✅ This customer is likely to STAY." + proba_text)

        except Exception as e:
            st.exception(e)