import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model and metadata ===
model = joblib.load("data/model_outputs/xgb_pipeline_model.pkl")
template_row = pd.read_csv("data/processed/borrowiq_final_model_ready.csv", nrows=1)

# === Streamlit UI ===
st.set_page_config(page_title="BorrowIQ: Default Risk Predictor", layout="centered")
st.title("💳 BorrowIQ: Default Risk Predictor")
st.markdown("Predict the probability of loan default using the final XGBoost model.")

# === User Inputs ===
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        annual_inc = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=60000)
        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, step=0.1, value=14.0)
        dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=45.0, step=0.1, value=18.0)
    with col2:
        purpose = st.selectbox("Purpose", options=[
            "credit_card", "debt_consolidation", "home_improvement", "major_purchase",
            "small_business", "other"
        ])
        term = st.selectbox("Loan Term", options=["36 months", "60 months"])
        verification_status = st.selectbox("Income Verification Status", options=["Verified", "Not Verified"])

    submitted = st.form_submit_button("Predict Default Probability")

# === Inference Logic ===
if submitted:
    input_df = template_row.copy()

    # Apply feature transformations consistent with training pipeline
    input_df['annual_inc_winsorized'] = min(annual_inc, 250000)  # use training cap
    input_df['dti_winsorized'] = min(dti, 45)  # use training cap
    input_df['annual_inc_winsorized_log'] = np.log1p(input_df['annual_inc_winsorized'])

    input_df['int_rate'] = int_rate
    input_df['term_ 36 months'] = 1 if term == "36 months" else 0
    input_df['purpose_grouped'] = (
        'debt' if purpose in ['credit_card', 'debt_consolidation']
        else 'home' if purpose in ['home_improvement', 'house', 'major_purchase']
        else 'business' if purpose == 'small_business'
        else 'other'
    )

    # Predict
    proba = model.predict_proba(input_df)[0, 1]
    st.subheader(f"📊 Estimated Default Probability: {proba:.2%}")

    if proba >= 0.35:
        st.error("⚠️ High Risk of Default. Consider stricter underwriting.")
    else:
        st.success("✅ Acceptable Risk Level.")

    st.markdown("---")
    st.markdown("**Model Info:** XGBoost | Threshold = 0.35 | Trained on stratified post-2016 LendingClub data")
