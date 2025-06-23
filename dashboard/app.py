# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================
# 📦 Load Model & Template Row
# ============================

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "data" / "model_outputs" / "xgb_pipeline_model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_cleaned_template():
    final_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "borrowiq_final_model_ready.csv"
    if not final_path.exists():
        st.error(f"❌ Template file not found at {final_path}")
        st.stop()
    df = pd.read_csv(final_path)
    return df.iloc[[0]].copy()  # Use the first full original row

xgb_pipeline = load_model()
template_row = load_cleaned_template()

# ============================
# 🧾 Input Form
# ============================

st.title("💳 BorrowIQ: Loan Default Predictor")
st.markdown("Estimate the likelihood of a loan default based on borrower information.")

with st.form("loan_form"):
    annual_inc = st.number_input("Annual Income (€)", min_value=0.0, value=45000.0)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=13.5)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=18.0)
    purpose = st.selectbox("Loan Purpose", options=["debt_consolidation", "credit_card", "home_improvement", "other"])
    term = st.selectbox("Loan Term", options=["36 months", "60 months"])
    verification_status = st.selectbox("Verification Status", options=["Verified", "Source Verified", "Not Verified"])
    submitted = st.form_submit_button("Predict")

# ============================
# 🧠 Run Prediction
# ============================

if submitted:
    try:
        threshold = 0.23
        st.caption(f"Model decision threshold: **{threshold:.2f}**")

        input_df = template_row.copy()
        input_df['annual_inc'] = annual_inc
        input_df['int_rate'] = int_rate
        input_df['dti'] = dti
        input_df['purpose'] = purpose
        input_df['term'] = term
        input_df['verification_status'] = verification_status

        # Feature engineering
        input_df['annual_inc_winsorized'] = min(annual_inc, template_row['annual_inc'].quantile(0.99))
        input_df['dti_winsorized'] = min(dti, template_row['dti'].quantile(0.99))
        input_df['annual_inc_log'] = np.log1p(annual_inc)

        input_df['int_rate_bin'] = pd.qcut(
            pd.concat([template_row['int_rate'], pd.Series([int_rate])]),
            5, labels=False, duplicates='drop'
        ).iloc[-1]

        input_df['income_bin'] = pd.qcut(
            pd.concat([template_row['annual_inc'], pd.Series([annual_inc])]),
            4, labels=["Low", "Medium", "High", "Very High"]
        ).iloc[-1]

        input_df['purpose_grouped'] = purpose
        if purpose in ['credit_card', 'debt_consolidation']:
            input_df['purpose_grouped'] = 'debt'
        elif purpose in ['home_improvement', 'house', 'major_purchase']:
            input_df['purpose_grouped'] = 'home'
        elif purpose in ['small_business']:
            input_df['purpose_grouped'] = 'business'
        elif purpose in ['medical', 'moving', 'vacation', 'wedding', 'renewable_energy', 'car', 'other']:
            input_df['purpose_grouped'] = 'other'

        input_df['int_rate_grade'] = pd.qcut(
            pd.concat([template_row['int_rate'], pd.Series([int_rate])]),
            7, labels=["A", "B", "C", "D", "E", "F", "G"]
        ).iloc[-1]

        # Predict with custom threshold
        default_proba = xgb_pipeline.predict_proba(input_df)[0, 1]
        prediction = int(default_proba >= threshold)

        # Output
        st.subheader("🔍 Prediction Result")
        st.write(f"**Probability of Default**: {default_proba:.2%}")
        st.write(f"**Decision**: {'❌ Reject' if prediction == 1 else '✅ Approve'}")

        if prediction == 1:
            st.warning("⚠️ High risk of default — consider further review.")
        else:
            st.success("👍 Low risk — eligible for approval.")

    except Exception as e:
        st.error("Prediction failed. Please check your inputs or pipeline.")
        st.exception(e)
