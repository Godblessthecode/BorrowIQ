# BorrowIQ: Real-Time Credit Default Modeling

**Author**: Alain Sorurian  
**Status**: Completed (MVP) — Built in public  
**Project Type**: Applied Machine Learning | Credit Risk  
**Dataset**: LendingClub Loan Data (2016–2018)

---

## 📌 Project Goal

BorrowIQ is a real-world credit risk pipeline that estimates the probability of default (PD) at **loan origination** using LendingClub data. The system is designed to reflect practical underwriting scenarios — only using features known at approval time — and includes robust explainability and a deployable frontend app.

---

## ✅ Completed Components

### 🔍 1. Data Preview Script

📄 `src/preview_data.py`

```bash
python src/preview_data.py
```

**Highlights:**
- Loads raw dataset (`accepted_2007_to_2018Q4.csv`)
- Displays shape, missing values, top columns with nulls
- Flags potential **leakage columns** based on heuristics

---

### 🧹 2. Data Preprocessing Pipeline

📄 `src/00_preprocess_borrowiq.py`

```bash
python src/00_preprocess_borrowiq.py
```

**Key Steps:**
- Filters to loans with status `Fully Paid` or `Charged Off`
- Target encoded: `Fully Paid` → 0, `Charged Off` → 1
- Filters loans to those **issued from 2016 onward**
- Drops columns with >80% missing data
- Drops known leakage features (e.g., `recoveries`, `last_pymnt_d`, etc.)
- Median imputation for numerics; fills objects with `'Unknown'`
- Saves to: `data/processed/borrowiq_cleaned.csv`

---

### 🧠 3. Feature Engineering Pipeline

📄 `src/02_feature_engineering_borrowiq.py`

```bash
python src/02_feature_engineering_borrowiq.py
```

**Adds:**
- Winsorization of `annual_inc` and `dti` (capped at 99th percentile)
- Log transform of capped income
- Binned income and interest into quartiles/terciles
- Interaction feature: `int_rate_grade` = `grade` × `int_rate_bin`
- Rare purpose grouping → `purpose_grouped`
- Drops raw `dti` and `annual_inc` post-transformation
- Saves final model-ready dataset to: `data/processed/borrowiq_final_model_ready.csv`

---

### 📊 4. Exploratory Data Analysis (EDA)

📓 `notebooks/01_eda_borrowiq.ipynb`

**Covers:**
- Class imbalance and stratified sample strategy
- Distribution plots, boxplots, bivariate comparisons
- Insights on `grade`, `term`, `int_rate`, and `purpose`

---

### 🤖 5. Model Training Pipeline

📓 `notebooks/03_modelling_borrowiq.ipynb`

**Includes:**
- Stratified train-test split
- Transformation pipeline (numeric + categorical encoders)
- Models trained:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Class-balanced thresholding and calibration
- Metrics logged:
  - ROC AUC
  - F1, Precision, Recall (esp. on defaulters)
  - Confusion Matrices
- Best model (`XGBoost`) saved to: `data/model_outputs/xgb_pipeline_model.pkl`

---

### 🧠 6. Model Explainability with SHAP

📓 `notebooks/04_explainability_borrowiq.ipynb`

**Covers:**
- Global SHAP feature importance
- Summary plot of key drivers
- SHAP dependence for features like `dti_winsorized`, `int_rate_grade`
- Local force plots for specific predictions
- Used optimized test set with stratified sampling

---

### 🖥️ 7. Interactive Streamlit App

📄 `dashboard/app.py`

```bash
streamlit run dashboard/app.py
```

**Features:**
- Accepts user inputs: income, interest rate, term, purpose, verification status
- Applies same feature engineering logic
- Loads saved `XGBoost` model
- Returns **default probability** with interpretation (threshold = 0.35)
- Flags high-risk vs low-risk predictions

---

## 📁 Folder Structure

```
borrowiq/
├── data/
│   ├── raw/                         # Original LendingClub data (.gitignored)
│   ├── processed/                   # Cleaned and feature-engineered datasets
│   └── model_outputs/              # Trained models and results
├── notebooks/
│   ├── 01_eda_borrowiq.ipynb       # Visual EDA and target balancing
│   ├── 03_modelling_borrowiq.ipynb # Model training, tuning, evaluation
│   └── 04_explainability_borrowiq.ipynb # SHAP explainability
├── src/
│   ├── preview_data.py
│   ├── 00_preprocess_borrowiq.py
│   ├── 02_feature_engineering_borrowiq.py
├── dashboard/
│   └── app.py                      # Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Repro Steps

### Step 1 — Preprocess raw data
```bash
python src/00_preprocess_borrowiq.py
```

### Step 2 — Feature Engineering
```bash
python src/02_feature_engineering_borrowiq.py
```

### Step 3 — Model Training (Jupyter Notebook)
```bash
jupyter notebook notebooks/03_modelling_borrowiq.ipynb
```

### Step 4 — Explainability
```bash
jupyter notebook notebooks/04_explainability_borrowiq.ipynb
```

### Step 5 — Run the Streamlit app
```bash
streamlit run dashboard/app.py
```

---

## 🚧 Known Caveats

- Imputation occurs pre-train-test split
- Threshold tuned manually — could be optimized via business cost matrix
- Raw income/DTI dropped only after transformation (watch for app alignment)

---

## 📎 Future Plans

- SHAP-based explanations within app
- More robust class imbalance handling (e.g. focal loss, SMOTE)
- CI/CD with GitHub Actions
- Dockerize full pipeline for deployment

---

## 🪪 License

MIT License — fork, modify, use freely.

