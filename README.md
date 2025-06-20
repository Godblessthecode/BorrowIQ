# BorrowIQ: Real-Time Credit Default Modeling

**Author**: Alain Sorurian  
**Status**: In Progress — building in public  
**Project Type**: Applied Machine Learning | Credit Risk  
**Dataset**: LendingClub Loan Data (2016–2018)

---

## 📌 Project Goal

BorrowIQ is an end-to-end credit risk modeling system designed to predict loan default probabilities at origination using historical consumer loan data. The project brings together:

- Financial domain expertise 
- Machine learning techniques (Logistic Regression, XGBoost, Decision Tree)
- Model explainability (SHAP values)
- Real-time scoring through a Streamlit dashboard

---

## ✅ Completed So Far

### 🔍 1. Data Preview Script

📄 `src/preview_data.py`

```bash
python src/preview_data.py
```

**What it does:**
- Loads the raw LendingClub dataset (`accepted_2007_to_2018Q4.csv`)
- Shows shape, columns, nulls, and a basic data profile

**Note**: Dataset must be placed in `data/raw/`. The `.gitignore` ensures raw data stays local.

---

### 🧹 2. Data Preprocessing

📄 `src/00_preprocess_borrowiq.py`

```bash
python src/00_preprocess_borrowiq.py
```

**What it does:**

- Keeps only `Fully Paid` and `Charged Off` loans
- Binarizes target: 0 = paid, 1 = default
- Filters loans from 2016 onward
- Drops columns with >80% missing
- Removes leakage columns
- Imputes missing numeric/categorical values
- Saves output to `data/processed/borrowiq_cleaned.csv`

---

## 🔨 Up Next (Live)

- `02_feature_engineering_borrowiq.py`: binning, scaling, winsorization  
- `01_eda_borrowiq.ipynb`: visual patterns, default heatmaps, variable exploration  
- GitHub Actions test automation for notebook execution  
- Deploying a lightweight Streamlit prototype

---

## 📁 Folder Structure (Live Snapshot)

```
borrowiq-default-predictor/
├── data/
│   ├── raw/                    # Ignored raw LendingClub data
│   └── processed/              # Cleaned data output
├── src/
│   ├── preview_data.py         # CSV inspection tool
│   └── 00_preprocess_borrowiq.py  # Cleaning + filtering logic
├── .gitignore                  # Raw data excluded
├── README.md                   # This file
├── requirements.txt            # Will be updated progressively
```

---

## 🪪 License

MIT License — fork freely and improve collaboratively.

