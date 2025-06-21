# BorrowIQ: Real-Time Credit Default Modeling

**Author**: Alain Sorurian  
**Status**: In Progress — building in public  
**Project Type**: Applied Machine Learning | Credit Risk  
**Dataset**: LendingClub Loan Data (2016–2018)

---

## 📌 Project Goal

BorrowIQ is a credit risk modeling pipeline that estimates the probability of default (PD) at **loan origination** using historical LendingClub data. The system is built to reflect realistic underwriting conditions — using only features that would have been known at the time of loan approval.

---

## ✅ Completed Components

### 🔍 1. Data Preview Script

📄 `src/preview_data.py`

```bash
python src/preview_data.py
```

**Functionality:**

- Loads raw dataset (`accepted_2007_to_2018Q4.csv`)
- Prints shape, columns, and missing values
- Highlights top 20 features with missing data
- Flags potential **leakage columns** using simple keyword heuristics  
  (e.g. `pymnt`, `rec`, `recover`, `last_`, etc.)

**Limitations:**

- Leakage detection is based on column name matching  
  → It will not detect structural or proxy leakage

> 📁 **Note**: Dataset must be placed in `data/raw/`. Raw files are `.gitignored`.

---

### 🧹 2. Data Preprocessing Script

📄 `src/00_preprocess_borrowiq.py`

```bash
python src/00_preprocess_borrowiq.py
```

**What it does:**

- Filters data to `Fully Paid` and `Charged Off` loans only
- Converts target: `Fully Paid` → `0`, `Charged Off` → `1`
- Keeps loans **issued from 2016 onward**  
  *(Note: LendingClub schema stabilized in 2016)*
- Drops columns with **>80% missing values**
- **Always removes known leakage columns** (e.g. `total_pymnt`, `recoveries`, `last_pymnt_d`, etc.)
- Fills missing values:
  - Numeric columns → median
  - Object columns → `'Unknown'`
- Saves output to: `data/processed/borrowiq_cleaned.csv`



## ⚠️ Known Caveats

- Imputation is applied **before train-test split** — not recommended for production modeling
- Keyword-based leakage checks **miss proxy features** like `int_rate`, `grade`, or `fico_range`
- No outlier treatment or feature scaling yet

---

## 🧪 Sample Usage

### Preview raw data:

```bash
python src/preview_data.py --path data/raw/accepted_2007_to_2018Q4.csv
```

### Run preprocessing (leakage dropped by default):

```bash
python src/00_preprocess_borrowiq.py
```

---

## 📁 Folder Structure

```
borrowiq-default-predictor/
├── data/
│   ├── raw/                      # Raw LendingClub data (.gitignored)
│   └── processed/                # Cleaned dataset outputs
├── src/
│   ├── preview_data.py           # Raw data inspection tool
│   └── 00_preprocess_borrowiq.py # Cleaning pipeline
├── .gitignore
├── README.md
├── requirements.txt
```

---

## 🔜 Roadmap

- `02_feature_engineering_borrowiq.py` (binning, scaling, encoding)
- `01_eda_borrowiq.ipynb` (EDA visualizations, SHAP planning)
- Streamlit scoring interface
- GitHub Actions for CI/CD

---

## 🪪 License

MIT License — use, fork, and adapt freely.