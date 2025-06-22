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

---

### 🧠 3. Feature Engineering Script

📄 `src/02_feature_engineering_borrowiq.py`

```bash
python src/02_feature_engineering_borrowiq.py
```

**What it adds:**

- ✅ Winsorization of `annual_inc` and `dti` at the 99th percentile
- 🔁 Log-transform of the winsorized income
- 📈 Binned income and interest rates into quartiles and terciles
- 🧠 Feature interaction: `int_rate_grade` = `grade` × `int_rate_bin`
- 🔍 Purpose grouping for low-frequency classes into `other_low_volume`
- 📋 Feature summary table printed to console

> Output is saved to: `data/processed/borrowiq_cleaned.csv` (overwritten)

---

### 🤖 4. Model Training Script

📄 `src/model_train.py`

```bash
python src/model_train.py --data data/processed/borrowiq_cleaned.csv
```

**What it does:**

- Loads the cleaned, feature-engineered dataset
- Drops high-cardinality and noisy columns (e.g. `emp_title`, `sub_grade`, `zip_code`, etc.)
- Encodes a predefined list of safe categorical variables
- Ensures all features are numeric to avoid model errors
- Splits into train/test sets with stratification
- Trains a `RandomForestClassifier` (by default)
- Prints:
  - 📊 Classification Report
  - 🎯 ROC AUC Score

> 💡 Script is modular and CLI-ready — easy to swap models, export outputs, or plug into future pipelines.

---

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

### Run feature engineering:

```bash
python src/02_feature_engineering_borrowiq.py
```

### Train baseline model:

```bash
python src/model_train.py --data data/processed/borrowiq_cleaned.csv
```

---

## 📁 Folder Structure

```
borrowiq-default-predictor/
├── data/
│   ├── raw/                      # Raw LendingClub data (.gitignored)
│   └── processed/                # Cleaned dataset outputs
├── src/
│   ├── preview_data.py
│   ├── 00_preprocess_borrowiq.py
│   ├── 02_feature_engineering_borrowiq.py
│   └── model_train.py
├── .gitignore
├── README.md
├── requirements.txt
```

---

## 🔜 Roadmap

- `01_eda_borrowiq.ipynb` (EDA visualizations, SHAP planning)
- Additional model experimentation + export
- Streamlit scoring interface
- GitHub Actions for CI/CD

---

## 🪪 License

MIT License — use, fork, and adapt freely.
