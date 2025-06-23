# src/00_preprocess_borrowiq.py

import argparse
import pandas as pd
from pathlib import Path

# --- Step 0: Feature summary (optional, mostly for inspection)
def generate_feature_summary(df):
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "num_missing": df.isnull().sum(),
        "pct_missing": df.isnull().mean() * 100,
        "num_unique": df.nunique(),
        "sample_values": df.apply(lambda x: x.dropna().unique()[:5])
    })
    return summary.sort_values(by="pct_missing", ascending=False)

# --- Step 1: Filter and convert target
def filter_and_encode_target(df):
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    return df

# --- Step 2: Filter by issue date
def filter_by_issue_date(df, year_threshold=2016):
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    return df[df['issue_d'].dt.year >= year_threshold]

# --- Step 3: Drop high-missing columns
def drop_high_missing(df, threshold=0.8):
    to_drop = df.columns[df.isnull().mean() > threshold].tolist()
    df = df.drop(columns=to_drop)
    print(f"🧹 Dropped {len(to_drop)} columns with >{int(threshold*100)}% missing: {to_drop}")
    return df

# --- Step 4: Drop known leakage columns
def drop_leakage_columns(df):
    leakage_cols = [
        'id', 'funded_amnt_inv', 'installment', 'total_pymnt', 'recoveries',
        'last_pymnt_d', 'last_credit_pull_d', 'collection_recovery_fee',
        'out_prncp', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'last_pymnt_amnt', 'total_pymnt_inv', 'last_fico_range_high', 'last_fico_range_low','debt_settlement_flag' 
    ]
    existing_leakage = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=existing_leakage)
    print(f"🛑 Dropped {len(existing_leakage)} leakage columns: {existing_leakage}")
    return df

# --- Step 5: Fill missing values
def impute_missing(df):
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')

    num_remaining = df.isnull().sum().gt(0).sum()
    print(f"🔧 Filled missing values. Remaining columns with missing: {num_remaining}")
    return df

# --- Full preprocessing pipeline
def preprocess(input_path, output_path):
    input_path = Path(input_path)
    assert input_path.exists(), f"❌ File not found: {input_path}"
    df = pd.read_csv(input_path, low_memory=False)

    print("📋 Feature summary (pre-cleaning):")
    print(generate_feature_summary(df).head(30))

    df = filter_and_encode_target(df)
    df = filter_by_issue_date(df)
    df = drop_high_missing(df)
    df = drop_leakage_columns(df)
    df = impute_missing(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing complete.")
    print(f"📊 Final shape: {df.shape}")
    print(f"💾 Saved to: {output_path}")

# --- CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Preprocess LendingClub dataset.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/accepted_2007_to_2018Q4.csv",
        help="Path to raw input CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/borrowiq_cleaned.csv",
        help="Path to save the processed CSV file"
    )
    args = parser.parse_args()
    preprocess(args.input, args.output)

if __name__ == "__main__":
    main()
