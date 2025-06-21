# src/00_preprocess_borrowiq.py

import argparse
import pandas as pd
from pathlib import Path

def preprocess(input_path, output_path):
    # 🧭 Load raw data
    input_path = Path(input_path)
    assert input_path.exists(), f"❌ File not found: {input_path}"
    df = pd.read_csv(input_path, low_memory=False)

    # 📊 Step 0: Feature summary before any preprocessing
    def generate_feature_summary(df):
        summary = pd.DataFrame({
            "dtype": df.dtypes,
            "num_missing": df.isnull().sum(),
            "pct_missing": df.isnull().mean() * 100,
            "num_unique": df.nunique(),
            "sample_values": df.apply(lambda x: x.dropna().unique()[:5])
        })
        return summary.sort_values(by="pct_missing", ascending=False)

    summary = generate_feature_summary(df)
    print("📋 Feature summary (pre-cleaning):")
    print(summary.head(30))  # Or export to CSV for deeper review

    # 🎯 Step 1: Binary target
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    # 📆 Step 2: Filter by issue date
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df = df[df['issue_d'].dt.year >= 2016]

    # 🧹 Step 3: Drop columns with >80% missing
    dropped_cols = df.columns[df.isnull().mean() > 0.8].tolist()
    df = df.drop(columns=dropped_cols)
    print(f"🧹 Dropped {len(dropped_cols)} columns with >80% missing: {dropped_cols}")

    # 🔐 Step 4: Always drop known leakage columns
    leakage_cols = [
        'id', 'funded_amnt_inv', 'installment', 'total_pymnt', 'recoveries',
        'last_pymnt_d', 'last_credit_pull_d', 'collection_recovery_fee',
        'out_prncp', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'last_pymnt_amnt', 'total_pymnt_inv', 'last_fico_range_high', 'last_fico_range_low'
    ]
    dropped = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=dropped)
    print(f"🛑 Dropped {len(dropped)} leakage columns: {dropped}")

    # 🛠 Step 5: Fill missing values
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')

    # 🧮 Report number of filled columns
    num_filled = df.isnull().sum().gt(0).sum()
    print(f"🔧 Filled missing values in {num_filled} columns.")

    # 💾 Step 6: Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing complete.")
    print(f"📊 Final shape: {df.shape}")
    print(f"💾 Saved to: {output_path}")

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
