# src/preview_data.py

import os
import pandas as pd
import argparse

def main():
    # Set up CLI argument parser
    parser = argparse.ArgumentParser(description="Preview LendingClub loan data.")
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join("data", "raw", "accepted_2007_to_2018Q4.csv"),
        help="Path to the raw CSV file (default: data/raw/accepted_2007_to_2018Q4.csv)"
    )
    args = parser.parse_args()
    file_path = args.path

    try:
        print(f"📥 Loading CSV file from: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        print("✅ File loaded successfully.\n")

        print(f"🧮 Shape of dataset: {df.shape}")
        print("\n📋 Column names:")
        print(df.columns.tolist())

        print("\n🔍 Preview of first rows:")
        print(df.head())

        print("\n🧾 Data info:")
        df.info()

        print("\n⚠️ Top 20 columns with missing values:")
        missing = df.isnull().sum().sort_values(ascending=False)
        print(missing[missing > 0].head(20))

        # 🔐 Check for potential data leakage columns
        print("\n🛑 Potential data leakage columns (based on keywords):")
        leakage_keywords = [
            'pymnt', 'rec', 'recover', 'settlement', 'hardship',
            'last_', 'next_', 'out_prncp', 'total_pymnt', 'collection',
            'debt_settlement', 'fico_range', 'id', 'url', 'desc'
        ]

        leakage_candidates = [col for col in df.columns
                              if any(kw in col.lower() for kw in leakage_keywords)]

        if leakage_candidates:
            for col in leakage_candidates:
                print(f" - {col}")
        else:
            print("✅ No obvious leakage columns detected based on keywords.")

    except FileNotFoundError:
        print("❌ File not found. Double-check the path or use --path to specify.")
    except PermissionError:
        print("❌ Permission denied. Is the file open in Excel or locked?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
