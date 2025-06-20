# src/preview_data.py

import pandas as pd
import sys

def main():
    # Define file path (raw string avoids Windows escape errors)
    file_path = r'C:\Users\Alain\OneDrive\Desktop\Alain\Code Projects\borrowiq\data\raw\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv'

    try:
        print("📥 Loading CSV file...")
        df = pd.read_csv(file_path, low_memory=False)
        print("✅ File loaded successfully.\n")

        # Show shape
        print(f"🧮 Shape of dataset: {df.shape}")

        # Show column names
        print("\n📋 Column names:")
        print(df.columns.tolist())

        # Show first few rows
        print("\n🔍 Preview of first rows:")
        print(df.head())

        # Show column data types and null counts
        print("\n🧾 Data info:")
        print(df.info())

        # Show missing values
        print("\n⚠️ Top 20 columns with missing values:")
        missing = df.isnull().sum().sort_values(ascending=False)
        print(missing[missing > 0].head(20))

    except FileNotFoundError:
        print("❌ File not found. Double-check the path.")
    except PermissionError:
        print("❌ Permission denied. Make sure the file isn’t open in Excel or locked by OneDrive.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

