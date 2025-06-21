from pathlib import Path
import pandas as pd

# 🧭 1. Dynamically find project root (works even if script is run from anywhere)
root_dir = Path(__file__).resolve().parents[1]

# 📁 2. Define portable file paths
input_path = root_dir / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
output_path = root_dir / "data" / "processed" / "borrowiq_cleaned.csv"

# 📥 3. Load raw data
assert input_path.exists(), f"❌ File not found: {input_path}"
df = pd.read_csv(input_path, low_memory=False)

# 🎯 Step 1: Define binary target
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# 📆 Step 2: Keep loans issued from 2016 onward
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
df = df[df['issue_d'].dt.year >= 2016]

# 🧹 Step 3: Drop columns with >80% missing
threshold = len(df) * 0.8
df = df.dropna(thresh=threshold, axis=1)

# 🔐 Step 4: Drop leakage columns
leakage_cols = [
    'id', 'funded_amnt_inv', 'installment', 'total_pymnt', 'recoveries',
    'last_pymnt_d', 'last_credit_pull_d', 'collection_recovery_fee',
    'out_prncp', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'last_pymnt_amnt', 'total_pymnt_inv', 'last_fico_range_high', 'last_fico_range_low'
]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# 🛠 Step 5: Fill missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('Unknown')

# 💾 Step 6: Save cleaned data
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("✅ Preprocessing complete.")
print(f"📊 Final shape: {df.shape}")
