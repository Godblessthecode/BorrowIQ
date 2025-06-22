# 04_feature_engineering_borrowiq.py

# ===============================
# 📦 LIBRARIES & DEPENDENCIES
# ===============================

import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# 🗂️ LOAD CLEANED DATA
# ===============================

root_dir = Path(__file__).resolve().parents[1]
input_path = root_dir / "data" / "processed" / "borrowiq_cleaned.csv"
output_path = root_dir / "data" / "processed" / "borrowiq_engineered.csv"

assert input_path.exists(), f"❌ Cleaned input file not found: {input_path}"

df = pd.read_csv(input_path)
print(f"✅ Loaded cleaned dataset — Shape: {df.shape}")

# ===============================
# 🧪 FEATURE ENGINEERING
# ===============================

# Winsorize high outliers
df['annual_inc_winsorized'] = df['annual_inc'].clip(upper=df['annual_inc'].quantile(0.99))
df['dti_winsorized'] = df['dti'].clip(upper=df['dti'].quantile(0.99))

# Log transformation of income
df['annual_inc_log'] = np.log1p(df['annual_inc'])

# Binning annual income
df['income_bin'] = pd.qcut(df['annual_inc'], 4, labels=["Low", "Medium", "High", "Very High"])

# Binning interest rate + custom grades
df['int_rate_bin'] = pd.qcut(df['int_rate'], q=5, labels=False, duplicates='drop')
df['int_rate_grade'] = pd.qcut(df['int_rate'], q=7, labels=["A", "B", "C", "D", "E", "F", "G"])

# Grouping loan purposes
df['purpose_grouped'] = df['purpose'].replace({
    'credit_card': 'debt',
    'debt_consolidation': 'debt',
    'home_improvement': 'home',
    'major_purchase': 'home',
    'small_business': 'business',
    'medical': 'other',
    'moving': 'other',
    'vacation': 'other',
    'wedding': 'other',
    'renewable_energy': 'other',
    'house': 'home',
    'car': 'other',
    'other': 'other',
})

# ===============================
# 💾 EXPORT ENGINEERED DATA
# ===============================

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete. Saved to: {output_path}")
