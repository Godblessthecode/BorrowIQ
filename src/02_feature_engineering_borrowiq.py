import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Config & Paths
# ===============================
root_dir = Path(__file__).resolve().parents[1]  # Adjusted for script in borrowiq/src

data_path = root_dir / "data" / "processed" / "borrowiq_cleaned.csv"
output_path = root_dir / "data" / "processed" / "borrowiq_cleaned.csv"  # Final output path

# ===============================
# Feature Engineering Functions
# ===============================
def load_data(path):
    assert path.exists(), f"❌ File not found: {path}"
    df = pd.read_csv(path)
    print(f"✅ Data loaded — Shape: {df.shape}")
    return df

def apply_winsorization(df, col, quantile=0.99):
    assert col in df.columns, f"Column {col} not found."
    cap = df[col][df[col] > 0].quantile(quantile)
    df[f'{col}_winsorized'] = df[col].clip(upper=cap)
    print(f"✅ Winsorized '{col}' at {quantile:.2%} quantile = {cap:.2f}")
    return df

def apply_log_transform(df, col):
    df[f'{col}_log'] = np.log1p(df[col])
    return df

def bin_income(df):
    df['income_bin'] = pd.qcut(df['annual_inc'], 4, labels=["Low", "Medium", "High", "Very High"])
    return df

def bin_interest_rate(df):
    if df['int_rate'].nunique() > 10:
        df['int_rate_bin'] = pd.qcut(df['int_rate'], q=3, labels=['low', 'medium', 'high'])
    else:
        df['int_rate_bin'] = pd.cut(df['int_rate'], bins=[0, 10, 15, np.inf], labels=['low', 'medium', 'high'])
    df['int_rate_bin'] = df['int_rate_bin'].astype(str)
    df['int_rate_grade'] = df['grade'].astype(str) + '_' + df['int_rate_bin']
    return df

def group_rare_purposes(df, threshold=0.01):
    value_counts = df['purpose'].value_counts()
    rare = value_counts[value_counts < threshold * len(df)].index
    df['purpose_grouped'] = df['purpose'].replace(rare, 'other_low_volume')
    return df

def feature_summary():
    return pd.DataFrame({
        'Feature': [
            'annual_inc_winsorized', 'dti_winsorized', 'annual_inc_winsorized_log',
            'income_bin', 'int_rate_bin', 'int_rate_grade', 'purpose_grouped'
        ],
        'Type': [
            'Numerical', 'Numerical', 'Numerical',
            'Categorical (Ordinal)', 'Categorical (Ordinal)', 'Categorical (Ordinal)', 'Categorical'
        ],
        'Description': [
            'Capped income at 99th percentile',
            'Capped DTI at 99th percentile',
            'Log transform of capped income',
            'Binned income into quartiles',
            'Binned interest rate into terciles',
            'Interaction of grade and interest bin',
            'Grouped rare purposes (<1%) into other_low_volume'
        ]
    })

# ===============================
# Main Execution Block
# ===============================
def main():
    df = load_data(data_path)
    df = apply_winsorization(df, 'annual_inc')
    df = apply_winsorization(df, 'dti')
    df = apply_log_transform(df, 'annual_inc_winsorized')
    df = bin_income(df)
    df = bin_interest_rate(df)
    df = group_rare_purposes(df)

    print("\n🔎 Feature Engineering Preview:")
    print(df[[
        'annual_inc', 'annual_inc_winsorized', 'annual_inc_winsorized_log',
        'dti', 'dti_winsorized', 'income_bin', 'int_rate', 'int_rate_bin', 'int_rate_grade', 'purpose', 'purpose_grouped']].head())

    print("\n📋 Feature Summary Table:")
    print(feature_summary())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Final model-ready dataset saved to: {output_path}")

if __name__ == '__main__':
    main()
