import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def load_data(path):
    """Load processed dataset from CSV."""
    df = pd.read_csv(path)
    print(f"✅ Loaded data from {path} — shape: {df.shape}")
    return df


def preprocess_features(df, target='loan_status'):
    """
    Drop high-cardinality or redundant features.
    Encode selected low-cardinality categoricals using one-hot encoding.
    Ensure all features are numeric.
    """
    drop_cols = ['emp_title', 'title', 'zip_code', 'addr_state', 'sub_grade']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    safe_categoricals = [
        'term', 'grade', 'home_ownership', 'verification_status',
        'purpose_grouped', 'income_bin', 'int_rate_bin', 'int_rate_grade'
    ]

    existing_categoricals = [col for col in safe_categoricals if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_categoricals, drop_first=True)
    print(f"🔄 Encoded {len(existing_categoricals)} categorical columns — resulting shape: {df_encoded.shape}")

    # Drop any remaining object-type columns that slipped through
    non_numeric = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        print(f"⚠️ Dropping remaining non-numeric columns: {non_numeric}")
        df_encoded = df_encoded.drop(columns=non_numeric)

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Align columns in train and test sets to avoid shape mismatch
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    print(f"✅ Data split — Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """Fit the model on training data."""
    model.fit(X_train, y_train)
    print(f"✅ Trained {model.__class__.__name__}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        print(f"🎯 ROC AUC: {roc:.4f}")
    else:
        print("⚠️ Model does not support probability predictions.")


# Optional: CLI entry point for script usage
if __name__ == '__main__':
    import argparse
    from sklearn.ensemble import RandomForestClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/borrowiq_cleaned.csv')
    args = parser.parse_args()

    df = load_data(args.data)
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(random_state=42)
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
# This code is designed to train a machine learning model on the processed BorrowIQ dataset.
   