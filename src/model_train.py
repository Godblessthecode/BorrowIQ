import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import argparse
import joblib
import os

from xgboost import XGBClassifier  # 👈 Replacing RandomForest
import warnings
warnings.filterwarnings("ignore")


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
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    print(f"✅ Data split — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train an XGBoost classifier with sensible defaults."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=3,  # 👈 account for class imbalance
        n_jobs=-1,
        random_state=42
    )
    print("⏳ Training XGBoost Classifier...")
    model.fit(X_train, y_train)
    print(f"✅ Trained {model.__class__.__name__}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_proba)
    print(f"🎯 ROC AUC: {roc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/borrowiq_final_model_ready.csv')
    parser.add_argument('--output_model', type=str, default='models/xgboost_model.pkl')
    args = parser.parse_args()

    df = load_data(args.data)
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model)
    print(f"📂 Model saved to {args.output_model}")
