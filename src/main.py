import argparse
import sys
import numpy as np
import pandas as pd

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model import ModelPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FeatureBot: Adult Income modular training pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/adult.csv",
        help="Path to Adult Income CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
        help="Model type to train",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for train/test split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/modular_model.pkl",
        help="Where to save the trained model",
    )

    args = parser.parse_args()

    print("\n================ FeatureBot Modular Pipeline ================\n")

    # 1. Load and validate data
    loader = DataLoader()
    print(f"📥 Loading data from {args.data} ...")
    df = loader.load_data(args.data)
    df = loader.clean_data(df)

    if not loader.validate_data(df):
        raise ValueError("Data validation failed.")

    df = loader.normalize_target(df)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Separate features/target
    X = df.drop("income", axis=1)
    y = df["income"]
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    # 3. Preprocess: missing values + encoding + scaling
    pre = DataPreprocessor()
    print("\n🔧 Preprocessing features...")

    # handle missing values first
    X_clean = pre.handle_missing_values(X)
    print("   ✓ Handled missing values")

    # split numeric / categorical
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_clean.select_dtypes(exclude="number").columns.tolist()

    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")

    # scale numeric
    if numeric_cols:
        X_num = pre.scale_features(X_clean[numeric_cols].values)
        print("   ✓ Scaled numeric features")
    else:
        X_num = None

    # encode categorical
    if categorical_cols:
        X_cat = pre.encode_categorical(X_clean[categorical_cols])
        print(f"   ✓ One-hot encoded categorical features ({X_cat.shape[1]} dims)")
    else:
        X_cat = None

    # combine back to one feature matrix
    if X_num is not None and X_cat is not None:
        X_final = np.hstack([X_num, X_cat])
    elif X_num is not None:
        X_final = X_num
    elif X_cat is not None:
        X_final = X_cat
    else:
        raise ValueError("No features available after preprocessing.")

    print(f"   Final feature matrix shape: {X_final.shape}")

    # 4. Train pipeline
    print(f"\n🤖 Training {args.model}...")
    pipeline = ModelPipeline(args.model)
    results = pipeline.run(X_final, y.values, test_size=args.test_size)

    print("\n✅ Evaluation metrics:")
    for k, v in results["metrics"].items():
        if k == "confusion_matrix":
            print(f"  {k}: {v}")
        elif v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 5. Save model
    pipeline.trainer.save_model(args.output)
    print(f"\n💾 Model saved to {args.output}")
    print("\n🎉 Modular pipeline run completed successfully.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
