import argparse
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

    # 1. Load and validate data
    loader = DataLoader()
    print(f"📥 Loading data from {args.data} ...")
    df = loader.load_data(args.data)
    df = loader.clean_data(df)

    if not loader.validate_data(df):
        raise ValueError("Data validation failed.")

    df = loader.normalize_target(df)
    print(f"✅ Loaded {len(df)} rows")

    # 2. Separate features/target and preprocess
    X = df.drop("income", axis=1)
    y = df["income"]

    pre = DataPreprocessor()
    X_clean = pre.handle_missing_values(X)

    # 3. Train pipeline
    pipeline = ModelPipeline(args.model)
    results = pipeline.run(X_clean.values, y.values, test_size=args.test_size)

    print("✅ Evaluation metrics:")
    for k, v in results["metrics"].items():
        if k == "confusion_matrix" or v is None:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    # 4. Save model
    pipeline.trainer.save_model(args.output)
    print(f"💾 Model saved to {args.output}")


if __name__ == "__main__":
    main()
