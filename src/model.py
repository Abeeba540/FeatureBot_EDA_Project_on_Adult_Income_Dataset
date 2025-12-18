import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
from typing import Tuple, Dict


class ModelTrainer:
    """Train and evaluate ML models"""

    def __init__(self, model_type: str = "logistic_regression"):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support predict_proba")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = (
            self.predict_proba(X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if y_proba is not None else None,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Cross-validation evaluation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
        }

    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        joblib.dump(self.model, filepath)

    @staticmethod
    def load_model(filepath: str):
        """Load model from disk"""
        return joblib.load(filepath)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])
        else:
            raise AttributeError("Model does not have feature importance")


class ModelPipeline:
    """End-to-end ML pipeline"""

    def __init__(self, model_type: str = "logistic_regression"):
        self.trainer = ModelTrainer(model_type)
        self.metrics = None

    def run(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Run complete pipeline"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.trainer.train(X_train, y_train)
        self.metrics = self.trainer.evaluate(X_test, y_test)

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": self.metrics,
        }
