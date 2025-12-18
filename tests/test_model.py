import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestModelTraining:
    @pytest.fixture
    def sample_data(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])
        return X, y
    
    def test_model_initialization(self):
        model = LogisticRegression(random_state=42)
        assert model is not None
    
    def test_model_training(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        assert hasattr(model, 'coef_')
    
    def test_model_prediction(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_predictions_are_binary(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_probability_predictions(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), 2)
    
    def test_probability_sum_to_one(self, sample_data):
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        sums = probabilities.sum(axis=1)
        assert np.allclose(sums, 1.0)

class TestModelEvaluation:
    @pytest.fixture
    def trained_model_data(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        return y, y_pred, y_proba
    
    def test_accuracy_score(self, trained_model_data):
        y, y_pred, _ = trained_model_data
        accuracy = accuracy_score(y, y_pred)
        assert 0 <= accuracy <= 1
    
    def test_precision_score(self, trained_model_data):
        y, y_pred, _ = trained_model_data
        precision = precision_score(y, y_pred, zero_division=0)
        assert 0 <= precision <= 1
    
    def test_recall_score(self, trained_model_data):
        y, y_pred, _ = trained_model_data
        recall = recall_score(y, y_pred, zero_division=0)
        assert 0 <= recall <= 1
    
    def test_f1_score(self, trained_model_data):
        y, y_pred, _ = trained_model_data
        f1 = f1_score(y, y_pred, zero_division=0)
        assert 0 <= f1 <= 1
    
    def test_roc_auc_score(self, trained_model_data):
        y, _, y_proba = trained_model_data
        auc = roc_auc_score(y, y_proba)
        assert 0 <= auc <= 1

class TestTrainTestSplit:
    def test_train_test_split_ratio(self):
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_stratified_split_preserves_distribution(self):
        X = np.random.rand(100, 5)
        y = np.array([0]*75 + [1]*25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)
        assert abs(train_ratio - 0.25) < 0.05
        assert abs(test_ratio - 0.25) < 0.05

class TestCrossValidation:
    def test_cross_val_score(self):
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model = LogisticRegression(random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
    
    def test_cross_val_mean_is_reasonable(self):
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model = LogisticRegression(random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        assert 0 <= mean_score <= 1
