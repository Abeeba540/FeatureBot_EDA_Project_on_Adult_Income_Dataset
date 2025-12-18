import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple

class DataPreprocessor:
    """Data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols])
        
        return df
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize numeric features"""
        return self.scaler.fit_transform(X)
    
    def encode_categorical(self, X_categorical: pd.DataFrame) -> np.ndarray:
        """One-hot encode categorical features"""
        return self.encoder.fit_transform(X_categorical)


@'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple

class DataPreprocessor:
    """Data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer_numeric.fit_transform(df[numeric_cols])
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.imputer_categorical.fit_transform(df[categorical_cols])
        
        return df
    
    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize numeric features"""
        return self.scaler.fit_transform(X)
    
    def encode_categorical(self, X_categorical: pd.DataFrame) -> np.ndarray:
        """One-hot encode categorical features"""
        return self.encoder.fit_transform(X_categorical)
    
    def remove_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score"""
        df = df.copy()
        if column in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        return df
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Simple undersampling for class balance"""
        df = X.copy()
        df['target'] = y
        
        class_counts = df['target'].value_counts()
        min_count = class_counts.min()
        
        balanced = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min_count, random_state=42)
        )
        
        X_balanced = balanced.drop('target', axis=1)
        y_balanced = balanced['target']
        
        return X_balanced, y_balanced
