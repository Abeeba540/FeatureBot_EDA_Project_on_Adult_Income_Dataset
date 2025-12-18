import pandas as pd
import numpy as np
from typing import Tuple


class DataLoader:
    """Load and validate adult income dataset"""

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load CSV data and normalize columns"""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Replace missing value indicators and handle nulls"""
        df = df.replace("?", np.nan)
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df.empty:
            return False
        if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) > 0.2:
            return False
        return True

    @staticmethod
    def normalize_target(df: pd.DataFrame, target_col: str = "income") -> pd.DataFrame:
        """Convert income to binary"""
        df[target_col] = df[target_col].astype(str).str.strip()
        df[target_col] = df[target_col].str.replace(".", "", regex=False)
        df[target_col] = df[target_col].map({"<=50K": 0, ">50K": 1})
        return df
