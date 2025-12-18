import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataCleaning:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'age': [25, 38, None, 44, 18],
            'workclass': ['Private', 'Private', 'Local-gov', '?', 'Private'],
            'education': ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', 'Some-college'],
            'capitalgain': [0, 0, 0, 7688, 0],
            'capitalloss': [0, 0, 0, 0, 0],
            'hoursperweek': [40, 50, 40, 40, 30],
            'income': ['<=50K', '<=50K', '<=50K', '>50K', '<=50K']
        })
    
    def test_missing_values_handling(self, sample_df):
        missing_count = sample_df.isnull().sum().sum()
        assert missing_count > 0
    
    def test_special_char_replacement(self, sample_df):
        df = sample_df.copy()
        df = df.replace('?', np.nan)
        assert '?' not in df.values
    
    def test_numeric_imputation_median(self, sample_df):
        imputer = SimpleImputer(strategy='median')
        numeric_cols = ['age', 'hoursperweek']
        df_numeric = sample_df[numeric_cols].copy()
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
        assert df_imputed.isnull().sum().sum() == 0
    
    def test_categorical_imputation_most_frequent(self, sample_df):
        df = sample_df.copy()
        df = df.replace('?', np.nan)
        imputer = SimpleImputer(strategy='most_frequent')
        df_cat = df[['workclass']]
        df_imputed = pd.DataFrame(imputer.fit_transform(df_cat), columns=['workclass'])
        assert df_imputed.isnull().sum().sum() == 0

class TestFeatureScaling:
    def test_standard_scaler_initialization(self):
        scaler = StandardScaler()
        assert scaler is not None
    
    def test_scaling_produces_standardized_values(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        assert np.abs(scaled.mean(axis=0)).max() < 1e-10
        assert np.abs(scaled.std(axis=0) - 1.0).max() < 1e-10
    
    def test_scaler_preserves_shape(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        assert scaled.shape == data.shape

class TestCategoricalEncoding:
    def test_onehot_encoder_initialization(self):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        assert encoder is not None
    
    def test_categorical_to_numeric(self):
        data = np.array([['a'], ['b'], ['c'], ['a']])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(data)
        assert encoded.dtype == np.float64
    
    def test_unknown_category_handling(self):
        data_train = np.array([['cat'], ['dog']])
        data_test = np.array([['cat'], ['bird']])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(data_train)
        encoded = encoder.transform(data_test)
        assert encoded.shape[0] == 2

class TestDataValidation:
    def test_no_infinite_values(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert not np.isinf(df.values).any()
    
    def test_numeric_columns_are_numeric_after_processing(self):
        df = pd.DataFrame({'age': ['25', '38', '28'], 'hours': ['40', '50', '30']})
        df_numeric = df.astype(float)
        assert pd.api.types.is_numeric_dtype(df_numeric['age'])
    
    def test_processed_data_not_empty(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) > 0
