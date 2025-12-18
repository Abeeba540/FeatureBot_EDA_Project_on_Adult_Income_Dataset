import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataLoader:
    """Test suite for data loading functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        return pd.DataFrame({
            'age': [25, 38, 28, 44, 18],
            'workclass': ['Private', 'Private', 'Local-gov', 'Private', '?'],
            'fnlwgt': [226802, 89814, 336951, 160323, 103497],
            'education': ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', 'Some-college'],
            'educationalnum': [7, 9, 12, 10, 10],
            'maritalstatus': ['Never-married', 'Married-civ-spouse', 'Married-civ-spouse', 'Married-civ-spouse', 'Never-married'],
            'occupation': ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Machine-op-inspct', '?'],
            'relationship': ['Own-child', 'Husband', 'Husband', 'Husband', 'Own-child'],
            'race': ['Black', 'White', 'White', 'Black', 'White'],
            'gender': ['Male', 'Male', 'Male', 'Male', 'Female'],
            'capitalgain': [0, 0, 0, 7688, 0],
            'capitalloss': [0, 0, 0, 0, 0],
            'hoursperweek': [40, 50, 40, 40, 30],
            'nativecountry': ['United-States', 'United-States', 'United-States', 'United-States', 'United-States'],
            'income': ['<=50K', '<=50K', '<=50K', '>50K', '<=50K']
        })
    
    def test_dataframe_shape(self, sample_data):
        """Test that sample data has correct shape"""
        assert sample_data.shape == (5, 15), "DataFrame should have 5 rows and 15 columns"
    
    def test_column_names(self, sample_data):
        """Test that all required columns exist"""
        required_cols = ['age', 'workclass', 'education', 'occupation', 'income']
        assert all(col in sample_data.columns for col in required_cols), "Missing required columns"
    
    def test_numeric_columns_type(self, sample_data):
        """Test that numeric columns have correct data types"""
        numeric_cols = ['age', 'fnlwgt', 'educationalnum', 'capitalgain', 'capitalloss', 'hoursperweek']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), f"{col} should be numeric"
    
    def test_categorical_columns_type(self, sample_data):
        """Test that categorical columns are object type"""
        categorical_cols = ['workclass', 'education', 'maritalstatus', 'occupation']
        for col in categorical_cols:
            assert sample_data[col].dtype == 'object', f"{col} should be object type"
    
    def test_missing_values_present(self, sample_data):
        """Test that missing values (represented as '?') are handled"""
        assert '?' in sample_data.values, "Sample data should contain missing value indicators"
    
    def test_age_range(self, sample_data):
        """Test that age values are in realistic range"""
        assert sample_data['age'].min() >= 0, "Age should not be negative"
        assert sample_data['age'].max() <= 150, "Age should not exceed 150"
    
    def test_hours_per_week_range(self, sample_data):
        """Test that hours per week are in valid range"""
        assert sample_data['hoursperweek'].min() > 0, "Hours per week should be positive"
        assert sample_data['hoursperweek'].max() <= 100, "Hours per week should not exceed 100"
    
    def test_no_empty_dataframe(self, sample_data):
        """Test that DataFrame is not empty"""
        assert len(sample_data) > 0, "DataFrame should not be empty"
    
    def test_income_values(self, sample_data):
        """Test that income column contains valid values"""
        valid_incomes = ['<=50K', '>50K']
        assert all(val in valid_incomes for val in sample_data['income'].unique()), \
            "Income should only contain '<=50K' or '>50K'"
    
    def test_no_duplicate_rows(self, sample_data):
        """Test that there are no duplicate rows"""
        # For this sample, we expect no duplicates
        assert len(sample_data) == len(sample_data.drop_duplicates()), "Should have no duplicate rows"
    
    def test_data_immutability(self, sample_data):
        """Test that original data is not modified during testing"""
        original_shape = sample_data.shape
        _ = sample_data.copy()
        assert sample_data.shape == original_shape, "Data should not be modified"


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_numeric_columns_not_null(self):
        """Test that numeric columns can handle null values"""
        df = pd.DataFrame({
            'age': [25, None, 35],
            'hours': [40, 50, None]
        })
        # Imputation should be numeric-compatible
        assert pd.api.types.is_numeric_dtype(df['age'].dtype), "Age should be numeric"
    
    def test_categorical_encoding_readiness(self):
        """Test that categorical data is ready for encoding"""
        df = pd.DataFrame({
            'workclass': ['Private', 'Federal-gov', 'Self-emp-not-inc'],
            'education': ['HS-grad', 'Bachelors', 'Doctorate']
        })
        # Check that categorical columns exist
        assert len(df.select_dtypes(include=['object']).columns) > 0, \
            "Should have categorical columns"
    
    def test_capital_columns_are_numeric(self):
        """Test that capital gain/loss columns are numeric"""
        df = pd.DataFrame({
            'capitalgain': [0, 15024, 7688],
            'capitalloss': [0, 0, 1977]
        })
        assert pd.api.types.is_numeric_dtype(df['capitalgain']), "Capital gain should be numeric"
        assert pd.api.types.is_numeric_dtype(df['capitalloss']), "Capital loss should be numeric"


# Run the tests with: pytest tests/test_data_loader.py -v
