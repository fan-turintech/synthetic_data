import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from syndata import (
    DataGenerator,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    DateTimeDistribution
)

# Add pytest marker to ensure discovery
pytestmark = pytest.mark.integration

class TestIntegration:
    def test_full_workflow(self):
        """Test the full workflow from creating a generator to producing data"""
        
        # Create generator
        generator = DataGenerator()
        
        # Add various column types
        generator.add_column('age', NormalDistribution(mean=35, std=10))
        generator.add_column('income', NormalDistribution(mean=50000, std=15000), missing_rate=0.1)
        generator.add_column('score', UniformDistribution(low=0, high=100))
        generator.add_column('category', CategoricalDistribution(
            categories=['A', 'B', 'C', 'D'], 
            probabilities=[0.4, 0.3, 0.2, 0.1]
        ))
        generator.add_column('is_active', BernoulliDistribution(p=0.7))
        generator.add_column('num_purchases', PoissonDistribution(lam=3))
        generator.add_column('registration_date', DateTimeDistribution(
            start_date='2020-01-01',
            end_date='2023-01-01'
        ))
        
        # Generate data
        n_samples = 1000
        df = generator.generate(n_samples)
        
        # Verify basics
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_samples
        assert list(df.columns) == ['age', 'income', 'score', 'category', 
                                   'is_active', 'num_purchases', 'registration_date']
        
        # Check expected data types
        assert pd.api.types.is_numeric_dtype(df['age'])
        assert pd.api.types.is_numeric_dtype(df['income'])
        assert pd.api.types.is_numeric_dtype(df['score'])
        assert pd.api.types.is_object_dtype(df['category'])
        assert pd.api.types.is_numeric_dtype(df['is_active'])
        assert pd.api.types.is_numeric_dtype(df['num_purchases'])
        assert pd.api.types.is_datetime64_dtype(df['registration_date']) or all(isinstance(x, datetime) for x in df['registration_date'])
        
        # Check missing values
        assert df['age'].isna().sum() == 0
        assert 80 <= df['income'].isna().sum() <= 120  # ~10% of 1000 samples
        
        # Check data ranges and distributions
        assert 0 <= df['score'].min() <= df['score'].max() <= 100
        assert set(df['category'].unique()).issubset({'A', 'B', 'C', 'D'})
        assert set(df['is_active'].unique()).issubset({0, 1})
        assert all(isinstance(x, (int, np.integer)) and x >= 0 for x in df['num_purchases'].dropna())
        
        # Verify registration_date range
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)
        dates = [d for d in df['registration_date'] if pd.notna(d)]
        assert all(start <= d <= end for d in dates)

    def test_data_reproducibility(self):
        """Test data reproducibility with fixed random seed"""
        
        # Set random seed
        np.random.seed(42)
        
        # Create and generate first dataset
        gen1 = DataGenerator()
        gen1.add_column('value', NormalDistribution(mean=0, std=1))
        df1 = gen1.generate(100)
        
        # Reset seed and create second dataset
        np.random.seed(42)
        gen2 = DataGenerator()
        gen2.add_column('value', NormalDistribution(mean=0, std=1))
        df2 = gen2.generate(100)
        
        # Check that the generated values are identical
        pd.testing.assert_series_equal(df1['value'], df2['value'])
