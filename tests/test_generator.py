import pytest
import pandas as pd
import numpy as np

from syndata.generator import DataGenerator
from syndata.distributions import (
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution,
    BernoulliDistribution
)

# Add pytest marker to ensure discovery
pytestmark = pytest.mark.generator

class TestDataGenerator:
    def test_initialization(self):
        generator = DataGenerator()
        assert generator.columns == {}

    def test_add_column(self):
        generator = DataGenerator()
        
        # Test adding a column with valid parameters
        generator.add_column('age', NormalDistribution(mean=30, std=5))
        assert 'age' in generator.columns
        assert generator.columns['age']['distribution'].mean == 30
        assert generator.columns['age']['distribution'].std == 5
        assert generator.columns['age']['missing_rate'] == 0.0
        
        # Test adding a column with missing rate
        generator.add_column('income', NormalDistribution(mean=50000, std=10000), missing_rate=0.1)
        assert 'income' in generator.columns
        assert generator.columns['income']['missing_rate'] == 0.1
        
        # Test invalid missing rate
        with pytest.raises(ValueError, match="Missing rate must be between 0 and 1"):
            generator.add_column('invalid', NormalDistribution(), missing_rate=1.5)

    def test_generate_empty(self):
        generator = DataGenerator()
        
        # Test generating with no columns
        with pytest.raises(ValueError, match="No columns defined"):
            generator.generate(100)
            
        # Test invalid sample count
        generator.add_column('test', NormalDistribution())
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            generator.generate(0)
            
    def test_generate_with_columns(self):
        generator = DataGenerator()
        
        # Add multiple columns with different distributions
        generator.add_column('age', NormalDistribution(mean=35, std=5))
        generator.add_column('score', UniformDistribution(low=0, high=100))
        generator.add_column('category', CategoricalDistribution(
            categories=['A', 'B', 'C'], 
            probabilities=[0.2, 0.3, 0.5]
        ))
        generator.add_column('active', BernoulliDistribution(p=0.7))
        
        # Generate data
        n_samples = 500
        df = generator.generate(n_samples)
        
        # Check DataFrame properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_samples
        assert list(df.columns) == ['age', 'score', 'category', 'active']
        
        # Check each column has the right data
        assert df['age'].dtype == np.float64
        assert df['score'].dtype == np.float64
        assert pd.api.types.is_object_dtype(df['category'])
        assert df['active'].dtype == np.int64
        
        # Check value ranges
        assert 20 <= df['age'].mean() <= 50
        assert 0 <= df['score'].min() <= df['score'].max() <= 100
        assert set(df['category'].unique()).issubset({'A', 'B', 'C'})
        assert set(df['active'].unique()).issubset({0, 1})

    def test_missing_values(self):
        generator = DataGenerator()
        
        # Add column with missing values
        missing_rate = 0.3
        generator.add_column('data', NormalDistribution(), missing_rate=missing_rate)
        
        # Generate data
        n_samples = 1000
        df = generator.generate(n_samples)
        
        # Check missing value rate is approximately as expected
        missing_count = df['data'].isna().sum()
        observed_rate = missing_count / n_samples
        
        # Allow some deviation due to randomness
        assert abs(observed_rate - missing_rate) < 0.05
