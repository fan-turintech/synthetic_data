import sys
import os
import pandas as pd

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from syndata import (
    DataGenerator,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    DateTimeDistribution
)

def main():
    # Create a data generator
    generator = DataGenerator()
    
    # Add columns with different distributions
    generator.add_column('age', NormalDistribution(mean=35, std=10))
    generator.add_column('income', NormalDistribution(mean=50000, std=15000), missing_rate=0.05)
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
    
    # Generate 1000 rows of synthetic data
    df = generator.generate(1000)
    
    # Display the first few rows
    print(df.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Display data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Save to CSV (optional)
    df.to_csv('synthetic_data.csv', index=False)
    print("\nData saved to 'synthetic_data.csv'")

if __name__ == '__main__':
    main()
