import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from .distributions import Distribution

class DataGenerator:
    """Main class for generating synthetic data"""
    
    def __init__(self):
        self.columns = {}
        
    def add_column(self, name: str, distribution: Distribution, missing_rate: float = 0.0):
        """
        Add a column to the synthetic dataset
        
        Parameters:
        -----------
        name : str
            Column name
        distribution : Distribution
            Distribution object that will generate data
        missing_rate : float, optional (default=0.0)
            Percentage of missing values in the column (between 0 and 1)
        """
        if missing_rate < 0 or missing_rate > 1:
            raise ValueError("Missing rate must be between 0 and 1")
            
        self.columns[name] = {
            'distribution': distribution,
            'missing_rate': missing_rate
        }
        
    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate a synthetic DataFrame
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        pd.DataFrame
            Generated synthetic data
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
            
        if not self.columns:
            raise ValueError("No columns defined. Use add_column() to define columns first.")
            
        data = {}
        
        for col_name, col_config in self.columns.items():
            # Generate data using the column's distribution
            col_data = col_config['distribution'].generate(n_samples)
            
            # Apply missing values if specified
            missing_rate = col_config['missing_rate']
            if missing_rate > 0:
                mask = np.random.random(size=n_samples) < missing_rate
                col_data = pd.Series(col_data)
                col_data[mask] = np.nan
            
            data[col_name] = col_data
            
        return pd.DataFrame(data)
