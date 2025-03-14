import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

class Distribution(ABC):
    """Base abstract class for all distribution types"""
    
    @abstractmethod
    def generate(self, size: int) -> np.ndarray:
        """Generate random data of the specified size"""
        pass
    
    @abstractmethod
    def validate_params(self) -> bool:
        """Validate the distribution parameters"""
        pass


class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution generator"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not isinstance(self.mean, (int, float)):
            raise ValueError("Mean must be a numeric value")
        if not isinstance(self.std, (int, float)) or self.std <= 0:
            raise ValueError("Standard deviation must be a positive numeric value")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        return np.random.normal(loc=self.mean, scale=self.std, size=size)


class UniformDistribution(Distribution):
    """Uniform distribution generator"""
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not isinstance(self.low, (int, float)) or not isinstance(self.high, (int, float)):
            raise ValueError("Low and high must be numeric values")
        if self.high <= self.low:
            raise ValueError("High must be greater than low")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=size)


class CategoricalDistribution(Distribution):
    """Categorical distribution generator"""
    
    def __init__(self, categories: List[Any], probabilities: Optional[List[float]] = None):
        self.categories = categories
        self.probabilities = probabilities
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not self.categories or not isinstance(self.categories, list):
            raise ValueError("Categories must be a non-empty list")
            
        if self.probabilities is not None:
            if not isinstance(self.probabilities, list):
                raise ValueError("Probabilities must be a list")
            if len(self.probabilities) != len(self.categories):
                raise ValueError("Probabilities must have the same length as categories")
            if not all(isinstance(p, (int, float)) and p >= 0 for p in self.probabilities):
                raise ValueError("All probabilities must be non-negative numbers")
            if abs(sum(self.probabilities) - 1.0) > 1e-10:
                raise ValueError("Probabilities must sum to 1")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        return np.random.choice(self.categories, size=size, p=self.probabilities)


class BernoulliDistribution(Distribution):
    """Bernoulli distribution generator (binary outcomes)"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not isinstance(self.p, (int, float)) or self.p < 0 or self.p > 1:
            raise ValueError("Probability p must be a number between 0 and 1")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        return np.random.binomial(n=1, p=self.p, size=size)


class PoissonDistribution(Distribution):
    """Poisson distribution generator"""
    
    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not isinstance(self.lam, (int, float)) or self.lam <= 0:
            raise ValueError("Lambda must be a positive number")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        return np.random.poisson(lam=self.lam, size=size)


class DateTimeDistribution(Distribution):
    """DateTime distribution generator"""
    
    def __init__(self, start_date: Union[str, datetime] = '2020-01-01', 
                 end_date: Union[str, datetime] = '2023-01-01',
                 format: str = '%Y-%m-%d'):
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, format)
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, format)
        else:
            self.end_date = end_date
            
        self.format = format
        self.validate_params()
        
    def validate_params(self) -> bool:
        if not isinstance(self.start_date, datetime) or not isinstance(self.end_date, datetime):
            raise ValueError("Start date and end date must be datetime objects or valid date strings")
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        return True
        
    def generate(self, size: int) -> np.ndarray:
        time_delta = (self.end_date - self.start_date).total_seconds()
        random_seconds = np.random.uniform(0, time_delta, size=size)
        return np.array([self.start_date + timedelta(seconds=s) for s in random_seconds])
