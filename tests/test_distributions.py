import pytest
import numpy as np
from datetime import datetime

from syndata.distributions import (
    Distribution,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    DateTimeDistribution
)

# Add pytest marker to ensure discovery
pytestmark = pytest.mark.distributions

class TestNormalDistribution:
    def test_initialization(self):
        # Test valid initialization
        dist = NormalDistribution(mean=10, std=2)
        assert dist.mean == 10
        assert dist.std == 2

        # Test default values
        dist = NormalDistribution()
        assert dist.mean == 0.0
        assert dist.std == 1.0

    def test_validation(self):
        # Test valid parameters
        dist = NormalDistribution(mean=0, std=1)
        assert dist.validate_params() is True

        # Test invalid std
        with pytest.raises(ValueError, match="Standard deviation must be a positive numeric value"):
            NormalDistribution(mean=0, std=-1)

        # Test invalid mean type
        with pytest.raises(ValueError, match="Mean must be a numeric value"):
            NormalDistribution(mean="invalid", std=1)

    def test_generate(self):
        dist = NormalDistribution(mean=50, std=10)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check statistical properties (approximate)
        assert 45 <= np.mean(data) <= 55  # Mean should be close to 50
        assert 8 <= np.std(data) <= 12    # Std should be close to 10


class TestUniformDistribution:
    def test_initialization(self):
        # Test valid initialization
        dist = UniformDistribution(low=5, high=10)
        assert dist.low == 5
        assert dist.high == 10

        # Test default values
        dist = UniformDistribution()
        assert dist.low == 0.0
        assert dist.high == 1.0

    def test_validation(self):
        # Test valid parameters
        dist = UniformDistribution(low=0, high=10)
        assert dist.validate_params() is True

        # Test invalid range (high <= low)
        with pytest.raises(ValueError, match="High must be greater than low"):
            UniformDistribution(low=10, high=5)

        # Test invalid types
        with pytest.raises(ValueError, match="Low and high must be numeric values"):
            UniformDistribution(low="invalid", high=10)

    def test_generate(self):
        dist = UniformDistribution(low=0, high=100)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check range
        assert np.min(data) >= 0
        assert np.max(data) <= 100
        
        # Check distribution (approximate)
        assert 45 <= np.mean(data) <= 55  # Mean should be close to 50


class TestCategoricalDistribution:
    def test_initialization(self):
        # Test valid initialization with equal probabilities
        categories = ['A', 'B', 'C']
        dist = CategoricalDistribution(categories=categories)
        assert dist.categories == categories
        assert dist.probabilities is None

        # Test with explicit probabilities
        probs = [0.2, 0.3, 0.5]
        dist = CategoricalDistribution(categories=categories, probabilities=probs)
        assert dist.categories == categories
        assert dist.probabilities == probs

    def test_validation(self):
        # Test valid parameters
        dist = CategoricalDistribution(categories=['A', 'B'], probabilities=[0.3, 0.7])
        assert dist.validate_params() is True

        # Test empty categories
        with pytest.raises(ValueError, match="Categories must be a non-empty list"):
            CategoricalDistribution(categories=[])

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Probabilities must have the same length as categories"):
            CategoricalDistribution(categories=['A', 'B', 'C'], probabilities=[0.5, 0.5])

        # Test invalid probabilities (negative)
        with pytest.raises(ValueError, match="All probabilities must be non-negative numbers"):
            CategoricalDistribution(categories=['A', 'B'], probabilities=[-0.2, 1.2])

        # Test invalid probabilities (sum != 1)
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            CategoricalDistribution(categories=['A', 'B'], probabilities=[0.3, 0.3])

    def test_generate(self):
        categories = ['A', 'B', 'C']
        probs = [0.2, 0.3, 0.5]
        dist = CategoricalDistribution(categories=categories, probabilities=probs)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check all values are in categories
        assert set(np.unique(data)).issubset(set(categories))
        
        # Check approximate frequency
        values, counts = np.unique(data, return_counts=True)
        freq_dict = dict(zip(values, counts/len(data)))
        
        # Check frequencies are roughly as expected (with wide margin for randomness)
        for cat, prob in zip(categories, probs):
            assert abs(freq_dict.get(cat, 0) - prob) < 0.1


class TestBernoulliDistribution:
    def test_initialization(self):
        # Test valid initialization
        dist = BernoulliDistribution(p=0.7)
        assert dist.p == 0.7

        # Test default value
        dist = BernoulliDistribution()
        assert dist.p == 0.5

    def test_validation(self):
        # Test valid parameter
        dist = BernoulliDistribution(p=0.3)
        assert dist.validate_params() is True

        # Test invalid p (out of range)
        with pytest.raises(ValueError, match="Probability p must be a number between 0 and 1"):
            BernoulliDistribution(p=1.5)

        # Test invalid p type
        with pytest.raises(ValueError, match="Probability p must be a number between 0 and 1"):
            BernoulliDistribution(p="invalid")

    def test_generate(self):
        dist = BernoulliDistribution(p=0.7)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check values are 0 or 1
        assert set(np.unique(data)).issubset({0, 1})
        
        # Check approximate probability
        assert 0.65 <= np.mean(data) <= 0.75  # Should be close to 0.7


class TestPoissonDistribution:
    def test_initialization(self):
        # Test valid initialization
        dist = PoissonDistribution(lam=3.5)
        assert dist.lam == 3.5

        # Test default value
        dist = PoissonDistribution()
        assert dist.lam == 1.0

    def test_validation(self):
        # Test valid parameter
        dist = PoissonDistribution(lam=2)
        assert dist.validate_params() is True

        # Test invalid lambda (negative)
        with pytest.raises(ValueError, match="Lambda must be a positive number"):
            PoissonDistribution(lam=-1)

        # Test invalid lambda type
        with pytest.raises(ValueError, match="Lambda must be a positive number"):
            PoissonDistribution(lam="invalid")

    def test_generate(self):
        lam = 5
        dist = PoissonDistribution(lam=lam)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check all values are non-negative integers
        assert all(isinstance(x, (int, np.integer)) for x in data)
        assert np.min(data) >= 0
        
        # Check mean is approximately lambda
        assert 4.5 <= np.mean(data) <= 5.5


class TestDateTimeDistribution:
    def test_initialization(self):
        # Test valid initialization with strings
        dist = DateTimeDistribution(
            start_date='2020-01-01', 
            end_date='2021-01-01'
        )
        assert dist.start_date == datetime(2020, 1, 1)
        assert dist.end_date == datetime(2021, 1, 1)

        # Test with datetime objects
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        dist = DateTimeDistribution(start_date=start, end_date=end)
        assert dist.start_date == start
        assert dist.end_date == end

    def test_validation(self):
        # Test valid parameters
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        dist = DateTimeDistribution(start_date=start, end_date=end)
        assert dist.validate_params() is True

        # Test invalid date order
        with pytest.raises(ValueError, match="End date must be after start date"):
            DateTimeDistribution(start_date='2021-01-01', end_date='2020-01-01')

    def test_generate(self):
        start_date = '2020-01-01'
        end_date = '2021-01-01'
        dist = DateTimeDistribution(start_date=start_date, end_date=end_date)
        data = dist.generate(1000)
        
        # Check shape
        assert len(data) == 1000
        
        # Check all values are datetime objects
        assert all(isinstance(x, datetime) for x in data)
        
        # Check range
        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)
        assert all(start <= x <= end for x in data)
