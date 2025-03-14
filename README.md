# Synthetic Data Generator

A Python library for generating synthetic data with various distributions.

This project is generated completely by AI.

## Features

- Generate data with various statistical distributions
- Support for common data types (numeric, categorical, boolean, datetime)
- Control the percentage of missing values
- Return data as pandas DataFrames for easy manipulation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic_data.git
cd synthetic_data

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from syndata.distributions import NormalDistribution, CategoricalDistribution
from syndata.generator import DataGenerator

# Create a data generator
gen = DataGenerator()

# Add columns with different distributions
gen.add_column("age", NormalDistribution(mean=35, std=12))
gen.add_column("income", NormalDistribution(mean=50000, std=15000))
gen.add_column("category", CategoricalDistribution(
    categories=["A", "B", "C"], 
    probabilities=[0.6, 0.3, 0.1]
))

# Generate 1000 samples
data = gen.generate(1000)
print(data.head())
```

## Benchmarks

The library includes a benchmarking script to test the performance of data generation with various configurations.

To run the benchmarks:

```bash
python benchmark.py
```

This will:
1. Test data generation with varying numbers of rows
2. Test data generation with varying numbers of columns
3. Compare performance across different distribution types
4. Save results as CSV files and plots in the `benchmark_results` directory

You can customize the benchmark parameters by modifying the variables in the `main()` function of `benchmark.py`.

## Available Distributions

- `NormalDistribution`: For continuous data following a normal (Gaussian) distribution
- `UniformDistribution`: For data uniformly distributed between a min and max value
- `CategoricalDistribution`: For categorical data with specified probabilities
- `BernoulliDistribution`: For binary outcomes (0 or 1)
- `PoissonDistribution`: For count data (non-negative integers)
- `DateTimeDistribution`: For date and time data within a specified range

## Examples

See the `examples` directory for more detailed examples.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- SciPy
