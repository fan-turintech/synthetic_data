import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

from syndata.distributions import (
    NormalDistribution, 
    UniformDistribution, 
    CategoricalDistribution,
    BernoulliDistribution,
    PoissonDistribution,
    DateTimeDistribution
)
from syndata.generator import DataGenerator

def benchmark_row_count(row_counts: List[int] = None, 
                       num_columns: int = 10, 
                       iterations: int = 3) -> pd.DataFrame:
    """
    Benchmark data generation with different row counts
    
    Parameters:
    -----------
    row_counts : List[int], optional
        List of row counts to test
    num_columns : int, optional
        Number of columns in the test dataset
    iterations : int, optional
        Number of iterations to run for each configuration
        
    Returns:
    --------
    pd.DataFrame
        Benchmark results
    """
    if row_counts is None:
        row_counts = [100, 1000, 10000, 100000, 1000000]
        
    results = []
    
    for n_rows in row_counts:
        print(f"Benchmarking {n_rows} rows with {num_columns} columns...")
        
        # Create a generator with specified number of columns
        generator = DataGenerator()
        for i in range(num_columns):
            # Alternate between different types of distributions
            if i % 5 == 0:
                generator.add_column(f"normal_{i}", NormalDistribution(mean=0, std=1))
            elif i % 5 == 1:
                generator.add_column(f"uniform_{i}", UniformDistribution(low=0, high=100))
            elif i % 5 == 2:
                generator.add_column(f"categorical_{i}", 
                                    CategoricalDistribution(['A', 'B', 'C', 'D'], 
                                                          [0.4, 0.3, 0.2, 0.1]))
            elif i % 5 == 3:
                generator.add_column(f"bernoulli_{i}", BernoulliDistribution(p=0.7))
            else:
                generator.add_column(f"poisson_{i}", PoissonDistribution(lam=5))
                
        # Run multiple iterations and measure time
        times = []
        for _ in range(iterations):
            start_time = time.time()
            df = generator.generate(n_rows)
            end_time = time.time()
            times.append(end_time - start_time)
            
        # Record results
        avg_time = sum(times) / len(times)
        results.append({
            'rows': n_rows,
            'columns': num_columns,
            'avg_time': avg_time,
            'rows_per_second': n_rows / avg_time
        })
        
    return pd.DataFrame(results)

def benchmark_column_count(column_counts: List[int] = None, 
                          n_rows: int = 10000, 
                          iterations: int = 3) -> pd.DataFrame:
    """
    Benchmark data generation with different column counts
    
    Parameters:
    -----------
    column_counts : List[int], optional
        List of column counts to test
    n_rows : int, optional
        Number of rows in the test dataset
    iterations : int, optional
        Number of iterations to run for each configuration
        
    Returns:
    --------
    pd.DataFrame
        Benchmark results
    """
    if column_counts is None:
        column_counts = [5, 10, 25, 50, 100]
        
    results = []
    
    for n_cols in column_counts:
        print(f"Benchmarking {n_cols} columns with {n_rows} rows...")
        
        # Create a generator with specified number of columns
        generator = DataGenerator()
        for i in range(n_cols):
            # Alternate between different types of distributions
            if i % 5 == 0:
                generator.add_column(f"normal_{i}", NormalDistribution(mean=0, std=1))
            elif i % 5 == 1:
                generator.add_column(f"uniform_{i}", UniformDistribution(low=0, high=100))
            elif i % 5 == 2:
                generator.add_column(f"categorical_{i}", 
                                    CategoricalDistribution(['A', 'B', 'C', 'D'], 
                                                          [0.4, 0.3, 0.2, 0.1]))
            elif i % 5 == 3:
                generator.add_column(f"bernoulli_{i}", BernoulliDistribution(p=0.7))
            else:
                generator.add_column(f"poisson_{i}", PoissonDistribution(lam=5))
                
        # Run multiple iterations and measure time
        times = []
        for _ in range(iterations):
            start_time = time.time()
            df = generator.generate(n_rows)
            end_time = time.time()
            times.append(end_time - start_time)
            
        # Record results
        avg_time = sum(times) / len(times)
        results.append({
            'rows': n_rows,
            'columns': n_cols,
            'avg_time': avg_time,
            'cols_per_second': n_cols / avg_time
        })
        
    return pd.DataFrame(results)

def benchmark_distribution_types(n_rows: int = 100000, iterations: int = 3) -> pd.DataFrame:
    """
    Benchmark performance of different distribution types
    
    Parameters:
    -----------
    n_rows : int, optional
        Number of rows to generate
    iterations : int, optional
        Number of iterations to run for each configuration
        
    Returns:
    --------
    pd.DataFrame
        Benchmark results
    """
    distribution_configs = [
        ("Normal", NormalDistribution(mean=0, std=1)),
        ("Uniform", UniformDistribution(low=0, high=100)),
        ("Categorical", CategoricalDistribution(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])),
        ("Bernoulli", BernoulliDistribution(p=0.7)),
        ("Poisson", PoissonDistribution(lam=5)),
        ("DateTime", DateTimeDistribution())
    ]
    
    results = []
    
    for dist_name, dist_obj in distribution_configs:
        print(f"Benchmarking {dist_name} distribution with {n_rows} rows...")
        
        generator = DataGenerator()
        generator.add_column("test_column", dist_obj)
        
        # Run multiple iterations and measure time
        times = []
        for _ in range(iterations):
            start_time = time.time()
            df = generator.generate(n_rows)
            end_time = time.time()
            times.append(end_time - start_time)
            
        # Record results
        avg_time = sum(times) / len(times)
        results.append({
            'distribution': dist_name,
            'rows': n_rows,
            'avg_time': avg_time,
            'rows_per_second': n_rows / avg_time
        })
        
    return pd.DataFrame(results)

def plot_benchmark_results(df: pd.DataFrame, 
                          x_col: str, 
                          y_col: str, 
                          title: str, 
                          xlabel: str, 
                          ylabel: str,
                          log_scale: bool = True) -> None:
    """Plot benchmark results"""
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
    
    if log_scale and all(df[x_col] > 0) and all(df[y_col] > 0):
        plt.xscale('log')
        plt.yscale('log')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

def save_results(benchmark_results: Dict[str, pd.DataFrame], output_dir: str = "benchmark_results"):
    """Save benchmark results to CSV files and plots"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in benchmark_results.items():
        # Save to CSV
        df.to_csv(f"{output_dir}/{name}.csv", index=False)
        
        # Create plots for each benchmark type
        if name == "row_benchmark":
            plot_benchmark_results(
                df, 
                x_col="rows", 
                y_col="avg_time", 
                title="Generation Time vs. Number of Rows", 
                xlabel="Number of Rows", 
                ylabel="Time (seconds)",
                log_scale=True
            )
            plt.savefig(f"{output_dir}/row_time_benchmark.png")
            
            plot_benchmark_results(
                df, 
                x_col="rows", 
                y_col="rows_per_second", 
                title="Generation Throughput vs. Number of Rows", 
                xlabel="Number of Rows", 
                ylabel="Rows per Second",
                log_scale=True
            )
            plt.savefig(f"{output_dir}/row_throughput_benchmark.png")
            
        elif name == "column_benchmark":
            plot_benchmark_results(
                df, 
                x_col="columns", 
                y_col="avg_time", 
                title="Generation Time vs. Number of Columns", 
                xlabel="Number of Columns", 
                ylabel="Time (seconds)",
                log_scale=False
            )
            plt.savefig(f"{output_dir}/column_time_benchmark.png")
            
        elif name == "distribution_benchmark":
            # Bar chart for distribution comparison
            plt.figure(figsize=(10, 6))
            plt.bar(df["distribution"], df["rows_per_second"])
            plt.title("Generation Performance by Distribution Type")
            plt.xlabel("Distribution Type")
            plt.ylabel("Rows per Second")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/distribution_benchmark.png")

def main():
    """Run benchmarks and save results"""
    print("Starting benchmarks...")
    
    # Customize these parameters based on your system capabilities
    row_counts = [100, 1000, 10000, 100000]  # You can add 1000000 if you have enough memory
    column_counts = [5, 10, 25, 50, 100]
    iterations = 3  # More iterations for more stable results
    
    # Run benchmarks
    row_df = benchmark_row_count(row_counts=row_counts, iterations=iterations)
    col_df = benchmark_column_count(column_counts=column_counts, iterations=iterations)
    dist_df = benchmark_distribution_types(iterations=iterations)
    
    # Store and save results
    benchmark_results = {
        "row_benchmark": row_df,
        "column_benchmark": col_df,
        "distribution_benchmark": dist_df
    }
    
    save_results(benchmark_results)
    
    print("Benchmarks completed. Results saved to 'benchmark_results' directory.")
    
    # Print summary of key findings
    print("\nKey findings:")
    print(f"- Max data generation speed: {row_df['rows_per_second'].max():.2f} rows per second")
    print(f"- Fastest distribution: {dist_df.loc[dist_df['rows_per_second'].idxmax(), 'distribution']} "
          f"({dist_df['rows_per_second'].max():.2f} rows/sec)")
    print(f"- Slowest distribution: {dist_df.loc[dist_df['rows_per_second'].idxmin(), 'distribution']} "
          f"({dist_df['rows_per_second'].min():.2f} rows/sec)")

if __name__ == "__main__":
    main()
