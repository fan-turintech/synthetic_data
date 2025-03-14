from setuptools import setup, find_packages

setup(
    name="syndata",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for generating synthetic data based on user requirements",
    keywords="synthetic, data, generation, pandas",
    python_requires=">=3.7",
)
