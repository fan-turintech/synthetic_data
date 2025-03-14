import pytest
import sys
import os
import numpy as np

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

# Print debugging information
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")
print(f"Available tests: {os.listdir(os.path.dirname(__file__))}")

# Add a seed fixture for reproducible tests
@pytest.fixture(scope="function")
def seed_random():
    """Fixture to set a fixed random seed for tests."""
    np.random.seed(42)
    yield
    # Reset the seed after the test
    np.random.seed(None)
