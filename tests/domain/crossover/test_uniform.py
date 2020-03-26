
# Standard library
from unittest.mock import patch

# 3rd party packages
import pytest
import numpy as np

# Local source
from parametrization_clean.domain.crossover.uniform import UniformCross


@patch('numpy.random.randint')
@pytest.mark.usefixtures('get_individuals')
def test_uniform(randint_mock, get_individuals):
    randint_mock.return_value = np.array([1, 0, 1, 0, 0])
    parent1 = get_individuals[0]
    parent2 = get_individuals[1]
    child1, child2 = UniformCross.crossover(parent1, parent2)
    assert child1.params == [1.0, 0.05, -0.5, 8.6, 2.1]
    assert child2.params == [0.5, 0.1, -0.76, 4.3, 2.4]
