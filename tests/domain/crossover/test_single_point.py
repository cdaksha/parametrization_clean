
# Standard library
from unittest.mock import patch

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.crossover.single_point import SinglePointCross
from tests.domain.crossover.test_double_pareto import get_root_individual


@patch('random.randrange')
@pytest.mark.usefixtures('get_individuals')
def test_single_point(randrange_mock, get_root_individual, get_individuals):
    randrange_mock.return_value = 2
    parent1 = get_individuals[0]
    parent2 = get_individuals[1]
    child1, child2 = SinglePointCross.crossover(parent1, parent2, root_individual=get_root_individual)
    assert child1.params == [1.0, 0.1, -0.76, 8.6, 2.1]
    assert child2.params == [0.5, 0.05, -0.5, 4.3, 2.4]

