
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.nakata import NakataMutate
from tests.domain.crossover.test_double_pareto import get_root_individual


@pytest.mark.usefixtures('get_individuals')
def test_single_point(get_root_individual, get_individuals):
    parent = get_individuals[0]
    child = NakataMutate.mutation(parent, get_root_individual,
                                  nakata_scale=0.1, nakata_rand_lower=-1.0, nakata_rand_higher=1.0)
    for param in child.params:
        low = min(0.9 * param, 1.1 * param)
        high = max(0.9 * param, 1.1 * param)
        assert low <= param <= high
