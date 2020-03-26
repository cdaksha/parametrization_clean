
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.nakata import NakataMutate


@pytest.mark.usefixtures('get_individuals')
def test_single_point(get_individuals):
    parent = get_individuals[0]
    child = NakataMutate.mutation(parent, scale=0.1, rand_lower=-1.0, rand_higher=1.0)
    for param in child.params:
        low = min(0.9 * param, 1.1 * param)
        high = max(0.9 * param, 1.1 * param)
        assert low <= param <= high
