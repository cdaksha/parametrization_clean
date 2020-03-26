
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate


@pytest.mark.usefixtures('get_individuals', 'param_bounds')
def test_central_uniform(get_individuals, param_bounds):
    parent = get_individuals[0]
    child = CentralUniformMutate.mutation(parent, param_bounds=param_bounds)
    for param, param_bounds in zip(child.params, param_bounds):
        delta = (param_bounds[1] - param_bounds[0]) / 4
        low = param_bounds[0] + delta
        high = param_bounds[1] - delta
        assert low <= param <= high
