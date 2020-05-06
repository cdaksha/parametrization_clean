
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate
from tests.domain.crossover.test_double_pareto import get_root_individual


@pytest.mark.usefixtures('get_individuals')
def test_polynomial(get_root_individual, get_individuals):
    parent = get_individuals[0]
    child = PolynomialMutate.mutation(parent, get_root_individual, polynomial_eta=60)
    assert parent.params != child.params
    assert parent.params == get_individuals[0].params
