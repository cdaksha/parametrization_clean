
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.mutation.polynomial import PolynomialMutate


@pytest.mark.usefixtures('get_individuals')
def test_polynomial(get_individuals):
    parent = get_individuals[0]
    child = PolynomialMutate.mutation(parent, polynomial_eta=60)
    assert parent.params != child.params
    assert parent.params == get_individuals[0].params
