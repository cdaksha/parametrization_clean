
# Standard library

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual


def test_individual_init():
    params = [0.5, 0.4, 1.0, -4.7, 8.6]
    cost = 19876.42
    individual = Individual(params=params, cost=cost)
    assert individual.params == params
    assert individual.cost == cost





