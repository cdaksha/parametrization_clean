
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.adaptation.srinivas import SrinivasAdapt


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@pytest.mark.usefixtures('get_individuals')
def test_srinivas_1(get_individuals, root_individual):
    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)

    average_cost = 500.0
    minimum_cost = min(costs)
    parent_costs = (get_individuals[2].cost, get_individuals[3].cost)
    crossover_rate, mutation_rates = SrinivasAdapt.adaptation(average_cost, minimum_cost, parent_costs)
    assert crossover_rate == 1.0
    assert mutation_rates[0] == 0.5
    assert mutation_rates[1] == 0.5


@pytest.mark.usefixtures('get_individuals')
def test_srinivas_2(get_individuals, root_individual):
    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)

    average_cost = 1000.0
    minimum_cost = min(costs)
    parent_costs = (get_individuals[2].cost, get_individuals[3].cost)
    crossover_rate, mutation_rates = SrinivasAdapt.adaptation(average_cost, minimum_cost, parent_costs)
    assert crossover_rate == pytest.approx(0.983, rel=1e-3)
    assert mutation_rates[0] == pytest.approx(0.491, rel=1e-3)
    assert mutation_rates[1] == 0.5
