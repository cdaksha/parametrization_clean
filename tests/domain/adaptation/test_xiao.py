
# Standard library

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@pytest.mark.usefixtures('get_individuals')
def test_xiao_1(get_individuals, root_individual):
    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)

    average_cost = 500.0
    minimum_cost = min(costs)
    parent_costs = (get_individuals[2].cost, get_individuals[3].cost)
    crossover_rate, mutation_rates = XiaoAdapt.adaptation(average_cost, minimum_cost, parent_costs,
                                                          crossover_rate=0.8, mutation_rate=0.2)
    assert crossover_rate == 0.8
    assert mutation_rates[0] == 0.2
    assert mutation_rates[1] == 0.2


@pytest.mark.usefixtures('get_individuals')
def test_xiao_2(get_individuals, root_individual):
    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)

    average_cost = 1000.0
    minimum_cost = min(costs)
    parent_costs = (get_individuals[2].cost, get_individuals[3].cost)
    crossover_rate, mutation_rates = XiaoAdapt.adaptation(average_cost, minimum_cost, parent_costs,
                                                          crossover_rate=0.8, mutation_rate=0.2)
    assert crossover_rate == pytest.approx(0.679, rel=1e-3)
    assert mutation_rates[0] == pytest.approx(0.170, rel=1e-3)
    assert mutation_rates[1] == 0.2
