
# Standard library
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.selection.tournament import TournamentSelect
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.use_case.population_propagator import PopulationPropagator
from parametrization_clean.domain.root_individual import RootIndividual


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@pytest.fixture()
@mock.patch('parametrization_clean.use_case.port.settings_repository.IAllSettings')
@pytest.mark.usefixtures('param_bounds')
def all_settings(all_settings_mock, param_bounds):
    all_settings_mock.strategy_settings.selection_strategy = TournamentSelect
    all_settings_mock.strategy_settings.mutation_strategy = GaussianMutate
    all_settings_mock.strategy_settings.crossover_strategy = DoubleParetoCross

    all_settings_mock.ga_settings.population_size = 4
    all_settings_mock.ga_settings.mutation_rate = 0.20
    all_settings_mock.ga_settings.crossover_rate = 0.80
    all_settings_mock.ga_settings.elitism = True

    all_settings_mock.mutation_settings.gauss_std = 0.10
    all_settings_mock.mutation_settings.nakata_rand_lower = -1.0
    all_settings_mock.mutation_settings.nakata_rand_higher = 1.0
    all_settings_mock.mutation_settings.nakata_scale = 0.10
    all_settings_mock.mutation_settings.polynomial_eta = 60
    all_settings_mock.mutation_settings.param_bounds = param_bounds

    all_settings_mock.crossover_settings.dpx_alpha = 10
    all_settings_mock.crossover_settings.dpx_beta = 1

    all_settings_mock.selection_settings.tournament_size = 2

    return all_settings_mock


def test_population_propagator_init(all_settings):
    propagator = PopulationPropagator(all_settings)

    assert propagator.ga_settings.population_size == 4
    assert propagator.ga_settings.mutation_rate == 0.20
    assert propagator.ga_settings.crossover_rate == 0.80
    assert propagator.ga_settings.elitism

    assert propagator.mutation_strategy == all_settings.strategy_settings.mutation_strategy
    assert propagator.crossover_strategy == all_settings.strategy_settings.crossover_strategy
    assert propagator.selection_strategy == all_settings.strategy_settings.selection_strategy

    assert propagator.mutation_settings_dict['gauss_std'] == 0.10
    assert propagator.mutation_settings_dict['nakata_rand_lower'] == -1.0
    assert propagator.mutation_settings_dict['nakata_rand_higher'] == 1.0
    assert propagator.mutation_settings_dict['nakata_scale'] == 0.10
    assert propagator.mutation_settings_dict['polynomial_eta'] == 60
    assert propagator.mutation_settings_dict['param_bounds'] == [[0.0, 2.0],
                                                                 [0.0, 0.5],
                                                                 [-1.0, 0.0],
                                                                 [3.0, 12.0],
                                                                 [1.0, 3.0]]

    assert propagator.crossover_settings_dict['dpx_alpha'] == 10
    assert propagator.crossover_settings_dict['dpx_beta'] == 1

    assert propagator.selection_settings_dict['tournament_size'] == 2


@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_initialize(get_individuals, root_individual, all_settings):
    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)
    propagator = PopulationPropagator(all_settings)
    sorted_costs = sorted(costs)

    children = propagator.initialize(get_individuals)
    assert len(children) == 2
    assert children[0].cost == sorted_costs[0]
    assert children[1].cost == sorted_costs[1]

    propagator.ga_settings.elitism = False
    children = propagator.initialize(get_individuals)
    assert len(children) == 0


@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_select(get_individuals, root_individual, all_settings):
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
    propagator = PopulationPropagator(all_settings)

    selected_individual = propagator.select(get_individuals)
    assert selected_individual in get_individuals


@mock.patch('random.random')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_cross(rand_mock, get_individuals, all_settings):
    propagator = PopulationPropagator(all_settings)
    parent1 = get_individuals[0]
    parent2 = get_individuals[1]

    rand_mock.return_value = 0.60
    child1, child2 = propagator.cross(parent1, parent2)
    assert child1.params != parent1.params and child1.params != parent2.params
    assert child2.params != parent2.params and child2.params != parent1.params

    rand_mock.return_value = 0.90
    child1, child2 = propagator.cross(parent1, parent2)
    assert child1.params == parent1.params and child2.params == parent2.params


@mock.patch('random.random')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_mutate(rand_mock, get_individuals, all_settings):
    propagator = PopulationPropagator(all_settings)
    parent = get_individuals[0]

    rand_mock.return_value = 0.10
    child = propagator.mutate(parent)
    assert child.params != parent.params

    rand_mock.return_value = 0.25
    child = propagator.mutate(parent)
    assert child.params == parent.params


@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_execute(get_individuals, root_individual, all_settings):
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
    propagator = PopulationPropagator(all_settings)
    parents = get_individuals

    children = propagator.execute(parents)

    assert len(children) == 4
