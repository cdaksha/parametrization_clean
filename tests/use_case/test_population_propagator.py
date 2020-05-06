
# Standard library
from typing import List
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.selection.tournament import TournamentSelect
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt
from parametrization_clean.use_case.population_propagator import PopulationPropagator
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.use_case.port.population_repository import IPopulationRepository
from tests.fixtures.domain import dft_energies, weights, root_ffield, param_keys


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@pytest.fixture()
def population_repository(root_individual):
    class TestPopulationRepository(IPopulationRepository):
        def get_root_individual(self) -> RootIndividual:
            return root_individual

        def get_population(self, generation_number: int) -> List[Individual]:
            pass

        def get_previous_n_populations(self, num_populations: int) -> List[Individual]:
            pass

        def write_individual(self, individual: Individual, **kwargs):
            pass

        def write_population(self, population: List[Individual], generation_number):
            pass

    return TestPopulationRepository()


@pytest.fixture()
@mock.patch('parametrization_clean.use_case.port.settings_repository.IAllSettings')
@pytest.mark.usefixtures('param_bounds')
def all_settings(all_settings_mock, param_bounds):
    all_settings_mock.strategy_settings.selection_strategy = TournamentSelect
    all_settings_mock.strategy_settings.mutation_strategy = GaussianMutate
    all_settings_mock.strategy_settings.crossover_strategy = DoubleParetoCross
    all_settings_mock.strategy_settings.adaptation_strategy = XiaoAdapt

    all_settings_mock.ga_settings.population_size = 4
    all_settings_mock.ga_settings.mutation_rate = 0.20
    all_settings_mock.ga_settings.crossover_rate = 0.80
    all_settings_mock.ga_settings.use_elitism = True
    all_settings_mock.ga_settings.use_adaptation = False
    all_settings_mock.ga_settings.use_neural_network = False

    all_settings_mock.mutation_settings.gauss_std = [0.10]
    all_settings_mock.mutation_settings.gauss_frac = [1.0]
    all_settings_mock.mutation_settings.nakata_rand_lower = -1.0
    all_settings_mock.mutation_settings.nakata_rand_higher = 1.0
    all_settings_mock.mutation_settings.nakata_scale = 0.10
    all_settings_mock.mutation_settings.polynomial_eta = 60
    all_settings_mock.mutation_settings.param_bounds = param_bounds

    all_settings_mock.crossover_settings.dpx_alpha = 10
    all_settings_mock.crossover_settings.dpx_beta = 1

    all_settings_mock.selection_settings.tournament_size = 2

    all_settings_mock.adaptation_settings.srinivas_k1 = 1.0
    all_settings_mock.adaptation_settings.srinivas_k2 = 0.5
    all_settings_mock.adaptation_settings.srinivas_k3 = 1.0
    all_settings_mock.adaptation_settings.srinivas_k4 = 0.5
    all_settings_mock.adaptation_settings.srinivas_default_mutation_rate = 0.005
    all_settings_mock.adaptation_settings.xiao_min_crossover_rate = 0.4
    all_settings_mock.adaptation_settings.xiao_min_mutation_rate = 0.1
    all_settings_mock.adaptation_settings.xiao_scale = 0.4

    return all_settings_mock


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_propagator_init(population_repository_mock, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    propagator = PopulationPropagator(all_settings, population_repository_mock)

    assert propagator.ga_settings.population_size == 4
    assert propagator.ga_settings.mutation_rate == 0.20
    assert propagator.ga_settings.crossover_rate == 0.80
    assert propagator.ga_settings.use_elitism
    assert not propagator.ga_settings.use_adaptation
    assert not propagator.ga_settings.use_neural_network

    assert propagator.crossover_rate == 0.80
    assert propagator.mutation_rates == [0.20, 0.20]

    assert propagator.mutation_strategy == all_settings.strategy_settings.mutation_strategy
    assert propagator.crossover_strategy == all_settings.strategy_settings.crossover_strategy
    assert propagator.selection_strategy == all_settings.strategy_settings.selection_strategy

    assert propagator.mutation_settings_dict['gauss_std'] == [0.10]
    assert propagator.mutation_settings_dict['gauss_frac'] == [1.0]
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

    assert propagator.adaptation_settings_dict['srinivas_k1'] == 1.0
    assert propagator.adaptation_settings_dict['srinivas_k2'] == 0.5
    assert propagator.adaptation_settings_dict['srinivas_k3'] == 1.0
    assert propagator.adaptation_settings_dict['srinivas_k4'] == 0.5
    assert propagator.adaptation_settings_dict['xiao_min_crossover_rate'] == 0.4
    assert propagator.adaptation_settings_dict['xiao_min_mutation_rate'] == 0.1
    assert propagator.adaptation_settings_dict['xiao_scale'] == 0.4


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_initialize(population_repository_mock, get_individuals, root_individual, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    costs = []
    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
        costs.append(individual.cost)
    propagator = PopulationPropagator(all_settings, population_repository_mock)
    sorted_costs = sorted(costs)

    children = propagator.initialize(get_individuals)
    assert len(children) == 2
    assert children[0].cost == sorted_costs[0]
    assert children[1].cost == sorted_costs[1]

    propagator.ga_settings.use_elitism = False
    children = propagator.initialize(get_individuals)
    assert len(children) == 0


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_select(population_repository_mock, get_individuals, root_individual, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
    propagator = PopulationPropagator(all_settings, population_repository_mock)

    selected_individual = propagator.select(get_individuals)
    assert selected_individual in get_individuals


@mock.patch('random.random')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_cross(rand_mock, population_repository, get_individuals, all_settings):
    propagator = PopulationPropagator(all_settings, population_repository)
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
def test_population_propagator_mutate(rand_mock, population_repository, get_individuals, all_settings):
    propagator = PopulationPropagator(all_settings, population_repository)
    parent1 = get_individuals[0]
    parent2 = get_individuals[1]

    propagator.mutation_rates = [0.20, 0.20]
    rand_mock.return_value = 0.10
    child1, child2 = propagator.mutate(parent1, parent2)
    assert child1.params != parent1.params
    assert child2.params != parent2.params

    propagator.mutation_rates = [0.20, 0.05]
    child1, child2 = propagator.mutate(parent1, parent2)
    assert child1.params != parent1.params
    assert child2.params == parent2.params


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_adapt(population_repository_mock, get_individuals, root_individual, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    propagator = PopulationPropagator(all_settings, population_repository_mock)
    propagator.ga_settings.use_adaptation = True

    # Two mock dictionaries passed in kwargs for adaptation...must remove duplicate keys
    propagator.ga_settings_dict.pop('_mock_return_value')
    propagator.ga_settings_dict.pop('_mock_parent')
    propagator.ga_settings_dict.pop('_mock_name')
    propagator.ga_settings_dict.pop('_mock_new_name')
    propagator.ga_settings_dict.pop('_mock_new_parent')
    propagator.ga_settings_dict.pop('_mock_sealed')
    propagator.ga_settings_dict.pop('_spec_class')
    propagator.ga_settings_dict.pop('_spec_set')
    propagator.ga_settings_dict.pop('_spec_signature')
    propagator.ga_settings_dict.pop('_mock_methods')
    propagator.ga_settings_dict.pop('_mock_children')
    propagator.ga_settings_dict.pop('_mock_wraps')
    propagator.ga_settings_dict.pop('_mock_delegate')
    propagator.ga_settings_dict.pop('_mock_called')
    propagator.ga_settings_dict.pop('_mock_call_args')
    propagator.ga_settings_dict.pop('_mock_call_count')
    propagator.ga_settings_dict.pop('_mock_call_args_list')
    propagator.ga_settings_dict.pop('_mock_mock_calls')
    propagator.ga_settings_dict.pop('method_calls')
    propagator.ga_settings_dict.pop('_mock_unsafe')
    propagator.ga_settings_dict.pop('_mock_side_effect')
    # Two mock dictionaries passed in kwargs for adaptation...must remove duplicate keys

    average_cost, minimum_cost = propagator.compute_statistics(get_individuals)
    parent1_cost = get_individuals[2].cost
    parent2_cost = get_individuals[3].cost
    propagator.adapt_cross_and_mutate_rates(average_cost, minimum_cost, (parent1_cost, parent2_cost))
    assert propagator.crossover_rate != 0.80
    assert propagator.mutation_rates[0] != 0.20
    assert propagator.mutation_rates[1] == 0.20


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_compute_statistics(population_repository_mock, get_individuals,
                                                  root_individual, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
    propagator = PopulationPropagator(all_settings, population_repository_mock)
    parents = get_individuals

    average_cost, minimum_cost = propagator.compute_statistics(parents)
    assert average_cost == pytest.approx(1037.955, rel=1e-3)
    assert minimum_cost == pytest.approx(239.558, rel=1e-3)


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_propagator_execute(population_repository_mock, get_individuals, root_individual, all_settings):
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)
    propagator = PopulationPropagator(all_settings, population_repository_mock)
    parents = get_individuals

    children = propagator.execute(parents)

    assert len(children) == 4
