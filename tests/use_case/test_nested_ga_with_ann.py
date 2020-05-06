
# Standard library
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.selection.tournament import TournamentSelect
from parametrization_clean.domain.mutation.gauss import GaussianMutate
from parametrization_clean.domain.crossover.double_pareto import DoubleParetoCross
from parametrization_clean.domain.adaptation.xiao import XiaoAdapt
from parametrization_clean.domain.cost.reax_error import ReaxError

# Fixtures
from tests.use_case.test_population_propagator import root_individual


@pytest.fixture()
@mock.patch('parametrization_clean.use_case.port.settings_repository.IAllSettings')
@pytest.mark.usefixtures('param_bounds')
def all_settings(all_settings_mock, param_bounds):
    all_settings_mock.strategy_settings.selection_strategy = TournamentSelect
    all_settings_mock.strategy_settings.mutation_strategy = GaussianMutate
    all_settings_mock.strategy_settings.crossover_strategy = DoubleParetoCross
    all_settings_mock.strategy_settings.adaptation_strategy = XiaoAdapt
    all_settings_mock.strategy_settings.error_strategy = ReaxError

    all_settings_mock.ga_settings.population_size = 4
    all_settings_mock.ga_settings.mutation_rate = 0.20
    all_settings_mock.ga_settings.crossover_rate = 0.80
    all_settings_mock.ga_settings.use_elitism = True
    all_settings_mock.ga_settings.use_adaptation = False

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

    all_settings_mock.neural_net_settings.verbosity = 2
    all_settings_mock.neural_net_settings.train_fraction = 0.80
    all_settings_mock.neural_net_settings.num_epochs = 1
    all_settings_mock.neural_net_settings.num_populations_to_train_on = 10
    all_settings_mock.neural_net_settings.num_nested_ga_iterations = 2
    all_settings_mock.neural_net_settings.minimum_validation_r_squared = 0.95

    return all_settings_mock


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_init(repository_mock, all_settings, get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals*10)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    assert propagator.neural_net_settings.verbosity == 2
    assert propagator.neural_net_settings.train_fraction == 0.80
    assert propagator.neural_net_settings.num_epochs == 1
    assert propagator.neural_net_settings.num_populations_to_train_on == 10
    assert propagator.neural_net_settings.num_nested_ga_iterations == 2
    assert propagator.neural_net_settings.minimum_validation_r_squared == 0.95

    assert propagator.root_individual == root_individual

    assert propagator.error_strategy == ReaxError
    assert propagator.population_size == 4


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_train_neural_net(repository_mock, all_settings,
                                                        get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    # Just see if the model and history are being created to see if neural net is actually being trained
    model, history = propagator.train_neural_net()
    assert model
    assert history


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_update_costs(repository_mock, all_settings,
                                                    get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    model, _ = propagator.train_neural_net()
    next_generation, _ = propagator.propagate_first(get_individuals, model)
    y_predicted = propagator.neural_net.predict_outputs(model, next_generation)
    propagator.update_costs(next_generation, y_predicted)

    old_costs = []
    new_costs = []
    for old_individual, new_individual in zip(get_individuals, next_generation):
        old_costs.append(old_individual.cost)
        new_costs.append(new_individual.cost)

    assert len(new_costs) == 4
    assert len(y_predicted) == 4
    assert len(y_predicted[0]) == 4
    for new_cost in new_costs:
        assert new_cost


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_propagate_first(repository_mock, all_settings,
                                                       get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    model, _ = propagator.train_neural_net()
    next_generation, best_master_parents = propagator.propagate_first(get_individuals, model)
    best_costs = [best_master_parents[0].cost, best_master_parents[1].cost]
    assert next_generation != get_individuals
    assert pytest.approx(239.557, rel=1e-3) in best_costs
    assert pytest.approx(871.989, rel=1e-3) in best_costs


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_propagate_remaining(repository_mock, all_settings,
                                                           get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    model, _ = propagator.train_neural_net()
    next_generation, _ = propagator.propagate_first(get_individuals, model)
    final_generation = propagator.propagate_remaining(next_generation, model)
    assert len(final_generation) == 4


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_propagator_final_ann_accuracy_is_poor(repository_mock, all_settings,
                                                              get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    _, history = propagator.train_neural_net()
    assert propagator.final_ann_accuracy_is_poor(history)
    history.history['val_r_square'][-1] = 0.96
    assert not propagator.final_ann_accuracy_is_poor(history)


@pytest.mark.usefixtures('get_individuals')
@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_genetic_neural_net_propagator_execute(repository_mock, all_settings,
                                               get_individuals, root_individual):
    # Conditional import machinery
    nested_ga_with_ann = pytest.importorskip('parametrization_clean.use_case.nested_ga_with_ann')

    for individual in get_individuals:
        individual.cost = individual.total_error(root_individual)

    repository_mock.get_previous_n_populations = mock.MagicMock(return_value=get_individuals)
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    propagator = nested_ga_with_ann.GeneticNeuralNetPropagator(all_settings, repository_mock)

    final_generation, model, history = propagator.execute(get_individuals)
    assert len(final_generation) == 4
    assert model
    assert history
