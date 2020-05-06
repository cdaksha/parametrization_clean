
# Standard library
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.use_case.population_initializer import PopulationInitializer
from tests.use_case.test_population_propagator import all_settings


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_initializer_init(population_repository_mock, all_settings, root_individual):
    all_settings.initialization_strategy = NakataMutate
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(population_repository=population_repository_mock,
                                        settings_repository=all_settings)
    assert initializer.population_repository == population_repository_mock
    assert initializer.population_size == 4

    assert initializer.mutation_settings_dict['gauss_std'] == [0.10]
    assert initializer.mutation_settings_dict['gauss_frac'] == [1.0]
    assert initializer.mutation_settings_dict['nakata_rand_lower'] == -1.0
    assert initializer.mutation_settings_dict['nakata_rand_higher'] == 1.0
    assert initializer.mutation_settings_dict['nakata_scale'] == 0.10
    assert initializer.mutation_settings_dict['polynomial_eta'] == 60
    assert initializer.mutation_settings_dict['param_bounds'] == [[0.0, 2.0],
                                                                  [0.0, 0.5],
                                                                  [-1.0, 0.0],
                                                                  [3.0, 12.0],
                                                                  [1.0, 3.0]]


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_initializer(population_repository_mock, all_settings, root_individual):
    all_settings.initialization_strategy = NakataMutate
    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(population_repository=population_repository_mock,
                                        settings_repository=all_settings)
    population = initializer.execute()
    assert len(population) == 4
    for case in population:
        assert case.params != root_individual.root_params


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('param_bounds')
def test_population_initializer_with_kwargs(population_repository_mock, all_settings, param_bounds, root_individual):
    all_settings.initialization_strategy = CentralUniformMutate
    all_settings.mutation_settings.param_bounds = param_bounds

    population_repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(population_repository=population_repository_mock,
                                        settings_repository=all_settings)
    population = initializer.execute()
    assert len(population) == 4
    for case in population:
        assert case.params != root_individual.root_params
        for param, param_bound in zip(case.params, param_bounds):
            assert param_bound[0] <= param <= param_bound[1]
