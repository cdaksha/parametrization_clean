
# Standard library
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.mutation.nakata import NakataMutate
from parametrization_clean.domain.mutation.central_uniform import CentralUniformMutate
from parametrization_clean.use_case.population_initializer import PopulationInitializer


@pytest.fixture()
@pytest.mark.usefixtures('dft_energies', 'weights', 'root_ffield', 'param_keys')
def root_individual(dft_energies, weights, root_ffield, param_keys):
    root_individual = RootIndividual(dft_energies, weights, root_ffield, param_keys)
    return root_individual


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_initializer_init(repository_mock, root_individual):
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(repository=repository_mock, initialization_strategy=NakataMutate)
    assert initializer.repository == repository_mock
    assert initializer.strategy == NakataMutate


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_initializer(repository_mock, root_individual):
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(repository=repository_mock, initialization_strategy=NakataMutate)
    population = initializer.execute(population_size=100)
    assert len(population) == 100
    for case in population:
        assert case.params != root_individual.root_params


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('param_bounds')
def test_population_initializer_with_kwargs(repository_mock, param_bounds, root_individual):
    repository_mock.get_root_individual = mock.MagicMock(return_value=root_individual)
    initializer = PopulationInitializer(repository=repository_mock, initialization_strategy=CentralUniformMutate)
    population = initializer.execute(population_size=50, param_bounds=param_bounds)
    assert len(population) == 50
    for case in population:
        assert case.params != root_individual.root_params
        for param, param_bound in zip(case.params, param_bounds):
            assert param_bound[0] <= param <= param_bound[1]
