
# Standard library
from unittest import mock

# 3rd party packages
import pytest

# Local source
from parametrization_clean.use_case.population_writer import PopulationWriter


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
def test_population_writer_init(repository_mock):
    population_writer = PopulationWriter(population_repository=repository_mock)
    assert population_writer.population_repository == repository_mock


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_writer_write_individual(repository_mock, get_individuals):
    population_writer = PopulationWriter(population_repository=repository_mock)
    population_writer.write_individual(get_individuals[0], child_dir="test/directory/child-1/")
    repository_mock.write_individual.assert_called_once_with(get_individuals[0], child_dir="test/directory/child-1/")


@mock.patch('parametrization_clean.use_case.port.population_repository.IPopulationRepository')
@pytest.mark.usefixtures('get_individuals')
def test_population_writer_write_population(repository_mock, get_individuals):
    population_writer = PopulationWriter(population_repository=repository_mock)
    population_writer.write_population(get_individuals, generation_number=1)
    repository_mock.write_population.assert_called_once_with(get_individuals, 1)
