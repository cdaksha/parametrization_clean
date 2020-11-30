
# Standard library
import os
from pathlib import Path
import shutil

# 3rd party packages
import pytest

# Local source
from parametrization_clean.infrastructure.repository.from_files import PopulationFileRepository
from parametrization_clean.domain.root_individual import RootIndividual, FirstGenerationRootIndividual
from parametrization_clean.infrastructure.utils.response_object import (ResponseSuccess,
                                                  ResponseWarning,
                                                  ResponseFailure)
from tests.use_case.test_population_propagator import all_settings
from tests.infrastructure.utils.test_reaxff_reader import reax_io_obj


@pytest.fixture(autouse=True)
@pytest.mark.usefixtures("reax_output_dir_path")
def module_setup_teardown(reax_output_dir_path):
    # Setup
    yield
    # Teardown
    if os.path.isdir(os.path.join(reax_output_dir_path, 'generation-3')):
        shutil.rmtree(os.path.join(reax_output_dir_path, 'generation-3'))


@pytest.fixture()
@pytest.mark.usefixtures("training_set_dir_path", "reax_output_dir_path")
def file_repository(all_settings, training_set_dir_path, reax_output_dir_path):
    return PopulationFileRepository(training_set_dir_path, reax_output_dir_path, all_settings,
                                    current_generation_number=3)


@pytest.fixture()
@pytest.mark.usefixtures("training_set_dir_path", "reax_output_dir_path")
def first_generation_file_repository(all_settings, training_set_dir_path, reax_output_dir_path):
    return PopulationFileRepository(training_set_dir_path, reax_output_dir_path, all_settings,
                                    current_generation_number=1)


@pytest.fixture()
@pytest.mark.usefixtures("training_set_dir_path", "reax_output_dir_path")
def second_generation_file_repository(all_settings, training_set_dir_path, reax_output_dir_path):
    return PopulationFileRepository(training_set_dir_path, reax_output_dir_path, all_settings,
                                    current_generation_number=2)


@pytest.mark.usefixtures("training_set_dir_path", "reax_output_dir_path")
def test_file_repository_init(file_repository, all_settings, training_set_dir_path, reax_output_dir_path):
    assert file_repository.training_set_path == Path(training_set_dir_path)
    assert file_repository.population_path == Path(reax_output_dir_path)
    assert file_repository.current_generation_number == 3
    assert file_repository.population_size == 4

    file_repository = PopulationFileRepository(training_set_dir_path,
                                               os.path.join(reax_output_dir_path, 'test_output_directory'),
                                               all_settings, current_generation_number=3)
    assert file_repository.training_set_path == Path(training_set_dir_path)
    assert file_repository.population_path == Path(os.path.join(reax_output_dir_path, 'test_output_directory'))
    assert file_repository.current_generation_number == 3
    assert file_repository.population_size == 4
    assert os.path.isdir(os.path.join(reax_output_dir_path, 'test_output_directory'))
    os.rmdir(os.path.join(reax_output_dir_path, 'test_output_directory'))


def test_get_root_individual_first_generation(reax_io_obj, first_generation_file_repository):
    root_individual = first_generation_file_repository.get_root_individual()

    ffield, _ = reax_io_obj.read_ffield()
    param_keys, _, _ = reax_io_obj.read_params()

    assert isinstance(root_individual, FirstGenerationRootIndividual)
    assert isinstance(root_individual.root_ffield, dict)
    assert root_individual.param_keys == param_keys


def test_get_root_individual_second_generation(reax_io_obj, second_generation_file_repository):
    if os.path.isdir(second_generation_file_repository.reference_path):
        shutil.rmtree(second_generation_file_repository.reference_path)

    root_individual = second_generation_file_repository.get_root_individual()

    ffield, _ = reax_io_obj.read_ffield()
    param_keys, _, _ = reax_io_obj.read_params()

    assert isinstance(root_individual, RootIndividual)
    assert isinstance(root_individual.root_ffield, dict)
    assert root_individual.param_keys == param_keys
    assert os.path.isdir(second_generation_file_repository.reference_path)
    assert os.path.exists(os.path.join(second_generation_file_repository.reference_path, "fort.99"))


def test_get_root_individual(reax_io_obj, file_repository):
    root_individual = file_repository.get_root_individual()

    ffield, _ = reax_io_obj.read_ffield()
    fort99_data = reax_io_obj.read_fort99()
    param_keys, _, _ = reax_io_obj.read_params()

    dft_energies = [row[1] for row in fort99_data]
    weights = [row[2] for row in fort99_data]

    assert os.path.isdir(file_repository.reference_path)
    assert os.path.exists(os.path.join(file_repository.reference_path, "fort.99"))

    assert isinstance(root_individual, RootIndividual)
    assert isinstance(root_individual.root_ffield, dict)
    assert list(root_individual.dft_energies) == dft_energies
    assert list(root_individual.weights) == weights
    assert root_individual.param_keys == param_keys


def test_get_population(file_repository):
    first_generation_population, successfully_retrieved_case_numbers = \
        file_repository.get_population(generation_number=1)
    assert len(first_generation_population) == 4
    assert len(successfully_retrieved_case_numbers) == 4
    assert first_generation_population[0].cost == pytest.approx(7253274.4676, rel=100)
    assert first_generation_population[1].cost == pytest.approx(4094926.0857, rel=100)
    assert isinstance(first_generation_population[2].cost, float)
    assert first_generation_population[-1].cost == pytest.approx(248741.8623, rel=100)

    file_repository.population_size = 5
    first_generation_population, _ = file_repository.get_population(generation_number=1)
    assert len(first_generation_population) == 4
    assert first_generation_population[0].cost == pytest.approx(7253274.4676, rel=100)
    assert first_generation_population[1].cost == pytest.approx(4094926.0857, rel=100)
    assert isinstance(first_generation_population[2].cost, float)
    assert first_generation_population[-1].cost == pytest.approx(248741.8623, rel=100)


def test_read_population_range(file_repository):
    first_and_second_generation_population = file_repository.read_population_range(lower_bound=1,
                                                                                   upper_bound=3)
    assert len(first_and_second_generation_population) == 8
    assert first_and_second_generation_population[0].cost == pytest.approx(7253274.4676, rel=100)
    assert first_and_second_generation_population[1].cost == pytest.approx(4094926.0857, rel=100)
    assert isinstance(first_and_second_generation_population[2].cost, float)
    assert first_and_second_generation_population[3].cost == pytest.approx(248741.8623, rel=100)
    assert first_and_second_generation_population[4].cost == pytest.approx(644975.3705, rel=100)
    assert first_and_second_generation_population[5].cost == pytest.approx(710538.2521, rel=100)
    assert isinstance(first_and_second_generation_population[6].cost, float)
    assert isinstance(first_and_second_generation_population[7].cost, float)


def test_get_previous_n_populations(file_repository):
    previous_two_populations = file_repository.get_previous_n_populations(num_populations=2)

    assert len(previous_two_populations) == 8
    assert previous_two_populations[0].cost == pytest.approx(7253274.4676, rel=100)
    assert previous_two_populations[1].cost == pytest.approx(4094926.0857, rel=100)
    assert isinstance(previous_two_populations[2].cost, float)
    assert previous_two_populations[3].cost == pytest.approx(248741.8623, rel=100)
    assert previous_two_populations[4].cost == pytest.approx(644975.3705, rel=100)
    assert previous_two_populations[5].cost == pytest.approx(710538.2521, rel=100)
    assert isinstance(previous_two_populations[6].cost, float)
    assert isinstance(previous_two_populations[7].cost, float)

    previous_one_population = file_repository.get_previous_n_populations(num_populations=1)
    assert len(previous_one_population) == 4
    assert previous_two_populations[0].cost == pytest.approx(644975.3705, rel=100)
    assert previous_two_populations[1].cost == pytest.approx(710538.2521, rel=100)
    assert isinstance(previous_two_populations[2].cost, float)
    assert isinstance(previous_two_populations[3].cost, float)


@pytest.mark.usefixtures('get_individuals')
def test_write_individual(file_repository, reax_output_dir_path):
    population, _ = file_repository.get_population(generation_number=1)
    individual = population[0]

    generation_path = os.path.join(reax_output_dir_path, 'generation-3')
    os.mkdir(generation_path)
    child_dir = os.path.join(generation_path, 'child-0')
    response = file_repository.write_individual(individual, child_dir=child_dir)

    assert isinstance(response, ResponseSuccess)
    assert response.message == "Child successfully written at {}.".format(child_dir)

    response = file_repository.write_individual(individual, child_dir=child_dir)
    assert isinstance(response, ResponseWarning)
    assert response.message == "Files in child directory {} were overwritten.".format(child_dir)

    response = file_repository.write_individual(individual)
    assert isinstance(response, ResponseFailure)
    assert response.message == "Child directory not provided."


def test_write_population(file_repository, reax_output_dir_path):
    population, _ = file_repository.get_population(generation_number=2)

    generation_path = os.path.join(reax_output_dir_path, 'generation-3')
    response = file_repository.write_population(population, generation_number=3)
    assert isinstance(response, ResponseSuccess)
    assert response.message == "Generation successfully written at {}".format(generation_path)

    shutil.rmtree(generation_path)
    os.mkdir(generation_path)
    response = file_repository.write_population(population, generation_number=3)
    assert isinstance(response, ResponseWarning)
    assert response.message == "Population generation directory {} overwritten.".format(generation_path)
