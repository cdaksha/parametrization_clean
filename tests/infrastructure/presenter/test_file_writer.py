# Standard library
import os

# 3rd party packages
import pytest

# Local source
from parametrization_clean.infrastructure.presenter.file_writer import DataWriter
from tests.infrastructure.repository.test_from_files import file_repository
from tests.use_case.test_population_propagator import all_settings
from parametrization_clean.infrastructure.config.default import DefaultSettings
from parametrization_clean.use_case.nested_ga_with_ann import GeneticNeuralNetPropagator


def read_csv(file_path, delimiter=','):
    data = []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            data.append([float(val) for val in line.split(delimiter)])
    return data


@pytest.mark.usefixtures("presenter_output_path")
def test_write_rows_to_csv(file_repository, presenter_output_path):
    csv_file = os.path.join(presenter_output_path, "test_output.csv")

    if os.path.isfile(csv_file):
        os.remove(csv_file)

    first_generation_population, successfully_retrieved_case_numbers = file_repository.get_population(
        generation_number=1)
    costs = [individual.cost for individual in first_generation_population]
    combined_data = [[first_val, second_val] for first_val, second_val in zip(successfully_retrieved_case_numbers,
                                                                              costs)]

    DataWriter.write_rows_to_csv(csv_file, combined_data, delimiter=',')
    read_data = read_csv(csv_file)
    for read_row, expected_row in zip(read_data, combined_data):
        for read_val, expected_val in zip(read_row, expected_row):
            assert read_val == expected_val

    os.remove(csv_file)
    DataWriter.write_rows_to_csv(csv_file, combined_data)
    read_data = read_csv(csv_file)
    for read_row, expected_row in zip(read_data, combined_data):
        for read_val, expected_val in zip(read_row, expected_row):
            assert read_val == expected_val

    os.remove(csv_file)
    DataWriter.write_rows_to_csv(csv_file, combined_data, "\t")
    read_data = read_csv(csv_file, delimiter='\t')
    for read_row, expected_row in zip(read_data, combined_data):
        for read_val, expected_val in zip(read_row, expected_row):
            assert read_val == expected_val


@pytest.mark.usefixtures("presenter_output_path")
def test_create_or_append(file_repository, presenter_output_path):
    file = os.path.join(presenter_output_path, "test_file.txt")

    if os.path.isfile(file):
        os.remove(file)

    first_generation_population, successfully_retrieved_case_numbers = file_repository.get_population(
        generation_number=1)
    costs = [individual.cost for individual in first_generation_population]
    combined_data = [[first_val, second_val] for first_val, second_val in zip(successfully_retrieved_case_numbers,
                                                                              costs)]

    formatted_first_row = "{},{:.3f}\n".format(combined_data[0][0], combined_data[0][1])
    DataWriter.create_or_append(formatted_first_row, file)
    read_data = read_csv(file)
    assert read_data[0][0] == combined_data[0][0]
    assert read_data[0][1] == round(combined_data[0][1], 3)

    formatted_second_row = "{},{:.3f}\n".format(combined_data[1][0], combined_data[1][1])
    DataWriter.create_or_append(formatted_second_row, file)
    read_data = read_csv(file)
    assert read_data[0][0] == combined_data[0][0]
    assert read_data[0][1] == round(combined_data[0][1], 3)
    assert read_data[1][0] == combined_data[1][0]
    assert read_data[1][1] == round(combined_data[1][1], 3)


@pytest.mark.usefixtures("presenter_output_path")
def test_write_summary(file_repository, presenter_output_path):
    summary_file = os.path.join(presenter_output_path, "00-gen-summary.txt")

    if os.path.isfile(summary_file):
        os.remove(summary_file)

    first_generation_population, successfully_retrieved_case_numbers = file_repository.get_population(
        generation_number=1)
    expected_population_size = 5
    DataWriter.write_summary(first_generation_population, expected_population_size,
                             successfully_retrieved_case_numbers, summary_file)

    with open(summary_file, "r") as in_file:
        for _ in range(3):
            next(in_file)
        best_error = float(next(in_file).split()[-1])
        assert best_error == 4094932.470
        best_case = int(next(in_file).split('-')[-1])
        assert best_case == 1


@pytest.mark.usefixtures("presenter_output_path", "reax_output_dir_path")
def test_write_outputs(file_repository, reax_output_dir_path, presenter_output_path):
    previous_generation_number = 1
    file_repository.current_generation_number = 2
    default_settings = DefaultSettings()
    default_settings.neural_net_settings.num_epochs = 1

    summary_file = os.path.join(presenter_output_path, "generation-" + str(previous_generation_number), "00-gen-summary.txt")
    generation_vs_error_file_path = os.path.join(presenter_output_path, '00-generation-vs-error.txt')
    neural_network_file_path = os.path.join(presenter_output_path, "00-ann-summary")

    if os.path.isfile(summary_file):
        os.remove(summary_file)
    if os.path.isfile(generation_vs_error_file_path):
        os.remove(generation_vs_error_file_path)
    if os.path.isfile(neural_network_file_path):
        os.remove(neural_network_file_path)

    first_generation_population, successfully_retrieved_case_numbers = file_repository.get_population(
        generation_number=previous_generation_number)

    file_repository.population_path = presenter_output_path
    DataWriter.write_outputs(first_generation_population, successfully_retrieved_case_numbers,
                             file_repository, default_settings, generation_number=2, history=None)

    os.remove(summary_file)
    os.remove(generation_vs_error_file_path)
    file_repository.population_path = reax_output_dir_path
    neural_net_propagator = GeneticNeuralNetPropagator(default_settings, file_repository)
    model, history = neural_net_propagator.train_neural_net()
    file_repository.population_path = presenter_output_path
    DataWriter.write_outputs(first_generation_population, successfully_retrieved_case_numbers,
                             file_repository, default_settings, generation_number=2, history=history)
