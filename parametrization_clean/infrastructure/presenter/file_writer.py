#!/usr/bin/env python

"""Present data from results of genetic algorithm/ANN. Allows user to write a list of lists to a CSV file (or
any other delimiter, as specified) through `write_rows_to_csv`, to create or append one row of data to a file through
`create_or_append`, or to write the summary of one generation of the genetic algorithm whose ReaxFF optimizations
have been completed (and therefore whose individuals have known costs/errors) through `write_summary`.
`write_outputs` allows generation of several outputs: total error as a function of generation number to
"00-generation-vs-error.txt", the summary of the previous generation to "00-gen-summary.txt", as well as summary
of the results of training the neural network to "00-ann-summary"---the last only if the neural network is enabled
and TensorFlow 2.0 is installed.
"""

# Standard library
import csv
import os

# 3rd party packages

# Local source


class DataWriter:

    @staticmethod
    def write_rows_to_csv(file_path, data, delimiter=','):
        with open(file_path, 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=delimiter)
            writer.writerows(data)

    @staticmethod
    def create_or_append(formatted_data_string, file_path):
        with open(file_path, 'a+') as out_file:
            out_file.write(formatted_data_string)

    @staticmethod
    def write_summary(population, expected_population_size, successfully_retrieved_case_numbers, file_path):
        """Generate report for a given generation at location `file_path`."""
        best_child = sorted(population)[0]

        num_retrieved = len(population)
        num_failed = expected_population_size - num_retrieved

        best_child_idx = population.index(best_child)
        with open(file_path, 'w') as out_file:
            out_file.write("---------- GENERATION SUMMARY ----------\n")
            out_file.write("Number of successful cases: {:.3f} ({:.2f}%)\n"
                           .format(num_retrieved, 100 * num_retrieved / expected_population_size))
            out_file.write("Number of failed cases: {:.3f} ({:.2f}%)\n"
                           .format(num_failed, 100 * num_failed / expected_population_size))
            out_file.write("Best Cost/Fitness Real ReaxFF Error (from Master GA): {:.3f}\n"
                           .format(best_child.cost))
            out_file.write("Corresponding best Master GA case: case-{}\n".
                           format(successfully_retrieved_case_numbers[best_child_idx]))

    @staticmethod
    def write_outputs(previous_population, successfully_retrieved_case_numbers, population_repository, user_settings,
                      generation_number, history=None):
        """Write summary of previous generation, append to generation # vs. best total error,
        and append to ANN summary.
        """
        previous_generation_number = generation_number - 1
        population_path = population_repository.population_path
        expected_population_size = user_settings.ga_settings.population_size
        if history:
            history_dict = history.history

        summary_file_path = os.path.join(population_path, 'generation-' + str(previous_generation_number),
                                         '00-gen-summary.txt')
        DataWriter.write_summary(previous_population, expected_population_size,
                                 successfully_retrieved_case_numbers, summary_file_path)

        generation_vs_error_file_path = os.path.join(population_path, '00-generation-vs-error.txt')
        neural_network_file_path = os.path.join(population_path, "00-ann-summary")
        if previous_generation_number == 1:
            # First generation -> initialize file
            DataWriter.create_or_append("Generation Number, Best Total Error\n", generation_vs_error_file_path)

        if history and previous_generation_number == user_settings.neural_net_settings.num_populations_to_train_on:
            # First time writing to ANN -> initialize file
            DataWriter.create_or_append("Generation #\t{}\n"
                                        .format(list(history_dict.keys())), neural_network_file_path)

        best_previous_error = sorted(previous_population)[0].cost
        DataWriter.create_or_append("{},{:.3f}\n".format(previous_generation_number, best_previous_error),
                                    generation_vs_error_file_path)

        if history:
            final_neural_network_results = [val[-1] for val in history_dict.values()]
            num_columns = len(final_neural_network_results)
            DataWriter.create_or_append(('{}\t' + '{:.3f}\t' * num_columns + '\n').format(
                previous_generation_number, *final_neural_network_results), neural_network_file_path)
