#!/usr/bin/env python

"""Main module to initialize/propagate genetic population and output files containing Individuals to execute
using ReaxFF.
"""

# Standard library

# 3rd party packages

# Local source
from use_case.population_initializer import PopulationInitializer
from use_case.population_propagator import PopulationPropagator
from use_case.population_writer import PopulationWriter
from infrastructure.config.local import UserSettings
from infrastructure.repository.from_files import PopulationFileRepository
from infrastructure.presenter.file_writer import DataWriter


def run_application(generation_number, training_path, population_path):
    user_settings = UserSettings()
    population_repository = PopulationFileRepository(training_path, population_path,
                                                     user_settings, generation_number)
    population_writer = PopulationWriter(population_repository)

    if generation_number == 1:
        # First generation of the genetic algorithm --> initialize first population
        population_initializer = PopulationInitializer(population_repository, user_settings)
        next_population = population_initializer.execute()
    else:
        # Generation number > 1. Propagate population by either using standalone GA or by using GA + nested ANN.
        previous_generation_number = generation_number - 1
        previous_population, successfully_retrieved_case_numbers = \
            population_repository.get_population(previous_generation_number)

        enough_generations_elapsed = generation_number >= user_settings.neural_net_settings.num_populations_to_train_on
        use_neural_network = user_settings.ga_settings.use_neural_network
        if use_neural_network and enough_generations_elapsed:
            from parametrization_clean.use_case.nested_ga_with_ann import GeneticNeuralNetPropagator
            population_propagator = GeneticNeuralNetPropagator(user_settings, population_repository)
            next_population, _, history = population_propagator.execute(previous_population)
        else:
            population_propagator = PopulationPropagator(user_settings, population_repository)
            next_population = population_propagator.execute(previous_population)
            history = None

        DataWriter.write_outputs(previous_population, successfully_retrieved_case_numbers, population_repository,
                                 user_settings, generation_number, history)

    response = population_writer.write_population(next_population, generation_number)

    return response
