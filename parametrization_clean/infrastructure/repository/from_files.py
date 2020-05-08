#!/usr/bin/env python

"""Concrete implementation of population repository interface. Reads files to get data."""

# Standard library
from typing import List, Tuple
from pathlib import Path
import os
import shutil

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.cost.reax_error import ReaxError
from parametrization_clean.domain.utils.helpers import get_param
from parametrization_clean.use_case.port.population_repository import IPopulationRepository
from parametrization_clean.use_case.port.settings_repository import IAllSettings
from parametrization_clean.infrastructure.utils.reax_reader import ReaxReader, write_ffield
from parametrization_clean.infrastructure.utils.reax_converter import Fort99Extractor
from parametrization_clean.infrastructure.utils.response_object import (ResponseSuccess, ResponseWarning,
                                                                        ResponseFailure)


class PopulationFileRepository(IPopulationRepository):
    GENERATION_FOLDER_PREFIX = "generation-"
    INDIVIDUAL_FOLDER_PREFIX = "child-"

    def __init__(self, training_set_path, population_path, settings_repository: IAllSettings,
                 current_generation_number):
        self.training_set_path = Path(training_set_path)
        self.population_path = Path(population_path)

        self.training_reax_reader = ReaxReader(training_set_path)
        self.population_reax_reader = ReaxReader(population_path)

        self.population_size = settings_repository.ga_settings.population_size
        self.current_generation_number = current_generation_number

        self.param_keys, _, _ = self.training_reax_reader.read_params()

        if not os.path.isdir(population_path):
            os.mkdir(population_path)

        _, self.atom_types = self.training_reax_reader.read_ffield()

    def get_root_individual(self) -> RootIndividual:
        root_ffield, _ = self.training_reax_reader.read_ffield()
        fort99_data = self.training_reax_reader.read_fort99()

        # TODO: PROBLEM - First generation also requires 'fort.99' to be present in training file directory!
        fort99_extractor = Fort99Extractor(fort99_data)
        dft_energies = fort99_extractor.get_dft_energies()
        weights = fort99_extractor.get_weights()
        return RootIndividual(dft_energies, weights, root_ffield, self.param_keys)

    def get_population(self, generation_number: int) -> Tuple[List[Individual], List[int]]:
        generation_dir_path = os.path.join(self.population_path, self.GENERATION_FOLDER_PREFIX + str(generation_number))
        root_individual = self.get_root_individual()

        population = []
        successfully_retrieved_case_numbers = []
        for case_number in range(self.population_size):
            child_dir = os.path.join(generation_dir_path, self.INDIVIDUAL_FOLDER_PREFIX + str(case_number))
            self.population_reax_reader.dir_path = child_dir

            try:
                child_fort99_data = self.population_reax_reader.read_fort99()
                child_ffield, _ = self.population_reax_reader.read_ffield()
                child_params = [get_param(key, child_ffield) for key in self.param_keys]

                fort99_extractor = Fort99Extractor(child_fort99_data)
                child_reax_energies = fort99_extractor.get_reax_energies()

                # noinspection PyTypeChecker
                child = Individual(child_params, child_reax_energies, root_individual, error_calculator=ReaxError)

                population.append(child)
                successfully_retrieved_case_numbers.append(case_number)

            except FileNotFoundError:
                # 'fort.99' file does not exist
                print("fort.99 not found in '{}'...continuing to next case".format(child_dir))
                continue

        return population, successfully_retrieved_case_numbers

    def get_previous_n_populations(self, num_populations: int) -> List[Individual]:
        """Read previous N populations before the current generation number.
        Precaution - ensure that lowest possible generation number is zero.
        """
        lower_bound = max(1, self.current_generation_number - num_populations)
        return self.read_population_range(lower_bound=lower_bound, upper_bound=self.current_generation_number)

    def write_individual(self, individual: Individual, **kwargs):
        """Write a `case` in the population to `child_dir`.

        NOTE: If `child_dir` does not exist, it is created.
        NOTE: control, exe, geo, params, trainset.in files are copied from training set directory to `child_dir`.
        NOTE: '0' is written to `iopt` in `child_dir` due to ReaxFF requirement.
        """
        child_dir = kwargs.get('child_dir', None)
        if not child_dir:
            return ResponseFailure.build_parameters_error(message="Child directory not provided.")

        files_overwritten = True
        if not os.path.isdir(child_dir):
            os.mkdir(child_dir)
            files_overwritten = False

        shutil.copy(os.path.join(self.training_set_path, 'control'), child_dir)
        shutil.copy(os.path.join(self.training_set_path, 'geo'), child_dir)
        shutil.copy(os.path.join(self.training_set_path, 'params'), child_dir)
        shutil.copy(os.path.join(self.training_set_path, 'trainset.in'), child_dir)

        with open(os.path.join(child_dir, 'iopt'), 'w') as out_file:
            out_file.write('0')

        try:
            write_ffield(os.path.join(self.training_set_path, 'ffield'), os.path.join(child_dir, 'ffield'),
                         individual.ffield, self.atom_types)
        except (OSError, IOError):
            return ResponseFailure.build_resource_error(
                message="I/O problem with creating child directory at {}".format(child_dir))
        except ValueError:
            return ResponseFailure.build_resource_error(
                message="Trouble writing ffield file in child directory {}".format(child_dir))

        if files_overwritten:
            return ResponseWarning(message="Files in child directory {} were overwritten."
                                   .format(child_dir))
        else:
            return ResponseSuccess(message="Child successfully written at {}.".format(child_dir))

    def write_population(self, population: List[Individual], generation_number):
        """Write a `population` of individuals to the population generation path."""
        generation_dir_path = os.path.join(self.population_path, self.GENERATION_FOLDER_PREFIX + str(generation_number))

        files_overwritten = True
        if not os.path.isdir(generation_dir_path):
            os.mkdir(generation_dir_path)
            files_overwritten = False

        responses = []
        for i, individual in enumerate(population):
            child_dir = os.path.join(generation_dir_path, self.INDIVIDUAL_FOLDER_PREFIX + str(i))
            responses.append(self.write_individual(individual, child_dir=child_dir))

        if any(responses):
            if files_overwritten:
                return ResponseWarning(message="Population generation directory {} overwritten."
                                       .format(generation_dir_path))
            else:
                return ResponseSuccess(message="Generation successfully written at {}"
                                       .format(generation_dir_path))
        else:
            return ResponseFailure.build_resource_error(
                message="No children in population directory {} were written successfully."
                .format(generation_dir_path))

    def read_population_range(self, lower_bound: int, upper_bound: int) -> List[Individual]:
        """Read population of Individuals from `_generation_path` for generations between
        [lower_bound, upper_bound) (lower bound inclusive, upper bound exclusive).
        """
        population = []
        for generation_number in range(lower_bound, upper_bound):
            generation_population, _ = self.get_population(generation_number)
            population.extend(generation_population)
        return population
