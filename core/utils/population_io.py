#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module to read/write populations with children/cases in the genetic algorithm.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List
import os
import shutil

# 3rd party packages

# Local source
from core.genetic_algorithm.individual import Individual
from core.genetic_algorithm.cost import PopulationCost
from core.genetic_algorithm.helpers import reax_error, unique_params, get_param
import core.utils.reax_io as reax_io
from core.settings.config import settings as s


class PopulationIO(object):

    def __init__(self, generation_num):
        """Class used for I/O operations for a Population of Children.

        NOTE: If the population dir_path does not exist, it is created.
        NOTE: If the children dir_path do not exist, they are created.

        Parameters
        ----------
        generation_num: int
            Number of current generation in the genetic pool. Starts with 0.
        """
        self._root_dir = s.path.root
        self._output_dir = s.path.population
        self._pop_size = s.GA.populationSize

        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)

        self.generation_num = generation_num

    @property
    def generation_num(self):
        return self._generation_num

    @generation_num.setter
    def generation_num(self, generation_num):
        self._generation_num = generation_num
        self._generation_path = os.path.join(self._output_dir, 'generation-' + str(self._generation_num - 1))
        self._next_gen_path = os.path.join(self._output_dir, 'generation-' + str(self._generation_num))

    def write_child(self, case: Individual, child_dir: str):
        """Write a `case` in the population to `child_dir`.

        NOTE: If `child_dir` does not exist, it is created.
        NOTE: control, exe, geo, params, trainset.in files are copied from `root_dir` to `child_dir`.
        NOTE: '0' is written to `iopt` in `child_dir` due to ReaxFF requirement.
        """
        if not os.path.isdir(child_dir):
            os.mkdir(child_dir)

        shutil.copy(os.path.join(self._root_dir, 'control'), child_dir)
        # shutil.copy(os.path.join(self._root_dir, 'exe'), child_dir)
        shutil.copy(os.path.join(self._root_dir, 'geo'), child_dir)
        shutil.copy(os.path.join(self._root_dir, 'params'), child_dir)
        shutil.copy(os.path.join(self._root_dir, 'trainset.in'), child_dir)

        with open(os.path.join(child_dir, 'iopt'), 'w') as outf:
            outf.write('0')

        reax_io.write_ffield(os.path.join(self._root_dir, 'ffield'),
                             os.path.join(child_dir, 'ffield'), case.atom_types, case.ffield)

    def write_population(self, population: List[Individual]):
        """Write a `population` of Children to `_generation_path`."""
        if not os.path.isdir(self._next_gen_path):
            os.mkdir(self._next_gen_path)
        else:
            # Next population directory already exists...overwriting
            print("Population directory already exists for generation number = {}...OVERWRITING!".format(self._generation_num))
            
        for i, case in enumerate(population):
            child_dir = os.path.join(self._next_gen_path, 'child-' + str(i))
            try:
                self.write_child(case, child_dir)
            except ValueError as e:
                print(e)

    def read_population(self) -> List[Individual]:
        """Read a population of Children from `_generation_path`, based on `generation_num`."""
        population = []

        root_reader = reax_io.ReaxIO(self._root_dir)
        param_keys, _, _ = unique_params(*root_reader.read_params())

        for case_number in range(self._pop_size):
            child_dir = os.path.join(self._generation_path, 'child-' + str(case_number))
            child_reader = reax_io.ReaxIO(child_dir)

            try:
                output_fort99 = child_reader.read_fort99()
                output_fort99_energy_sections = child_reader.read_fort99_energy_sections()

                ffield, _ = child_reader.read_ffield()
                params = [get_param(key, ffield) for key in param_keys]

                reax_predictions = [row[0] for row in output_fort99]
                true_vals = [row[1] for row in output_fort99]
                weights = [row[2] for row in output_fort99]

                all_errors = [reax_error(reax_pred=row[0], true_val=row[1], weight=row[2])
                              for row in output_fort99]

                child = Individual(params)
                child.errors = all_errors
                child.reax_predictions = reax_predictions
                child.energy_curve_data = reax_io.retrieve_energy_curve_data(root_reader, output_fort99_energy_sections)
                child.true_vals = true_vals
                child.weights = weights
                child.index = case_number

                population.append(child)

            except FileNotFoundError:
                # 'fort.99' file does not exist
                print("fort.99 not found in '{}'...continuing to next case".format(child_dir))
                continue
            except IndexError:
                # sometimes, 'fort.99' ReaxFF values are so bad that they are printed out as stars!
                print("ReaxFF values so bad for child-{} that they are printed out as stars! Continuing to next case".format(case_number))
            except ValueError:
                # sometimes, values in the 'ffield' file merge columns and become unusable - maybe use regexp in future?
                print("Column merging occurring for child-{}! Cannot use data. Continuing to next case.".format(case_number))

        # For now, calculating the composite costs and entering them here
        # Retrieving and assigning composite costs
        cost_controller = PopulationCost(population)
        composite_costs = cost_controller.get_costs()
        cost_controller.assign_costs(composite_costs)

        return population

    def read_population_range(self, lower_bound: int, upper_bound: int) -> List[Individual]:
        """Read a population of Children from `_generation_path` for generations between
        [lower_bound, upper_bound) (lower bound inclusive, upper bound exclusive).
        """
        population = []
        for i in range(lower_bound, upper_bound):
            print("Reading data from generation {}...".format(i))
            self._generation_path = os.path.join(self._output_dir, 'generation-' + str(i))
            generation_population = self.read_population()
            population.extend(generation_population)
        return population

    def read_all_previous(self) -> List[Individual]:
        """Read a population of Children from `_generation_path` for all generations before `generation_num`."""
        return self.read_population_range(lower_bound=0, upper_bound=self.generation_num)

    def read_previous_n_populations(self, num: int) -> List[Individual]:
        """Read previous N populations before `generation_num`.
        Precaution - ensure that lowest possible generation number is zero.
        """
        lower_bound = max(1, self.generation_num - num)
        return self.read_population_range(lower_bound=lower_bound, upper_bound=self.generation_num)
