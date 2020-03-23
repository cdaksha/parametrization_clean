#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pahari's 1st stage parameter reduction technique applied to `params` file.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List, Tuple
import os

# Local source
from core.utils.population_io import PopulationIO
from core.genetic_algorithm.individual import Individual
from core.settings.config import settings as s
from core.utils.report import ReportWriter


def get_params_to_test(param_bounds: List) -> Tuple[List, List, List]:
    """Given a list of parameter bounds (with min, max), return a tuple with a
     list containing the min, a list containing the average, and a list containing the max.
     """
    param_low = []
    param_mid = []
    param_high = []
    for bounds in param_bounds:
        low = bounds[0]
        high = bounds[1]
        param_low.append(low)
        param_high.append(high)
        param_mid.append((low + high) / 2)
    return param_low, param_mid, param_high


def get_params_to_test2(params) -> Tuple[List, List, List]:
    """Test +/- 15% from the base parameter values."""
    param_low = []
    param_mid = []
    param_high = []
    for param in params:
        param_low.append(param - 0.15 * param)
        param_mid.append(param)
        param_high.append(param + 0.15 * param)
    return param_low, param_mid, param_high


def create_param_reduction_population(param_low, param_mid, param_high, params):
    """Vary each parameter in params file while holding the rest constant and create reaxFF files for submission."""
    temp = [param for param in params]
    population = []
    for i in range(len(params)):
        # Creating reaxFF optimizations to submit for low, mid, and high cases
        for val in (param_low[i], param_mid[i], param_high[i]):
            temp[i] = val
            population.append(Individual(params=temp))
        # Resetting back to original params
        temp[i] = params[i]
    return population


def create_param_reduction_files(param_reduction_population, output_dir='param-reduction'):
    """Write population to file."""
    pop_io_obj = PopulationIO(generation_num=-1)
    generation_path = os.path.join(s.path.population, output_dir)
    if not os.path.isdir(generation_path):
        os.mkdir(generation_path)
    # counter = 0
    for i, case in enumerate(param_reduction_population):
        # if i % 3 == 0:
        #     counter += 1
        # child_dir = os.path.join(generation_path, 'child' + str(counter) + str(i % 3))
        child_dir = os.path.join(generation_path, 'child-' + str(i))
        pop_io_obj.write_child(case, child_dir)


def run_param_reduction(output_dir='param-reduction'):
    """Vary each parameter while holding the rest constant (using min, average, and max), then write population to
    `output_dir`.
    """
    root_child = Individual()
    param_bounds = root_child.param_bounds
    params = root_child.params
    # param_low, param_mid, param_high = get_params_to_test(param_bounds)
    param_low, param_mid, param_high = get_params_to_test2(params)
    population = create_param_reduction_population(param_low, param_mid, param_high, params)
    create_param_reduction_files(population, output_dir)


def read_and_write_param_reduction(input_dir='param-reduction'):
    pop_io_obj = PopulationIO(generation_num=-1)
    pop_io_obj._pop_size = get_number_of_params()
    pop_io_obj._generation_path = os.path.join(s.path.population, input_dir)
    parents = pop_io_obj.read_population()
    case_numbers = [parent.index for parent in parents]
    sorted_errors_file = os.path.join(pop_io_obj._generation_path, '00-all-errors.txt')
    numbers = []
    costs = []
    for case_number, parent in zip(case_numbers, parents):
        current_val_str = str(int(case_number / 3) + 1) + '-' + str(case_number % 3)
        numbers.append(current_val_str)
        costs.append(parent.cost)
        ReportWriter.write_rows_to_csv(sorted_errors_file, zip(numbers, costs))


def get_number_of_params():
    """For bash script."""
    root_child = Individual()
    num_params = len(root_child.params)
    return num_params * 3


# if __name__ == '__main__':
#     output_dir = 'param-reduction'
#     run_param_reduction(output_dir)
