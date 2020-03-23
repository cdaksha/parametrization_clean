#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module to extract generation data from population directory.
Data is then loaded into two files that can be used to train the ANN.
First file contains features and second file contains output variables.

__author__ = "Chad Daksha"
"""

# Standard library
import os

# 3rd party packages
import pandas as pd

# Local source
import core.utils.population_io as pop_io
from core.settings.config import settings as s


def read_population_range(lower_bound, upper_bound):
    # Lower bound inclusive, upper bound exclusive
    pop_io_obj = pop_io.PopulationIO(generation_num=-1)
    return pop_io_obj.read_population_range(lower_bound, upper_bound)


def write_population(population, x_path, y_path):
    X = pd.DataFrame([individual.params for individual in population])
    y = pd.DataFrame([individual.reax_predictions for individual in population])
    X.to_csv(x_path)
    y.to_csv(y_path)


def run():
    # Change root & population paths here if needed
    s.path.root = s.path.root
    s.path.population = "/lustre/scratch/daksha/202002-ZnO-results/base_case/3"

    #lower_bound = input('Enter lower bound (inclusive) for population to read:')
    #upper_bound = input('Enter upper bound (exclusive) for population to read:')
    lower_bound = '100'
    upper_bound = '201'
    population = read_population_range(int(lower_bound), int(upper_bound))

    x_path = os.path.join(s.path.population, "x_data.csv")
    y_path = os.path.join(s.path.population, "y_data.csv")
    write_population(population, x_path, y_path)

    print("Done!")


if __name__ == "__main__":
    run()

