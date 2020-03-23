#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that contains selection methods to be used for a Child.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List
import random
from itertools import accumulate

# Third party
import numpy as np

# Local source
from core.genetic_algorithm import individual as c
from core.genetic_algorithm.helpers import sort_by_cost
from core.settings.config import settings as s


def get_selection(selection_type: str = 'tournament'):
    """Factory to select selection type. Default is tournament selection."""
    selection_types = {
        'tournament': tournament_selection,
        'truncation': truncation_selection,
        'linear_SUS': linear_rank_SUS,

    }
    return selection_types[selection_type]


def tournament_selection(population: List[c.Individual]) -> c.Individual:
    """Select `tournament_size` individuals from `population` at random and return best individual by fitness.
    Currently, the selection pressure = 1.
    """
    tournament_size = s.AlgorithmParameters.selection.tournament.size
    # Randomly sample `tournament_size` number of indices
    idx = random.sample(range(len(population)), tournament_size)

    selected = [population[i] for i in idx]  # Extract children based on indices

    return sort_by_cost(selected)[0]


def truncation_selection(population: List[c.Individual]) -> List[c.Individual]:
    """Order the population by fitness, then pick `num_select` individuals to pass to the next generation."""
    truncation_fraction = s.AlgorithmParameters.selection.truncation.probability
    num_select = int(truncation_fraction * s.GA.populationSize)
    return sort_by_cost(population)[:num_select]


def roulette_wheel_selection() -> c.Individual:
    """Relate fitness level to probability of selection."""
    raise NotImplementedError('ERROR - roulette wheel selection currently unavailable!')


def linear_rank_SUS(population: List[c.Individual]):
    """Use linear ranking to compute probabilities of selection, then use stochastic universal sampling."""
    sorted_population, cumulative_probabilities = linear_rank(population)
    selected_population = SUS(sorted_population, cumulative_probabilities)
    print("{fmt}LINEAR SELECTION + STOCHASTIC UNIVERSAL SAMPLING{fmt}".format(fmt='-'*50))
    print("Cumulative Probabilities: {}".format(cumulative_probabilities))
    return selected_population


def linear_rank(population: List[c.Individual]):
    """Return list of new fitness values based on ranking."""
    sorted_population = sort_by_cost(population)[::-1]  # reverse list to have best cost @ last index
    population_size = len(population)
    selection_pressure = 2.0  # S.P. of 2.0 gives same intensity as tournament size of 2
    selection_pdf = [(2 - selection_pressure) / population_size +
                     2 * (selection_pressure - 1) * i / (population_size * (population_size - 1))
                     for i in range(population_size)]
    print("Linear Ranking PDF: {}".format(selection_pdf))
    cumulative_probabilities = list(accumulate(selection_pdf))
    print("Linear Ranking CDF: {}".format(cumulative_probabilities))
    return sorted_population, cumulative_probabilities


def SUS(population: List[c.Individual], cumulative_probabilities) -> List[c.Individual]:
    """Return list of parents to consider for selection."""
    cdf_with_zero = [0] + cumulative_probabilities
    num_to_select = len(population) - 2  # subtracting 2 due to elitism
    dist_between_ptrs = 1.0 / num_to_select
    print("Distance Between Pointers: {}".format(dist_between_ptrs))
    ptr = random.uniform(0, dist_between_ptrs)

    ptrs = [ptr] + [ptr + dist_between_ptrs for _ in range(num_to_select - 1)]
    print("Pointers: {}".format(ptrs))
    ptrs = list(accumulate(ptrs))
    print("Cumulative Pointers for Binning: {}".format(ptrs))
    indices = np.digitize(ptrs, cumulative_probabilities)
    print("Indices of Parents to Select: {}".format(indices))
    new_population = [population[index] for index in indices]

    return new_population
