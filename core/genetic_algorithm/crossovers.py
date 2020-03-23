#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that contains crossover methods to be used for a Child.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import Tuple
import random

# 3rd party packages
import numpy as np

# Local source
import core.genetic_algorithm.individual as c
from core.settings.config import settings as s


def get_cross(cross_type: str = 'one_point'):
    """Factory to select crossover type. Default is single-point crossover."""
    cross_types = {
        'one_point': one_point_cross,
        'two_point': two_point_cross,
        'uniform': uniform_cross,
        'DPX': dpx_cross,
    }
    return cross_types[cross_type]


def one_point_cross(parent1: c.Individual, parent2: c.Individual) -> Tuple[c.Individual, c.Individual]:
    """Execute single-point crossover. Cut the parameter vectors from two children at random positions and join to
    yield two new vectors (children). See Pahari's 2012 GA paper for more information.

    NOTE: `params` are mutated in place.

    Parameters
    ----------
    parent1: Child
        The first individual participating in crossover.
    parent2: Child
        The second individual participating in crossover.

    Returns
    -------
    Two new Children generated from the mating/crossover.
    """
    choice = random.randrange(0, len(parent1.params))

    child1 = c.Individual(parent1.params[:choice] + parent2.params[choice:])
    child2 = c.Individual(parent2.params[:choice] + parent1.params[choice:])

    return child1, child2


def two_point_cross(parent1: c.Individual, parent2: c.Individual) -> Tuple[c.Individual, c.Individual]:
    """Execute two-point crossover."""
    idx1 = random.randrange(0, len(parent1.params))
    idx2 = random.randrange(0, len(parent1.params))

    while idx1 == idx2:  # Ensuring two indices are not same
        idx2 = random.randrange(0, len(parent1.params))

    smaller_idx = min(idx1, idx2)
    bigger_idx = max(idx1, idx2)

    child1 = c.Individual(parent1.params[:smaller_idx] + parent2.params[smaller_idx:bigger_idx]
                          + parent1.params[bigger_idx:])
    child2 = c.Individual(parent2.params[:smaller_idx] + parent1.params[smaller_idx:bigger_idx]
                          + parent2.params[bigger_idx:])

    return child1, child2


def uniform_cross(parent1: c.Individual, parent2: c.Individual) -> Tuple[c.Individual, c.Individual]:
    """Execute uniform crossover. Take the ith row of parent1 and randomly swap bits with the ith row of p2."""
    sieve = np.random.randint(2, size=len(parent1.params))  # Array of 0's and 1's
    not_sieve = sieve ^ 1  # Complement of sieve

    child1 = c.Individual(list(parent1.params * sieve + parent2.params * not_sieve))
    child2 = c.Individual(list(parent1.params * not_sieve + parent2.params * sieve))

    return child1, child2


def dpx_cross(parent1: c.Individual, parent2: c.Individual) -> Tuple[c.Individual, c.Individual]:
    """Double Pareto crossover from Thakur's 2014 - "A new GA for global optimization of multimodal continuous
    functions."
    NOTE: Thakur is unclear about modified beta if/then command.
    I have ASSUMED the following:
    if u >= 1/2:
        >> use first expression
    else: (u < 1/2):
        >> use second expression
    This SHOULD be correct, as x = 0 corresponds to f(x), which is the Pareto density function, equal to 0.5!
    """
    dpx_alpha = s.AlgorithmParameters.crossover.dpx.alpha
    dpx_beta = s.AlgorithmParameters.crossover.dpx.beta

    child1_params = []
    child2_params = []
    for parent1_param, parent2_param in zip(parent1.params, parent2.params):
        u = random.uniform(0, 1)

        if u >= 1/2:
            modified_beta = dpx_alpha * dpx_beta * (1 - (2 * u) ** (-1 / dpx_alpha))
        else:
            modified_beta = dpx_alpha * dpx_beta * ((1 - (2 * u)) ** (-1 / dpx_alpha) - 1)

        child1_param = ((parent1_param + parent2_param) + modified_beta * abs(parent1_param - parent2_param)) / 2
        child2_param = ((parent1_param + parent2_param) - modified_beta * abs(parent1_param - parent2_param)) / 2

        child1_params.append(child1_param)
        child2_params.append(child2_param)

    child1 = c.Individual(child1_params)
    child2 = c.Individual(child2_params)
    return child1, child2

