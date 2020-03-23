#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that contains methods to implement Adaptive Genetic Algorithm.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import Tuple, List
import math

# 3rd party packages

# Local source
from core.settings.config import settings as s


def get_adaptation(adaptation_type: str = 'lei'):
    """Factory to select mutation type. Default is polynomial mutation. Note that Nakata mutation is present in
    c.Child class.
    """
    adaptation_types = {
        'lei': lei_adapt,
        'srinivas': srinivas_adapt,
        'xiao': xiao_adapt,
    }
    return adaptation_types[adaptation_type]


def lei_adapt(ave_cost: float, max_cost: float, min_cost: float) -> Tuple[float, float]:
    """Uses improved Adaptive GA algorithm from Lei and Tingzhi's image segmentation paper.
    The probability of crossover and mutation are adapted according to "locally convergent" and
    "concentrated with each other" criteria, based on parameters B and A, respectively.
    This algorithm considers the fitness of the population as a whole, rather than determining crossover and
    mutation rates separately for each individual in the population.
    """
    mutation_rate = s.GA.mutation.probability
    cross_rate = s.GA.crossover.probability

    converged_measure = min_cost / max_cost
    concentrated_measure = min_cost / ave_cost
    print("CONVERGED MEASURE: {}".format(converged_measure))
    print("CONCENTRATED MEASURE: {}".format(concentrated_measure))
    if (converged_measure > s.AlgorithmParameters.adaptation.lei.B
            and concentrated_measure > s.AlgorithmParameters.adaptation.lei.A):  # generation considered convergent
        cross_rate = min(1.0, cross_rate / (1 - converged_measure))
        mutation_rate = min(1.0, mutation_rate / (1 - converged_measure))

    return cross_rate, mutation_rate


def srinivas_adapt(ave_cost: float, min_cost: float, parent_costs: Tuple) -> Tuple[float, List]:
    """Uses improved AGA from Srinivas' paper. In this paper, four constants: K1, K2, K3, K4 are used to
    adapt the crossover and mutation probabilities.
    This algorithm considers the fitness of each individual rather than measuring the population as a whole.

    Returns one crossover probability and two mutation probabilities (corresponding to two parents).
    """
    k1 = s.AlgorithmParameters.adaptation.srinivas.k1
    k2 = s.AlgorithmParameters.adaptation.srinivas.k2
    k3 = s.AlgorithmParameters.adaptation.srinivas.k3
    k4 = s.AlgorithmParameters.adaptation.srinivas.k4

    better_parent_cost = min(parent_costs)
    if better_parent_cost <= ave_cost:
        cross_rate = k1 * (better_parent_cost - min_cost) / (ave_cost - min_cost)
    else:
        cross_rate = k3

    mutation_rates = []
    for cost in parent_costs:
        if cost <= ave_cost:
            new_mutation_rate = k2 * (cost - min_cost) / (ave_cost - min_cost)
            if new_mutation_rate < s.AlgorithmParameters.adaptation.srinivas.default_mutation_rate:
                new_mutation_rate = s.AlgorithmParameters.adaptation.srinivas.default_mutation_rate
            mutation_rates.append(new_mutation_rate)
        else:
            mutation_rates.append(k4)

    return cross_rate, mutation_rates


def xiao_adapt(median_cost: float, min_cost: float, parent_costs: Tuple) -> Tuple[float, List]:
    """Uses Xiao's improved adaptive genetic algorithm based on the arctan function.
    This algorithm considers the fitness of each individual rather than measuring the population as a whole.

    Returns one crossover probability and two mutation probabilities (corresponding to two parents).
    """
    # TODO: Use Median Cost instead of Average Cost due to outliers?
    min_cross_prob = s.AlgorithmParameters.adaptation.xiao.crossover_min_probability
    max_cross_prob = s.GA.crossover.probability
    min_mutate_prob = s.AlgorithmParameters.adaptation.xiao.mutation_min_probability
    max_mutate_prob = s.GA.mutation.probability
    C = s.AlgorithmParameters.adaptation.xiao.C

    better_parent_cost = min(parent_costs)
    if better_parent_cost < median_cost:
        cross_rate = (min_cross_prob + max_cross_prob) / 2 + \
                     (min_cross_prob - max_cross_prob) / math.pi * \
                     math.atan(C * (2 * better_parent_cost - min_cost - median_cost) /
                               (2 * (min_cost - median_cost)))
    else:
        cross_rate = max_cross_prob

    mutation_rates = []
    for cost in parent_costs:
        if cost < median_cost:
            mutation_rate = (min_mutate_prob + max_mutate_prob) / 2 + \
                            (min_mutate_prob - max_mutate_prob) / math.pi * \
                            math.atan(C * (2 * cost - min_cost - median_cost) /
                                      (2 * (min_cost - median_cost)))
        else:
            mutation_rate = max_mutate_prob
        mutation_rates.append(mutation_rate)

    return cross_rate, mutation_rates
