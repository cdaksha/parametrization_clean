#!/usr/bin/env python

# Standard library
from typing import Tuple
import random

# 3rd party packages

# Local source
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class DoubleParetoCross(ICrossoverStrategy):

    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, root_individual: RootIndividual,
                  **kwargs) -> Tuple[Individual, Individual]:
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
        alpha = kwargs.get('dpx_alpha', 10)
        beta = kwargs.get('dpx_beta', 1)

        child1_params = []
        child2_params = []
        for parent1_param, parent2_param in zip(parent1.params, parent2.params):
            u = random.uniform(0, 1)
            if u >= 1 / 2:
                modified_beta = alpha * beta * (1 - (2 * u) ** (-1 / alpha))
            else:
                modified_beta = alpha * beta * ((1 - (2 * u)) ** (-1 / alpha) - 1)

            child1_param = ((parent1_param + parent2_param) + modified_beta * abs(parent1_param - parent2_param)) / 2
            child2_param = ((parent1_param + parent2_param) - modified_beta * abs(parent1_param - parent2_param)) / 2

            child1_params.append(child1_param)
            child2_params.append(child2_param)

        child1 = Individual(child1_params, root_individual=root_individual)
        child2 = Individual(child2_params, root_individual=root_individual)
        return child1, child2
