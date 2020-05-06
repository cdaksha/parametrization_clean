#!/usr/bin/env python

# Standard library
from typing import Tuple

# 3rd party packages
import numpy as np

# Local source
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class UniformCross(ICrossoverStrategy):

    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, root_individual: RootIndividual,
                  **kwargs) -> Tuple[Individual, Individual]:
        """Execute uniform crossover. Take the ith row of parent1 and randomly swap bits with the ith row of p2."""
        sieve = np.random.randint(2, size=len(parent1.params))  # Array of 0's and 1's
        not_sieve = sieve ^ 1  # Complement of sieve

        child1 = Individual(list(parent1.params * sieve + parent2.params * not_sieve), root_individual=root_individual)
        child2 = Individual(list(parent1.params * not_sieve + parent2.params * sieve), root_individual=root_individual)

        return child1, child2
