#!/usr/bin/env python

# Standard library
from typing import Tuple
import random

# 3rd party packages


# Local source
from parametrization_clean.domain.crossover.strategy import ICrossoverStrategy
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class SinglePointCross(ICrossoverStrategy):

    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, root_individual: RootIndividual,
                  **kwargs) -> Tuple[Individual, Individual]:
        """Execute single-point crossover. Cut the parameter vectors from two children at random positions
        and join to yield two new vectors (children).

        :param parent1: The first individual participating in crossover.
        :param parent2: The second individual participating in crossover.
        :return: Two new Children generated from the mating/crossover.
        """
        choice = random.randrange(0, len(parent1.params))
        child1 = Individual(parent1.params[:choice] + parent2.params[choice:], root_individual=root_individual)
        child2 = Individual(parent2.params[:choice] + parent1.params[choice:], root_individual=root_individual)
        return child1, child2
