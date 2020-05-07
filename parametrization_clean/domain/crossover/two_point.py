#!/usr/bin/env python

# Standard library
from typing import Tuple
import random

# 3rd party packages


# Local source
from domain.crossover.strategy import ICrossoverStrategy
from domain.crossover.factory import CrossoverFactory
from domain.individual import Individual
from domain.root_individual import RootIndividual


@CrossoverFactory.register('two_point')
class TwoPointCross(ICrossoverStrategy):

    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, root_individual: RootIndividual,
                  **kwargs) -> Tuple[Individual, Individual]:
        """Execute two-point crossover."""
        idx = random.sample(len(parent1.params), 2)
        smaller_id = min(idx)
        bigger_id = max(idx)
        child1 = Individual(parent1.params[:smaller_id] + parent2.params[smaller_id:bigger_id]
                            + parent1.params[bigger_id:], root_individual=root_individual)
        child2 = Individual(parent2.params[:smaller_id] + parent1.params[smaller_id:bigger_id]
                            + parent2.params[bigger_id:], root_individual=root_individual)
        return child1, child2
