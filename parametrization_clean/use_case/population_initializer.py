#!/usr/bin/env python

"""Module with class to initialize first generation ReaxFF genetic algorithm by applying mutations to
a reference training case.
"""

# Standard library
from typing import List

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.mutation.strategy import IMutationStrategy
from parametrization_clean.use_case.port.population_repository import IPopulationRepository


class PopulationInitializer(object):

    def __init__(self, repository: IPopulationRepository, initialization_strategy: IMutationStrategy):
        self.repository = repository
        self.strategy = initialization_strategy

    def execute(self, population_size, **kwargs) -> List[Individual]:
        root_individual = self.repository.get_root_individual()
        individual = Individual.from_root_individual(root_individual)
        return [self.strategy.mutation(individual, **kwargs) for _ in range(population_size)]
