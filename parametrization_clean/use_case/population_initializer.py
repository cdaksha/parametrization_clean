#!/usr/bin/env python

"""Module with class to initialize first generation ReaxFF genetic algorithm by applying mutations to
a reference training case.
"""

# Standard library
from typing import List

# 3rd party packages

# Local source
from domain.individual import Individual
from use_case.port.population_repository import IPopulationRepository
from use_case.port.settings_repository import IAllSettings


class PopulationInitializer(object):

    def __init__(self, population_repository: IPopulationRepository, settings_repository: IAllSettings):
        self.population_repository = population_repository
        self.strategy = settings_repository.strategy_settings.initialization_strategy
        self.population_size = settings_repository.ga_settings.population_size
        self.mutation_settings_dict = vars(settings_repository.mutation_settings)

    def execute(self) -> List[Individual]:
        root_individual = self.population_repository.get_root_individual()
        individual = Individual.from_root_individual(root_individual)
        population = [self.strategy.mutation(individual, root_individual, **self.mutation_settings_dict)
                      for _ in range(self.population_size)]
        return population
