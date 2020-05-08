#!/usr/bin/env python

"""Module with class to initialize first generation ReaxFF genetic algorithm by applying mutations to
a reference training case, then output the corresponding first generation of individuals using the population
repository.

PROBLEM: Reference training set must also contain 'fort.99', which is an output file from ReaxFF optimization.
The first generation really shouldn't require the user to already have a fort.99 file; this should be revised in the
future.
"""

# Standard library
from typing import List

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.use_case.port.population_repository import IPopulationRepository
from parametrization_clean.use_case.port.settings_repository import IAllSettings


class PopulationInitializer(object):

    def __init__(self, population_repository: IPopulationRepository, settings_repository: IAllSettings):
        self.population_repository = population_repository
        self.strategy = settings_repository.strategy_settings.initialization_strategy
        self.population_size = settings_repository.ga_settings.population_size
        self.mutation_settings_dict = vars(settings_repository.mutation_settings)

    def execute(self) -> List[Individual]:
        # TODO: Population initialization shouldn't require dft energies, weights (ex. info from fort.99)
        root_individual = self.population_repository.get_root_individual()
        individual = Individual.from_root_individual(root_individual)
        population = [self.strategy.mutation(individual, root_individual, **self.mutation_settings_dict)
                      for _ in range(self.population_size)]
        return population
