#!/usr/bin/env python

"""Generate output for individuals in the genetic algorithm."""

# Standard library

# 3rd party packages

# Local source
from parametrization_clean.use_case.port.population_repository import IPopulationRepository


class PopulationWriter:

    def __init__(self, population_repository: IPopulationRepository):
        self.population_repository = population_repository

    def write_individual(self, individual, **kwargs):
        return self.population_repository.write_individual(individual, **kwargs)

    def write_population(self, population, generation_number):
        return self.population_repository.write_population(population, generation_number)
