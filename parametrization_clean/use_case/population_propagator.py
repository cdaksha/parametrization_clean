#!/usr/bin/env python

"""Combines domain logic to propagate (generational) genetic algorithm. To be used if the generation number
is greater than one. Applies selection, crossover, mutation, and adaptation operators on the parents (individuals
from the previous generation) to generate better offspring/children (individuals for the next generation).
"""

# Standard library
from typing import List, Tuple
import random
import statistics

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.use_case.port.settings_repository import IAllSettings
from parametrization_clean.use_case.port.population_repository import IPopulationRepository


class PopulationPropagator:

    def __init__(self, settings_repository: IAllSettings, population_repository: IPopulationRepository):
        self.ga_settings = settings_repository.ga_settings

        self.crossover_rate = self.ga_settings.crossover_rate
        self.mutation_rates = [self.ga_settings.mutation_rate, self.ga_settings.mutation_rate]

        self.mutation_strategy = settings_repository.strategy_settings.mutation_strategy
        self.crossover_strategy = settings_repository.strategy_settings.crossover_strategy
        self.selection_strategy = settings_repository.strategy_settings.selection_strategy
        self.adaptation_strategy = settings_repository.strategy_settings.adaptation_strategy

        self.ga_settings_dict = vars(settings_repository.ga_settings)
        self.mutation_settings_dict = vars(settings_repository.mutation_settings)
        self.crossover_settings_dict = vars(settings_repository.crossover_settings)
        self.selection_settings_dict = vars(settings_repository.selection_settings)
        self.adaptation_settings_dict = vars(settings_repository.adaptation_settings)

        self.root_individual = population_repository.get_root_individual()

    def execute(self, parents: List[Individual]) -> List[Individual]:
        """Create next generation using parents."""
        average_cost, minimum_cost = self.compute_statistics(parents)

        children = self.initialize(parents)
        while len(children) < self.ga_settings.population_size:
            parent1 = self.select(parents)
            parent2 = self.select(parents)

            self.adapt_cross_and_mutate_rates(average_cost, minimum_cost, (parent1.cost, parent2.cost))

            child1, child2 = self.cross(parent1, parent2)
            child1, child2 = self.mutate(child1, child2)

            children.extend([child1, child2])

        return children

    def initialize(self, parents: List[Individual]) -> List[Individual]:
        children = []
        if self.ga_settings.use_elitism:
            children.extend(sorted(parents)[0:2])
        return children

    def select(self, parents: List[Individual]) -> Individual:
        return self.selection_strategy.selection(parents, **self.selection_settings_dict)

    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        child1, child2 = parent1, parent2
        if random.random() < self.crossover_rate:
            child1, child2 = self.crossover_strategy.crossover(parent1, parent2, self.root_individual,
                                                               **self.crossover_settings_dict)
        return child1, child2

    def mutate(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        child1, child2 = parent1, parent2
        if random.random() < self.mutation_rates[0]:
            child1 = self.mutation_strategy.mutation(parent1, self.root_individual, **self.mutation_settings_dict)
        if random.random() < self.mutation_rates[1]:
            child2 = self.mutation_strategy.mutation(parent2, self.root_individual, **self.mutation_settings_dict)
        return child1, child2

    def adapt_cross_and_mutate_rates(self, average_cost: float, minimum_cost: float, parent_costs: Tuple[float, float]):
        if self.ga_settings.use_adaptation:
            self.crossover_rate, self.mutation_rates = \
                self.adaptation_strategy.adaptation(average_cost, minimum_cost, parent_costs,
                                                    **self.adaptation_settings_dict, **self.ga_settings_dict)

    @staticmethod
    def compute_statistics(population: List[Individual]) -> Tuple[float, float]:
        costs = [individual.cost for individual in population]
        return statistics.mean(costs), min(costs)
