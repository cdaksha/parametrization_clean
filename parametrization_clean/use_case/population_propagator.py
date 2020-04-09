#!/usr/bin/env python

"""Combines domain logic to propagate (generational) genetic algorithm."""

# Standard library
from typing import List, Tuple
import random

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.use_case.port.settings_repository import IAllSettings


class PopulationPropagator:

    def __init__(self, settings_repository: IAllSettings):
        self.ga_settings = settings_repository.ga_settings

        self.mutation_strategy = settings_repository.strategy_settings.mutation_strategy
        self.crossover_strategy = settings_repository.strategy_settings.crossover_strategy
        self.selection_strategy = settings_repository.strategy_settings.selection_strategy

        self.mutation_settings_dict = vars(settings_repository.mutation_settings)
        self.crossover_settings_dict = vars(settings_repository.crossover_settings)
        self.selection_settings_dict = vars(settings_repository.selection_settings)

    def execute(self, parents: List[Individual]) -> List[Individual]:
        """Create next generation using parents."""

        children = self.initialize(parents)
        while len(children) < self.ga_settings.population_size:
            parent1 = self.select(parents)
            parent2 = self.select(parents)

            child1, child2 = self.cross(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            children.extend([child1, child2])

        return children

    def initialize(self, parents: List[Individual]) -> List[Individual]:
        children = []
        if self.ga_settings.elitism:
            children.extend(sorted(parents)[0:2])
        return children

    def select(self, parents: List[Individual]) -> Individual:
        return self.selection_strategy.selection(parents, **self.selection_settings_dict)

    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        child1, child2 = parent1, parent2
        if random.random() < self.ga_settings.crossover_rate:
            child1, child2 = self.crossover_strategy.crossover(parent1, parent2,
                                                               **self.crossover_settings_dict)
        return child1, child2

    def mutate(self, parent: Individual) -> Individual:
        child = parent
        if random.random() < self.ga_settings.mutation_rate:
            child = self.mutation_strategy.mutation(parent, **self.mutation_settings_dict)
        return child
