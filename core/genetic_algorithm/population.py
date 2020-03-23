#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that uses Child to create a Population.

__author__ = "Chad Daksha"
"""

# Standard library
from typing import List
import random
import statistics

# 3rd party packages


# Local source
from core.genetic_algorithm.individual import Individual
from core.genetic_algorithm.cost import PopulationCost
from core.genetic_algorithm.selection import get_selection
from core.genetic_algorithm.helpers import sort_by_cost
from core.genetic_algorithm.crossovers import get_cross
from core.genetic_algorithm.mutations import get_mutation, central_uniform_mutate, nakata_mutate
from core.genetic_algorithm.adaptive import get_adaptation
from core.settings.config import settings as s


class Population(object):

    def __init__(self):
        """Class to initialize/propagate populations in the genetic pool.

        NOTE: if `scale` is not provided, `child.param_increments` will be used.

        Parameters
        ----------
        """
        self.pop_size = s.GA.populationSize

        # GA operations
        self.selection = get_selection(s.GA.selection.algorithm)
        self.crossover = get_cross(s.GA.crossover.algorithm)
        self.mutation = get_mutation(s.GA.mutation.algorithm)
        self.adaptation = get_adaptation(s.GA.adaptation.algorithm)

        self.mutation_rate = [s.GA.mutation.probability, s.GA.mutation.probability]
        self.crossover_rate = s.GA.crossover.probability

        # Birth place of all children
        self._root = Individual()

    def initialize(self) -> List[Individual]:
        """Create initial population of genetic pool by mutating `_root`, the birth place of all children."""
        # return [central_uniform_mutate(self._root) for _ in range(self.pop_size)]
        return [central_uniform_mutate(self._root) for _ in range(self.pop_size)]

    def propagate(self, parents: List[Individual]) -> List[Individual]:
        """Create next generation using parents.

        (frac_select * pop_size)                  = # parents tournament selected
        (cross_rate * pop_size)                   = # crossovers applied to selected parents
        (1 - frac_select - cross_rate) * pop_size = # mutations applied to selected parents
        """
        # Performing Adaptive GA
        costs = [parent.cost for parent in parents]
        median_cost = statistics.median(costs)
        mean_cost = sum(costs) / len(parents)
        min_cost = min(costs)
        max_cost = max(costs)
        print("Min Cost: {}\tMean Cost: {}\tMedian Cost: {}\tMax Cost: {}"
              .format(min_cost, mean_cost, median_cost, max_cost))
        print("Original crossover and mutation rates: {} and {}".format(self.crossover_rate, self.mutation_rate))
        # TODO: lei_adapt and srinivas_adapt have different function signatures...cannot use same lines of code!
        # Lei Adaptive GA
        # if s.adaptation:  # adaptive GA is turned on
        #     cross_rate, mutation_rate = self.adaptation(mean_cost, max_cost, min_cost)
        #     self.crossover_rate = cross_rate
        #     self.mutation_rate = [mutation_rate, mutation_rate]
        # print("Adapted crossover and mutation rates: {} and {}".format(self.crossover_rate, self.mutation_rate))

        # SUS Sampling
        # linear_SUS_selection = get_adaptation('linear_SUS')
        # selected_parents = linear_SUS_selection(parents)
        # shuffled_selected_parents = random.shuffle(selected_parents)
        # counter = 0

        # Building next generation
        # children = []
        children = sort_by_cost(parents)[0:2]  # Elitism - preserve best two parents for next generation
        count_same_parents_used = 0
        while len(children) < self.pop_size:
            parent1 = self.selection(parents)
            parent2 = self.selection(parents)
            # parent1 = shuffled_selected_parents[counter]
            # counter += 1
            # parent2 = shuffled_selected_parents[counter]
            # counter += 1

            # Debugging
            if parent1.params == parent2.params:
                count_same_parents_used += 1

            # Srinivas or Xiao Adaptive GA (individual-based)
            if s.adaptation:
                cross_rate, mutation_rates = self.adaptation(median_cost, min_cost, (parent1.cost, parent2.cost))
                self.crossover_rate = cross_rate
                self.mutation_rate = mutation_rates
                print('Adapted: cross {}, mutate {}'.format(cross_rate, mutation_rates))

            if random.random() < self.crossover_rate:  # Crossover rate should be high, ~0.65-0.85
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            if random.random() < self.mutation_rate[0]:  # Mutation rate should be low
                child1 = self.mutation(child1)
            if random.random() < self.mutation_rate[1]:
                child2 = self.mutation(child2)

            children.append(child1)
            children.append(child2)

        print("NUMBER OF TIMES THAT SAME PARENTS USED FOR CROSSOVER/MUTATION: {}/100".format(count_same_parents_used))

        return children
