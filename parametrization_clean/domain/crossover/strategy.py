#!/usr/bin/env python

"""
Module that contains interface for crossover methods to be used for an Individual.
New crossover strategies can be added as classes, so long as they implement the abstraction presented here.

__author__ = "Chad Daksha"
"""

# Standard library
import abc
from typing import Tuple

# 3rd party packages


# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class ICrossoverStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def crossover(parent1: Individual, parent2: Individual, root_individual: RootIndividual,
                  **kwargs) -> Tuple[Individual, Individual]:
        raise NotImplementedError
