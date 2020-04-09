#!/usr/bin/env python

"""
Module that contains interface for repository used to get data on Individual(s) in the genetic pool.

__author__ = "Chad Daksha"
"""

# Standard library
import abc
from typing import List

# 3rd party packages


# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.individual import Individual


class IPopulationRepository(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_root_individual(self) -> RootIndividual:
        """Get RootIndividual data from reference training set."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_population(self, generation_number: int) -> List[Individual]:
        """Get population of Individuals based on the (unique) generation number."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_populations(self, lower_bound: int, upper_bound: int) -> List[Individual]:
        """Get a population of Individuals from generations between generation numbers of
        [lower_bound, upper_bound) (lower bound inclusive, upper bound exclusive).
        """
        raise NotImplementedError
