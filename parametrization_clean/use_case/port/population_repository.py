#!/usr/bin/env python

"""Module that contains interface for repository used to get data on Individual(s) in the genetic pool.
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
    def get_previous_n_populations(self, num_populations: int) -> List[Individual]:
        """Read previous N populations relative to the current generation.
        Precaution - ensure that lowest possible generation number is zero.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_individual(self, individual: Individual, **kwargs):
        """Write an individual in the population to the repository."""
        raise NotImplementedError

    @abc.abstractmethod
    def write_population(self, population: List[Individual], generation_number):
        """Write a population of individuals to the repository."""
        raise NotImplementedError
