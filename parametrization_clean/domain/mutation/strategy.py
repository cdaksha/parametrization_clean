#!/usr/bin/env python

"""
Module that contains interface for mutation methods to be used for an Individual.
New mutation strategies can be added as classes, so long as they implement the abstraction presented here.

__author__ = "Chad Daksha"
"""

# Standard library
import abc

# 3rd party packages


# Local source
from parametrization_clean.domain.individual import Individual
from parametrization_clean.domain.root_individual import RootIndividual


class IMutationStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        raise NotImplementedError
