#!/usr/bin/env python

"""
Module that contains interface for mutation methods to be used for an Individual.

__author__ = "Chad Daksha"
"""

# Standard library
import abc

# 3rd party packages


# Local source
from domain.individual import Individual
from domain.root_individual import RootIndividual


class IMutationStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def mutation(parent: Individual, root_individual: RootIndividual, **kwargs) -> Individual:
        raise NotImplementedError
