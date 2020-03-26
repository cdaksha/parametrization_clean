#!/usr/bin/env python

"""
Module that contains interface for crossover methods to be used for an Individual.

__author__ = "Chad Daksha"
"""

# Standard library
import abc
from typing import Tuple

# 3rd party packages


# Local source
from parametrization_clean.domain.individual import Individual


class ICrossoverStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def crossover(parent1: Individual, parent2: Individual, **kwargs) -> Tuple[Individual, Individual]:
        raise NotImplementedError
