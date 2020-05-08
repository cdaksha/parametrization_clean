#!/usr/bin/env python

"""Module with interface for selection operators to be performed on Individuals.
New selection strategies can be added as classes, so long as they implement the abstraction presented here.
"""

# Standard library
import abc
from typing import List

# 3rd party packages

# Local source
from parametrization_clean.domain.individual import Individual


class ISelectionStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def selection(population: List[Individual], **kwargs) -> Individual:
        raise NotImplementedError
