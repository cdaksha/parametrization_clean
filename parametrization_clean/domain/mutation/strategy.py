#!/usr/bin/env python

"""
Module that contains interface for mutation methods to be used for an Individual.

__author__ = "Chad Daksha"
"""

# Standard library
import abc

# 3rd party packages


# Local source
from parametrization_clean.domain.individual import Individual


class IMutationStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def mutation(parent: Individual, **kwargs) -> Individual:
        raise NotImplementedError
