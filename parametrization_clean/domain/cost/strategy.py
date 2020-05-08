#!/usr/bin/env python

"""Module with interface for computation of objective function in evaluating the fitness of individuals
in the genetic algorithm.
New error calculation/objective function strategies can be added as classes,
so long as they implement the abstraction presented here.
"""

# Standard library
import abc

# 3rd party packages

# Local source


class IErrorStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def error(reax_val, dft_val, weight, **kwargs) -> float:
        raise NotImplementedError
