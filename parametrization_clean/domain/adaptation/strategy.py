#!/usr/bin/env python

"""
Module that contains interface for adaptive genetic algorithm methods to be used for an Individual.
New adaptation strategies can be added as classes, so long as they implement the abstraction presented here.

__author__ = "Chad Daksha"
"""

# Standard library
import abc
from typing import List, Tuple

# 3rd party packages

# Local source


class IAdaptationStrategy(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def adaptation(average_cost: float, minimum_cost: float, parent_costs: Tuple[float, float], **kwargs) \
            -> Tuple[float, List[float]]:
        raise NotImplementedError
