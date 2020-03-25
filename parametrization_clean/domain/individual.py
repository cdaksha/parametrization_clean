#!/usr/bin/env python

"""Module with class to structure/maintain genetic_algorithm population's "individual"/child."""

# Standard library
from typing import List

# 3rd party packages

# Local source


class Individual(object):

    def __init__(self, params: List[float], cost: float):
        """Individual/Case in the Genetic Algorithm. Unique based on its own params and corresponding ffield.
        Used as a data storage object.

        :param params: list containing param values mapped from param_keys to ffield.
        :param cost: Floating point total error of the Individual.
        """
        self.params = params
        self.cost = cost
