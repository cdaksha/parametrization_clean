#!/usr/bin/env python

"""Module with class to structure/maintain genetic_algorithm population's "individual"/child."""

# Standard library
from typing import List
from copy import deepcopy

# 3rd party packages

# Local source
from parametrization_clean.domain.root_individual import RootIndividual
from parametrization_clean.domain.cost.strategy import IErrorStrategy
from parametrization_clean.domain.cost.reax_error import ReaxError
from parametrization_clean.domain.helpers import set_param


class Individual(object):

    def __init__(self, params: List[float], reax_energies: List[float] = None,
                 root_individual: RootIndividual = None, error_calculator: IErrorStrategy = ReaxError):
        """Individual/Case in the Genetic Algorithm. Unique based on its own params and corresponding ffield.
        Used as a data storage object.

        :param params: List containing param values mapped from param_keys to ffield.
        :param reax_energies: List containing ReaxFF energies (output from fort.99).
        :param root_individual: Root individual with stored dft energies, weights, etc.
        :param error_calculator: Strategy for error computation. Default = ReaxFF Error.
        """
        self.params = params
        self.reax_energies = reax_energies

        self.ffield = self.update_ffield(root_individual) if root_individual else None
        self.cost = self.total_error(root_individual, error_calculator) if root_individual else None

    def total_error(self, root_individual: RootIndividual, error_calculator: IErrorStrategy) -> float:
        return sum(error_calculator.error(reax_val, dft_val, weight) for reax_val, dft_val, weight in
                   zip(self.reax_energies, root_individual.dft_energies, root_individual.weights))

    def update_ffield(self, root_individual: RootIndividual):
        """Update Individual's `ffield` based on `params` (to be used after mutation/mating).
        Uses deepcopy to avoid mutating the original root ffield.
        """
        ffield = deepcopy(root_individual.root_ffield)
        for param, key in zip(self.params, root_individual.param_keys):
            set_param(key, ffield, param)
        return ffield

    # For usage of min/max functions in determining best/worst individuals
    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost
