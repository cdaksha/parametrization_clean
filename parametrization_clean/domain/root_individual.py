#!/usr/bin/env python

"""Module with immutable structure to store reference training set information.
Potentially includes DFT energies, weights, parameter boundaries (low, high)."""

# Standard library
from typing import List, Dict


# 3rd party packages

# Local source
from parametrization_clean.domain.utils.helpers import get_param


class Borg(object):
    """Avoid Singleton anti-pattern, but ensure shared state (if many instances are created)."""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class RootIndividual(Borg):

    def __init__(self, dft_energies: List[float], weights: List[float],
                 root_ffield: Dict[int, List], param_keys: List[List[int]]):
        """Root individual to store data from reference training set that needs to be stored only once.
        For example, the weights of the error, DFT energies, parameter bounds only need to be stored once.
        Input structures are converted to tuples to prevent mutation.

        Parameters
        ----------
        dft_energies: List[float]
            Stored as a tuple containing DFT energies; used to compute error.
        weights: List[float]
            Stored as a tuple containing weights, used to compute error.
        root_ffield: Dict[int, List]
            Dictionary mapping ReaxFF force field section number to parameters contained in that section.
        param_keys: List[List[int]]
            List of keys mapping to values in the ffield object.
        """
        super().__init__()
        self.dft_energies = tuple(dft_energies)
        self.weights = tuple(weights)
        self.root_ffield = root_ffield
        self.param_keys = param_keys

        # Root parameters from the reference training set
        self.root_params = self.extract_params()

    def extract_params(self) -> List[float]:
        return [get_param(key, self.root_ffield) for key in self.param_keys]


class FirstGenerationRootIndividual(RootIndividual):

    def __init__(self, root_ffield: Dict[int, List], param_keys: List[List[int]]):
        """Root individual for the first generation. Does NOT require DFT energies or weights, as they are not
        necessary for population initialization.

        Parameters
        ----------
        root_ffield: Dict[int, List]
            Dictionary mapping ReaxFF force field section number to parameters contained in that section.
        param_keys: List[List[int]]
            List of keys mapping to values in the ffield object.
        """
        super().__init__(dft_energies=[], weights=[], root_ffield=root_ffield, param_keys=param_keys)
